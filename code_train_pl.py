import torch
import torch.utils.data
from torch import nn, optim
from torchvision import *
import socket
import os
import time
from datetime import datetime
import numpy as np
import argparse
from args.setup import set_logger, set_trainer, parse_args
# from tensorboardX import SummaryWriter
from utils_funcs import set_seed, ToFloat
import utils.torch.modules as modules
import utils.torch.rand as random
import pytorch_lightning as pl
from dataset import get_indices, CodesNpzDataset
from torchmetrics import MeanMetric
import wandb
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from datasets.DataModule import CodesModule
class Model(pl.LightningModule):
    def __init__(self, args, xs=(3, 32, 32), nz=1, zchannels=16, nprocessing=1, kernel_size=3, resdepth=2,
                 reswidth=256, dropout_p=0., tag=''):
        super().__init__()
        # default: disable compressing mode
        # if activated, tensors will be flattened
        self.compressing = False

        # hyperparameters
        self.xs = xs
        self.nz = nz
        self.zchannels = zchannels
        self.nprocessing = nprocessing
        # latent height/width is always 16,
        # the number of channels depends on the dataset
        # zdim has CxH/2 x H/2
        self.zdim = (self.zchannels, int(xs[1]/2), int(xs[1]/2))
        self.resdepth = resdepth
        self.reswidth = reswidth
        self.kernel_size = kernel_size

        # apply these two factors (i.e. on the ELBO) in sequence and it results in "bits/dim"
        # factor to convert "nats" to bits
        self.bitsscale = np.log2(np.e)
        # factor to divide by the data dimension
        self.perdimsscale = 1. / np.prod(self.xs)

        # calculate processing layers convolutions options
        # kernel/filter is 5, so in order to ensure same-size outputs, we have to pad by 2
        padding_proc = (5 - 1) / 2
        assert padding_proc.is_integer()
        padding_proc = int(padding_proc)

        # calculate other convolutions options
        padding = (self.kernel_size - 1) / 2
        assert padding.is_integer()
        padding = int(padding)

        # create loggers
        self.tag = tag


        # set-up current "best elbo"
        self.best_elbo = np.inf

        # distribute ResNet blocks over latent layers
        resdepth = [0] * (self.nz)
        i = 0
        for _ in range(self.resdepth):
            i = 0 if i == (self.nz) else i
            resdepth[i] += 1
            i += 1

        # reduce initial variance of distributions corresponding
        # to latent layers if latent nz increases
        scale = 1.0 / (self.nz ** 0.5)

        # store activations
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
        self.act = nn.ELU()
        self.actresnet = nn.ELU()

        # Below we build up the main model architecture of the inference- and generative-models
        # All the architecure components are built up from different custom are existing PyTorch modules

        # <===== INFERENCE MODEL =====>
        # the bottom (zi=1) inference model
        self.infer_in = nn.Sequential(
            # shape: [1,32,32] -> [4,16,16]
            # shape: [1,4,4] => [4,2,2]
            modules.Squeeze2d(factor=2),

            # shape: [4,2,2] -> [32,1,1]
            modules.WnConv2d(4 * xs[0],
                             self.reswidth,
                             5,
                             1,
                             padding_proc,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )
        self.infer_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.infer_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        # shape: [32,16,16] -> [1,16,16]
        self.infer_mu = modules.WnConv2d(self.reswidth,
                                         self.zchannels,
                                         self.kernel_size,
                                         1,
                                         padding,
                                         init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # shape: [32,16,16] -> [1,16,16]
        self.infer_std = modules.WnConv2d(self.reswidth,
                                          self.zchannels,
                                          self.kernel_size,
                                          1,
                                          padding,
                                          init_scale=scale if self.nz > 1 else 2 ** 0.5 * scale)

        # <===== DEEP INFERENCE MODEL =====>
        # the deeper (zi > 1) inference models
        self.deepinfer_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepinfer_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepinfer_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        self.deepinfer_std = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale if i < self.nz - 2 else 2 ** 0.5 * scale)
            )
            for i in range(self.nz - 1)])

        # <===== DEEP GENERATIVE MODEL =====>
        # the deeper (zi > 1) generative models
        self.deepgen_in = nn.ModuleList([
            # shape: [1,16,16] -> [32,16,16]
            nn.Sequential(
                modules.WnConv2d(self.zchannels,
                                 self.reswidth,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=1.0,
                                 loggain=True),
                self.act
            )
            for _ in range(self.nz - 1)])

        self.deepgen_res = nn.ModuleList([
            # shape: [32,16,16] -> [32,16,16]
            nn.Sequential(
                modules.ResNetBlock(self.reswidth,
                                    self.reswidth,
                                    self.kernel_size,
                                    1,
                                    padding,
                                    resdepth[i + 1],
                                    dropout_p,
                                    self.actresnet),
                self.act
            ) if resdepth[i + 1] > 0 else modules.Pass()
            for i in range(self.nz - 1)])

        self.deepgen_mu = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding,
                                 init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        self.deepgen_std = nn.ModuleList([
            # shape: [32,16,16] -> [1,16,16]
            nn.Sequential(
                modules.WnConv2d(self.reswidth,
                                 self.zchannels,
                                 self.kernel_size,
                                 1,
                                 padding, init_scale=scale)
            )
            for _ in range(self.nz - 1)])

        # <===== GENERATIVE MODEL =====>
        # the bottom (zi = 1) inference model
        self.gen_in = nn.Sequential(
            # shape: [1,16,16] -> [32,16,16]
            modules.WnConv2d(self.zchannels,
                             self.reswidth,
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=1.0,
                             loggain=True),
            self.act
        )

        self.gen_res1 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                self.kernel_size,
                                1,
                                padding,
                                resdepth[0],
                                dropout_p,
                                self.actresnet),
            self.act
        ) if resdepth[0] > 0 else modules.Pass()

        self.gen_res0 = nn.Sequential(
            # shape: [32,16,16] -> [32,16,16]
            modules.ResNetBlock(self.reswidth,
                                self.reswidth,
                                5,
                                1,
                                padding_proc,
                                self.nprocessing,
                                dropout_p,
                                self.actresnet),
            self.act
        ) if self.nprocessing > 0 else modules.Pass()

        self.gen_mu = nn.Sequential(
            # shape: [32,16,16] -> [4,16,16]
            modules.WnConv2d(self.reswidth,
                             4 * xs[0],
                             self.kernel_size,
                             1,
                             padding,
                             init_scale=0.1),
            # shape: [4,16,16] -> [1,32,23]
            modules.UnSqueeze2d(factor=2)
        )

        # the scale parameter of the bottom (zi = 1) generative model is modelled unconditional
        self.gen_std = nn.Parameter(torch.Tensor(*self.xs))
        nn.init.zeros_(self.gen_std)

        self.args = args
        self.root_process = True
        if os.getenv("LOCAL_RANK", 0) != 0:
            self.root_process = False
        if self.root_process:
            print('root  process')
        self.elbo_metric = MeanMetric(dist_sync_on_step=True)
        # ema is not really necessary for training
        # ema = modules.EMA(0.999)
        # with torch.no_grad():
        #     for name, param in self.named_parameters():
        #         # only parameters optimized using gradient-descent are relevant here
        #         if param.requires_grad:
        #             # register (1) parameters
        #             ema.register_ema(name, param.data)
        #             # register (2) parameters
        #             ema.register_default(name, param.data)
        # # not really working
        # self.ema = ema


    # function to set the model to compression mode
    def compress(self, compress=True):
        self.compressing = compress

    # function that only takes in the layer number and returns a distribution based on that
    def infer(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            # for bot is Bx1x8x8
            h = given

            # if compressing, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()

            # bottom latent layer
            if i == 0:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.xs)
                # also, when NOT compressing, the input is not scaled from [0,255] to [-1,1]
                # [0,511] [-1,1]
                else:
                    h = (h - 255.5) / 255.5
                    # h = (h - 127.5) / 127.5

                # input convolution
                h = self.infer_in(h)

                # processing ResNet blocks
                # for bot Bx64x4x4
                h = self.infer_res0(h)

                # other ResNet blocks
                h = self.infer_res1(h)

                # mu parameter of the conditional Logistic distribution
                mu = self.infer_mu(h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.infer_std(h) + 2.)

            # deeper latent layers
            else:
                # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
                if self.compressing:
                    h = h.view((-1,) + self.zdim)

                # input convolution
                h = self.deepinfer_in[i - 1](h)

                # other ResNet blocks
                h = self.deepinfer_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepinfer_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * self.sigmoid(self.deepinfer_std[i - 1](h) + 2.)

            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.zdim)).type(type)
                scale = scale.view(np.prod(self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that only takes in the layer number and returns a distribution based on that
    def generate(self, i):
        # nested function that takes in the "given" value of the conditional Logistic distribution
        # and returns the mu and scale parameters of that distribution
        def distribution(given):
            h = given

            # if compressing, the input is flattened, so we'll have to convert it back to a Tensor
            # also, the input might not be float32, so we'll have to convert it first
            if self.compressing:
                type = h.type()
                h = h.float()
                h = h.view((-1,) + self.zdim)

            # bottom latent layer
            if i == 0:
                # input convolution
                h = self.gen_in(h)

                # processing ResNet blocks
                h = self.gen_res1(h)

                # other ResNet blocks
                h = self.gen_res0(h)

                # mu parameter of the conditional Logistic distribution

                mu = self.gen_mu(h)

                # scale parameter of the conditional Logistic distribution
                # set a minimal value for the scale parameter of the bottom generative model
                # it doesn' really matter, just a minimum value
                scale = ((2. / 511.) / 8.) + modules.softplus(self.gen_std)
                # scale = ((2. / 255.) / 8.) + modules.softplus(self.gen_std)

            # deeper latent layers
            else:
                # input convolution
                h = self.deepgen_in[i - 1](h)

                # other ResNet blocks
                h = self.deepgen_res[i - 1](h)

                # mu parameter of the conditional Logistic distribution
                mu = self.deepgen_mu[i - 1](h)

                # scale parameter of the conditional Logistic distribution
                # clamp the output of the scale parameter between [0.1, 1.0] for stability
                scale = 0.1 + 0.9 * modules.softplus(self.deepgen_std[i - 1](h) + np.log(np.exp(1.) - 1.))


            if self.compressing:
                # if compressing, the "batch-size" can only be 1
                assert mu.shape[0] == 1

                # flatten the Tensors back and convert back to the input datatype
                mu = mu.view(np.prod(self.xs if i == 0 else self.zdim)).type(type)
                scale = scale.view(np.prod(self.xs if i == 0 else self.zdim)).type(type)
            return mu, scale

        return distribution

    # function that takes as input the data and outputs all the components of the ELBO + the latent samples
    def loss(self, x):
        # say x is the size of unsqueeze codes Bx1xHxW, for bot 8x8
        # tensor to store inference model losses
        # [nz, B, C]
        logenc = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the generative model losses
        logdec = torch.zeros((self.nz, x.shape[0], self.zdim[0]), device=x.device)

        # tensor to store the latent samples
        zsamples = torch.zeros((self.nz, x.shape[0], np.prod(self.zdim)), device=x.device)

        for i in range(self.nz):
            # inference model
            # get the parameters of inference distribution i given x (if i == 0) or z (otherwise)
            mu, scale = self.infer(i)(given=x if i == 0 else z)

            # sample untransformed sample from Logistic distribution (mu=0, scale=1)
            eps = random.logistic_eps(mu.shape, device=mu.device)
            # reparameterization trick: transform using obtained parameters
            z_next = random.transform(eps, mu, scale)

            # store the inference model loss
            # mu and scale for bot are all 4x4
            # nz xB x16 or to say nz xB x4x4
            zsamples[i] = z_next.flatten(1)
            logq = torch.sum(random.logistic_logp(mu, scale, z_next), dim=2)
            logenc[i] += logq

            # generative model
            # get the parameters of inference distribution i given z
            # mu 8x8; scale 4x4
            mu, scale = self.generate(i)(given=z_next)

            # store the generative model loss
            if i == 0:
                # if bottom (zi = 1) generative model, evaluate loss using discretized Logistic distribution
                logp = torch.sum(random.discretized_logistic_logp(mu, scale, x), dim=1)
                logrecon = logp

            else:
                logp = torch.sum(random.logistic_logp(mu, scale, x if i == 0 else z), dim=2)
                logdec[i - 1] += logp

            z = z_next

        # store the prior loss
        logp = torch.sum(random.logistic_logp(torch.zeros(1, device=x.device), torch.ones(1, device=x.device), z), dim=2)
        logdec[self.nz - 1] += logp

        # convert from "nats" to bits
        logenc = torch.mean(logenc, dim=1) * self.bitsscale
        logdec = torch.mean(logdec, dim=1) * self.bitsscale
        logrecon = torch.mean(logrecon) * self.bitsscale
        return logrecon, logdec, logenc, zsamples

    # function to sample from the model (using the generative model)
    def sample(self, device, epoch, num=64):
        # sample "num" latent variables from the prior
        z = random.logistic_eps(((num,) + self.zdim), device=device)

        # sample from the generative distribution(s)
        for i in reversed(range(self.nz)):
            mu, scale = self.generate(i)(given=z)
            eps = random.logistic_eps(mu.shape, device=device)
            z_prev = random.transform(eps, mu, scale)
            z = z_prev

        # scale up from [-1,1] to [0,255]
        # [-1,1] to [0, 511]
        # x_cont = (z * 127.5) + 127.5
        x_cont = (z * 255.5) + 255.5

        # ensure that [0,255]
        # ensure that  [0, 511]
        # x = torch.clamp(x_cont, 0, 255)
        x = torch.clamp(x_cont, 0, 511)
        # scale from [0,255] to [0,1] and convert to right shape
        # scale from [0,511] to [0,1] and convert to right shape
        # x_sample = x.float() / 255.
        x_sample = x.float() / 511.
        x_sample = x_sample.view((num,) + self.xs)

        # make grid out of "num" samples
        x_grid = utils.make_grid(x_sample)

        # log
        # self.logger.add_image('x_sample', x_grid, epoch)

    # function to sample a reconstruction of input data
    def reconstruct(self, x_orig, device, epoch):
        # take only first 32 datapoints of the input
        # otherwise the output image grid may be too big for visualization
        x_orig = x_orig[:32, :, :, :].to(device)

        # sample from the bottom (zi = 1) inference model
        mu, scale = self.infer(0)(given=x_orig)
        eps = random.logistic_eps(mu.shape, device=device)
        z = random.transform(eps, mu, scale)  # sample zs

        # sample from the bottom (zi = 1) generative model
        mu, scale = self.generate(0)(given=z)
        x_eps = random.logistic_eps(mu.shape, device=device)
        x_cont = random.transform(x_eps, mu, scale)

        # scale up from [-1.1] to [0,255]
        x_cont = (x_cont * 127.5) + 127.5
        # scale up from [-1.1] to [0,511]
        x_cont = (x_cont * 255.5) + 255.5
        # esnure that [0,255]
        # x_sample = torch.clamp(x_cont, 0, 255)
        x_sample = torch.clamp(x_cont, 0, 511)
        # scale from [0,255] to [0,1] and convert to right shape
        x_sample = x_sample.float() / 511.
        x_orig = x_orig.float() / 511.

        # concatenate the input data and the sampled reconstructions for comparison
        x_with_recon = torch.cat((x_orig, x_sample))
        # make a grid out of the original data and the reconstruction samples
        x_with_recon = x_with_recon.view((2 * x_orig.shape[0],) + self.xs)
        x_grid = utils.make_grid(x_with_recon)

        # log
        # self.logger.add_image('x_reconstruct', x_grid, epoch)



    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.args.lr)
        return [optimizer]


    def training_step(self, batch, batch_idx):
        # after_zero_diff = self.backbone.get_summed_diff()
        # self.backbone.zero_buffer()

        top, bot, y = batch

        # get number of batches
        nbatches = self.trainer.num_training_batches

        # switch to parameters not affected by exponential moving average decay
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         param.data = self.ema.get_default(name).to(self.device)

        # enumerate over the batches

        if self.args.code_level == 'top':
            batch = top
        elif self.args.code_level == 'bot':
            batch = bot
        # keep track of the global step
        global_step = self.global_step

        # update the learning rate according to schedule
        opt = self.optimizers().optimizer
        for param_group in opt.param_groups:
            lr = param_group['lr']
            lr = lr_step(global_step, lr, decay=self.args.decay)
            param_group['lr'] = lr


        # evaluate the data under the model and calculate ELBO components
        logrecon, logdec, logenc, zsamples = self.loss(batch)

        # free bits technique, in order to prevent posterior collapse
        bits_pc = 1.
        tmp = torch.ones((self.nz, self.zdim[0])).cuda()
        kl = torch.sum(torch.max(-logdec + logenc, bits_pc * tmp))

        # compute the inference- and generative-model loss
        logdec = torch.sum(logdec, dim=1)
        logenc = torch.sum(logenc, dim=1)

        # construct ELBO
        elbo = -logrecon + kl

        # scale by image dimensions to get "bits/dim"
        elbo *= self.perdimsscale
        logrecon *= self.perdimsscale
        logdec *= self.perdimsscale
        logenc *= self.perdimsscale

        loss = elbo
        # take gradient step
        total_norm = nn.utils.clip_grad_norm_(self.parameters(), 1., norm_type=2)
        # # do ema update on parameters used for evaluation
        # if self.root_process:
        #     with torch.no_grad():
        #         for name, param in self.named_parameters():
        #             if param.requires_grad:
        #                 self.ema(name, param.data)

        self.elbo_metric.update(elbo.item())

        # print the average loss of the epoch to the console

        log_data = {
            'train_loss': loss.item(),
        }

        self.log_dict(log_data)
        return loss

    def training_epoch_end(self, outputs):

        # save training params, to be able to return to these values after evaluation
        # with torch.no_grad():
        #     for name, param in self.named_parameters():
        #         if param.requires_grad:
        #             self.ema.register_default(name, param.data)
        avg_loss = self.elbo_metric.compute()
        # mean metric will not call reset automatically, so do it manually
        self.elbo_metric.reset()
        if self.root_process:
            print('epoch: {}, avg_loss: {}'.format(self.current_epoch, avg_loss))



def warmup(model, device, data_loader, warmup_batches, root_process, args):
    # convert model to evaluation mode (no Dropout etc.)
    model.eval()

    # prepare initialization batch
    for batch_idx, (top, bot, _) in enumerate(data_loader):
        if args.code_level == 'top':
            image = top
        elif args.code_level == 'bot':
            image = bot
        # stack image with to current stack
        warmup_images = torch.cat((warmup_images, image), dim=0) \
            if batch_idx != 0 else image

        # stop stacking batches if reaching limit
        if batch_idx + 1 == warmup_batches:
            break

    # set the stack to current device
    # input size
    warmup_images = warmup_images.to(device)
    # print('warm image device:', warmup_images.device)
    # do one 'special' forward pass to initialize parameters
    with modules.init_mode():
        logrecon, logdec, logenc, _ = model.loss(warmup_images)



# learning rate schedule
def lr_step(step, curr_lr, decay=0.99995, min_lr=5e-4):
    # only decay after certain point
    # and decay down until minimal value
    if curr_lr > min_lr:
        curr_lr *= decay
        return curr_lr
    return curr_lr

class CodesToTensor:
    def __call__(self, codes):

        return codes/511.0

    def __repr__(self):
        return self.__class__.__name__ + '()'

def main(args):
    # hyperparameters, input from command line

    # store hyperparameters in variables

    batch_size = args.batch_size
    nz = args.nz
    zchannels = args.zchannels
    nprocessing = args.nprocessing

    blocks = args.blocks
    width = args.width

    dropout = args.dropout
    kernel = args.kernel

    assert nz > 0

    batch_size //= args.gpus
    # string-tag for logging
    tag = f'nz{nz}'
    transform_ops = transforms.Compose([ToFloat()])
    if args.code_level == 'top':
        factor = 1
    elif args.code_level == 'bot':
        factor = 2
    if args.dataset =='img64':
        base = 8
    elif args.dataset == 'cifar':
        base = 4
    elif args.dataset == 'img224':
        base = 28

    height = base * factor
    xs = (1, height, height)

    print("Load model")

    # build model from hyperparameters
    model = Model(args, xs=xs,
                  kernel_size=kernel,
                  nprocessing=nprocessing,
                  nz=nz,
                  zchannels=zchannels,
                  resdepth=blocks,
                  reswidth=width,
                  dropout_p=dropout,
                  tag=tag,
                  ).cuda()

    if args.tag == 'full':
        args.tag = ''
    save_path = f'params/code{args.tag}/nz{nz}_{args.code_level}_{args.dataset}_ckpt'

    codes_path = args.dataset_path

    # codes_ds = CodesNpzDataset(codes_path, transform=transform_ops)
    #
    # print('----- dataset loaded')
    # #
    # train_set = codes_ds
    #
    #
    codes_module = CodesModule(codes_path, args.batch_size, args.num_workers)
    codes_module.setup()
    train_set = codes_module.train_ds
    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size//args.gpus, shuffle=True, drop_last=True, num_workers=args.num_workers // args.gpus)

    device = 'cuda'
    root_process = True
    local_rank = os.getenv("LOCAL_RANK", 0)
    if local_rank != 0:
        root_process = False
    if root_process:
        print(save_path)
        os.makedirs(save_path, exist_ok=True)

    ckpt_path = None
    if args.resume:
        print('-----------load resume--------------')
        ckpt_path = os.path.join(save_path, 'last.ckpt')

    else:
        # data-dependent initialization
        warmup(model, device, train_loader, 25, root_process, args)
        del train_loader
        print("Data Dependent Initialization") if root_process else print("Data Dependent Initialization with ya!")

    root_dir = 'results'
    # wandb_logger = set_logger(args, root_dir)
    wandb_logger = None
    trainer = set_trainer(args, wandb_logger, save_path)

    trainer.fit(model, codes_module, ckpt_path=ckpt_path)

    print('--------------finish training')
    trainer.save_checkpoint(f'{save_path}/checkpoint.pt')

    # wandb.finish()

if __name__ == '__main__':
    args = parse_args()
    # set random seed
    set_seed(args.seed)
    # pl.seed_everything(args.seed)
    print(args)

    main(args)
