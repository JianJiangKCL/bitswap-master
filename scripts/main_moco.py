# MOCO training with knn vilidation

import warnings
import os
import shutil
import time

import argparse
import builtins
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp

import torchvision.models as models


from moco.builder import MoCo  
from utils import AverageMeter, ProgressMeter, calc_accuracy, knn_evaluate
from moco.custom_data_loader import get_dataloader, get_dataloader_with_KNN

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch MOCO Training')
parser.add_argument('--dataset', default='cifar10',
                    help='path to dataset')
parser.add_argument('--ckpt-path', default='./checkpoints',
                    help='path to save checkpoints')
parser.add_argument('--ckpt-name', default=None, required=True,
                    help='name to save checkpoints')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-s', '--simplify-model', default=True,
                    help='To simplify the resnet models or not, as the datatsets are in lower resolution')
parser.add_argument('-j', '--workers', default=16, type=int,
                    help='number of data loading workers (default: 32)')

parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')

parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel, (4 GPUs: 128)')
parser.add_argument('--lr', '--learning-rate', default=0.06, type=float,
                    metavar='LR', help='initial learning rate, 8 GPUs: 0.03, 4 GPUs: 0.015', dest='lr')
parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
                    help='learning rate schedule (when to drop lr by 10x)')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')

parser.add_argument('--wd', '--weight-decay', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=100, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')

parser.add_argument('--world-size', default=1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', default=True, type=bool,
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

# moco specific configs:
parser.add_argument('--moco-dim', default=128, type=int,
                    help='feature dimension (default: 128)')
parser.add_argument('--moco-k', default=4096, type=int,
                    help='queue size; number of negative keys (default: 65536)')
parser.add_argument('--moco-m', default=0.99, type=float,
                    help='moco momentum of updating key encoder (default: 0.999)')
parser.add_argument('--moco-t', default=0.1, type=float,
                    help='softmax temperature (default: 0.07 v1, 0.2 v2)')

# options for moco v2
parser.add_argument('--mlp', default=True,
                    help='use mlp head')
parser.add_argument('--aug-plus', default=True,
                    help='use moco v2 data augmentation')
parser.add_argument('--cos', default=True,
                    help='use cosine lr schedule')

# options for moco v2
parser.add_argument('--knn-monitor', action='store_true',
                    help='use knn to evaluate while training')
parser.add_argument('--knn-k', default=200, type=int,
                    help='use moco v2 data augmentation')
parser.add_argument('--knn-t', default=0.1, type=float,
                    help='soft t in KNN')
parser.add_argument('--knn_interval', default=1, type=int,
                    help='knn evaluation frequency')

def main_worker(gpu, n_gpus_per_node, args):
    args.gpu = gpu

    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print('Use GPU: {} for training'.format(args.gpu))
    
    if args.distributed:
        if args.dist_url == 'env://' and args.rank == -1:
            args.rank = int(os.environ['RANK'])
        if args.multiprocessing_distributed:
            args.rank = args.rank * n_gpus_per_node + gpu
        
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        print('Initialize process Process Group {}'.format(args.rank))
        
    print('>= creating model {}'.format(args.arch))
    backbone = models.__dict__[args.arch]
    model = MoCo(backbone, args.moco_dim, args.moco_k, args.moco_m,
                args.moco_t, args.mlp, args.simplify_model)
    
    if args.distributed:
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            args.batch_size = int(args.batch_size / n_gpus_per_node)
            args.workers = int((args.workers + n_gpus_per_node - 1) / n_gpus_per_node)
            model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        else:
            model.cuda()
            model = nn.parallel.DistributedDataParallel(model)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    criterion = nn.CrossEntropyLoss().cuda(args.gpu)
    optimizer = torch.optim.SGD(model.parameters(), args.lr, 
                                momentum=args.momentum, weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if not args.knn_monitor:
        train_loader, train_sampler = get_dataloader(args)
    else:
        train_loader, memory_loader, test_loader, train_sampler = get_dataloader_with_KNN(args)

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        train(train_loader, model, criterion, optimizer, epoch, args)

        if args.knn_monitor and (epoch+1) % args.knn_interval == 0:
            knn_top1 = knn_evaluate(model.module.encoder_q, memory_loader, test_loader, args.knn_k, args.knn_t)
            # print('Epoch {}: knn top1 accuracy: {:.2f}'.format(epoch+1, knn_top1))
        if not args.multiprocessing_distributed or (args.multiprocessing_distributed 
                and args.rank % n_gpus_per_node == 0):
            os.makedirs(args.ckpt_path, exist_ok=True)
            # save_path = os.path.join(args.ckpt_path, 
            #     'ckpt-{}-{}-lr{:f}-b{:d}-k{:d}-t{:.2f}.pth.tar'.format(
            #         args.arch, args.dataset, args.lr, args.batch_size, args.moco_k, args.moco_t))
            save_path = os.path.join(args.ckpt_path, args.ckpt_name)
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': args.arch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, is_best=False, filename=save_path)


def train(train_loader, model, criterion, optimzier, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':.2f')
    top5 = AverageMeter('Acc@5', ':.2f')
    progress = ProgressMeter(
        len(train_loader), [batch_time, data_time, losses, top1, top5], prefix='Epoch:[{}].'.format(epoch)
    )

    model.train()

    end = time.time()
    for i, (imgs, _) in enumerate(train_loader):
        data_time.update(time.time() - end)
        if args.gpu is not None:
            imgs[0] = imgs[0].cuda(args.gpu, non_blocking=True)
            imgs[1] = imgs[1].cuda(args.gpu, non_blocking=True)
        
        output, target = model(imgs[0], imgs[1])
        loss = criterion(output, target)

        acc1, acc5 = calc_accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), imgs[0].size(0))
        top1.update(acc1[0], imgs[0].size(0))
        top5.update(acc5[0], imgs[0].size(0))

        optimzier.zero_grad()
        loss.backward()
        optimzier.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if (i+1) % args.print_freq == 0:
        #     progress.display(i) 
    progress.display(i)

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    if args.cos:
        lr *= 0.5 * (1 + math.cos(math.pi * epoch / args.epochs))
    else:
        for milestone in args.schedule:
            lr *= 0.1 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == 'env://' and args.world_size == -1:
        args.world_size = int(os.environ['WORLD_SIZE'])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    n_gpus_per_node = torch.cuda.device_count()
    print('Using {} GPUs for training'.format(n_gpus_per_node))

    if args.multiprocessing_distributed:
        args.world_size = n_gpus_per_node * args.world_size
        #Number of processes to spawn.
        mp.spawn(main_worker, nprocs=n_gpus_per_node, args=(n_gpus_per_node, args))
    else:
        main_worker(args.gpu, n_gpus_per_node, args)