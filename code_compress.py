from utils.torch.rand import *
from utils.torch.modules import ImageNet
from code_train import Model

from discretization import *
from torchvision import datasets, transforms
import random
import time
import argparse
from tqdm import tqdm
import pickle
from torch.utils.data import DataLoader, Subset
class ANS:
	def __init__(self, pmfs, bits=31, quantbits=8):
		self.device = pmfs.device
		self.bits = bits
		self.quantbits = quantbits

		# mask of 2**bits - 1 bits
		self.mask = (1 << bits) - 1

		# normalization constants
		self.lbound = 1 << 32
		self.tail_bits = (1 << 32) - 1

		self.seq_len, self.support = pmfs.shape

		# compute pmf's and cdf's scaled up by 2**n
		multiplier = (1 << self.bits) - (1 << self.quantbits)
		self.pmfs = (pmfs * multiplier).long()

		# add ones to counter zero probabilities
		self.pmfs += torch.ones_like(self.pmfs)

		# add remnant to the maximum value of the probabilites
		self.pmfs[torch.arange(0, self.seq_len),torch.argmax(self.pmfs, dim=1)] += ((1 << self.bits) - self.pmfs.sum(1))

		# compute cdf's
		self.cdfs = torch.cumsum(self.pmfs, dim=1) # compute CDF (scaled up to 2**n)
		self.cdfs = torch.cat([torch.zeros([self.cdfs.shape[0], 1], dtype=torch.long, device=self.device), self.cdfs], dim=1) # pad with 0 at the beginning

		# move cdf's and pmf's the cpu for faster encoding and decoding
		self.cdfs = self.cdfs.cpu().numpy()
		self.pmfs = self.pmfs.cpu().numpy()

		assert self.cdfs.shape == (self.seq_len, self.support + 1)
		assert np.all(self.cdfs[:,-1] == (1 << bits))

	def encode(self, x, symbols):
		for i, s in enumerate(symbols):
			pmf = int(self.pmfs[i,s])
			if x[-1] >= ((self.lbound >> self.bits) << 32) * pmf:
				x.append(x[-1] >> 32)
				x[-2] = x[-2] & self.tail_bits
			x[-1] = ((x[-1] // pmf) << self.bits) + (x[-1] % pmf) + int(self.cdfs[i, s])
		return x

	def decode(self, x):
		sequence = np.zeros((self.seq_len,), dtype=np.int64)
		for i in reversed(range(self.seq_len)):
			masked_x = x[-1] & self.mask
			s = np.searchsorted(self.cdfs[i,:-1], masked_x, 'right') - 1
			sequence[i] = s
			x[-1] = int(self.pmfs[i,s]) * (x[-1] >> self.bits) + masked_x - int(self.cdfs[i, s])
			if x[-1] < self.lbound:
				x[-1] = (x[-1] << 32) | x.pop(-2)
		sequence = torch.from_numpy(sequence).to(self.device)
		return x, sequence

def check_states(states):
	test = np.array(states, dtype=np.uint32)
	states = test.tolist()
	return states
def compress(quantbits, nz, bitswap, gpu, args):
	# model and compression params
	# zdim = 1 * 16 * 16
	# the size of latent z; is X_h/2
	if args.code_level == 'top':
		factor = 1
	elif args.code_level == 'bot':
		factor = 2
	if args.dataset == 'img64':
		base = 8
	elif args.dataset == 'cifar':
		base = 4
		codes_path = 'np_codes_uint16_via50VQVAEn512.npz'
	elif args.dataset == 'img224':
		base = 28
		codes_path = 'sub_down_img224_all_viaCifar.npz'
	height = base * factor
	xs = (1, height, height)
	width = height

	zdim = int(args.zchannels * height/2 * width/2)
	zrange = torch.arange(zdim)
	#my the size of image
	# xdim = 32 ** 2 * 1
	xdim = height* width * 1
	print(height, width)
	# print('zarnage', zrange)
	print('xdim', xdim)
	xrange = torch.arange(xdim)
	ansbits = 31 # ANS precision
	# his demo also use float64 for type
	type = torch.float64 # datatype throughout compression
	device = f"cuda:{gpu}" # gpu
	reswidth = 64

	print(f"{'Bit-Swap' if bitswap else 'BB-ANS'} - CODE - {nz} latent layers - {quantbits} bits quantization")

	# seed for replicating experiment and stability
	np.random.seed(100)
	random.seed(50)
	torch.manual_seed(50)
	torch.cuda.manual_seed(50)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True



	print("Load data..")
	# <=== DATA ===>
	# transform_ops = transforms.Compose([transforms.Pad(2), transforms.ToTensor(), ToInt()])
	from code_train import CodesToTensor
	from dataset import CodesNpzDataset

	# transform_ops = transforms.Compose([ToFloat()])
	transform_ops = None

	# codes_path = 'np_codes_uint16_via50VQVAEn512.npz'
	args.codes_path = codes_path
	test_set = CodesNpzDataset(codes_path, transform=transform_ops)
	print('len dataset', len(test_set))
	# compression experiment params
	experiments = args.n_group
	# experiments = 2
	# ndatapoints = 10
	ndatapoints = int(len(test_set)/experiments)
	print('ndatapoint', ndatapoints)
	# sample (experiments, ndatapoints) from test set with replacement
	# in fact, below if-condition will always go "the first condition" as the right file path should end with .npy
	# always overwrite the indices is better, o/w  this may load other setting's indices;
	# if not os.path.exists("bitstreams/code/indices"):
	# randindices = np.random.choice(len(test_set.targets), size=(experiments, ndatapoints), replace=False)
	randindices = np.random.choice(20, size=(experiments, ndatapoints), replace=False)
	if not os.path.exists(f'bitstreams/'):
		os.makedirs(f'bitstreams/code')
	# np.save("bitstreams/code/indices", randindices)
	# else:
	# 	randindices = np.load("bitstreams/code/indices")

	# decomperss to True to validate
	decompress = False

	# <=== MODEL ===>
	model = Model(xs=(1, height, width), nz=nz, zchannels=args.zchannels, nprocessing=4, kernel_size=args.kernel, resdepth=args.blocks, reswidth=reswidth).to(device)
	checkpoint = torch.load(args.vae_path,
	                        map_location=lambda storage, location: storage
	                        )
	model.load_state_dict(checkpoint['model'])
	print('epoch', checkpoint['epoch'])
	model.eval()

	print("Discretizing")
	# get discretization bins for latent variables
	zendpoints, zcentres = discretize(nz, quantbits, type, device, model,args)

	# get discretization bins for discretized logistic
	xbins = ImageBins(type, device, xdim)
	xendpoints = xbins.endpoints()
	xcentres = xbins.centres()



	print("Setting up metrics..")
	# metrics for the results
	nets = np.zeros((experiments, ndatapoints), dtype=np.float)
	elbos = np.zeros((experiments, ndatapoints), dtype=np.float)
	cma = np.zeros((experiments, ndatapoints), dtype=np.float)
	total = np.zeros((experiments, ndatapoints), dtype=np.float)

	print("Compression..")
	# for ei in range(experiments):

	ei = args.slurm_id
	print(f"Experiment {ei + 1}")
	subset = Subset(test_set, randindices[ei])
	test_loader = DataLoader(
		dataset=subset,
		batch_size=1, shuffle=False, drop_last=True)
	datapoints = list(test_loader)

	# < ===== COMPRESSION ===>
	# initialize compression
	model.compress()
	#
	# it means it starts with 10000 sta tes.

	# default it generates int64 data, but it set it to np.uint32
	# but int changed the state back to int64
	# clear no value overflow
	excess_state_len = 10000
	state = list(map(int, np.random.randint(low=1 << 16, high=(1 << 32) - 1, size=excess_state_len, dtype=np.uint32))) # fill state list with 'random' bits

	#todo  value overflow, what's so used for
	state[-1] = state[-1] << 32
	#state = check_states(state)
	initialstate = state.copy()
	restbits = None

	# <===== SENDER =====>
	iterator = tqdm(range(len(datapoints)), desc="Sender", disable=args.disable_tqdm)
	for xi in iterator:
		if args.code_level == 'top':
			x = datapoints[xi][0]
		elif args.code_level == 'bot':
			x = datapoints[xi][1]
		# (x, _, _) = datapoints[xi]
		x = x.to(device).view(xdim)

		# calculate ELBO
		with torch.no_grad():
			model.compress(False)
			logrecon, logdec, logenc, _ = model.loss(x.view((-1,) + model.xs))
			elbo = -logrecon + torch.sum(-logdec + logenc)
			model.compress(True)

		if bitswap:
			# < ===== Bit-Swap ====>
			# inference and generative model
			for zi in range(nz):
				# inference model

				input = zcentres[zi - 1, zrange, zsym] if zi > 0 else xcentres[xrange, x.long()]
				mu, scale = model.infer(zi)(given=input)
				# print('--------mu', mu.size())
				cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t() # most expensive calculation?
				pmfs = cdfs[:, 1:] - cdfs[:, :-1]
				pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

				# decode z
				# pop from ANS
				state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)

				# save excess bits for calculations
				if xi == zi == 0:
					restbits = state.copy()
					assert len(restbits) > 1, "too few initial bits" # otherwise initial state consists of too few bits
				excess_state_len = len(state) if len(state) < excess_state_len else excess_state_len
				# generative model
				# zrange = zrange.to(torch.long)
				z = zcentres[zi, zrange, zsymtop]
				mu, scale = model.generate(zi)(given=z)
				cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t() # most expensive calculation?
				pmfs = cdfs[:, 1:] - cdfs[:, :-1]
				pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

				# encode z or x
				# state = ANS(pmfs, bits=ansbits, quantbits=(quantbits if zi > 0 else 8)).encode(state, zsym if zi > 0 else x.long())
				state = ANS(pmfs, bits=ansbits, quantbits=(quantbits if zi > 0 else 9)).encode(state,
				                                                                               zsym if zi > 0 else x.long())
				zsym = zsymtop


		# prior
		cdfs = logistic_cdf(zendpoints[-1].t(), torch.zeros(1, device=device, dtype=type), torch.ones(1, device=device, dtype=type)).t()
		pmfs = cdfs[:, 1:] - cdfs[:, :-1]
		pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)

		# encode prior
		state = ANS(pmfs, bits=ansbits, quantbits=quantbits).encode(state, zsymtop)

		# calculating bits
		totaladdedbits = (len(state) - len(initialstate)) * 32
		totalbits = (len(state) - (len(restbits) - 1)) * 32

		# logging
		nets[ei, xi] = (totaladdedbits / xdim) - nets[ei, :xi].sum()
		elbos[ei, xi] = elbo.item() / xdim
		cma[ei, xi] = totalbits / (xdim * (xi + 1))
		total[ei, xi] = totalbits

		iterator.set_postfix_str(s=f"N:{nets[ei,:xi+1].mean():.2f}±{nets[ei,:xi+1].std():.2f}, D:{nets[ei,:xi+1].mean()-elbos[ei,:xi+1].mean():.4f}, C: {cma[ei,:xi+1].mean():.2f}, T: {totalbits:.0f}", refresh=False)

	# write state to file
	del state[0:excess_state_len - 1]
	os.makedirs(f"bitstreams/code/nz{nz}/{'Bit-Swap' if bitswap else 'BB-ANS'}", exist_ok=True)
	with open(f"bitstreams/code/nz{nz}/{'Bit-Swap' if bitswap else 'BB-ANS'}/{'Bit-Swap' if bitswap else 'BB-ANS'}_{args.dataset}_{args.code_level}_{quantbits}bits_nz{nz}_ndata{ndatapoints}_experiment{ei + 1}", "wb") as fp:
		# #state = check_states(state)
		pickle.dump(state, fp)
	# if not decompress:
	# 	continue
	# state = None
	# # open state file
	# with open(f"bitstreams/code/nz{nz}/{'Bit-Swap' if bitswap else 'BB-ANS'}/{'Bit-Swap' if bitswap else 'BB-ANS'}_{args.dataset}_{args.code_level}_{quantbits}bits_nz{nz}_ndata{ndatapoints}_experiment{ei + 1}", "rb") as fp:
	# 	state = pickle.load(fp)
	#
	#
	#
	# # <===== RECEIVER =====>
	# datapoints.reverse()
	# iterator = tqdm(range(len(datapoints)), desc="Receiver", postfix=f"decoded {None}",disable=args.disable_tqdm)
	# for xi in iterator:
	# 	(x, _, _) = datapoints[xi]
	# 	x = x.to(device).view(xdim)
	#
	# 	# prior
	# 	cdfs = logistic_cdf(zendpoints[-1].t(), torch.zeros(1, device=device, dtype=type), torch.ones(1, device=device, dtype=type)).t()
	# 	pmfs = cdfs[:, 1:] - cdfs[:, :-1]
	# 	pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)
	#
	# 	# decode z
	# 	state, zsymtop = ANS(pmfs, bits=ansbits, quantbits=quantbits).decode(state)
	#
	# 	if bitswap:
	# 		# < ===== Bit-Swap ====>
	# 		# inference and generative model
	# 		for zi in reversed(range(nz)):
	# 			# generative model
	# 			z = zcentres[zi, zrange, zsymtop]
	# 			mu, scale = model.generate(zi)(given=z)
	# 			cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t()  # most expensive calculation?
	# 			pmfs = cdfs[:, 1:] - cdfs[:, :-1]
	# 			pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)
	#
	# 			# decode z or x
	# 			# state, sym = ANS(pmfs, bits=ansbits, quantbits=quantbits if zi > 0 else 8).decode(state)
	# 			state, sym = ANS(pmfs, bits=ansbits, quantbits=quantbits if zi > 0 else 9).decode(state)
	# 			# inference model
	# 			input = zcentres[zi - 1, zrange, sym] if zi > 0 else xcentres[xrange, sym]
	# 			mu, scale = model.infer(zi)(given=input)
	# 			cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t() # most expensive calculation?
	# 			pmfs = cdfs[:, 1:] - cdfs[:, :-1]
	# 			pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)
	#
	# 			# encode z
	# 			state = ANS(pmfs, bits=ansbits, quantbits=quantbits).encode(state, zsymtop)
	#
	# 			zsymtop = sym
	# 		k=1
	# 		assert torch.all(x.long() == zsymtop), f"decoded datapoint does not match {xi + 1}"
	#
	# 	else:
	# 		# < ===== BB-ANS ====>
	# 		# inference and generative model
	# 		zs = [zsymtop]
	# 		for zi in reversed(range(nz)):
	# 			# generative model
	# 			z = zcentres[zi, zrange, zsymtop]
	# 			mu, scale = model.generate(zi)(given=z)
	# 			cdfs = logistic_cdf((zendpoints[zi - 1] if zi > 0 else xendpoints).t(), mu, scale).t() # most expensive calculation?
	# 			pmfs = cdfs[:, 1:] - cdfs[:, :-1]
	# 			pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)
	#
	# 			# decode z or x
	# 			# state, sym = ANS(pmfs, bits=ansbits, quantbits=quantbits if zi > 0 else 8).decode(state)
	# 			state, sym = ANS(pmfs, bits=ansbits, quantbits=quantbits if zi > 0 else 9).decode(state)
	# 			zs.append(sym)
	# 			zsymtop = sym
	#
	# 		zsymtop = zs.pop(0)
	# 		for zi in reversed(range(nz)):
	# 			# inference model
	# 			sym = zs.pop(0) if zi > 0 else zs[0]
	#
	# 			input = zcentres[zi - 1, zrange, sym] if zi > 0 else xcentres[xrange, sym]
	# 			mu, scale = model.infer(zi)(given=input)
	# 			cdfs = logistic_cdf(zendpoints[zi].t(), mu, scale).t() # most expensive calculation?
	# 			pmfs = cdfs[:, 1:] - cdfs[:, :-1]
	# 			pmfs = torch.cat((cdfs[:, 0].unsqueeze(1), pmfs, 1. - cdfs[:, -1].unsqueeze(1)), dim=1)
	#
	# 			# encode z
	# 			state = ANS(pmfs, bits=ansbits, quantbits=quantbits).encode(state, zsymtop)
	#
	# 			zsymtop = sym
	# 		# check if decoded datapoint matches the real datapoint
	# 		assert torch.all(x.long() == zs[0]), f"decoded datapoint does not match {xi + 1}"
	# 	iterator.set_postfix_str(s=f"decoded {len(datapoints) - xi}")

	# check if the initial state matches the output state
	# assert initialstate == state

	print(f"N:{nets.mean():.4f}±{nets.std():.2f}, E:{elbos.mean():.4f}±{elbos.std():.2f}, D:{nets.mean() - elbos.mean():.6f}")

	# save experiments
	os.makedirs(f"plots/code{nz}", exist_ok=True)
	np.save(f"plots/code{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_nets",nets)
	np.save(f"plots/code{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_elbos", elbos)
	np.save(f"plots/code{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_cmas",cma)
	np.save(f"plots/code{nz}/{'bitswap' if bitswap else 'bbans'}_{quantbits}bits_total", total)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', default=0, type=int)  # assign to gpu
	parser.add_argument('--nz', default=2, type=int)  # choose number of latent variables
	parser.add_argument('--zchannels', default=4, type=int, help="number of channels for the latent variables")
	parser.add_argument('--quantbits', default=9, type=int)  # choose discretization precision
	parser.add_argument('--blocks', default=4, type=int, help="number of ResNet blocks")
	parser.add_argument('--bitswap', default=1, type=int)  # choose whether to use Bit-Swap or not
	parser.add_argument('--vae_path', default=None, type=str)
	parser.add_argument('--ndata', default=100, type=int)
	parser.add_argument('--kernel', default=3, type=int)
	parser.add_argument('--dataset',  type=str)
	parser.add_argument('--code_level', type=str)
	parser.add_argument("--disable_tqdm", type=int, default=0)
	parser.add_argument("--slurm_id", type=int, default=3)
	parser.add_argument("--n_group", type=int, default=4)
	# parser.add_argument('--codes_path', type=str)
	args = parser.parse_args()
	print(args)

	gpu = args.gpu
	nz = args.nz
	quantbits = args.quantbits
	bitswap = args.bitswap
	# xs = [1, 4, 4]
	# for nz in [nz]:
	# 	for bits in [quantbits]:
	# 		for bitswap in [bitswap]:
	compress(quantbits, nz, bitswap, gpu, args)