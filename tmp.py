import torch
import numpy as np
import pickle
import os
bits = 9
dataset = 'code'
nz = 2
type = 'bitswap'
quantbits = 9
bitswap = True
ndatapoints = 10
# elbo = np.load(f'plots/{dataset}{nz}/{type}_{bits}bits_elbos.npy')
# total_bits = np.load(f'plots/{dataset}{nz}/{type}_{bits}bits_total.npy')
# path = f"bitstreams/code/nz{nz}/{'Bit-Swap' if bitswap else 'BB-ANS'}/{'Bit-Swap' if bitswap else 'BB-ANS'}_{quantbits}bits_nz{nz}_ndata{ndatapoints}"
path = "bitstreams/code/nz2/Bit-Swap/checkpoints/Bit-Swap_9bits_nz2_ndata50000"

# indices = np.load(f"bitstreams/code/indices.npy")
h=3
# with open(path, 'rb') as fp:
#     k = 1
#     from code_compress import check_states
#
#     # move tail bits (everything after 32 bits) to new state stream
#     state = pickle.load(fp)
#     state.append(state[-1] >> 32)
#     state[-2] = state[-2] & ((1 << 32) - 1)
#
#     # append number of blocks, height, the width and file extension of the image to the state
#     # state.append(blocks.shape[0])
#     # state.append(h)
#     # state.append(w)
#
#     # save compressed image
#     state_array = np.array(state, dtype=np.uint32)
#     np.savez("top_code_bits", state_array)
#     size_bitswap = os.path.getsize("top_code_bits.npz") * 8
#     print(size_bitswap)
#     # state =
#     # # check_states(state)
#     #
#     # max_va = max(state)
# k=6
# with open(
#         f"bitstreams/code/nz{nz}/{'Bit-Swap' if bitswap else 'BB-ANS'}/{'Bit-Swap' if bitswap else 'BB-ANS'}_{quantbits}bits_nz{nz}_ndata{ndatapoints}",
#         "wb") as fp:
#     # check_states(state)
#     pickle.dump(test, fp)
#
# data_p = test[0]
# print(isinstance(data_p, np.int32),'32')
# print(isinstance(data_p, np.int64))
# experiments= 2
# ndatapoints = 5
# targets = [i for  i in range(10)]
codes_path = 'np_codes_uint16_via50VQVAEn512.npz'
# codes_path = 'codes_img64_viacifar.npz'
from dataset import CodesNpzDataset
codes_ds = CodesNpzDataset(codes_path)
sample_size = 50000
top_code = torch.from_numpy(codes_ds.code_t[:sample_size])
top_code = top_code.flatten()
total = len(top_code)
uni_label, counts = torch.unique(top_code, return_counts=True)
probs = counts/total
def H(probs):
    tmp = probs * torch.log2(1/probs)
    entropy = torch.sum(tmp)
    return entropy
top_entropy = H(probs)
print('top entrop', top_entropy)

## bottom entropy
bot_code = torch.from_numpy(codes_ds.code_b[:sample_size])
bot_code = bot_code.flatten()
total = len(bot_code)
uni_label, counts = torch.unique(bot_code, return_counts=True)
probs_bot = counts/total
def H(probs):
    tmp = probs * torch.log2(1/probs)
    entropy = torch.sum(tmp)
    return entropy.item()
bot_entropy = H(probs_bot)
print('bot entropy', bot_entropy)
two_codes = torch.cat([top_code, bot_code], dim=0)
uni_label, counts = torch.unique(two_codes, return_counts=True)
probs_two = counts/total
two_entropy = H(probs_two)
print('two entropy', two_entropy)
# for label, cnt in zip(uni_label,counts):
#     print(label.item(), cnt.item(), cnt.item()/total)
# k=1