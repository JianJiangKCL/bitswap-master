import torch
import numpy as np
import pickle
from torchvision import *
import os

def process(path, to_save_name):
    with open(path, 'rb') as fp:
        k = 1
        # move tail bits (everything after 32 bits) to new state stream
        state = pickle.load(fp)
        state.append(state[-1] >> 32)
        state[-2] = state[-2] & ((1 << 32) - 1)
        # append number of blocks, height, the width and file extension of the image to the state
        # state.append(blocks.shape[0])
        # state.append(h)
        # state.append(w)
        # save compressed image
        print(len(state))
        state_array = np.array(state, dtype=np.uint32)
        # np.savez_compressed will get the same results
        np.savez(to_save_name, state_array)
        size_bitswap = os.path.getsize(f"{to_save_name}.npz") * 8
        print(size_bitswap)
        return size_bitswap

#############################
# for multiple experiments
# path_prefix = 'bitstreams/code/nz8/Bit-Swap/Bit-Swap_img224_bot_9bits_nz8_ndata32214_experiment'
# to_save_name_prefix = 'bot_img224_1100epoch'
# total_bits = 0
# for e in range(1, 5):
#     path = f"{path_prefix}{e}"
#     print(path)
#     to_save_name = f"{to_save_name_prefix}_{e}"
#     e_size = process(path, to_save_name)
#     total_bits += e_size
# print(f"total bits: {total_bits} bits\n "
#       f" {float(total_bits/8/1024/1024)} in MB")
##############################


# path = "bitstreams/code/nz8/Bit-Swap/Bit-Swap_9bits_nz8_ndata50000"
path = "bitstreams/code/nz8/Bit-Swap/Bit-Swap_9bits_nz8_ndata50000" # works
# path = "bitstreams/code/nz8/Bit-Swap/Bit-Swap_9bits_nz8_bot_cifar_400epoch_ndata50000"
to_save_name = 'bot_cifar_nz8_top_cifar_2200epoch_data50000'

path = "bitstreams/code/nz8/Bit-Swap/Bit-Swap_img224_9bits_nz8_ndata128856_top"
to_save_name = 'top_img224_2100epoch'
process(path, to_save_name)