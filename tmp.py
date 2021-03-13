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
with open(path, 'rb') as fp:
    k = 1
    from code_compress import check_states

    # move tail bits (everything after 32 bits) to new state stream
    state = pickle.load(fp)
    state.append(state[-1] >> 32)
    state[-2] = state[-2] & ((1 << 32) - 1)

    # append number of blocks, height, the width and file extension of the image to the state
    # state.append(blocks.shape[0])
    # state.append(h)
    # state.append(w)

    # save compressed image
    state_array = np.array(state, dtype=np.uint32)
    np.savez_compressed("my_test_c", state_array)
    size_bitswap = os.path.getsize("my_test_c.npz") * 8
    print(size_bitswap)
    # state =
    # # check_states(state)
    #
    # max_va = max(state)
k=6
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
