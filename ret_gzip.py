
import numpy as np
import os
import gzip
from os.path import isfile, join
from dataset import CodesNpzDataset
# dataset = 'cifar'
dataset = 'img224'
is_compress = True

if dataset == 'cifar':

	codes_path = 'np_codes_uint16_via50VQVAEn512.npz'
	base = 4
elif dataset == 'img224':
	codes_path = 'sub_down_img224_all_viaCifar.npz'
	base = 28
if is_compress:

	codes_ds = CodesNpzDataset(codes_path)
	top_code = codes_ds.code_t#[0:100, :, :, :]
	bot_code = codes_ds.code_b
	B = len(top_code)
	# tobytes() or not will not impact the results
	# top_bits = top_code.tobytes()
	# back_top = np.frombuffer(top_bits, dtype=np.int16).reshape(top_shape)
	# the number of the byte array of compressed_top is just the memory cost
	compressed_top = gzip.compress(top_code)
	compressed_bot = gzip.compress(bot_code)
	byte_compressed_top = len(compressed_top)
	divider_top = float(B * base* base)/8
	byte_compressed_bot = len(compressed_bot)
	divider_bot = float(B * (base*2) * (base*2)) / 8

	print(f"{byte_compressed_top} compressed bytes; {byte_compressed_top/divider_top} bits per dim" )
	print(f"{byte_compressed_bot} compressed bytes; {byte_compressed_bot / divider_bot} bits per dim")
	# print(len(compressed_bot))
	# test_ = b''.join(compressed_top)
	back_top = gzip.decompress(compressed_top)
	back_top = np.frombuffer(back_top, dtype=np.int16).reshape(-1,1,base,base)
	with gzip.open(f"{dataset}_top.gz", "wb") as f:
		f.write(compressed_top)
	with gzip.open(f"{dataset}_bot.gz", "wb") as f:
		f.write(compressed_bot)

total_bist = 0

for level in ["top","bot"]:
	tmp =os.path.getsize( f"{dataset}_{level}.gz") * 8
	total_bist += tmp
	print(f"{level}  size {tmp} bits")

print(f"{total_bist} in bits")
#todo test uncompress
#
# obj =lzma.LZMAFile(f"{dataset}_bot.xz", mode="rb")
# bits_data = obj.read()
#
# # ori_bits_data = decompressor.decompress(bits_data)
# back_top = np.frombuffer(bits_data, dtype=np.int16).reshape(-1,1,base,base)
# # ori_data= int(ori_bits_data)
# k=1