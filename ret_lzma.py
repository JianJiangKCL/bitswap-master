import lzma
import numpy as np
import bz2
import os
from os.path import isfile, join
from dataset import CodesNpzDataset
dataset = 'cifar'
# dataset = 'img224'
is_compress = True
decompressor = lzma.LZMADecompressor()
if dataset == 'cifar':

	codes_path = 'np_codes_uint16_via50VQVAEn512.npz'
	base = 4
elif dataset == 'img224':
	codes_path = 'sub_down_img224_all_viaCifar.npz'
	base = 28
codes_ds = CodesNpzDataset(codes_path)
top_code = codes_ds.code_t#[0:100, :, :, :]
bot_code = codes_ds.code_b
B = len(top_code)
if is_compress:


	# top_bits = top_code.tobytes()
	# back_top = np.frombuffer(top_bits, dtype=np.int16).reshape(top_shape)
	# the number of the byte array of compressed_top is just the memory cost
	compressed_top = lzma.compress(top_code)
	compressed_bot = lzma.compress(bot_code)
	byte_compressed_top = len(compressed_top)
	divider_top = float(B * base* base)/8
	print(f"{byte_compressed_top} compressed bytes; {byte_compressed_top/divider_top} bits per dim" )
	print(len(compressed_bot))
	# test_ = b''.join(compressed_top)
	back_top = decompressor.decompress(compressed_top)
	back_top = np.frombuffer(back_top, dtype=np.int16).reshape(-1,1,base,base)
	with lzma.open(f"{dataset}_top.xz", "wb") as f:
		f.write(compressed_top)
	# with lzma.open(f"{dataset}_bot.xz", "wb") as f:
	# 	f.write(compressed_bot)

total_bist = 0

for level in ["top","bot"]:
	tmp =os.path.getsize( f"{dataset}_{level}.xz") * 8
	total_bist += tmp
	print(f"{level}  size {tmp} bits")
	byte_compressed = tmp
	if level == "top":
		divider = float(B * base * base)
	elif level == "bot":
		divider = float(B * (base*2) * (base*2))
	print(f"{byte_compressed} compressed bytes; {byte_compressed / divider} bits per dim for level {level}")


print(f"{total_bist} in bits")

#
# obj =lzma.LZMAFile(f"{dataset}_bot.xz", mode="rb")
# bits_data = obj.read()
#
# # ori_bits_data = decompressor.decompress(bits_data)
# back_top = np.frombuffer(bits_data, dtype=np.int16).reshape(-1,1,base,base)
# # ori_data= int(ori_bits_data)
# k=1