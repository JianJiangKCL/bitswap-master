import os
import pickle
from collections import namedtuple

import torch

import os
import os.path
import torch.utils.data as data

from PIL import Image
import numpy as np
import random
from torchvision import transforms
import joblib

CodeRow = namedtuple('CodeRow', ['code_t', 'code_b', 'label'])

def get_indices(dataset, class_name):
	indices = []
	#todo
	# max_imgs = 65000
	for i in range(len(dataset.targets)):
		label = dataset.targets[i]
		if label in class_name:
			indices.append(i)
			# used for selecting

	random.shuffle(indices)
	# indices = indices[0:max_imgs]
	return indices
def write(txn, key, value):
	ret = txn.put(str(key).encode('utf-8'), pickle.dumps(value), overwrite=False)
	return ret


def get(txn, key):
	value = txn.get(str(key).encode('utf-8'))
	return value


def unpickle(file):
	with open(file, 'rb') as fo:
		dict = np.load(fo)
	return dict


trans2PIL = transforms.ToPILImage(mode='RGB')


def denormalize(x):
	x = x * torch.Tensor((0.1, 0.1, 0.1)).view(3, 1, 1)
	x = x + torch.tensor((0.5, 0.5, 0.5)).view(3, 1, 1)
	return x


class RuntimeDataset(data.Dataset):
	trans2PIL = transforms.ToPILImage(mode='RGB')

	def __init__(self, data, transform, type='RGB'):
		self.data = data
		self.transform = transform
		self.PIL = False
		if type == 'RGB':
			self.PIL = True

	def __getitem__(self, index):
		x = self.data[index]
		if self.PIL:
			x = denormalize(x)
			x = trans2PIL(x)
		x = self.transform(x)
		return x

	def __len__(self):
		return len(self.data)


class Denormalize(object):
	def __init__(self, mean, std):
		self.mean = mean
		self.std = std

	def __call__(self, x):
		x = x * torch.Tensor(self.std).view(3, 1, 1)  # .cuda()
		x = x + torch.tensor(self.mean).view(3, 1, 1)  # .cuda()
		return x


class ChangeRange(object):
	# input is tensor
	def __init__(self, max_value):
		self.max_value = max_value

	def __call__(self, code):
		code = code.float().div(self.max_value).unsqueeze(0)
		return code


transform_toPIL = transforms.Compose([
	# todo normal aug need rotate
	Denormalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
	transforms.ToPILImage(mode='RGB')
])


class RuntimeEvaDataset(data.Dataset):

	def __init__(self, data, transform_AUG):
		self.data = data
		self.transform_AUG = transform_AUG

	def __getitem__(self, index):
		x = self.data[index]
		x = transform_toPIL(x)
		x = self.transform_AUG(x)
		return x

	def __len__(self):
		return len(self.data)


class ImageNet_downsampled(data.Dataset):
	'''
	‘data’ - numpy array with uint8 numbers of shape samples x 3072 (32*32*3) [h*w*3]. First 1024 numbers represent red               channel, next 1024 numbers green channel, last 1024 numbers represent blue channel.
	‘labels’- number representing image class, indexing starts at 1 and it uses mapping from the map_clsloc.txt     file provided in original Imagenet devkit
	‘mean’ - mean image computed over all training samples, included for convenience, usually first preprocessing step removes mean from all images.
	'''

	def __init__(self, data_folder, idx, transform=None, img_size=32):
		self.transform = transform

		# batch file idx starts from 1 to 11.
		# for idx in range(1, 11):
		self.img_size = img_size
		self.decode_img_size = img_size
		img_size2 = img_size * img_size
		print('data_file', data_folder)
		if idx == -1:
			data_file = os.path.join(data_folder, 'val_data')

			d = np.load(data_file + '.npz')
			x = d['data']
			# y index start from 1
			y = d['labels']
		elif idx == 0:
			if img_size == 32:
				# 32x32x3
				x = np.empty([0, 3072],
				             dtype=np.uint8)  # default dtype is float64 that will cause huge memory during concatenate
			# 64x64x32
			elif img_size == 64:
				x = np.empty([0, 12288],
				             dtype=np.uint8)  # default dtype is float64 that will cause huge memory during concatenate
			y = np.empty([0])
			for b_id in range(1, 11):  # (1,11) i.e.,from 1 to 10
				print('>>>>>>>>processing train batch ', b_id)
				data_file = os.path.join(data_folder, 'train_data_batch_')
				d = np.load(data_file + str(b_id) + '.npz')
				tmp_x = d['data']
				# y index start from 1
				tmp_y = d['labels']
				x = np.concatenate([x, tmp_x], axis=0)
				y = np.concatenate([y, tmp_y], axis=0)

		else:
			data_file = os.path.join(data_folder, 'train_data_batch_')
			d = np.load(data_file + str(idx) + '.npz')

			# d = unpickle(data_file + str(idx) + '.npz')
			x = d['data']
			# y index start from 1
			y = d['labels']

		# [:,R-G-B]
		x = np.dstack((x[:, :img_size2], x[:, img_size2:2 * img_size2], x[:, 2 * img_size2:]))
		# BxCxHxW
		x = x.reshape((x.shape[0], img_size, img_size, 3)).transpose(0, 3, 1, 2)
		# if idx == 1:
		self.data = x

		self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC #

		self.targets = y
		k = 1

	def set_decode_img_size(self, size):
		self.decode_img_size = size

	def __getitem__(self, index):

		img, target = self.data[index], self.targets[index]

		img = Image.fromarray(img)
		# target = torch.LongTensor([target])
		if self.transform is not None:
			img = self.transform(img)
		# if self.img_size == 32 and self.decode_img_size == 64:
		# 	sample_1 = img[:, 0:32, 0:32]
		# 	sample_2 = img[:, 32:64, 0:32]
		# 	sample_3 = img[:, 0:32, 32:64]
		# 	sample_4 = img[:, 32:64, 32:64]
		# 	# 4*3 x 32x32
		# 	img = torch.cat([sample_1,sample_2, sample_3, sample_4],dim=0)
		return img, target

	def __len__(self):
		return len(self.data)


class CodesNpzDataset(data.Dataset):

	def __init__(self, data_path, transform=None, max_value=255):
		self.transform = transform
		format = data_path[-2:]
		if format == 'pz':
			d = np.load(data_path)
		elif format == 'xz':
			d = joblib.load(data_path)

		self.code_t = d['code_t']
		self.code_b = d['code_b']
		self.targets = d['label']
		self.transform = transform
		self.max_value = max_value
		# BxCxHxW
		# unsqueeze codes in one shot rather than everytime getitem
		self.code_t = np.expand_dims(self.code_t, axis=1)
		self.code_b = np.expand_dims(self.code_b, axis=1)
	def change_range(self, code):
		code = code.float().div(self.max_value).unsqueeze(0)

		return code

	def __getitem__(self, index):

		code_t, code_b, target = self.code_t[index], self.code_b[index], self.targets[index]

		target = torch.LongTensor([target])

		code_t, code_b = torch.from_numpy(code_t), torch.from_numpy(code_b)
		# code_t, code_b = self.change_range(code_t), self.change_range(code_b)
		if self.transform is not None:
			# code_t = code_t / self.max_value * 2.0 - 1.0
			# code_b = code_b / self.max_value * 2.0 - 1.0
			code_t, code_b = self.transform(code_t), self.transform(code_b)
		# make codes to CxHxW
		# return code_t.unsqueeze(0)#, code_b.unsqueeze(0), target

		return code_t, code_b, target
	def __len__(self):
		return len(self.code_t)




