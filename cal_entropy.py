import torch
import numpy as np
import pickle
from torchvision import *
import os
from dataset import CodesNpzDataset
def H(probs):
    tmp = probs * torch.log2(1 / probs)
    entropy = torch.sum(tmp)
    return entropy.item()
def I(probs):
    tmp = -torch.log2(probs)
    info = torch.sum(tmp)
    return info.item()
def cal_entropy(x):
    # x = torch.from_numpy(x)
    # x = x.flatten()
    total = len(x)
    _, counts = torch.unique(x, return_counts=True)
    probs = counts/ total
    entropy = H(probs)
    return entropy
def cal_info(x):
    # x = torch.from_numpy(x)
    # x = x.flatten()
    total = len(x)
    _, counts = torch.unique(x, return_counts=True)
    probs = counts/ total
    info = I(probs)
    return info
# codes_path = 'np_codes_uint16_via50VQVAEn512.npz'
# codes_path = 'codes_img64_viacifar.npz'
# codes_path = 'img224_codes_viaCifar_deflate.npz'
# codes_path = 'img224_all_codes_viaCifar_deflate.npz'
codes_path = 'sub_down_img224_all_viaCifar.npz'
codes_ds = CodesNpzDataset(codes_path)
uni_label, counts = torch.unique(torch.from_numpy(codes_ds.targets), return_counts=True)
print(f"num samples {len(codes_ds.targets)}")

#####
# top code
###

top_code = torch.from_numpy(codes_ds.code_t)#[0:100,:, :,:]
top_code = top_code.flatten()
top_entropy = cal_entropy(top_code)
top_info = cal_info(top_code)
# total_top = len(top_code)
# _, counts_top = torch.unique(top_code, return_counts=True)
# probs = counts_top/total_top
# top_entropy = H(probs)
print('top entrop', top_entropy)
print('top info', top_info)
#############
# bottom entropy
####
bot_code = torch.from_numpy(codes_ds.code_b)
bot_code = bot_code.flatten()

bot_entropy = cal_entropy(bot_code)
# total_bot = len(bot_code)
# _, counts_bot = torch.unique(bot_code, return_counts=True)
# probs_bot = counts_bot/total_bot
#
# bot_entropy = H(probs_bot)
print('bot entropy', bot_entropy)
######
# two-level entropy
###
two_codes = torch.cat([top_code, bot_code], dim=0)
two_entropy = cal_entropy(two_codes)
# _, counts_two = torch.unique(two_codes, return_counts=True)
# probs_two = counts_two/(total_top + total_bot)
# two_sum_prob = probs_two.sum()
# two_entropy = H(probs_two)
print('two entropy', two_entropy)


####
# incremental entropy
####
# whole_size = len(codes_ds.targets)
# for sample_size in range(25000, whole_size +1, 500):
#     print('=============== =============')
#     print('=============== sample_size',sample_size, '=============')
#     top_code = torch.from_numpy(codes_ds.code_t[:sample_size])
#     top_code = top_code.flatten()
#     total_top = len(top_code)
#     uni_label, counts = torch.unique(top_code, return_counts=True)
#     probs = counts/total_top
#     def H(probs):
#         tmp = probs * torch.log2(1/probs)
#         entropy = torch.sum(tmp)
#         return entropy
#     top_entropy = H(probs)
#     print('top entrop', top_entropy)
#
#     ## bottom entropy
#     bot_code = torch.from_numpy(codes_ds.code_b[:sample_size])
#     bot_code = bot_code.flatten()
#     total_bot = len(bot_code)
#     uni_label, counts = torch.unique(bot_code, return_counts=True)
#     probs_bot = counts/total_bot
#
#     bot_entropy = H(probs_bot)
#     print('bot entropy', bot_entropy)
################################
# image-level entropy
####
# class ToInt:
#     def __call__(self, pic):
#         return pic * 255
#
#
# # set data pre-processing transforms
# transform_ops = transforms.Compose([transforms.ToTensor(), ToInt()])
# train_set = datasets.CIFAR10(root="D:\Dataset\cifar100", train=True, transform=transform_ops, download=True)
# img = train_set.data
# img = torch.from_numpy(img)
# img = img.flatten()
# totoal_img = len(img)
# uni_label, counts = torch.unique(img, return_counts=True)
# probs_img = counts/totoal_img
# img_sum_prob = probs_img.sum()
# img_entropy = H(probs_img)
# print('img entropy', img_entropy)