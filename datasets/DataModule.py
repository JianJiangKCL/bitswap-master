from datasets.BaseDM import BaseDataModule
import os
from datasets.transform import transform_image224_train, transform_image224_test, transform_image224_4Tensor
from torchvision.datasets import ImageFolder
import torch

import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from dataset import  CodesNpzDataset
from utils_funcs import  ToFloat
from torchvision import transforms

def get_indices(dataset, class_name):

    class_name = np.array(class_name)
    ind = np.zeros_like(dataset.targets)
    for target in class_name:
        # np element can only be equal to np element instead of python object
        tmp_ind = dataset.targets == target
        # print('===== tmp_ind', tmp_ind)
        ind = ind | tmp_ind
    return ind.nonzero()[0]


def get_indices_for_old(dataset, class_name, num_per_cls):
    indices = []
    cnts = {}
    for cls in class_name:
        cnts[cls] = 0

    for i in range(len(dataset.targets)):
        label = dataset.targets[i]
        if label in class_name:
            if cnts[label] < num_per_cls:
                cnts[label] += 1
                indices.append(i)

    random.shuffle(indices)
    return indices


transform_ops = transforms.Compose([ToFloat()])


class CodesModule(pl.LightningDataModule):
    def __init__(self, dataset_path, batch_size, num_workers):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        self.train_ds = CodesNpzDataset(self.dataset_path, transform=transform_ops)
        self.test_ds = self.train_ds

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers, pin_memory=True)

    # def val_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)
    #
    # def test_dataloader(self):
    #     return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.num_workers)


