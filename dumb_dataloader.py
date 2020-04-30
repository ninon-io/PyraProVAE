import torch
from torch.utils.data.dataset import Dataset
import os
import numpy as np
import math
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
from guppy import hpy

test_split = 0.2
life_seed = 42
shuffle_data_set = True


class Toy_set(Dataset):
    def __init__(self, length=100, transform=None):
        self.sample = np.random.rand(128, 128, 100)
        self.len = length
        self.transform = transform

    def __getitem__(self, index):
        sample = torch.from_numpy((self.sample[index]))
        if self.transform:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return self.len


def get_data_loader(bar_dir, frame_bar=100, batch_size=16, export=False):
    data_set = Toy_set()
    data_set_size = len(data_set)
    indices = list(range(data_set_size))
    split = int(np.floor(test_split * data_set_size))
    if shuffle_data_set:
        np.random.seed(life_seed)
        np.random.shuffle(indices)
    train_indices, test_indices = indices[split:], indices[:split]
    # create corresponding subsets
    train_sampler = SubsetRandomSampler(train_indices)
    test_sampler = SubsetRandomSampler(test_indices)
    train_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=train_sampler,
                                               num_workers=0, pin_memory=True, shuffle=False, drop_last=True)
    test_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, sampler=test_sampler,
                                              num_workers=0, pin_memory=True, shuffle=False, drop_last=True)

    return train_loader, test_loader, train_sampler, test_sampler


