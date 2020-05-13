import pathlib
import torch.utils.data as data
import torch
from config import *
import os
import random
import glob
import numpy as np

class VoxelDataset(data.Dataset):
    def __init__(self):
        self.all_list = []
        self.cat_list = []
        for i in INDICES:
            cat = CATEGORIES[i]
            path_names = (glob.glob('data/new_voxels/{}_*.npy'.format(cat)))
            for path in path_names:
                voxel = np.load(path)
                self.all_list.append(voxel)
                self.cat_list.append(i)

    def __len__(self):
        return len(self.all_list)

    def __getitem__(self, index):
        return self.all_list[index], self.cat_list[index] # if self.type == "all" else target
