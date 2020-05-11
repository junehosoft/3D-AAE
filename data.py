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


class GeneratedDataset(data.Dataset):
    def __init__(self, base_dir, transform=None, target_transform=None):
        """

        :param images: list of images
        :param transform:
        :param target_transform:
        """
        self.transform = transform
        self.target_transform = target_transform
        path = pathlib.Path(base_dir)
        # match all images in directory
        self.img_paths = [
            str(p) for p in
            list(path.glob('**/*.jpg')) + list(path.glob('**/*.png'))]

    def __len__(self):
        return len(self.img_paths)

    def read_img(self, img_path):
        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                img = img.convert('RGB')
        return img

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img = self.read_img(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, img_path
