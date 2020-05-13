import argparse
import os
import numpy as np
import math
import itertools
import pprint
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from config import *
from utils import *
from data import VoxelDataset
from aae import *  # import the model
from aae_train import class_noise

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True
def test(tag, model_type, generator, test_type):
    device = torch.device("cuda" if cuda else "cpu")

    # load the model
    
    # generate fixed noise vector
    n_row = 2
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    if test_type == 'sample':
    #fixed_noise = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
        decoder = Decoder().to(device)
        decoder.load_state_dict(load_model('decoder', CONFIG_AS_STR, tag, device))
        decoder.eval()

        if cuda:
            decoder.cuda()
        noise = np.zeros((N_CLASSES, LATENT_DIM))
        means, covs = class_noise(LATENT_DIM, N_CLASSES)
        for n in range(N_CLASSES):
            noise[n, :] = np.random.multivariate_normal(mean=means[n], cov=covs[n],size=1)

        z = Variable(Tensor(noise))
        path = "/".join([str(c) for c in [GENERATED_BASE, 'aae', "_".join([CONFIG_AS_STR, opt.tag]), "test"]])
        os.makedirs(path, exist_ok=True)
        sample_image(decoder, n_row, path, 'inter', fixed_noise=z, individual=True)
    elif test_type == 'encode':
        dataloader = torch.utils.data.DataLoader(
            VoxelDataset(),
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )
        encoder = Encoder().to(device)
        encoder.load_state_dict(load_model('encoder', CONFIG_AS_STR, tag, device))
        encoder.eval()

        if cuda:
            encoder.cuda()
        z = []
        for i in range(N_CLASSES):
            z.append([])
        with torch.no_grad():
            for i, (voxels, labels) in enumerate(dataloader):
                voxels = Variable(voxels.type(Tensor))
                encoded_voxel = encoder(voxels).cpu().numpy()
                for j in range(len(labels)):
                    z[labels[j]].append(encoded_voxel[j, :])
        fig = plt.figure()
        for i in range(N_CLASSES):
            z[i] = np.array(z[i])
            plt.scatter(z[i][:, 0], z[i][:, 1], label=CATEGORIES[INDICES[i]])
        plt.title("Distribution of training samples in latent space")
        plt.legend()
        plt.savefig('plots/{}.png'.format("_".join([CONFIG_AS_STR, opt.tag, 'distribution'])))
    elif test_type == 'full':
        decoder = Decoder().to(device)
        decoder.load_state_dict(load_model('decoder', CONFIG_AS_STR, tag, device))
        decoder.eval()
        encoder = Encoder().to(device)
        encoder.load_state_dict(load_model('encoder', CONFIG_AS_STR, tag, device))
        encoder.eval()
        dataloader = torch.utils.data.DataLoader(
            VoxelDataset(),
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=NUM_WORKERS,
            shuffle=True,
        )
        if cuda:
            decoder.cuda()
            encoder.cuda()
        path = "/".join([str(c) for c in [GENERATED_BASE, 'aae', "_".join([CONFIG_AS_STR, opt.tag]), "test"]])
        os.makedirs(path, exist_ok=True)
        with torch.no_grad():
            for i, (voxels, labels) in enumerate(dataloader):
                voxels = Variable(voxels.type(Tensor))
                encoded_voxel = encoder(voxels)
                sample_image(decoder, n_row, path, i, fixed_noise=encoded_voxel, individual=True)
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", type=str, default='none')
    parser.add_argument("--type", type=str, help="model type eg. aae")
    parser.add_argument("--model", type=str, help="generator name eg. decoder or generator")
    parser.add_argument("--test", type=str, default='sample')
    opt = parser.parse_args()

    test(opt.tag, opt.type, opt.model, opt.test)
