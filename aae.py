from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os
import numpy as np
import math
import itertools

import torchvision.transforms as transforms
from torchvision.utils import save_image
from config import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

cuda = True if torch.cuda.is_available() else False

def reparameterization(mu, logvar):
    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), LATENT_DIM))))
    z = sampled_z * std + mu
    return z

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.convolve = nn.Sequential(
            nn.Conv3d(1, INTER_CH_1, 4, stride=2, padding=1),
            nn.BatchNorm3d(INTER_CH_1),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(INTER_CH_1, INTER_CH_2, 4, stride=2, padding=1),
            nn.BatchNorm3d(INTER_CH_2),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(INTER_CH_2, INTER_CH_3, 4, stride=2, padding=1),
            nn.BatchNorm3d(INTER_CH_3),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(INTER_CH_3, INTER_CH_4, 4, stride=2, padding=1),
            nn.BatchNorm3d(INTER_CH_4),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(INTER_CH_4, LATENT_DIM, 4, stride=1),
        )


    def forward(self, voxel):
        voxel = voxel.view(voxel.shape[0], 1, IMG_SIZE, IMG_SIZE, IMG_SIZE)
        x = self.convolve(voxel)
        return x.view(x.shape[0], LATENT_DIM)


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.convolve = nn.Sequential(
            nn.ConvTranspose3d(LATENT_DIM, INTER_CH_4, 4, stride=1),
            nn.BatchNorm3d(INTER_CH_4),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(INTER_CH_4, INTER_CH_3, 4, stride=2, padding=1),
            nn.BatchNorm3d(INTER_CH_3),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(INTER_CH_3, INTER_CH_2, 4, stride=2, padding=1),
            nn.BatchNorm3d(INTER_CH_2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(INTER_CH_2, INTER_CH_1, 4, stride=2, padding=1),
            nn.BatchNorm3d(INTER_CH_1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose3d(INTER_CH_1, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        z = z.view(z.shape[0], LATENT_DIM, 1, 1, 1)
        voxel = self.convolve(z)
        return voxel.view(voxel.shape[0], IMG_SIZE, IMG_SIZE, IMG_SIZE)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            # nn.Linear(LATENT_DIM + N_CLASSES, INTER_DIM_1),
            nn.Linear(LATENT_DIM, INTER_DIM_1),
            nn.LeakyReLU(inplace=True),
            nn.Linear(INTER_DIM_1, INTER_DIM_3),
            nn.LeakyReLU(inplace=True),
            nn.Linear(INTER_DIM_3, 1),
            nn.Sigmoid(),
        )

    def forward(self, z): #labels):
        #validity = self.model(torch.cat((z, labels), -1))
        validity = self.model(z)
        return validity

def sample_image(decoder, n_row, path, name, fixed_noise=None, individual=False):
    """Saves a grid of generated digits"""
    # Sample noise
    with torch.no_grad():
        if fixed_noise is not None:
            z = fixed_noise
        else:
            Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
            z = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
        gen_vox = decoder(z)
        gen_vox = gen_vox.cpu().numpy()
        if individual:
            for i in range(gen_vox.shape[0]): # save them one by one
                fig = plt.figure()
                ax = fig.gca(projection='3d')
                ax.voxels(np.around(gen_vox[i]), facecolors='red')
                plt.axis('off')
                plt.savefig("{}/{}_{}.png".format(path, name, i), bbox_inches='tight')
            # save_image(gen_imgs.data[i, :, :, :], "%s/%s_%d.png" % (path, name, i), normalize=True)
    #else: # create grid
    #    save_image(gen_imgs.data, "%s/%s.png" % (path, name), nrow=n_row, normalize=True)


#def sample_image_fixed(decoder, fixed_noise, n_row, name):
#    """Saves a grid of generated digits"""
#    gen_imgs = decoder(fixed_noise)
#    save_image(gen_imgs.data, "images/%s.png" % name, nrow=n_row, normalize=True)





