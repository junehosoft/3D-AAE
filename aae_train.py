import argparse
import os
import numpy as np
import math
import itertools
from datetime import date
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import torchvision.transforms as transforms
from torchvision.utils import save_image
import torchvision
from config import *
from utils import *

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from data import VoxelDataset
from aae import * # import the model 

import torch.nn as nn
import torch.nn.functional as F
import torch

# ----------
#  Training
# ----------
cuda = True if torch.cuda.is_available() else False

def class_noise(class_num, dim, size):
	'''if class_num == CATEGORIES[0]:
		mu = 3
	else:
		mu = -3
	mean = np.ones(dim) * mu
	cov = np.diag(np.ones(dim))
	arr = np.random.multivariate_normal(mean, cov, size)'''
	l = class_num
	half = int(dim/2)
	m1 = 10*np.cos((l*2*np.pi)/10)
	m2 = 10*np.sin((l*2*np.pi)/10)
	mean = [m1, m2]
	mean = np.tile(mean, half)
	v1 = [np.cos((l*2*np.pi)/10), np.sin((l*2*np.pi)/10)]
	v2 = [-np.sin((l*2*np.pi)/10), np.cos((l*2*np.pi)/10)]
	a1 = 8
	a2 = .4
	M =np.vstack((v1,v2)).T
	S = np.array([[a1, 0], [0, a2]])
	c = np.dot(np.dot(M, S), np.linalg.inv(M))
	cov = np.zeros((dim, dim))
	for i in range(half):
		cov[i*2:(i+1)*2, i*2:(i+1)*2] = c
	#cov = cov*cov.T
	vec = np.random.multivariate_normal(mean=mean, cov=cov,
										size=size)
	return vec

def sample_noise(size):
	noise_vector = np.zeros((size, LATENT_DIM))
	section = int(size/N_CLASSES)
	for i in range(section):
		noise_vector[i, :] = np.array(np.random.uniform(0, 1, LATENT_DIM))
	#for i in range(N_CLASSES):
		# noise_vector[i*section:min((i+1)*section, size), :] = class_noise(i, LATENT_DIM, min(section, size-section*i))


	return noise_vector

def make_one_hot_real(size):
	section = int(size/N_CLASSES)
	indices = []
	for i in range(N_CLASSES):
		indices.append(range(i * section, min((i+1)*section, size)))
	arr = one_hot_encode(indices, size)

	return arr

def one_hot_encode(index_arr, size):
	'''arr = np.zeros((len(index1) + len(index2), N_CLASSES))
	arr[index1, 0] = 1
	arr[index2, 1] = 1'''
	arr = np.zeros((size, N_CLASSES))
	for i in range(len(index_arr)):
		arr[index_arr[i], i] = 1
	return arr

def train(b1, b2):
	# Use binary cross-entropy loss
	adversarial_loss = torch.nn.BCELoss()
	pixelwise_loss = torch.nn.L1Loss()

	device = torch.device("cuda" if cuda else "cpu")
	# Initialize generator and discriminator
	encoder = Encoder().to(device)
	decoder = Decoder().to(device)
	discriminator = Discriminator().to(device)

	if cuda:
		encoder.cuda()
		decoder.cuda()
		discriminator.cuda()
		adversarial_loss.cuda()
		pixelwise_loss.cuda()

	# Configure data loader
	# os.makedirs("../../data/deepfashion", exist_ok=True)
	dataloader = torch.utils.data.DataLoader(
		VoxelDataset(),
		batch_size=TRAIN_BATCH_SIZE,
		num_workers=NUM_WORKERS,
		shuffle=True,
	)
	print("done loading data")
	# Optimizers
	optimizer_G = torch.optim.Adam(
		itertools.chain(encoder.parameters(), decoder.parameters()), lr=LR_G, betas=(b1, b2)
	)
	optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=LR_D, betas=(b1, b2))

	Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

	# generate fixed noise vector
	#fixed_noise = Variable(Tensor(np.random.normal(0, 1, (n_row ** 2, LATENT_DIM))))
	#noise_vector = np.zeros((n_row**2, LATENT_DIM))
	#noise_vector[:50,:] = class_noise(CATEGORIES[0], LATENT_DIM, 50)
	#noise_vector[50:,:] = class_noise(CATEGORIES[1], LATENT_DIM, 50)
	n_row = 1
	fixed_noise = Variable(Tensor(sample_noise(n_row**2)))
	# make directory for saving images
	path = "/".join([str(c) for c in [GENERATED_BASE, "aae", "_".join([CONFIG_AS_STR, opt.tag]), 'train']])
	os.makedirs(path, exist_ok=True)

	# save losses across all
	G_losses = []
	D_losses = []
	D_x = []
	#one_hot_label = one_hot_encode(range(int(TRAIN_BATCH_SIZE/2)), range(int(TRAIN_BATCH_SIZE/2), TRAIN_BATCH_SIZE))
	# one_hot_label = make_one_hot_real(TRAIN_BATCH_SIZE)
	print("done getting hot labels")
	# training loop 
	for epoch in range(N_EPOCHS):
		for i, (voxels, labels) in enumerate(dataloader):
			# Configure input
			real_voxels = Variable(voxels.type(Tensor))

			# Adversarial ground truths
			valid = Variable(Tensor(voxels.shape[0], 1).fill_(1.0), requires_grad=False)
			fake = Variable(Tensor(voxels.shape[0], 1).fill_(0.0), requires_grad=False)

			# ---------------------
			#  Train Generator
			# ---------------------

			optimizer_D.zero_grad()

			# Sample noise as discriminator ground truth
			#z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], LATENT_DIM))))
			z = Variable(Tensor(sample_noise(voxels.shape[0])))
			#if imgs.shape[0] == TRAIN_BATCH_SIZE:
			#	real_labels = Variable(Tensor(one_hot_label))
			#else:
			#	real_labels = Variable(Tensor(make_one_hot_real(imgs.shape[0])))
			#print("made one hot labels again")
			encoded_voxels = encoder(real_voxels)
			indices = []
			#for j in range(N_CLASSES):
			#	indices.append(np.where(labels==CATEGORIES[j])[0])
			#fake_labels = Variable(Tensor(one_hot_encode(indices, imgs.shape[0])))
			#print("made fake labels")
			# Measure discriminator's ability to classify real from generated samples
			real_loss = adversarial_loss(discriminator(z), valid) #real_labels), valid)
			#print(encoded_voxels.shape)
			#print(fake.shape)
			#guess
			fake_loss = adversarial_loss(discriminator(encoded_voxels.detach()), fake)# fake_labels), fake)
			d_loss = 0.5 * (real_loss + fake_loss)

			d_loss.backward()
			optimizer_D.step()

			if i % N_CRITIC == 0:
				# -----------------
				#  Train Discriminator
				# -----------------

				optimizer_G.zero_grad()

				encoded_voxels = encoder(real_voxels)
				decoded_voxels = torch.round(decoder(encoded_voxels))

				# Loss measures generator's ability to fool the discriminator
				g_loss = 0.5 * adversarial_loss(discriminator(encoded_voxels), valid) + 0.5 * pixelwise_loss(
					decoded_voxels, real_voxels
				)

				g_loss.backward()
				optimizer_G.step()

			batches_done = epoch * len(dataloader) + i

			if batches_done % 1 == 0:
				print(
					"[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
					% (epoch, N_EPOCHS, i, len(dataloader), d_loss.item(), g_loss.item())
				)
			
			'''if batches_done % SAMPLE_INTERVAL == 0:
				name = gen_name(batches_done)
				if FIXED_NOISE:
					sample_image(decoder=decoder, n_row=n_row, path=path, name=name, fixed_noise=fixed_noise, individual=True)
				else:
					sample_image(decoder=decoder, n_row=n_row, path=path, name=name)'''

			# save losses
			G_losses.append(g_loss.item())
			D_losses.append(d_loss.item())
			D_x.append(batches_done)

		'''if epoch % 10 == 0:
			config_mid = gen_name(CATEGORIES_AS_STR, LATENT_DIM, IMG_SIZE, epoch, LR, TRAIN_BATCH_SIZE, N_CRITIC)
			print("Saved Encoder to {}".format(save_model(encoder, "caae_encoder", config_mid, today)))
			print("Saved Decoder to {}".format(save_model(decoder, "caae_decoder", config_mid, today)))
			print("Saved Discriminator to {}".format(save_model(discriminator, "caae_discriminator", config_mid, today)))
			plot_losses("caae", G_losses, D_losses, config_mid, today)'''
	
	plot_losses("caae", G_losses, D_losses, D_x, CONFIG_AS_STR, opt.tag)
	return encoder, decoder, discriminator

if __name__=="__main__":
	os.makedirs("images", exist_ok=True)
	parser = argparse.ArgumentParser()
	parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
	parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
	#parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
	parser.add_argument("--tag", type=str, default='none')
	opt = parser.parse_args()
	print(opt)
	tag = opt.tag

	encoder, decoder, discriminator = train(opt.b1, opt.b2)
	# ----------
	#  Save Model and create Training Log
	# ----------
	# TODO: save this to a folder logs
	print(opt)
	print("Saved Encoder to {}".format(save_model(encoder, "encoder", CONFIG_AS_STR, tag)))
	print("Saved Decoder to {}".format(save_model(decoder, "decoder", CONFIG_AS_STR, tag)))
	print("Saved Discriminator to {}".format(save_model(discriminator, "discriminator", CONFIG_AS_STR, tag)))