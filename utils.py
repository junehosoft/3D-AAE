import os
import time
import torch
import numpy as np
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
import matplotlib.pyplot as plt

def save_model(model, model_type, config, date):
	os.makedirs("model", exist_ok=True)
	save_path = "model/{}.pkl".format(gen_name(model_type, config, date))
	torch.save(model.state_dict(), save_path)
	return save_path

def load_model(model_type, config, date, device):
	model_path = "model/{}.pkl".format(gen_name(model_type, config, date))
	return torch.load(model_path, map_location=device)

def plot_losses(model_name, G_losses, D_losses, D_x, config, tag):
	plt.figure(figsize=(10,5))
	plt.title("Generator and Discriminator Loss During Training")
	G_x = range(len(G_losses))
	plt.plot(G_x, G_losses,label="Generator loss")
	plt.plot(D_x, D_losses,label="Discriminator loss")
	plt.xlabel("iterations")
	plt.ylabel("Loss")
	plt.legend()
	plt.savefig('plots/%s.png' % gen_name(model_name, config, tag))


def gen_name(model_name, *args):
	name = model_name
	for i in range(len(args)):
		name = name + "_" + str(args[i])
	return name

def class_noise(dim, n_classes):
	# l = class_num
	half = int(dim/2)
	means = []
	covs = []
	for l in range(n_classes):
		m1 = n_classes*np.cos((l*2*np.pi)/n_classes)
		m2 = n_classes*np.sin((l*2*np.pi)/n_classes)
		mean = [m1, m2]
		mean = np.tile(mean, half)
		v1 = [np.cos((l*2*np.pi)/n_classes), np.sin((l*2*np.pi)/n_classes)]
		v2 = [-np.sin((l*2*np.pi)/n_classes), np.cos((l*2*np.pi)/n_classes)]
		a1 = 8
		a2 = .4
		M =np.vstack((v1,v2)).T
		S = np.array([[a1, 0], [0, a2]])
		c = np.dot(np.dot(M, S), np.linalg.inv(M))
		cov = np.zeros((dim, dim))
		for i in range(half):
			cov[i*2:(i+1)*2, i*2:(i+1)*2] = c
		means.append(mean)
		covs.append(cov)
	#cov = cov*cov.T
	#vec = np.random.multivariate_normal(mean=mean, cov=cov,size=size)
	return means, covs

def sample_noise(size, latent_dim):
	noise_vector = np.zeros((size, latent_dim))
	#section = int(size/N_CLASSES)
	for i in range(size):
		noise_vector[i, :] = np.array(np.random.normal(0, 1, latent_dim))
	#for i in range(N_CLASSES):
		# noise_vector[i*section:min((i+1)*section, size), :] = class_noise(i, LATENT_DIM, min(section, size-section*i))
	return noise_vector