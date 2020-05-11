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
