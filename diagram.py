from config import *
from utils import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    means, covs = class_noise(2, N_CLASSES)
    noise = []
    for n in range(N_CLASSES):
        noise.append(np.random.multivariate_normal(mean=means[n], cov=covs[n],size=50))

    fig = plt.figure()
    for i in range(N_CLASSES):
        n = noise[i]
        plt.scatter(n[:, 0], n[:, 1], label=CATEGORIES[INDICES[i]])
    plt.title("Desired distribution of training samples in latent space")
    plt.legend()
    plt.savefig('desire.png')