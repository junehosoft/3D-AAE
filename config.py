# -*- coding: utf-8 -*-
INTER_DIM_0 = 64
INTER_DIM_1 = 32
INTER_DIM_2 = 16
INTER_DIM_3 = 8
INTER_DIM_4 = 4

INTER_CH_1 = 64
INTER_CH_2 = 128
INTER_CH_3 = 256
INTER_CH_4 = 512

TEST_BATCH_SIZE = 8
FIXED_NOISE = True

TRAIN_BATCH_SIZE = 8
LR_G = 0.0001
LR_D = 0.0025
N_EPOCHS = 5
SAMPLE_INTERVAL = 3
NUM_WORKERS = 2
DATASET_BASE = 'data/'
GENERATED_BASE = 'images/'
IMG_SIZE = 64
CATEGORIES = ['bed', 'bookcase', 'chair', 'desk', 'sofa', 'table', 'wardrobe']
INDICES = [5]
N_CLASSES = len(INDICES)
CATEGORIES_AS_STR = ",".join([str(c) for c in INDICES])
LATENT_DIM = 200
N_CRITIC = 1
CONFIG_AS_STR = "_".join([str(c) for c in [CATEGORIES_AS_STR, LATENT_DIM, N_EPOCHS]])
