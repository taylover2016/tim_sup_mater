# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import math
import time
import random
import os
import copy
import pathlib

import torch
from torch import nn
from torch.nn import functional as F
import torch.optim as optim
from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
from torchvision.io import read_image
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torch.utils.data.dataset import random_split

from torchvision import models
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

from model import model


# Define a custom dataset
class PhaseDataset(Dataset):
    """ Custom Phase Dataset """

    def __init__(self, csv_file, root_dir, transform=None):
        """
          Args:
              csv_file (string): Path to the csv file with annotations.
              root_dir (string): Directory with all the images.
              transform (callable, optional): Optional transform to be applied
                  on a sample.
        """
        self.data_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_name = self.data_frame.iloc[idx, 0]
        phase_name = self.data_frame.iloc[idx, 1]

        image = (1 / 255) * read_image(img_name).float()

        phase = torch.load(phase_name)
        phase = phase.unsqueeze(0)

        if self.transform:
            image, phase = self.transform(image, phase)

        return image, phase

# Define some transforms for training and testing
def train_transform(image, phase, seed=123456):
    """
    This function returns a composition of data augmentations to a single training image.
    """
    random.seed(seed)
    torch.random.manual_seed(seed)

    # Step 1:  With a probability of 0.5, apply horizontal flip.
    if random.random() < 0.5:
        image = TF.hflip(image)
        phase = TF.hflip(phase)

    # Step 2: With a probability of 0.5, apply vertical flip
    if random.random() < 0.5:
        image = TF.vflip(image)
        phase = TF.vflip(phase)

    Normalizer = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = Normalizer(image)

    return image, phase

def test_transform(image, phase):
    Normalizer = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    image = Normalizer(image)

    return image, phase


# Load datasets
train_path = "./Ideal/train"
test_path = "./Ideal/test"

dataset_seed = 42

train_data = PhaseDataset(
    csv_file=train_path+'/data.csv',
    root_dir=train_path,
    transform=train_transform
)

val_test_data = PhaseDataset(
    csv_file=test_path+'/data.csv',
    root_dir=test_path,
    transform=test_transform
)

train_set = train_data
val_test_set = val_test_data

test_len = int((1/4)*len(val_test_set))
val_len = len(val_test_set) - test_len
val_set, test_set = random_split(val_test_set, [val_len, test_len])


datasets = {'train': train_set,
            'val': val_set,
            'test': test_set,
            }
dataset_sizes = {x: len(datasets[x]) for x in datasets.keys()}

print(dataset_sizes['val'])


# Create a DataLoader for both train and test dataset
train_dataloader = DataLoader(train_set, batch_size=512, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=512, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=512, shuffle=True)


myModel = model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

test_model = model.to(device)

PATH_weights = './model.pth'
test_model.load_state_dict(torch.load(PATH_weights))
test_model.eval()

image, phase_true = test_set[0]
phase_pred = test_model(image.unsqueeze(0).to(device))
phase_true = phase_true.to(device)

print(phase_true)
print(phase_pred)
