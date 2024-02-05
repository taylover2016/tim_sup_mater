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


def train_model(model, lr, train_dls, val_dls, num_epochs=30):
    start_time = time.time()

    criterion = nn.MSELoss()

    optimizer = optim.AdamW(
        [
            {'params': model.parameters(), 'lr': lr}
        ], weight_decay=0.1
    )
    scheduler = CosineAnnealingWarmupRestarts(
        optimizer,
        first_cycle_steps=num_epochs,
        cycle_mult=1,
        max_lr=lr,
        min_lr=lr/1000,
        warmup_steps=3,
        gamma=0.5
    )

    best_model_wts = None
    best_loss = np.inf
    loss_dict = {
        'train': [],
        'val': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch + 1}/{num_epochs}')
        print('-' * 10)
        print("Current lr: "+str(scheduler.get_lr()))

        # For every epoch, train and then evaluate on the validation set
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dls = train_dls
            else:
                model.eval()
                dls = val_dls

            running_loss = 0

            # Iterate over DataLoader
            for img, phase_true in dls:
                img = img.to(device)
                phase_true = phase_true.to(device)

                # Zero out the gradients
                if phase == 'train':
                    optimizer.zero_grad()

                # Forward pass
                # Compute gradients only in train mode
                with torch.set_grad_enabled(phase == 'train'):
                    phase_pred = model(img)

                    loss = criterion(phase_true, phase_pred)

                # Only backprop in train mode
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Compute the loss over the entire batch
                running_loss += loss.item() * img.size(0)

            # Compute loss for the epoch
            epoch_loss = running_loss / dataset_sizes[phase]
            print(f'{phase} Loss: {epoch_loss:.9f}')
            loss_dict[phase].append(epoch_loss)

            if phase == 'val':
                scheduler.step()

            # Make a deep copy of the model and save it if it's better
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.module.state_dict())
                PATH_weights = './model.pth'
                torch.save(best_model_wts, PATH_weights)

        print()

    time_elapsed = time.time() - start_time
    print(f'Training whole model complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best loss on validation set: {best_loss:9f}')

    # load best model weights
    model.module.load_state_dict(best_model_wts)
    return model, loss_dict, best_model_wts



myModel = model()

# Train the whole model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

model = nn.DataParallel(myModel)
model = model.to(device)

# First train the decoder
train_epoch = 30
lr = 1e-3

best_model, training_losses, best_weights_tmp = train_model(
        model=model,
        lr=lr,
        train_dls=train_dataloader,
        val_dls=val_dataloader,
        num_epochs=train_epoch
    )

# Plot the loss history
epochs = np.arange(train_epoch)
training_losses = {
        'train': np.array(training_losses['train']),
        'val': np.array(training_losses['val'])
    }

plt.figure()
plt.plot(epochs, training_losses['train'], color='b', label='train')
plt.plot(epochs, training_losses['val'], color='r', label='val')
plt.title('Loss for training the model with lr ' + str(lr))
plt.legend()
plt.show()
print()

