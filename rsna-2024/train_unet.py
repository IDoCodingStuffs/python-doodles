import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import Resize
from pathlib import Path
from torch import optim, nn
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, random_split, Dataset
from tqdm import tqdm

import pandas as pd
import numpy as np
import os
import glob
import pydicom
import matplotlib.pyplot as plt
from time import time_ns

from unet import UNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _model_validation_loss(model,
                           val_loader,
                           loss_fn,
                           label_map,
                           desc=None):
    model.eval()
    total_loss = 0
    for images, labels in tqdm(val_loader, desc=desc):
        labels = torch.tensor([label_map[label] for label in labels])
        labels = labels.to(device)

        output = model(images.to(device))
        loss = loss_fn(output, labels)
        total_loss += loss.item()
    return total_loss


def _train_model_with_validation(model,
                                 optimizer,
                                 loss_fn,
                                 train_loader,
                                 val_loader,
                                 label_map,
                                 train_loader_desc=None,
                                 model_desc="my_model",
                                 epochs=10):
    epoch_losses = []
    epoch_validation_losses = []
    for epoch in range(epochs):
        epoch_loss = 0
        model.train()

        for images, labels in tqdm(train_loader, desc=train_loader_desc):
            labels = torch.tensor([label_map[label] for label in labels])
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images.to(device))
            # !TODO: This is sus
            loss = loss_fn(output.squeeze(1,2), labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_validation_loss = _model_validation_loss(model, val_loader, loss_fn)
        if len(epoch_validation_losses) == 0 or epoch_validation_loss < min(epoch_validation_losses):
            torch.save(model, "/Users/victorsahin/PycharmProjects/pythonProject/models/" + model_desc + ".pt")
        epoch_validation_losses.append(epoch_validation_loss)
        epoch_losses.append(epoch_loss)

    return epoch_losses, epoch_validation_losses


def train(training_dataloader, val_dataloader, label_map, epoch_count=50) -> UNet:
    # Just the first one that comes to mind. To be tooned
    model = UNet(n_channels=1, n_classes=3)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    loss_fn = nn.CrossEntropyLoss()

    train_losses, val_losses = _train_model_with_validation(model,
                                                            optimizer,
                                                            loss_fn,
                                                            training_dataloader,
                                                            val_dataloader,
                                                            label_map,
                                                            train_loader_desc=None,
                                                            model_desc="unet",
                                                            epochs=epoch_count)

    plt.plot(train_losses)
    plt.plot(val_losses)
    plt.title("Train and val losses per epoch")

    return model

#my_model = train()
#pass
