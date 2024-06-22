import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import logging
import seaborn as sn
from itertools import chain
from transformers import AutoImageProcessor, AutoModelForImageClassification

from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


# !TODO: Optional
def freeze_model_backbone(model: nn.Module):
    for param in model.backbone.model.parameters():
        param.requires_grad = False
    # for param in model.backbone.model.fc.parameters():
    #     param.requires_grad = True


# !TODO: Optional
def unfreeze_model_backbone(model: nn.Module):
    for param in model.backbone.model.parameters():
        param.requires_grad = True


def model_validation_loss(model, val_loader, loss_fns):
    model.eval()
    total_loss = 0

    for images, label in val_loader:
        # !TODO: Do this in the data loader
        label = label.to(device)

        output = model(images.to(device))
        for index, loss_fn in enumerate(loss_fns):
            loss = loss_fn(output[index], label[index])
            total_loss += loss.cpu().item()

    total_loss = total_loss / len(val_loader)

    return total_loss


def dump_plots_for_loss_and_acc(losses, val_losses, data_subset_label, model_label):
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend(loc="center right")
    plt.title(data_subset_label)
    # plt.savefig(f'./figures/{model_label}_{time.time_ns() // 1e9}_loss.png')
    plt.savefig(f'./figures/{model_label}_loss.png')
    plt.close()


def train_model_with_validation(model, optimizers, schedulers, loss_fns, train_loader, val_loader,
                                train_loader_desc=None,
                                model_desc="my_model", epochs=10):
    epoch_losses = []
    epoch_validation_losses = []

    # freeze_model_backbone(model)

    for epoch in tqdm(range(epochs), desc=train_loader_desc):
        epoch_loss = 0
        model.train()

        # if epoch >= 10:
        #     unfreeze_model_backbone(model)

        for images, label in train_loader:
            # !TODO: Do this in the data loader
            label = label.to(device)
            # label = label.type(torch.LongTensor).to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            output = model(images.to(device))

            # !TODO: Refactor
            # !TODO: Track separately
            for index, loss_fn in enumerate(loss_fns):
                loss = loss_fn(output[index], label[index])
                epoch_loss += loss.cpu().item()
                loss.backward(retain_graph=True)

            for optimizer in optimizers:
                optimizer.step()

        epoch_loss = epoch_loss / len(train_loader)

        epoch_validation_loss = model_validation_loss(model, val_loader, loss_fns)

        for scheduler in schedulers:
            scheduler.step()

        if (epoch + 1) % 20 == 0 or (epoch_validation_losses and epoch_validation_loss < min(epoch_validation_losses)):
            os.makedirs(f'./models/{model_desc}', exist_ok=True)
            torch.save(model, f'./models/{model_desc}/{model_desc}' + "_" + str(epoch) + ".pt")

        epoch_validation_losses.append(epoch_validation_loss)
        epoch_losses.append(epoch_loss)

        dump_plots_for_loss_and_acc(epoch_losses, epoch_validation_losses,
                                    train_loader_desc, model_desc)
        print(f"Training Loss for epoch {epoch}: {epoch_loss:.6f}")
        print(f"Validation Loss for epoch {epoch}: {epoch_validation_loss:.6f}")

    return epoch_losses, epoch_validation_losses
