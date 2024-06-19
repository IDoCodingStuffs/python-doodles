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
from fastmri.models.unet import Unet

from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResnetBackbone(nn.Module):
    def __init__(self, out_features=512, pretrained_weights=None):
        super(ResnetBackbone, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=num_ftrs, out_features=out_features)
        torch.nn.init.xavier_uniform(self.model.fc.weight)

    def forward(self, x):
        return self.model(x)


class ConvToFC(nn.Module):
    def __init__(self, in_channels=1, in_dims=512, out_dims=512):
        super(ConvToFC, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=(in_dims, 1))
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_dims, out_dims)
    def forward(self, x):
        return self.fc(self.relu(self.conv(x)))


class UNetBackbone(nn.Module):
    def __init__(self, out_features=512, pretrained_weights=None):
        super(UNetBackbone, self).__init__()
        self.model = Unet(num_pool_layers=4, drop_prob=0.05, in_chans=1, out_chans=1)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))

    def forward(self, x):
        return self.model(x)


class FCHead(nn.Module):
    def __init__(self, drop_rate=0.1, num_classes=3):
        super(FCHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            # nn.BatchNorm1d(256),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class RNSAModel2_5D(nn.Module):
    hidden_size = 256
    num_layers = 3

    def __init__(self, num_classes=2, num_levels=5, drop_rate=0.2, resnet_weights=None):
        super(RNSAModel2_5D, self).__init__()
        self.backbone = UNetBackbone(pretrained_weights=resnet_weights)
        self.temporal = nn.LSTM(input_size=512, hidden_size=self.hidden_size, dropout=drop_rate, num_layers=self.num_layers,
                                batch_first=True,
                                bidirectional=True)
        self.head = FCHead(num_classes=num_levels * num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x_3d):
        hidden = None

        # Iterate over each frame of a video in a video of batch * frames * channels * height * width
        for t in range(x_3d.size(1)):
            x = self.backbone(x_3d[:, t])
            # Pass latent representation of frame through lstm and update hidden state
            # out, hidden = self.temporal(x.unsqueeze(0), hidden)
            out, hidden = self.temporal(x.squeeze(0), hidden)

            # Get the last hidden state (hidden is a tuple with both hidden and cell state in it)

        x = self.head(hidden[0][-1])
        return self.activation(x)


def freeze_model_backbone(model: RNSAModel2_5D):
    for param in model.backbone.model.parameters():
        param.requires_grad = False
    # for param in model.backbone.fc.parameters():
    #     param.requires_grad = True


def unfreeze_model_backbone(model: RNSAModel2_5D):
    for param in model.backbone.model.parameters():
        param.requires_grad = True


def model_validation_loss(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0

    for images, label in val_loader:
        # !TODO: Do this in the data loader
        label = label.type(torch.FloatTensor).to(device)

        output = model(images.to(device))
        loss = loss_fn(output, label)
        total_loss += loss.item()

    total_loss = total_loss / len(val_loader.dataset)

    return total_loss


def dump_plots_for_loss_and_acc(losses, val_losses, data_subset_label, model_label):
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend(loc="center right")
    plt.title(data_subset_label)
    # plt.savefig(f'./figures/{model_label}_{time.time_ns() // 1e9}_loss.png')
    plt.savefig(f'./figures/{model_label}_loss.png')
    plt.close()


def train_model_with_validation(model, optimizers, schedulers, loss_fn, train_loader, val_loader, train_loader_desc=None,
                                model_desc="my_model", epochs=10):
    epoch_losses = []
    epoch_validation_losses = []

    freeze_model_backbone(model)

    for epoch in tqdm(range(epochs), desc=train_loader_desc):
        epoch_loss = 0
        model.train()

        # if epoch >= 10:
        if epoch >= 0:
            unfreeze_model_backbone(model)

        for images, label in train_loader:
            # !TODO: Do this in the data loader
            label = label.type(torch.FloatTensor).to(device)

            for optimizer in optimizers:
                optimizer.zero_grad()

            output = model(images.to(device))

            loss = loss_fn(output, label)
            epoch_loss += loss.detach().item()
            loss.backward(retain_graph=True)

            for optimizer in optimizers:
                optimizer.step()

        epoch_loss = epoch_loss / len(train_loader.dataset)

        epoch_validation_loss = model_validation_loss(model, val_loader, loss_fn)

        for scheduler in schedulers:
            scheduler.step()

        if epoch % 5 == 0:
            torch.save(model, "./models/" + model_desc + "_" + str(epoch) + ".pt")

        epoch_validation_losses.append(epoch_validation_loss)
        epoch_losses.append(epoch_loss)

        dump_plots_for_loss_and_acc(epoch_losses, epoch_validation_losses,
                                    train_loader_desc, model_desc)
        print(f"Training Loss for epoch {epoch}: {epoch_loss:.6f}")
        print(f"Validation Loss for epoch {epoch}: {epoch_validation_loss:.6f}")

    return epoch_losses, epoch_validation_losses


def train_model_for_series(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_training_data(data_basepath)

    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomVerticalFlip(p=0.3),
        # transforms.RandomRotation([0, 90]),
        transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
            v2.Identity(),
        ], p=[0.25, 0.75]),
        v2.RandomPhotometricDistort(p=0.3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
    ])

    trainloader, valloader, len_train, len_val = create_series_level_datasets_and_loaders(training_data,
                                                                                          data_subset_label,
                                                                                          transform_val,
                                                                                          # Try overfitting first
                                                                                          transform_val,
                                                                                          data_basepath + "train_images",
                                                                                          num_workers=0, batch_size=1)

    NUM_EPOCHS = 40

    model = RNSAModel2_5D().to(device)
    optimizers = [torch.optim.Adam(model.head.parameters(), lr=1e-3),
                  torch.optim.Adam(model.temporal.parameters(), lr=5e-4),
                  torch.optim.Adam(model.backbone.parameters(), lr=1e-4)]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=5e-5),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], NUM_EPOCHS, eta_min=2e-6),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[2], NUM_EPOCHS, eta_min=5e-7),
        ]

    criterion = nn.BCELoss()

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criterion,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {data_subset_label}",
                                epochs=NUM_EPOCHS)

    return model


def train():
    model_t2stir = train_model_for_series("Sagittal T2/STIR", "resnet18_lstm_t2stir")
    # model_t1 = train_model_for_series("Sagittal T1", "resnet18_lstm_t1")
    # model_t2 = train_model_for_series("Axial T2", "resnet18_lstm_t2")


if __name__ == '__main__':
    train()
