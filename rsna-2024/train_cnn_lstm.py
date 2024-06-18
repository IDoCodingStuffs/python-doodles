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

from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomResNet(nn.Module):
    def __init__(self, out_features=512, pretrained_weights=None):
        super(CustomResNet, self).__init__()
        self.model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=num_ftrs, out_features=out_features)
        torch.nn.init.xavier_uniform(self.model.fc.weight)

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


class CustomLSTM(nn.Module):
    hidden_size = 256
    num_layers = 3

    def __init__(self, num_classes=3, num_levels=5, drop_rate=0.2, resnet_weights=None):
        super(CustomLSTM, self).__init__()
        self.cnn = CustomResNet(pretrained_weights=resnet_weights)
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_size, dropout=drop_rate, num_layers=self.num_layers,
                            batch_first=True,
                            bidirectional=True)
        self.heads = [FCHead().to(device) for i in range(num_levels)]

    def forward(self, x_3d):
        hidden = None

        # Iterate over each frame of a video in a video of batch * frames * channels * height * width
        for t in range(x_3d.size(1)):
            x = self.cnn(x_3d[:, t])
            # Pass latent representation of frame through lstm and update hidden state
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

            # Get the last hidden state (hidden is a tuple with both hidden and cell state in it)

        return [head(hidden[0][-1]) for head in self.heads]


def freeze_model_initial_layers(model: CustomLSTM):
    for param in model.cnn.model.parameters():
        param.requires_grad = False
    for param in model.cnn.model.fc.parameters():
        param.requires_grad = True


def model_validation_loss(model, val_loader, loss_fns):
    model.eval()
    total_loss = 0

    for images, label in val_loader:
        # !TODO: Do this in the data loader
        label = label.type(torch.FloatTensor).to(device)

        output = model(images.to(device))
        for index, loss_fn in enumerate(loss_fns):
            loss = loss_fn(output[index].squeeze(0), label.squeeze(0)[index])
            total_loss += loss.item()

    total_loss = total_loss / len(val_loader.dataset) / len(loss_fns)

    return total_loss


def dump_plots_for_loss_and_acc(losses, val_losses, data_subset_label, model_label):
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend(loc="center right")
    plt.title(data_subset_label)
    # plt.savefig(f'./figures/{model_label}_{time.time_ns() // 1e9}_loss.png')
    plt.savefig(f'./figures/{model_label}_loss.png')
    plt.close()


def train_model_with_validation(model, optimizer, scheduler, loss_fns, train_loader, val_loader, train_loader_desc=None,
                                model_desc="my_model", epochs=10):
    epoch_losses = []
    epoch_validation_losses = []

    for epoch in tqdm(range(epochs), desc=train_loader_desc):
        epoch_loss = 0
        model.train()

        for images, label in train_loader:
            # !TODO: Do this in the data loader
            label = label.type(torch.FloatTensor).to(device)

            optimizer.zero_grad()
            output = model(images.to(device))

            for index, loss_fn in enumerate(loss_fns):
                loss = loss_fn(output[index].squeeze(0), label.squeeze(0)[index])
                epoch_loss += loss.detach().item()
                loss.backward(retain_graph=True)

            optimizer.step()

        epoch_loss = epoch_loss / len(train_loader.dataset) / 5

        epoch_validation_loss = model_validation_loss(model, val_loader, loss_fns)
        scheduler.step()

        if len(epoch_validation_losses) == 0 or epoch_validation_loss < min(epoch_validation_losses):
            torch.save(model, "./models/" + model_desc + ".pt")

        epoch_validation_losses.append(epoch_validation_loss)
        epoch_losses.append(epoch_loss)

        dump_plots_for_loss_and_acc(epoch_losses, epoch_validation_losses,
                                    train_loader_desc, model_desc)
        print(f"Training Loss for epoch {epoch}: {epoch_loss:.4f}")
        print(f"Validation Loss for epoch {epoch}: {epoch_validation_loss:.4f}")

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
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    trainloader, valloader, len_train, len_val = create_series_level_datasets_and_loaders(training_data,
                                                                                          data_subset_label,
                                                                                          transform_val,
                                                                                          # Try overfitting first
                                                                                          transform_val,
                                                                                          data_basepath + "train_images",
                                                                                          num_workers=4, batch_size=1)
    weights_path = './models/resnet50-19c8e357.pth'
    NUM_EPOCHS = 40

    model = CustomLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=23e-6)

    freeze_model_initial_layers(model)

    criteria = [nn.BCEWithLogitsLoss() for i in range(5)]

    train_model_with_validation(model,
                                optimizer,
                                scheduler,
                                criteria,
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
