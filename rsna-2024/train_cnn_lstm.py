import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm
import logging
from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
label_map = {'normal_mild': 0, 'moderate': 1, 'severe': 2}


class CustomResNet(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(CustomResNet, self).__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=num_ftrs, out_features=512)

    def forward(self, x):
        return self.model(x)


class CustomLSTM(nn.Module):
    hidden_size = 512
    num_layers = 3

    def __init__(self, num_classes=3, drop_rate=0.3, resnet_weights=None):
        super(CustomLSTM, self).__init__()
        self.resnet = CustomResNet(pretrained_weights=resnet_weights)
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.1),
            nn.Linear(256, num_classes),
        )
    def forward(self, x_3d):
        hidden = None

        # Iterate over each frame of a video in a video of batch * frames * channels * height * width
        for t in range(x_3d.size(1)):
            with torch.no_grad():
                x = self.resnet(x_3d[:, t])
                # Pass latent representation of frame through lstm and update hidden state
            out, hidden = self.lstm(x.unsqueeze(0), hidden)

            # Get the last hidden state (hidden is a tuple with both hidden and cell state in it)
        x = self.head(hidden[0][-1])

        return x


def freeze_model_initial_layers(model: CustomLSTM):
    for param in model.resnet.model.parameters():
        param.requires_grad = False
    for param in model.resnet.model.fc.parameters():
        param.requires_grad = True


def model_validation_loss(model, val_loader, loss_fn):
    model.eval()
    total_loss = 0
    acc = 0
    for images, labels in val_loader:
        labels = torch.tensor([label_map[label] for label in labels])
        labels = labels.to(device)

        output = model(images.to(device))
        loss = loss_fn(output, labels)
        total_loss += loss.item()

        acc += torch.sum(torch.argmax(output) == labels).item()

    acc = acc / len(val_loader.dataset)

    return total_loss, acc


def dump_plots_for_loss_and_acc(losses, val_losses, acc, val_acc, data_subset_label, model_label):
    plt.plot(losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend(loc="center right")
    plt.title(data_subset_label)
    # plt.savefig(f'./figures/{model_label}_{time.time_ns() // 1e9}_loss.png')
    plt.savefig(f'./figures/{model_label}_loss.png')
    plt.close()

    plt.plot(acc, label="train")
    plt.plot(val_acc, label="val")
    plt.title(data_subset_label)
    plt.legend(loc="center right")
    plt.savefig(f'./figures/{model_label}_acc.png')
    plt.close()


def train_model_with_validation(model, optimizer, scheduler, loss_fn, train_loader, val_loader, train_loader_desc=None,
                                model_desc="my_model", epochs=10):
    epoch_losses = []
    epoch_validation_losses = []
    epoch_accs = []
    epoch_val_accs = []

    for epoch in tqdm(range(epochs), desc=train_loader_desc):
        epoch_loss = 0
        epoch_acc = 0
        model.train()

        for images, labels in train_loader:
            labels = torch.tensor([label_map[label] for label in labels])
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(images.to(device))
            loss = loss_fn(output, labels)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            epoch_acc += torch.sum(torch.argmax(output) == labels).item()

        epoch_acc = epoch_acc / len(train_loader.dataset)

        epoch_validation_loss, epoch_validation_acc = model_validation_loss(model, val_loader, loss_fn)
        scheduler.step()

        if len(epoch_validation_losses) == 0 or epoch_validation_loss < min(epoch_validation_losses):
            torch.save(model, "./models/" + model_desc + ".pt")

        epoch_validation_losses.append(epoch_validation_loss)
        epoch_losses.append(epoch_loss)
        epoch_accs.append(epoch_acc)
        epoch_val_accs.append(epoch_validation_acc)

        dump_plots_for_loss_and_acc(epoch_losses, epoch_accs, epoch_validation_losses, epoch_val_accs, train_loader_desc, model_desc)

    return epoch_losses, epoch_validation_losses, epoch_accs, epoch_val_accs


def train_model_for_series(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_training_data(data_basepath)

    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    trainloader, valloader, len_train, len_val = create_series_level_datasets_and_loaders(training_data,
                                                                                          data_subset_label,
                                                                                          transform_train,
                                                                                          transform_train,
                                                                                          data_basepath + "train_images",
                                                                                          num_workers=4)
    weights_path = './models/resnet50-19c8e357.pth'
    NUM_EPOCHS = 30

    # model = CustomLSTM(resnet_weights=weights_path).to(device)
    model = CustomLSTM().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, NUM_EPOCHS, eta_min=23e-6)

    freeze_model_initial_layers(model)
    criterion = nn.CrossEntropyLoss()

    train_model_with_validation(model,
                                optimizer,
                                scheduler,
                                criterion,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {data_subset_label}",
                                epochs=NUM_EPOCHS)

    return model


def train():
    model_t2stir = train_model_for_series("Sagittal T2/STIR", "resnet18_lstm_t2stir")
    model_t2 = train_model_for_series("Axial T2", "resnet18_lstm_t2")
    model_t1 = train_model_for_series("Sagittal T1", "resnet18_lstm_t1")


if __name__ == '__main__':
    train()
