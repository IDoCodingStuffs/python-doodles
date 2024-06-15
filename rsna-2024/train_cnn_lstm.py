import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from rsna_dataloader import *

data_basepath = "C://Users/Victor/Documents/python-doodles/data/rsna-2024-lumbar-spine-degenerative-classification/"
training_data = retrieve_training_data(data_basepath)

transform_train = transforms.Compose([
    transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])

trainloader_t1, valloader_t1, len_train_t1, len_val_t1 = create_series_level_datasets_and_loaders(training_data, 'Sagittal T1', transform_train, transform_train, data_basepath + "train_images")
trainloader_t2, valloader_t2, len_train_t2, len_val_t2 = create_series_level_datasets_and_loaders(training_data, 'Axial T2', transform_train, transform_train, data_basepath + "train_images")
trainloader_t2stir, valloader_t2stir, len_train_t2stir, len_val_t2stir = create_series_level_datasets_and_loaders(training_data, 'Sagittal T2/STIR', transform_train, transform_train, data_basepath + "train_images")


class CustomResNet(nn.Module):
    def __init__(self, pretrained_weights=None):
        super(CustomResNet, self).__init__()
        self.model = models.resnet50(pretrained=False)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features=num_ftrs, out_features=512)

    def forward(self, x):
        return self.model(x)


class CustomLSTM(nn.Module):
    hidden_size = 256
    num_layers = 3

    def __init__(self, num_classes=3, resnet_weights=None):
        super(CustomLSTM, self).__init__()
        self.resnet = CustomResNet(pretrained_weights=resnet_weights)
        self.lstm = nn.LSTM(input_size=512, hidden_size=self.hidden_size, num_layers=self.num_layers, batch_first=True)
        self.tail = nn.Sequential(
            nn.Linear(self.hidden_size, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
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
        x = self.tail(hidden[0][-1])

        return x

