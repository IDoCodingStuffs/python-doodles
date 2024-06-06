import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
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


class CustomImageDataset(Dataset):
    def __init__(self, labels, images, transform=None, target_transform=None):
        self.labels = labels
        self.images = images
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def load_dataset():
    # Load data, create dataset
    train_data = pd.read_csv('data/rsna-2024-lumbar-spine-degenerative-classification/train.csv')

    train_images = os.listdir('data/rsna-2024-lumbar-spine-degenerative-classification/train_images')
    train_images = list(filter(lambda x: x.find('.DS') == -1, train_images))
    train_images = [(x, f"data/rsna-2024-lumbar-spine-degenerative-classification/train_images/{x}") for x in
                    train_images]

    image_metadata_set = {p[0]: {'folder_path': p[1],
                             'SeriesInstanceUIDs': []
                             }
                      for p in train_images}

    for m in image_metadata_set:
        image_metadata_set[m]['SeriesInstanceUIDs'] = list(
            filter(lambda x: x.find('.DS') == -1,
                   os.listdir(image_metadata_set[m]['folder_path'])
                   )
        )

    df_meta_f = pd.read_csv('data/rsna-2024-lumbar-spine-degenerative-classification/train_series_descriptions.csv')

    for k in tqdm(image_metadata_set):
        for s in image_metadata_set[k]['SeriesInstanceUIDs']:
            if 'SeriesDescriptions' not in image_metadata_set[k]:
                image_metadata_set[k]['SeriesDescriptions'] = []
            try:
                image_metadata_set[k]['SeriesDescriptions'].append(
                    df_meta_f[(df_meta_f['study_id'] == int(k)) &
                              (df_meta_f['series_id'] == int(s))]['series_description'].iloc[0])
            except:
                print("Failed on", s, k)

    return train_data, image_metadata_set


def get_imageset_for_patient(sample_patient, train_data, image_metadata_set):
    # !TODO: Index by some patient id
    ptobj = image_metadata_set[str(sample_patient['study_id'])]

    im_list_dcm = {}
    for idx, i in enumerate(ptobj['SeriesInstanceUIDs']):
        im_list_dcm[i] = {'images': [], 'description': ptobj['SeriesDescriptions'][idx]}
        images = glob.glob(f"{ptobj['folder_path']}/{ptobj['SeriesInstanceUIDs'][idx]}/*.dcm")
        for j in sorted(images, key=lambda x: int(x.split('/')[-1].replace('.dcm', ''))):
            im_list_dcm[i]['images'].append({
                'SOPInstanceUID': j.split('/')[-1].replace('.dcm', ''),
                'dicom': pydicom.dcmread(j)})

    return {e['description']: [x['dicom'].pixel_array for x in e['images']] for e in im_list_dcm.values()}


def show_images(imageset):
    for key, value in imageset.items():
        display_images(value, key)


def display_images(images, title, max_images_per_row=4):
    # Calculate the number of rows needed
    num_images = len(images)
    num_rows = (num_images + max_images_per_row - 1) // max_images_per_row  # Ceiling division

    # Create a subplot grid
    fig, axes = plt.subplots(num_rows, max_images_per_row, figsize=(5, 1.5 * num_rows))

    # Flatten axes array for easier looping if there are multiple rows
    if num_rows > 1:
        axes = axes.flatten()
    else:
        axes = [axes]  # Make it iterable for consistency

    # Plot each image
    for idx, image in enumerate(images):
        ax = axes[idx]
        ax.imshow(image, cmap='gray')  # Assuming grayscale for simplicity, change cmap as needed
        ax.axis('off')  # Hide axes

    # Turn off unused subplots
    for idx in range(num_images, len(axes)):
        axes[idx].axis('off')
    fig.suptitle(title, fontsize=16)

    plt.tight_layout()
    plt.show()


LABELS = ['Normal/Mild', 'Moderate', 'Severe']
EXCLUDE_COLS = ["study_id"]
def convert_to_output_vector(train_data: pd.DataFrame):
    # !TODO: Naive approach
    # !TODO: Submission requires bin probabilities, rather than 1d score, need to think about that
    train_data_features = train_data[[e for e in train_data.columns if e not in EXCLUDE_COLS]]
    train_data_features = [[LABELS.index(e_) for e_ in e] for e in train_data_features.values]
    return train_data_features

def train():
    EPOCH_COUNT = 10

    study_data, image_metadata_set = load_dataset()
    study_data = study_data.dropna()
    # First approach: just get each individual image, tack on expected labels 0-2, train away
    study_data["features"] = convert_to_output_vector(study_data)
    study_data = study_data[["study_id", "features"]]

    # !TODO: Split within same study?
    # !TODO: Tripartite split vs bipartite?
    train_data, val_data = train_test_split(study_data, test_size=0.2)

    num_classes = len(study_data["features"].iloc[0])
    model = UNet(n_channels=1, n_classes=num_classes)

    exp = []
    train_set = []

    for i in range(10):
        image_examples = get_imageset_for_patient(train_data.iloc[i], train_data, image_metadata_set)
        train_set += [np.array(e, dtype=np.int32) for e in image_examples["Sagittal T1"]]
        exp += [train_data["features"].iloc[i] for e in image_examples["Sagittal T1"]]

    # !TODO: Loader args?
    train_dataset = CustomImageDataset(exp, train_set)
    train_loader = DataLoader(train_dataset, shuffle=True)
    #pred = model(list(train_loader)[0].float().unsqueeze(0))

    # Just the first one that comes to mind. To be tooned
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.L1Loss()

    for epoch in range(EPOCH_COUNT):
        total_loss = 0
        start = time_ns()
        for image, target in train_loader:
            # !TODO: Admit any image size
            if image.shape != (1, 512, 512):
                continue
            optimizer.zero_grad()
            output = model(image.float().unsqueeze(0))
            loss = loss_fn(output.squeeze(0).squeeze(0), torch.tensor(target).unsqueeze(0))
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        end = time_ns()
        print("Loss at epoch", epoch, total_loss)
        print("Seconds elapsed at epoch", epoch, (end - start) // 1e9)

train()
