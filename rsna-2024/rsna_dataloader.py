import random

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import cv2

import pydicom
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.transforms import v2

label_map = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
conditions = ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing", "Left Subarticular Stenosis",
              "Right Subarticular Stenosis", "Spinal Canal Stenosis"]


class PerImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.label = (dataframe[["image_path", "level", "severity"]]
                      .groupby("image_path")
                      # !TODO: Unhardcode
                      .filter(lambda x: len(x) == 5))

        self.label = (self.label.groupby("image_path")
                      .agg({"image_path": "unique",
                            "level": lambda x: ",".join(x),
                            "severity": lambda x: ",".join(x)}))

        # !TODO: refactor and properly formulate
        self.label["weight"] = [sum(2 ** np.argmax(self.label_as_tensor(e[0]).numpy(), axis=1)) + 1 for e in
                                self.label["image_path"]]

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image_path = self.label['image_path'].iloc[index][0]
        image = load_dicom(image_path)

        if self.transform:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = self.transform(image)

        return image, self.label_as_tensor(image_path)

    def label_as_tensor(self, image_path):
        row = self.label[self.label["image_path"] == image_path]
        levels = row["level"].str.split(",").values[0]
        levels = sorted([(val, index) for index, val in enumerate(levels)])

        label = row["severity"].values[0].split(",")
        label = [label_map[label[index]] for level, index in levels]

        label = self._one_hot_encode_multihead(label)

        return torch.tensor(label).type(torch.FloatTensor)

    def _multi_hot_encode(self, label):
        ret = []
        for l in label:
            # !TODO: Unhardcode
            for i in range(1, 3):
                ret.append(1 if i <= l else 0)
        return ret

    def _one_hot_encode_multihead(self, label):
        ret = [[1 if i == l else 0 for i in range(3)] for l in label]
        return ret

    def _one_hot_encode_multihead(self, label):
        ret = [[1 if i == l else 0 for i in range(3)] for l in label]
        return ret


class CoordinateDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.dataframe = (dataframe[['study_id', "series_id", "level", "x", "y", "image_path"]]
                          .drop_duplicates()
                          .dropna())
        self.levels = sorted(self.dataframe["level"].unique())
        self.coords = dataframe[["image_path", "level", "x", "y"]].drop_duplicates().dropna()
        self.coords = self.coords.groupby('image_path').filter(lambda x: len(x) == len(self.levels)).reset_index(
            drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        image_path = self.coords['image_path'][index]
        image = load_dicom(image_path)

        label = self._get_coords_given_image_path(image_path)

        if self.transform:
            image, label = self.transform(image, label)

        return image, label

    def _get_coords_given_image_path(self, image_path):
        subset = self.coords[(self.coords["image_path"] == image_path)]
        subset = subset.sort_values(by="level")

        ret = [0 for i in range(len(self.levels) * 2)]
        for index, row in subset.iterrows():
            ret[self.levels.index(row["level"]) * 2] = row["x"]
            ret[self.levels.index(row["level"]) * 2 + 1] = row["y"]

        return ret


class SeriesLevelDataset(Dataset):
    def __init__(self, base_path: str, dataframe: pd.DataFrame, transform=None):
        self.base_path = base_path

        # !TODO: Impute later
        self.dataframe = dataframe.groupby(['study_id', "series_id"]).filter(lambda x: len(x["level"].unique()) == 5)
        self.dataframe = (self.dataframe[['study_id', "series_id", "severity", "level"]]
                          .drop_duplicates())
        self.series = self.dataframe[['study_id', "series_id"]].drop_duplicates().reset_index(drop=True)

        self.transform = transform

        self.levels = sorted(self.dataframe["level"].unique())
        self.labels = dict()

        for name, group in self.dataframe.groupby(["study_id", "series_id"]):
            # !TODO: Better imputation
            label_indices = [0 for e in range(len(self.levels))]
            for index, row in group.iterrows():
                if row["severity"] in label_map:  # and row["condition"] in conditions:
                    label_index = self.levels.index(row["level"])
                    label_indices[label_index] = label_map[row["severity"]]

            self.labels[name] = []
            for label in label_indices:
                curr = [0 if label != i else 1 for i in range(3)]
                self.labels[name].append(curr)


        # !TODO:Refactor
        # !TODO: Revisit
        self.weights = []
        for index, row in self.series.iterrows():
            curr = self.labels[(row["study_id"], row["series_id"])]
            if np.argmax(curr[0]) != 0 or np.argmax(curr[-1]) != 0:
                self.weights.append(20)
            elif np.argmax(curr[1]) != 0:
                self.weights.append(10)
            elif np.argmax(curr[2]) != 0:
                self.weights.append(5)
            elif np.argmax(curr[3]) != 0:
                self.weights.append(3)
            else:
                self.weights.append(1)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        curr = self.series.iloc[index]
        image_paths = retrieve_image_paths(self.base_path, curr["study_id"], curr["series_id"])
        image_paths = sorted(image_paths, key=lambda x: self._get_image_index(x))
        label = np.array(self.labels[(curr["study_id"], curr["series_id"])])

        # Pad to max 29
        images = np.array([self.transform(load_dicom(image_path)) if self.transform
                           else load_dicom(image_path) for image_path in image_paths])

        # front_buffer = (29 - len(images)) // 2
        # rear_buffer = (29 - len(images)) // 2 + ((29 - len(images)) % 2)

        # +1 is for the BERT-like pooling
        # images = np.pad(images, ((front_buffer + 1, rear_buffer), (0, 0), (0, 0), (0, 0)))
        images = np.pad(images, ((1, 0), (0, 0), (0, 0), (0, 0)))

        return images, torch.tensor(label).type(torch.FloatTensor)

    def _get_image_index(self, image_path):
        return int(image_path.split("/")[-1].split("\\")[-1].replace(".dcm", ""))


# !TODO: Use inheritance
class SeriesLevelCoordinateDataset(Dataset):
    def __init__(self, base_path: str, dataframe: pd.DataFrame, transform=None):
        self.base_path = base_path
        self.dataframe = (dataframe[['study_id', "series_id", "x", "y", "level"]]
                          .drop_duplicates())
        self.transform = transform
        self.series = dataframe[['study_id', "series_id"]].drop_duplicates().reset_index(drop=True)
        self.levels = sorted(self.dataframe["level"].unique())
        self.labels = dict()
        for name, group in self.dataframe.groupby(["study_id", "series_id"]):
            # !TODO: Refine this
            labels = [0 for e in range(len(self.levels) * 2)]
            for index, row in group.iterrows():
                level_index = self.levels.index(row["level"])
                label_indices = (level_index * 2, level_index * 2 + 1)
                labels[label_indices[0]] = row["x"]
                labels[label_indices[1]] = row["y"]

            self.labels[name] = torch.tensor(labels)

    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        curr = self.series.iloc[index]
        image_paths = retrieve_image_paths(self.base_path, curr["study_id"], curr["series_id"])
        image_paths = sorted(image_paths, key=lambda x: self._get_image_index(x))
        images = np.array([self.transform(load_dicom(image_path)) if self.transform else load_dicom(image_path)
                           for image_path in image_paths])

        label = self.labels[(curr["study_id"], curr["series_id"])]

        return images, label

    def _get_image_index(self, image_path):
        return int(image_path.split("/")[-1].split("\\")[-1].replace(".dcm", ""))


class TrainingTransform(nn.Module):
    def __init__(self, image_size=(224, 224), num_channels=3):
        super(TrainingTransform, self).__init__()
        self.image_size = image_size

        self.to_uint8 = transforms.Lambda(lambda x: (x * 255).astype(np.uint8))
        self.to_pil = transforms.ToPILImage()
        # !TODO: Refactor image dims
        self.resize = transforms.Resize(self.image_size)
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.vflip = transforms.RandomVerticalFlip(p=0.5)

        self.gaussian_blur = transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3)),
            transforms.GaussianBlur(kernel_size=7, sigma=(0.1, 3)),
            v2.Identity(),
        ], p=[0.2, 0.2, 0.2, 0.4])

        self.grayscale = transforms.Grayscale(num_output_channels=num_channels)
        self.to_tensor = transforms.ToTensor()

    def forward(self, image):
        image = self.to_uint8(image)
        image = self.to_pil(image)
        image = self.resize(image)
        image = self.grayscale(image)
        # image = self.hflip(image)
        # image = self.vflip(image)
        image = self.gaussian_blur(image)
        image = self.to_tensor(image)

        return image


class ValidationTransform(nn.Module):
    def __init__(self, image_size=(224, 224), num_channels=3):
        super(ValidationTransform, self).__init__()
        self.image_size = image_size

        self.to_uint8 = transforms.Lambda(lambda x: (x * 255).astype(np.uint8))
        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize(image_size)
        self.grayscale = transforms.Grayscale(num_output_channels=num_channels)
        self.to_tensor = transforms.ToTensor()

    def forward(self, image):
        image = self.to_uint8(image)
        image = self.to_pil(image)
        image = self.resize(image)
        image = self.grayscale(image)

        image = self.to_tensor(image)

        return image


# !TODO: Avoid duplication
def create_series_level_datasets_and_loaders(df: pd.DataFrame,
                                             series_description: str,
                                             transform_train: nn.Module,
                                             transform_val: nn.Module,
                                             base_path: str,
                                             split_factor=0.2,
                                             random_seed=42,
                                             batch_size=1,
                                             num_workers=0):
    filtered_df = df[df['series_description'] == series_description]

    train_studies, val_studies = train_test_split(filtered_df["study_id"], test_size=split_factor,
                                                  random_state=random_seed)
    train_df = filtered_df[filtered_df["study_id"].isin(train_studies)]
    val_df = filtered_df[filtered_df["study_id"].isin(val_studies)]
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    random.seed(random_seed)
    train_dataset = SeriesLevelDataset(base_path, train_df, transform_train)
    val_dataset = SeriesLevelDataset(base_path, val_df, transform_val)

    train_picker = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_picker,num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset, val_dataset


def create_series_level_coordinate_datasets_and_loaders(df: pd.DataFrame,
                                                        series_description: str,
                                                        transform_train: transforms.Compose,
                                                        transform_val: transforms.Compose,
                                                        base_path: str,
                                                        split_factor=0.2,
                                                        random_seed=42,
                                                        batch_size=1,
                                                        num_workers=0):
    filtered_df = df[df['series_description'] == series_description]

    train_df, val_df = train_test_split(filtered_df, test_size=split_factor, random_state=random_seed)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = SeriesLevelCoordinateDataset(base_path, train_df, transform_train)
    val_dataset = SeriesLevelCoordinateDataset(base_path, val_df, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset, val_dataset


def create_series_level_test_datasets_and_loaders(df: pd.DataFrame,
                                                  series_description: str,
                                                  transform_val: transforms.Compose,
                                                  base_path: str,
                                                  random_seed=42,
                                                  batch_size=1):
    filtered_df = df[df['series_description'] == series_description].reset_index(drop=True)

    val_dataset = SeriesLevelDataset(base_path, filtered_df, transform_val)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return val_loader


def create_coordinate_datasets_and_loaders(df: pd.DataFrame,
                                           series_description: str,
                                           transform_train: nn.Module,
                                           transform_val: nn.Module,
                                           base_path: str,
                                           split_factor=0.2,
                                           random_seed=42,
                                           num_workers=0,
                                           batch_size=8):
    filtered_df = df[df['series_description'] == series_description]

    # Split by study ids
    train_study_ids, val_study_ids = train_test_split(filtered_df['study_id'].unique(), test_size=split_factor,
                                                      random_state=random_seed)
    train_df = filtered_df[filtered_df["study_id"].isin(train_study_ids)].reset_index(drop=True)
    val_df = filtered_df[filtered_df["study_id"].isin(val_study_ids)].reset_index(drop=True)

    train_dataset = CoordinateDataset(train_df, transform_train)
    val_dataset = CoordinateDataset(val_df, transform_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, train_dataset, val_dataset


def create_datasets_and_loaders(df: pd.DataFrame,
                                series_description: str,
                                transform_train: nn.Module,
                                transform_val: nn.Module,
                                split_factor=0.2,
                                random_seed=42,
                                num_workers=0,
                                batch_size=8):
    filtered_df = df[df['series_description'] == series_description]
    train_series, val_series = train_test_split(filtered_df["series_id"], test_size=split_factor,
                                                random_state=random_seed)
    train_df = filtered_df[filtered_df["series_id"].isin(train_series)].reset_index(drop=True)
    val_df = filtered_df[filtered_df["series_id"].isin(val_series)].reset_index(drop=True)

    train_dataset = PerImageDataset(train_df, transform_train)
    val_dataset = PerImageDataset(val_df, transform_val)

    train_sampler = WeightedRandomSampler(weights=train_dataset.label["weight"], num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, train_dataset, val_dataset


def retrieve_image_paths(base_path, study_id, series_id):
    series_dir = os.path.join(base_path, str(study_id), str(series_id))
    images = os.listdir(series_dir)
    image_paths = [os.path.join(series_dir, img) for img in images]
    return image_paths


def display_dicom_with_coordinates(image_paths: list, label_df: pd.DataFrame):
    fig, axs = plt.subplots(1, len(image_paths), figsize=(18, 6))

    for idx, path in enumerate(image_paths):  # Display images
        study_id = int(path.replace("\\", "/").split('/')[-3])
        series_id = int(path.replace("\\", "/").split('/')[-2])

        # Filter label coordinates for the current study and series
        filtered_labels = label_df[(label_df['study_id'] == study_id) & (label_df['series_id'] == series_id)]

        # Read DICOM image
        ds = pydicom.dcmread(path)

        # Plot DICOM image
        axs[idx].imshow(ds.pixel_array, cmap='gray')
        axs[idx].set_title(f"Study ID: {study_id}, Series ID: {series_id}")
        axs[idx].axis('off')

        # Plot coordinates
        for _, row in filtered_labels.iterrows():
            axs[idx].plot(row['x'], row['y'], 'ro', markersize=5)

    plt.tight_layout()
    plt.show()


# Load DICOM files from a folder
def load_dicom_files(path_to_folder):
    files = [os.path.join(path_to_folder, f) for f in os.listdir(path_to_folder) if f.endswith('.dcm')]
    files.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('-')[-1]))
    return files


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data


def get_bounding_boxes_for_label(label, box_offset_from_center=5):
    offset = box_offset_from_center
    ret = []
    for i in range(0, len(label), 2):
        curr = []
        curr.append(label[i] - offset)
        curr.append(label[i + 1] - offset)
        curr.append(label[i] + offset)
        curr.append(label[i + 1] + offset)
        ret.append(torch.cat([e.reshape(1) for e in curr]))

    return torch.cat(ret)


def retrieve_training_data(train_path):
    def reshape_row(row):
        data = {'study_id': [], 'condition': [], 'level': [], 'severity': []}

        for column, value in row.items():
            if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description']:
                parts = column.split('_')
                condition = ' '.join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
                data['study_id'].append(row['study_id'])
                data['condition'].append(condition)
                data['level'].append(level)
                data['severity'].append(value)

        return pd.DataFrame(data)

    train = pd.read_csv(train_path + 'train.csv')
    label = pd.read_csv(train_path + 'train_label_coordinates.csv')
    train_desc = pd.read_csv(train_path + 'train_series_descriptions.csv')
    test_desc = pd.read_csv(train_path + 'test_series_descriptions.csv')
    sub = pd.read_csv(train_path + 'sample_submission.csv')

    new_train_df = pd.concat([reshape_row(row) for _, row in train.iterrows()], ignore_index=True)
    merged_df = pd.merge(new_train_df, label, on=['study_id', 'condition', 'level'], how='inner')
    final_merged_df = pd.merge(merged_df, train_desc, on=['series_id', 'study_id'], how='inner')
    final_merged_df['severity'] = final_merged_df['severity'].map(
        {'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})

    final_merged_df['row_id'] = (
            final_merged_df['study_id'].astype(str) + '_' +
            final_merged_df['condition'].str.lower().str.replace(' ', '_') + '_' +
            final_merged_df['level'].str.lower().str.replace('/', '_')
    )

    # Create the image_path column
    final_merged_df['image_path'] = (
            f'{train_path}/train_images/' +
            final_merged_df['study_id'].astype(str) + '/' +
            final_merged_df['series_id'].astype(str) + '/' +
            final_merged_df['instance_number'].astype(str) + '.dcm'
    )

    return final_merged_df


def clean_training_data(train_data, train_path):
    def check_exists(path):
        return os.path.exists(path)

    # Define a function to check if a study ID directory exists
    def check_study_id(row):
        study_id = row['study_id']
        path = f'{train_path}/train_images/{study_id}'
        return check_exists(path)

    # Define a function to check if a series ID directory exists
    def check_series_id(row):
        study_id = row['study_id']
        series_id = row['series_id']
        path = f'{train_path}/train_images/{study_id}/{series_id}'
        return check_exists(path)

    # Define a function to check if an image file exists
    def check_image_exists(row):
        image_path = row['image_path']
        return check_exists(image_path)

    # Apply the functions to the train_data dataframe
    train_data['study_id_exists'] = train_data.apply(check_study_id, axis=1)
    train_data['series_id_exists'] = train_data.apply(check_series_id, axis=1)
    train_data['image_exists'] = train_data.apply(check_image_exists, axis=1)

    # Filter train_data
    train_data = train_data[
        (train_data['study_id_exists']) & (train_data['series_id_exists']) & (train_data['image_exists'])]
    train_data = train_data.dropna()

    return train_data
