import random
import matplotlib.pyplot as plt
import os
from os.path import abspath
import numpy as np
import pandas as pd
import glob
import pydicom
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
from torchvision.transforms import v2
from scipy import ndimage
from enum import Enum
import cv2

LABEL_MAP = {'normal_mild': 0, 'moderate': 1, 'severe': 2}
CONDITIONS = {
    "Sagittal T2/STIR": ["Spinal Canal Stenosis"],
    "Axial T2": ["Left Subarticular Stenosis", "Right Subarticular Stenosis"],
    "Sagittal T1": ["Left Neural Foraminal Narrowing", "Right Neural Foraminal Narrowing"],
}
MAX_IMAGES_IN_SERIES = {
    "Sagittal T2/STIR": 29,
    # !TODO: Might need a 3D model for this one
    "Axial T2": 192,
    "Sagittal T1": 38,
}

RESIZING_CHANNELS = {
    "Sagittal T2/STIR": 25,
    # !TODO: Might need a 3D model for this one
    "Axial T2": 100,
    "Sagittal T1": 35,
}

DOWNSAMPLING_TARGETS = {
    "Sagittal T2/STIR": 25,
    "Axial T2": 10,
    "Sagittal T1": 10,
}


class PerImageDataset(Dataset):
    def __init__(self, dataframe, base_path="./data/rsna-2024-lumbar-spine-degenerative-classification/train_images",
                 transform=None):
        self.to_pil = transforms.ToPILImage()
        self.to_tensor = transforms.ToTensor()
        self.grayscale = transforms.Grayscale(num_output_channels=3)

        self.transform = transform

        self.dataframe = dataframe
        self.dataframe["image_path"] = self.dataframe.apply(lambda x: retrieve_image_paths(base_path, x[0], x[1]),
                                                            axis=1)
        self.dataframe = self._expand_paths(self.dataframe)

        self.label = (self.dataframe.groupby("image_path")
                      .agg({"image_path": "first",
                            "level": lambda x: ",".join(x),
                            "severity": lambda x: ",".join(x)}))

        self.weights = self._get_weights()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        image_path = self.label['image_path'].iloc[index]
        image = load_dicom(image_path)

        if self.transform:
            image = self.transform(image=image)['image']

        return self.to_tensor(image), self.label_as_tensor(image_path)

    def _get_weights(self):
        # !TODO: refactor and properly formulate
        weights = []
        for path in self.label["image_path"]:
            curr = self.label_as_tensor(path).numpy()
            # L5/S1 severe
            if np.argmax(curr[-1]) == 2:
                weights.append(100)
            # L1/L2 severe
            elif np.argmax(curr[0]) == 2:
                weights.append(90)
            # L5/S1 moderate
            elif np.argmax(curr[-1]) == 1:
                weights.append(38)
            # L2/L3 severe
            elif np.argmax(curr[1]) == 2:
                weights.append(34)
            # L1/L2 moderate
            elif np.argmax(curr[0]) == 1:
                weights.append(31)
            # L3/L4 severe
            elif np.argmax(curr[2]) == 2:
                weights.append(13)
            # L2/L3 moderate
            elif np.argmax(curr[1]) == 1:
                weights.append(11)
            # L3/L4 moderate
            elif np.argmax(curr[2]) == 1:
                weights.append(7)
            # L4/L5 severe
            elif np.argmax(curr[3]) == 2:
                weights.append(6)
            # L4/L5 moderate
            elif np.argmax(curr[3]) == 1:
                weights.append(6)
            # All mild
            else:
                weights.append(1)
        return weights

    def _expand_paths(self, df):
        lens = [len(item) for item in df['image_path']]
        return pd.DataFrame(
            {col: np.repeat(df[col].values, lens) if col != "image_path" else np.concatenate(df[col].values) for col in
             df.columns})

    def label_as_tensor(self, image_path):
        row = self.label[self.label["image_path"] == image_path]
        levels = row["level"].str.split(",").values[0]
        levels = sorted([(val, index) for index, val in enumerate(levels)])

        label = row["severity"].values[0].split(",")
        label = [LABEL_MAP[label[index]] for level, index in levels]

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


class SeriesDataType(Enum):
    SEQUENTIAL_VARIABLE_LENGTH = 1
    SEQUENTIAL_VARIABLE_LENGTH_WITH_CLS = 2

    SEQUENTIAL_FIXED_LENGTH = 3
    SEQUENTIAL_FIXED_LENGTH_WITH_CLS = 4
    SEQUENTIAL_FIXED_LENGTH_PADDED = 5
    SEQUENTIAL_FIXED_LENGTH_RESIZED = 6
    SEQUENTIAL_FIXED_LENGTH_DOWNSAMPLED = 7

    CUBE_3D_RESIZED = 8
    CUBE_3D_DOWNSAMPLED_PADDED = 9
    CUBE_3D_RESIZED_PADDED = 10


class SeriesLevelDataset(Dataset):
    def __init__(self,
                 base_path: str,
                 dataframe: pd.DataFrame,
                 data_type=SeriesDataType.SEQUENTIAL_VARIABLE_LENGTH_WITH_CLS,
                 data_series="Sagittal T2/STIR",
                 transform=None,
                 transform_3d=None,
                 is_train=False,
                 downsample_ratio=1):
        self.base_path = base_path
        self.type = data_type
        self.data_series = data_series
        self.is_train = is_train

        self.dataframe = (dataframe[['study_id', "series_id", "condition", "severity", "level"]]
                          .drop_duplicates())
        self.dataframe["mirrored"] = False

        # Axial flipping
        if is_train and data_series == "Axial T2":
            df_copy = self.dataframe.copy()
            df_copy["mirrored"] = True
            self.dataframe = pd.concat([self.dataframe, df_copy])

        self.series = self.dataframe[['study_id', "series_id", "mirrored"]].drop_duplicates().reset_index(drop=True)

        self.transform = transform
        self.transform_3d = transform_3d
        self.downsample_ratio = downsample_ratio

        self.levels = sorted(self.dataframe["level"].unique())

        if data_series == "Sagittal T2/STIR":
            self.labels = self._get_t2stir_labels()
            self.weights = self._get_t2stir_weights()

        elif data_series == "Sagittal T1":
            self.labels = self._get_t1_labels()
            self.weights = self._get_t1_weights()

        if data_series == "Axial T2":
            self.labels = self._get_t2_labels()
            self.weights = self._get_t2_weights()

    def __len__(self):
        return len(self.series)

    def __getitem__(self, index):
        curr = self.series.iloc[index]
        label = np.array(self.labels[(curr["study_id"], curr["series_id"], curr["mirrored"])])
        images_basepath = os.path.join(self.base_path, str(curr["study_id"]), str(curr["series_id"]))

        images = load_dicom_series(images_basepath, self.transform, self.downsample_ratio)

        if curr["mirrored"]:
            images = np.array([cv2.flip(image, 1) for image in images])

        images = self._reshape_by_data_type(images)

        if self.transform_3d is not None:
            images = self.transform_3d(image=images)["image"]

        return images, torch.tensor(label).type(torch.FloatTensor)

    def _reshape_by_data_type(self, images):
        if self.type == SeriesDataType.SEQUENTIAL_FIXED_LENGTH_PADDED:
            front_buffer = (MAX_IMAGES_IN_SERIES[self.data_series] - len(images)) // 2
            rear_buffer = (MAX_IMAGES_IN_SERIES[self.data_series] - len(images)) // 2 + (
                    (MAX_IMAGES_IN_SERIES[self.data_series] - len(images)) % 2)

            images = np.pad(images, ((front_buffer, rear_buffer), (0, 0), (0, 0)))

        elif self.type == SeriesDataType.SEQUENTIAL_FIXED_LENGTH_WITH_CLS:
            front_buffer = (MAX_IMAGES_IN_SERIES[self.data_series] - len(images)) // 2
            rear_buffer = (MAX_IMAGES_IN_SERIES[self.data_series] - len(images)) // 2 + (
                    (MAX_IMAGES_IN_SERIES[self.data_series] - len(images)) % 2)

            images = np.pad(images, ((front_buffer + 1, rear_buffer), (0, 0), (0, 0)))

        elif self.type == SeriesDataType.SEQUENTIAL_FIXED_LENGTH_RESIZED:
            resize_target = RESIZING_CHANNELS[self.data_series]
            images = ndimage.zoom(images, (len(images) / resize_target, 1, 1))
            # Clip last
            images = images[:resize_target]
            # Pad offset
            images = np.pad(images, ((0, resize_target - len(images)), (0, 0), (0, 0)))

        elif self.type == SeriesDataType.SEQUENTIAL_VARIABLE_LENGTH_WITH_CLS:
            images = np.pad(images, ((1, 0), (0, 0), (0, 0)))


        elif self.type == SeriesDataType.SEQUENTIAL_FIXED_LENGTH_DOWNSAMPLED:
            if len(images) < DOWNSAMPLING_TARGETS[self.data_series]:
                images = np.pad(images, ((0, DOWNSAMPLING_TARGETS[self.data_series] - len(images)), (0, 0), (0, 0)))
            np.random.shuffle(images)
            images = images[:DOWNSAMPLING_TARGETS[self.data_series]]

        return images

    def _get_t2stir_labels(self):
        labels = dict()
        for name, group in self.dataframe.groupby(["study_id", "series_id", "mirrored"]):
            label_indices = [0 for e in range(len(self.levels))]
            for index, row in group.iterrows():
                if row["severity"] in LABEL_MAP:  # and row["condition"] in conditions:
                    label_index = self.levels.index(row["level"])
                    label_indices[label_index] = LABEL_MAP[row["severity"]]

            labels[name] = []
            for label in label_indices:
                curr = [0 if label != i else 1 for i in range(3)]
                labels[name].append(curr)

        return labels

    def _get_t1_labels(self):
        labels = dict()
        for name, group in self.dataframe.groupby(["study_id", "series_id", "mirrored"]):
            label_indices = [0 for e in range(len(self.levels) * 2)]
            for index, row in group.iterrows():
                if row["severity"] in LABEL_MAP:
                    if not row["mirrored"]:
                        label_index = self.levels.index(row["level"]) * 2 + (0 if "Left" in row["condition"] else 1)
                    else:
                        label_index = self.levels.index(row["level"]) * 2 + (1 if "Left" in row["condition"] else 0)
                    label_indices[label_index] = LABEL_MAP[row["severity"]]

            labels[name] = []
            for label in label_indices:
                # One hot encode
                curr = [0 if label != i else 1 for i in range(3)]
                labels[name].append(curr)

        return labels

    def _get_t2_labels(self):
        labels = dict()
        for name, group in self.dataframe.groupby(["study_id", "series_id", "mirrored"]):
            label_indices = [0 for e in range(len(self.levels) * 2)]
            for index, row in group.iterrows():
                if row["severity"] in LABEL_MAP:
                    label_index = self.levels.index(row["level"]) * 2 + (0 if "Left" in row["condition"] else 1)
                    label_indices[label_index] = LABEL_MAP[row["severity"]]

            labels[name] = []
            for label in label_indices:
                # One hot encode
                curr = [0 if label != i else 1 for i in range(3)]
                labels[name].append(curr)

        return labels

    def _get_t2stir_weights(self):
        # !TODO: refactor and properly formulate
        weights = []
        for name, group in self.dataframe.groupby(["study_id", "series_id", "mirrored"]):
            curr = self.labels[name]
            # L5/S1 severe
            if np.argmax(curr[-1]) == 2:
                weights.append(100)
            # L1/L2 severe
            elif np.argmax(curr[0]) == 2:
                weights.append(90)
            # L5/S1 moderate
            elif np.argmax(curr[-1]) == 1:
                weights.append(38)
            # L2/L3 severe
            elif np.argmax(curr[1]) == 2:
                weights.append(34)
            # L1/L2 moderate
            elif np.argmax(curr[0]) == 1:
                weights.append(31)
            # L3/L4 severe
            elif np.argmax(curr[2]) == 2:
                weights.append(13)
            # L2/L3 moderate
            elif np.argmax(curr[1]) == 1:
                weights.append(11)
            # L3/L4 moderate
            elif np.argmax(curr[2]) == 1:
                weights.append(7)
            # L4/L5 severe
            elif np.argmax(curr[3]) == 2:
                weights.append(6)
            # L4/L5 moderate
            elif np.argmax(curr[3]) == 1:
                weights.append(6)
            # All mild
            else:
                weights.append(1)
        return weights

    def _get_t1_weights(self):
        # !TODO: refactor and properly formulate
        weights = []
        for name, group in self.dataframe.groupby(["study_id", "series_id", "mirrored"]):
            curr = self.labels[name]
            # L1/L2 L severe
            if np.argmax(curr[0]) == 2:
                weights.append(700)
            # L2/L3 R severe
            elif np.argmax(curr[3]) == 2:
                weights.append(650)
            # L2/L3 L severe
            elif np.argmax(curr[2]) == 2:
                weights.append(280)
            # L3/L4 R severe
            elif np.argmax(curr[5]) == 2:
                weights.append(40)
            # L3/L4 L severe
            elif np.argmax(curr[4]) == 2:
                weights.append(38)
            # L1/L2 L or R moderate
            elif np.argmax(curr[0]) == 1:
                weights.append(30)
            elif np.argmax(curr[1]) == 1:
                weights.append(30)
            # L4/L5 L or R severe
            elif np.argmax(curr[6]) == 2:
                weights.append(15)
            elif np.argmax(curr[7]) == 2:
                weights.append(15)
            # L2/L3 L or R moderate
            elif np.argmax(curr[2]) == 1:
                weights.append(10)
            elif np.argmax(curr[3]) == 1:
                weights.append(10)
            # L5/S1 L or R severe
            elif np.argmax(curr[8]) == 2:
                weights.append(8)
            elif np.argmax(curr[9]) == 2:
                weights.append(8)
            # L3/L4 L or R moderate
            elif np.argmax(curr[4]) == 1:
                weights.append(5)
            elif np.argmax(curr[5]) == 1:
                weights.append(5)
            # L5/S1 L or R moderate
            elif np.argmax(curr[8]) == 1:
                weights.append(3)
            elif np.argmax(curr[9]) == 1:
                weights.append(3)
            # L4/L5 L or R moderate
            elif np.argmax(curr[6]) == 1:
                weights.append(2)
            elif np.argmax(curr[7]) == 1:
                weights.append(2)
            # All mild
            else:
                weights.append(1)
        return weights

    def _get_t2_weights(self):
        # !TODO: refactor and properly formulate
        weights = []
        for name, group in self.dataframe.groupby(["study_id", "series_id", "mirrored"]):
            curr = self.labels[name]
            # L1/L2 L severe
            if np.argmax(curr[0]) == 2:
                weights.append(700)
            # L2/L3 R severe
            elif np.argmax(curr[3]) == 2:
                weights.append(650)
            # L2/L3 L severe
            elif np.argmax(curr[2]) == 2:
                weights.append(280)
            # L3/L4 R severe
            elif np.argmax(curr[5]) == 2:
                weights.append(40)
            # L3/L4 L severe
            elif np.argmax(curr[4]) == 2:
                weights.append(38)
            # L1/L2 L or R moderate
            elif np.argmax(curr[0]) == 1:
                weights.append(30)
            elif np.argmax(curr[1]) == 1:
                weights.append(30)
            # L4/L5 L or R severe
            elif np.argmax(curr[6]) == 2:
                weights.append(15)
            elif np.argmax(curr[7]) == 2:
                weights.append(15)
            # L2/L3 L or R moderate
            elif np.argmax(curr[2]) == 1:
                weights.append(10)
            elif np.argmax(curr[3]) == 1:
                weights.append(10)
            # L5/S1 L or R severe
            elif np.argmax(curr[8]) == 2:
                weights.append(8)
            elif np.argmax(curr[9]) == 2:
                weights.append(8)
            # L3/L4 L or R moderate
            elif np.argmax(curr[4]) == 1:
                weights.append(5)
            elif np.argmax(curr[5]) == 1:
                weights.append(5)
            # L5/S1 L or R moderate
            elif np.argmax(curr[8]) == 1:
                weights.append(3)
            elif np.argmax(curr[9]) == 1:
                weights.append(3)
            # L4/L5 L or R moderate
            elif np.argmax(curr[6]) == 1:
                weights.append(2)
            elif np.argmax(curr[7]) == 1:
                weights.append(2)
            # All mild
            else:
                weights.append(1)
        return weights


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


class PatientLevelDataset(Dataset):
    def __init__(self,
                 base_path: str,
                 dataframe: pd.DataFrame,
                 data_type=SeriesDataType.CUBE_3D_RESIZED,
                 transform=None,
                 transform_3d=None,
                 is_train=False,
                 downsample_ratio=1):
        self.base_path = base_path
        self.type = data_type
        self.is_train = is_train

        self.dataframe = (dataframe[['study_id', "series_id", "series_description", "condition", "severity", "level"]]
                          .drop_duplicates())

        self.subjects = self.dataframe[['study_id']].drop_duplicates().reset_index(drop=True)

        self.transform = transform
        self.transform_3d = transform_3d
        self.downsample_ratio = downsample_ratio

        self.levels = sorted(self.dataframe["level"].unique())
        self.labels = self._get_labels()

    def __len__(self):
        return len(self.subjects)

    def __getitem__(self, index):
        curr = self.subjects.iloc[index]
        label = np.array(self.labels[(curr["study_id"])])
        images_basepath = os.path.join(self.base_path, str(curr["study_id"]))
        images = []

        series_and_images = load_dicom_subject(images_basepath, self.transform, self.downsample_ratio)
        for series_desc in CONDITIONS.keys():
            series = self.dataframe.loc[
                (self.dataframe["study_id"] == curr["study_id"]) &
                (self.dataframe["series_description"] == series_desc)]['series_id'].iloc[0]

            series_images = [item[1] for item in series_and_images if item[0] == series][0]
            series_images = self._reshape_by_data_type(series_images)
            if self.transform_3d is not None:
                series_images = self.transform_3d(image=series_images)["image"]

            images.append(torch.Tensor(series_images))

        return torch.stack(images), torch.tensor(label).type(torch.FloatTensor)

    def _reshape_by_data_type(self, images):
        width = len(images[0])
        if self.type == SeriesDataType.CUBE_3D_RESIZED:
            images = ndimage.interpolation.zoom(images, (width / len(images), 1, 1))
        elif self.type in (SeriesDataType.CUBE_3D_DOWNSAMPLED_PADDED, SeriesDataType.CUBE_3D_RESIZED_PADDED):
            if len(images) > width:
                if self.type == SeriesDataType.CUBE_3D_DOWNSAMPLED_PADDED:
                    images = images[::2, :, :]
                elif self.type == SeriesDataType.CUBE_3D_RESIZED_PADDED:
                    images = ndimage.interpolation.zoom(images, (width / len(images), 1, 1))

            front_buffer = (width - len(images)) // 2
            rear_buffer = (width - len(images)) // 2 + ((width - len(images)) % 2)

            images = np.pad(images, ((front_buffer, rear_buffer), (0, 0), (0, 0)))

        return images

    def _get_labels(self):
        labels = dict()
        for name, group in self.dataframe.groupby(["study_id"]):
            group = group[["condition", "level", "severity"]].drop_duplicates().sort_values(["condition", "level"])
            label_indices = []
            for index, row in group.iterrows():
                if row["severity"] in LABEL_MAP:
                    label_indices.append(LABEL_MAP[row["severity"]])
                else:
                    raise ValueError()

            # !TODO: Clean
            study_id = name[0]
            labels[study_id] = []
            for label in label_indices:
                curr = [0 if label != i else 1 for i in range(3)]
                labels[study_id].append(curr)
            labels[study_id] = np.array(labels[study_id]).flatten()
        return labels


class TrainingTransform(nn.Module):
    def __init__(self, image_size=(224, 224), num_channels=3):
        super(TrainingTransform, self).__init__()
        self.image_size = image_size

        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize(self.image_size)
        self.hflip = transforms.RandomHorizontalFlip(p=0.5)
        self.vflip = transforms.RandomVerticalFlip(p=0.5)

        self.gaussian_blur = transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3)),
            v2.Identity(),
        ], p=[0.2, 0.2, 0.6])

        self.gaussian_noise = transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 3)),
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 3)),
            v2.Identity(),
        ], p=[0.2, 0.2, 0.6])

        self.grayscale = transforms.Grayscale(num_output_channels=num_channels)
        self.to_tensor = transforms.ToTensor()

    def forward(self, image):
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

        self.to_pil = transforms.ToPILImage()
        self.resize = transforms.Resize(image_size)
        self.grayscale = transforms.Grayscale(num_output_channels=num_channels)
        self.to_tensor = transforms.ToTensor()

    def forward(self, image):
        image = self.to_pil(image)
        image = self.resize(image)
        image = self.grayscale(image)

        image = self.to_tensor(image)

        return image


def create_series_level_datasets_and_loaders(df: pd.DataFrame,
                                             series_description: str,
                                             transform_train,
                                             transform_val,
                                             base_path: str,
                                             transform_3d_train=None,
                                             split_factor=0.2,
                                             random_seed=42,
                                             batch_size=1,
                                             num_workers=0,
                                             data_type=SeriesDataType.SEQUENTIAL_VARIABLE_LENGTH_WITH_CLS):
    filtered_df = df[
        (df['series_description'] == series_description) & (df["condition"].isin(CONDITIONS[series_description]))]

    # By defauly, 8-1.5-.5 split
    train_studies, val_studies = train_test_split(filtered_df["study_id"].unique(), test_size=split_factor,
                                                  random_state=random_seed)
    val_studies, test_studies = train_test_split(val_studies, test_size=0.25, random_state=random_seed)

    train_df = filtered_df[filtered_df["study_id"].isin(train_studies)]
    val_df = filtered_df[filtered_df["study_id"].isin(val_studies)]
    test_df = filtered_df[filtered_df["study_id"].isin(test_studies)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    random.seed(random_seed)
    train_dataset = SeriesLevelDataset(base_path, train_df,
                                        transform=transform_train,
                                        transform_3d=transform_3d_train,
                                        data_type=data_type,
                                        data_series=series_description,
                                        is_train=True
                                        )
    val_dataset = SeriesLevelDataset(base_path, val_df,
                                      transform=transform_val, data_type=data_type, data_series=series_description)
    test_dataset = SeriesLevelDataset(base_path, test_df,
                                       transform=transform_val, data_type=data_type, data_series=series_description)

    train_picker = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_picker, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # !TODO: Refactor
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_subject_level_datasets_and_loaders(df: pd.DataFrame,
                                             transform_train,
                                             transform_val,
                                             base_path: str,
                                             transform_3d_train=None,
                                             split_factor=0.2,
                                             random_seed=42,
                                             batch_size=1,
                                             num_workers=0,
                                             data_type=SeriesDataType.SEQUENTIAL_VARIABLE_LENGTH_WITH_CLS):
    # By defauly, 8-1.5-.5 split
    df = df.dropna()
    # This drops any subjects with nans
    filtered_df = pd.DataFrame(columns=df.columns)
    for series_desc in CONDITIONS.keys():
        subset = df[df['series_description'] == series_desc]
        if series_desc == "Sagittal T2/STIR":
            subset = subset[subset.groupby(["study_id"]).transform('size') == 5]
        else:
            subset = subset[subset.groupby(["study_id"]).transform('size') == 10]
        filtered_df = pd.concat([filtered_df, subset])
    filtered_df = filtered_df[filtered_df.groupby(["study_id"]).transform('size') == 25]

    train_studies, val_studies = train_test_split(filtered_df["study_id"].unique(), test_size=split_factor,
                                                  random_state=random_seed)
    val_studies, test_studies = train_test_split(val_studies, test_size=0.25, random_state=random_seed)

    train_df = filtered_df[filtered_df["study_id"].isin(train_studies)]
    val_df = filtered_df[filtered_df["study_id"].isin(val_studies)]
    test_df = filtered_df[filtered_df["study_id"].isin(test_studies)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    random.seed(random_seed)
    train_dataset = PatientLevelDataset(base_path, train_df,
                                        transform=transform_train,
                                        transform_3d=transform_3d_train,
                                        data_type=data_type,
                                        is_train=True
                                        )
    val_dataset = PatientLevelDataset(base_path, val_df,
                                      transform=transform_val, data_type=data_type)
    test_dataset = PatientLevelDataset(base_path, test_df,
                                       transform=transform_val, data_type=data_type)

    #train_picker = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
    #train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_picker, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # !TODO: Refactor
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def create_series_level_coordinate_datasets_and_loaders(df: pd.DataFrame,
                                                        series_description: str,
                                                        transform_train: transforms.Compose,
                                                        transform_val: transforms.Compose,
                                                        base_path: str,
                                                        split_factor=0.2,
                                                        random_seed=42,
                                                        batch_size=1,
                                                        num_workers=0):
    filtered_df = df[
        (df['series_description'] == series_description) & (df['condition'].isin(CONDITIONS[series_description]))]

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

    val_dataset = PatientLevelDataset(base_path, filtered_df, transform_val)

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
    filtered_df = df[
        (df['series_description'] == series_description) & (df['condition'].isin(CONDITIONS[series_description]))]
    # By defauly, 8-1.5-.5 split
    train_studies, val_studies = train_test_split(filtered_df["study_id"].unique(), test_size=split_factor,
                                                  random_state=random_seed)
    val_studies, test_studies = train_test_split(val_studies, test_size=0.25, random_state=random_seed)

    train_df = filtered_df[filtered_df["study_id"].isin(train_studies)]
    val_df = filtered_df[filtered_df["study_id"].isin(val_studies)]
    test_df = filtered_df[filtered_df["study_id"].isin(test_studies)]

    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    train_dataset = PerImageDataset(train_df, transform=transform_train)
    val_dataset = PerImageDataset(val_df, transform=transform_val)
    test_dataset = PerImageDataset(test_df, transform=transform_val)

    train_sampler = WeightedRandomSampler(weights=train_dataset.weights, num_samples=len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


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


def load_dicom(path):
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    data = (data - data.min()) / (data.max() - data.min() + 1e-6) * 255
    data = np.uint8(data)
    return data


def load_dicom_series(path, transform=None, downsampling_rate=1):
    files = glob.glob(os.path.join(path, '*.dcm'))
    files = sorted(files, key=lambda x: int(x.split('/')[-1].split("\\")[-1].split('.')[0]))
    slices = [pydicom.dcmread(fname) for fname in files]
    # slices = sorted(slices, key=lambda s: s.SliceLocation)
    if transform is not None:
        data = np.array([transform(image=cv2.convertScaleAbs(slice.pixel_array))["image"] for slice in slices])
    else:
        data = np.array([cv2.convertScaleAbs(slice.pixel_array) for slice in slices])

    if downsampling_rate > 1:
        data = np.array([slice[::downsampling_rate, ::downsampling_rate] for slice in data])

    data = np.array(data)
    return data


def load_dicom_subject(path, transform=None, downsampling_rate=1):
    series_list = glob.glob(os.path.join(path, "*"))
    return [(int(os.path.basename(series)), load_dicom_series(series, transform)) for series in series_list]


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
    # !TODO: refactor
    def reshape_row(row):
        data = {col: [] for col in row.axes[0] if col in
                ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description', 'image_paths']}
        data["level"] = []
        data["condition"] = []
        data["severity"] = []

        for column, value in row.items():
            if column not in ['study_id', 'series_id', 'instance_number', 'x', 'y', 'series_description',
                              'image_paths']:
                parts = column.split('_')
                condition = ' '.join([word.capitalize() for word in parts[:-2]])
                level = parts[-2].capitalize() + '/' + parts[-1].capitalize()
                data['condition'].append(condition)
                data['level'].append(level)
                data['severity'].append(value)
            else:
                # !TODO: Seriously, refactor
                for i in range(25):
                    data[column].append(value)

        return pd.DataFrame(data)

    train = pd.read_csv(train_path + 'train.csv')
    train_desc = pd.read_csv(train_path + 'train_series_descriptions.csv')

    train_df = pd.merge(train, train_desc, on="study_id")

    train_df = pd.concat([reshape_row(row) for _, row in train_df.iterrows()], ignore_index=True)
    train_df['severity'] = train_df['severity'].map(
        {'Normal/Mild': 'normal_mild', 'Moderate': 'moderate', 'Severe': 'severe'})

    return train_df


def retrieve_coordinate_training_data(train_path):
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
