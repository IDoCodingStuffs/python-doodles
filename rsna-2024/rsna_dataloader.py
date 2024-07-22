import random

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import glob
import pydicom
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from scipy import ndimage
from enum import Enum
import cv2
import torchio as tio
import itk

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



class PatientLevelDataset(Dataset):
    def __init__(self,
                 base_path: str,
                 dataframe: pd.DataFrame,
                 transform_3d=None,
                 is_train=False,
                 use_mirror_trick=False):
        self.base_path = base_path
        self.is_train = is_train
        self.use_mirror_trick = use_mirror_trick

        self.dataframe = (dataframe[['study_id', "series_id", "series_description", "condition", "severity", "level"]]
                          .drop_duplicates())

        self.subjects = self.dataframe[['study_id']].drop_duplicates().reset_index(drop=True)

        self.transform_3d = transform_3d

        self.levels = sorted(self.dataframe["level"].unique())
        self.labels = self._get_labels()

    def __len__(self):
        return len(self.subjects) * 2 if self.use_mirror_trick else 1

    def __getitem__(self, index):
        is_mirror = index >= len(self.subjects)
        curr = self.subjects.iloc[index % len(self.subjects)]

        label = np.array(self.labels[(curr["study_id"])])
        images_basepath = os.path.join(self.base_path, str(curr["study_id"]))
        images = []

        for series_desc in CONDITIONS.keys():
            # !TODO: Multiple matching series
            series = self.dataframe.loc[
                (self.dataframe["study_id"] == curr["study_id"]) &
                (self.dataframe["series_description"] == series_desc)].sort_values("series_id")['series_id'].iloc[0]

            series_path = os.path.join(images_basepath, str(series))
            series_images = read_series_as_volume(series_path)

            if is_mirror:
                series_images = np.flip(series_images, axis=1 if series_desc == "Axial T2" else 0)
                temp = label[:10].copy()
                label[:10] = label[10:20].copy()
                label[10:20] = temp

            if self.transform_3d is not None:
                series_images = self.transform_3d(np.expand_dims(series_images, 0)) #.data

            images.append(torch.tensor(series_images, dtype=torch.half).squeeze(0))

        return torch.stack(images), torch.tensor(label, dtype=torch.long)

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

            study_id = name[0]

            # One-hot
            # labels[study_id] = []
            # for label in label_indices:
            #     curr = [0 if label != i else 1 for i in range(3)]
            #     labels[study_id].append(curr)
            # labels[study_id] = np.array(labels[study_id]).flatten()
            labels[study_id] = label_indices

        return labels

def create_subject_level_datasets_and_loaders(df: pd.DataFrame,
                                              base_path: str,
                                              transform_3d_train=None,
                                              transform_3d_val=None,
                                              split_factor=0.2,
                                              random_seed=42,
                                              batch_size=1,
                                              num_workers=0,
                                              pin_memory=True):
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
                                        transform_3d=transform_3d_train,
                                        is_train=True
                                        )
    val_dataset = PatientLevelDataset(base_path, val_df,
                                      transform_3d=transform_3d_val)
    test_dataset = PatientLevelDataset(base_path, test_df,
                                       transform_3d=transform_3d_val)

    # train_picker = WeightedRandomSampler(train_dataset.weights, num_samples=len(train_dataset))
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_picker, num_workers=num_workers)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # !TODO: Refactor
    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


def read_series_as_volume(dirName, verbose=False):
    cache_path = os.path.join(dirName, "cached.npy")
    if os.path.exists(cache_path):
        return np.load(cache_path)

    PixelType = itk.ctype("signed short")
    Dimension = 3

    ImageType = itk.Image[PixelType, Dimension]

    namesGenerator = itk.GDCMSeriesFileNames.New()
    namesGenerator.SetUseSeriesDetails(True)
    namesGenerator.AddSeriesRestriction("0008|0021")
    namesGenerator.SetGlobalWarningDisplay(False)
    namesGenerator.SetDirectory(dirName)

    seriesUID = namesGenerator.GetSeriesUIDs()

    if verbose:
        if len(seriesUID) < 1:
            print("No DICOMs in: " + dirName)

        print("The directory: " + dirName)
        print("Contains the following DICOM Series: ")
        for uid in seriesUID:
            print(uid)

    reader = None
    dicomIO = None
    for i in range(10):
        for uid in seriesUID:
            seriesIdentifier = uid
            if verbose:
                print("Reading: " + seriesIdentifier)
            fileNames = namesGenerator.GetFileNames(seriesIdentifier)

            reader = itk.ImageSeriesReader[ImageType].New()
            dicomIO = itk.GDCMImageIO.New()
            reader.SetImageIO(dicomIO)
            reader.SetFileNames(fileNames)
            reader.ForceOrthogonalDirectionOff()
        if reader is not None:
            break

    if reader is None or dicomIO is None:
        raise FileNotFoundError(f"Empty path? {os.path.abspath(dirName)}")
    reader.Update()
    data = itk.GetArrayFromImage(reader.GetOutput())

    del namesGenerator
    del dicomIO
    del reader

    np.save(cache_path, data)

    return data


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
