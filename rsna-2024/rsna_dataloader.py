import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

import pydicom
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms


TRANSFORM = transforms.Compose([
    transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


class CustomDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        image_path = self.dataframe['image_path'][index]
        image = load_dicom(image_path)  # Define this function to load your DICOM images
        label = self.dataframe['severity'][index]

        if self.transform:
            image = self.transform(image)

        return image, label


def create_datasets_and_loaders(df, series_description, transform_train, transform_val, batch_size=8):
    filtered_df = df[df['series_description'] == series_description]

    train_df, val_df = train_test_split(filtered_df, test_size=0.2, random_state=42)
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_dataset = CustomDataset(train_df, transform_train)
    val_dataset = CustomDataset(val_df, transform_val)

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return trainloader, valloader, len(train_df), len(val_df)


def generate_image_paths(df, data_dir):
    image_paths = []
    for study_id, series_id in zip(df['study_id'], df['series_id']):
        study_dir = os.path.join(data_dir, str(study_id))
        series_dir = os.path.join(study_dir, str(series_id))
        images = os.listdir(series_dir)
        image_paths.extend([os.path.join(series_dir, img) for img in images])
    return image_paths


def display_dicom_with_coordinates(image_paths, label_df):
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

    return train_data
