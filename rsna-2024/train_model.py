import math

import timm
import timm_3d
import torchvision
import albumentations as A
import volumentations as V

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)
torchvision.disable_beta_transforms_warning()

CONFIG = dict(
    n_levels=5,
    # backbone="tf_efficientnetv2_b3",
    # backbone="tiny_vit_21m_512",
    backbone="efficientnet_b4",
    vit_backbone_path="./models/tiny_vit_21m_512_t2stir/tiny_vit_21m_512_t2stir_70.pt",
    efficientnet_backbone_path="./models/tf_efficientnetv2_b3_t2stir/tf_efficientnetv2_b3_t2stir_85.pt",
    # img_size=(512, 512),
    img_size=(384, 384),
    in_chans=1,
    drop_rate=0.05,
    drop_rate_last=0.3,
    drop_path_rate=0.,
    aug_prob=0.7,
    out_dim=3,
    epochs=100,
    batch_size=8,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)
DATA_BASEPATH = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)


class CNN_Model(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_b3", pretrained=True):
        super(CNN_Model, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"] * CONFIG["n_levels"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            global_pool='avg',
            in_chans=CONFIG["in_chans"],
        )

    def forward(self, x):
        return self.encoder(x).reshape((-1, 5, 3))


class CNN_Model_Multichannel(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_b3", in_chans=29, num_levels=5, pretrained=True):
        super(CNN_Model_Multichannel, self).__init__()

        self.num_levels = num_levels
        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"] * self.num_levels,
            features_only=False,
            global_pool='avg',
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            in_chans=CONFIG["in_chans"] * in_chans,
        )

    def forward(self, x):
        return self.encoder(x).reshape((-1, self.num_levels, 3))


class CNN_Model_3D(nn.Module):
    def __init__(self, backbone="efficientnet_lite0", pretrained=False):
        super(CNN_Model_3D, self).__init__()

        self.encoder = timm_3d.create_model(
            backbone,
            num_classes=CONFIG["out_dim"] * CONFIG["n_levels"] * 2,
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            in_chans=CONFIG["in_chans"],
        ).to(CONFIG["device"])

    def forward(self, x):
        return self.encoder(x.unsqueeze(1)).reshape((-1, 10, 3))


def train_model_for_series_per_image(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_training_data(data_basepath)

    transform_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=CONFIG["aug_prob"]),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=CONFIG["aug_prob"]),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=CONFIG["aug_prob"]),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=CONFIG["aug_prob"]),
        A.Resize(*CONFIG["img_size"]),
        A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8,
                        p=CONFIG["aug_prob"]),
        A.Normalize(mean=0.5, std=0.5),
        # A.ToRGB()
    ])

    transform_val = A.Compose([
        A.Resize(*CONFIG["img_size"]),
        A.Normalize(mean=0.5, std=0.5),
        # A.ToRGB()
    ])

    (trainloader, valloader, testloader,
     trainset, valset, testset) = create_datasets_and_loaders(training_data,
                                                              data_subset_label,
                                                              transform_train,
                                                              transform_val,
                                                              num_workers=0,
                                                              split_factor=0.3,
                                                              batch_size=8)

    NUM_EPOCHS = CONFIG["epochs"]

    # model = VIT_Model().to(device)
    model = CNN_Model().to(device)

    optimizers = [
        torch.optim.Adam(model.encoder.parameters(), lr=1e-3),
    ]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=1e-6),
    ]

    criteria = [
        FocalLoss(alpha=0.2, gamma=3).to(device) for i in range(CONFIG["n_levels"])
    ]

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {data_subset_label}",
                                epochs=NUM_EPOCHS)

    return model


def train_model_for_series(data_subset_label: str, model_label: str):
    transform_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=CONFIG["aug_prob"]),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=CONFIG["aug_prob"]),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=CONFIG["aug_prob"]),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=CONFIG["aug_prob"]),
        A.Resize(*CONFIG["img_size"]),
        A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8,
                        p=CONFIG["aug_prob"]),
        A.Normalize(mean=0.5, std=0.5),
    ])

    transform_val = A.Compose([
        A.Resize(*CONFIG["img_size"]),
        A.Normalize(mean=0.5, std=0.5),
    ])

    transform_3d_train = V.Compose([
        V.OneOf(
            [
                V.RotateAroundAxis3d(rotation_limit=math.pi / 4, axis=(1, 0, 0)),
                V.RotateAroundAxis3d(rotation_limit=math.pi / 20, axis=(0, 1, 0)),
                V.RotateAroundAxis3d(rotation_limit=math.pi / 20, axis=(0, 0, 1))
            ], p=CONFIG["aug_prob"]
        )
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_series_level_datasets_and_loaders(TRAINING_DATA,
                                                                           data_subset_label,
                                                                           transform_train,
                                                                           transform_val,
                                                                           transform_3d_train=None,
                                                                           base_path=os.path.join(
                                                                               DATA_BASEPATH,
                                                                               "train_images"),
                                                                           num_workers=12,
                                                                           split_factor=0.3,
                                                                           batch_size=8,
                                                                           data_type=SeriesDataType.SEQUENTIAL_FIXED_LENGTH_PADDED
                                                                           )

    NUM_EPOCHS = CONFIG["epochs"]
    model = CNN_Model_Multichannel(backbone=CONFIG["backbone"],
                                   in_chans=MAX_IMAGES_IN_SERIES[data_subset_label],
                                   num_levels=(5 if "T2/STIR" in data_subset_label else 10)).to(device)

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3)
    ]

    schedulers = [
    ]

    criteria = [
        FocalLoss(alpha=(0.2 if "T2/STIR" in data_subset_label else 0.1)).to(device) for i in
        range(CONFIG["n_levels"] * (1 if "T2/STIR" in data_subset_label else 2))
    ]

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {data_subset_label}",
                                epochs=NUM_EPOCHS,
                                freeze_backbone_initial_epochs=0,
                                )

    return model


def train_model_3d(data_subset_label: str, model_label: str):
    # !TODO: Use volumentations
    transform_train = A.Compose([
        A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2), contrast_limit=(-0.2, 0.2), p=CONFIG["aug_prob"]),
        A.OneOf([
            A.MotionBlur(blur_limit=5),
            A.MedianBlur(blur_limit=5),
            A.GaussianBlur(blur_limit=5),
            A.GaussNoise(var_limit=(5.0, 30.0)),
        ], p=CONFIG["aug_prob"]),

        A.OneOf([
            A.OpticalDistortion(distort_limit=1.0),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            A.ElasticTransform(alpha=3),
        ], p=CONFIG["aug_prob"]),

        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, border_mode=0, p=CONFIG["aug_prob"]),
        A.Resize(*CONFIG["img_size"]),
        A.CoarseDropout(max_holes=16, max_height=64, max_width=64, min_holes=1, min_height=8, min_width=8,
                        p=CONFIG["aug_prob"]),
        A.Normalize(mean=0.5, std=0.5),
    ])

    transform_val = A.Compose([
        A.Resize(*CONFIG["img_size"]),
        A.Normalize(mean=0.5, std=0.5),
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_series_level_datasets_and_loaders(TRAINING_DATA,
                                                                           data_subset_label,
                                                                           transform_train,
                                                                           transform_val,
                                                                           base_path=os.path.join(
                                                                               DATA_BASEPATH,
                                                                               "train_images"),
                                                                           num_workers=2,
                                                                           split_factor=0.3,
                                                                           batch_size=1,
                                                                           data_type=SeriesDataType.CUBE_3D)

    NUM_EPOCHS = CONFIG["epochs"]

    model = CNN_Model_3D()

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]

    schedulers = [
    ]

    criteria = [
        FocalLoss(alpha=0.1).to(device) for i in range(CONFIG["n_levels"] * 2)
    ]

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {data_subset_label}",
                                epochs=NUM_EPOCHS,
                                freeze_backbone_initial_epochs=0,
                                empty_cache_every_n_iterations=2)

    return model


def train():
    # model_t2stir = train_model_for_series("Sagittal T2/STIR", "efficientnet_b4_multichannel_shuffled_t2stir")
    model_t1 = train_model_for_series("Sagittal T1", "efficientnet_b4_multichannel_shuffled_t1")
    # model_t2 = train_model_3d("Axial T2", "efficientnet_b0_3d_t2")


if __name__ == '__main__':
    train()
