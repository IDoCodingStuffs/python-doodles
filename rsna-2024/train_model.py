import timm
import torchvision
import albumentations as A
import timm_3d
import math

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
    epochs=200,
    batch_size=8,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)


class CNN_Model(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_b3", pretrained=False):
        super(CNN_Model, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"] * CONFIG["n_levels"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            in_chans=CONFIG["in_chans"],
        )

    def forward(self, x):
        return self.encoder(x).reshape((-1, 5, 3))


class CNN_Model_Multichannel(nn.Module):
    def __init__(self, backbone="tf_efficientnetv2_b3", pretrained=False):
        super(CNN_Model_Multichannel, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"] * CONFIG["n_levels"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            # !TODO: Refactor
            in_chans=CONFIG["in_chans"] * 29,
        )

    def forward(self, x):
        return self.encoder(x).reshape((-1, 5, 3))


class CNN_Model_3D(nn.Module):
    def __init__(self, backbone="tf_efficientnet_b0", pretrained=False):
        super(CNN_Model_3D, self).__init__()

        self.encoder = timm_3d.create_model(
            backbone,
            num_classes=CONFIG["out_dim"] * CONFIG["n_levels"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            in_chans=CONFIG["in_chans"],
        ).to(CONFIG["device"])

    def forward(self, x):
        return self.encoder(x.unsqueeze(1)).reshape((-1, 5, 3))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        pos_emb = self.pe[:x.size(0)]
        x = x + pos_emb
        return self.dropout(x)


class EfficientNetModel_Series(nn.Module):
    def __init__(self, backbone: CNN_Model, num_inter_layers=4):
        super(EfficientNetModel_Series, self).__init__()

        self.backbone = backbone
        self.backbone.encoder.classifier = nn.Identity()
        self.backbone.forward = self._backbone_forward
        self.num_inter_layers = num_inter_layers

        hdim = self.backbone.encoder.conv_head.out_channels
        self.pos_emb = PositionalEncoding(d_model=hdim, dropout=CONFIG["drop_rate"])
        self.attention_layer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(hdim, nhead=8, dropout=CONFIG["drop_rate"]),
            num_layers=self.num_inter_layers,
            norm=nn.LayerNorm(hdim, eps=1e-05, elementwise_affine=True)
        )
        self.head = NormMLPClassifierHead(hdim, CONFIG["n_levels"] * CONFIG["out_dim"])

    def _backbone_forward(self, x):
        return self.backbone.encoder(x)

    def forward(self, x):
        feat = self.backbone(x.squeeze(0).unsqueeze(1))
        feat = feat.unsqueeze(0)

        feat = self.pos_emb(feat)
        feat = self.attention_layer(feat)

        feat = self.head(feat[:, 0])

        return feat.reshape((-1, CONFIG["n_levels"], CONFIG["out_dim"]))


class VIT_Model(nn.Module):
    def __init__(self, backbone="tiny_vit_21m_512", pretrained=False):
        super(VIT_Model, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"] * CONFIG["n_levels"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained
        )

    def forward(self, x):
        return self.encoder(x).reshape((-1, 5, 3))


class NormMLPClassifierHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(NormMLPClassifierHead, self).__init__()

        self.out_dim = out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim, eps=1e-05, elementwise_affine=True),
            nn.Dropout(p=CONFIG["drop_rate_last"], inplace=True),
            nn.Linear(in_features=in_dim, out_features=out_dim, bias=True),
        )
        self.head.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.head(x)


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
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_coordinate_training_data(data_basepath)

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

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_series_level_datasets_and_loaders(training_data,
                                                                           data_subset_label,
                                                                           transform_train,
                                                                           transform_val,
                                                                           base_path=os.path.join(
                                                                               data_basepath,
                                                                               "train_images"),
                                                                           num_workers=12,
                                                                           split_factor=0.3,
                                                                           batch_size=8,
                                                                           data_type=SeriesDataType.SEQUENTIAL_FIXED_LENGTH
                                                                           )

    NUM_EPOCHS = CONFIG["epochs"]

    # model_per_image = torch.load(CONFIG["efficientnet_backbone_path"])
    # model = EfficientNetModel_Series(backbone=model_per_image).to(device)
    model = CNN_Model_Multichannel(backbone=CONFIG["backbone"]).to(device)

    optimizers = [
        # torch.optim.Adam(model.backbone.parameters(), lr=1e-4),
        # torch.optim.Adam(model.attention_layer.parameters(), lr=1e-3),
        # torch.optim.Adam(model.head.parameters(), lr=1e-3)
        torch.optim.Adam(model.parameters(), lr=1e-3)
    ]

    schedulers = [
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=1e-5),
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], NUM_EPOCHS, eta_min=5e-4),
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[2], NUM_EPOCHS, eta_min=1e-4),
    ]

    criteria = [
        FocalLoss(alpha=0.2).to(device) for i in range(CONFIG["n_levels"])
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
                                freeze_backbone_initial_epochs=0
                                )

    return model


def train_model_3d(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_coordinate_training_data(data_basepath)

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

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_series_level_datasets_and_loaders(training_data,
                                                                           data_subset_label,
                                                                           transform_train,
                                                                           transform_val,
                                                                           base_path=os.path.join(
                                                                               data_basepath,
                                                                               "train_images"),
                                                                           num_workers=0,
                                                                           split_factor=0.3,
                                                                           batch_size=1)

    NUM_EPOCHS = CONFIG["epochs"]

    model = CNN_Model_3D()

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=1e-5),
    ]

    criteria = [
        FocalLoss(alpha=0.2).to(device) for i in range(CONFIG["n_levels"])
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
                                empty_cache_every_n_iterations=2,
                                freeze_backbone_initial_epochs=0)

    return model


def train():
    model_t2stir = train_model_for_series("Sagittal T2/STIR", "efficientnet_b4_multichannel_t2stir")
    # model_t1 = train_model_for_series("Sagittal T1", "efficientnet_b0_lstm_t1")
    # model_t2 = train_model_for_series("Axial T2", "efficientnet_b0_lstm_t2")


if __name__ == '__main__':
    train()
