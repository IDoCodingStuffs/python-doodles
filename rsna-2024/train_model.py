import timm
import torchvision
import albumentations as A

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)
torchvision.disable_beta_transforms_warning()

CONFIG = dict(
    n_levels=5,
    # backbone="tf_efficientnetv2_b0",
    backbone="tiny_vit_21m_512",
    img_size=(512, 512),
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
    def __init__(self, backbone, pretrained=False):
        super(CNN_Model, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            in_chans=CONFIG["in_chans"],
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hdim, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(CONFIG["drop_rate_last"]),
                nn.LeakyReLU(0.1),
                nn.Linear(256, CONFIG["out_dim"]),
                nn.Softmax(),
            )
            for i in range(CONFIG["n_levels"])])

    def forward(self, x):
        feat = self.encoder(x)
        return torch.stack([head(feat) for head in self.heads], dim=1)


class CNN_LSTM_Model(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(CNN_LSTM_Model, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained,
            in_chans=CONFIG["in_chans"],
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=CONFIG["drop_rate"], bidirectional=True, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(CONFIG["drop_rate_last"]),
                nn.LeakyReLU(0.1),
                nn.Linear(256, CONFIG["out_dim"]),
                nn.Softmax(),
            )
            for i in range(CONFIG["n_levels"])])

    def forward(self, x):
        feat = self.encoder(x)
        feat, _ = self.lstm(feat)
        return torch.stack([head(feat) for head in self.heads], dim=1)


class CNN_LSTM_Model_Series(nn.Module):
    def __init__(self, backbone: CNN_LSTM_Model, encoder_feature_size=1280):
        super(CNN_LSTM_Model_Series, self).__init__()

        self.encoder = backbone
        self.lstm = nn.LSTM(encoder_feature_size, 256, num_layers=2, dropout=CONFIG["drop_rate"], bidirectional=True,
                             batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.LayerNorm(256),
                nn.Dropout(CONFIG["drop_rate_last"]),
                nn.LeakyReLU(0.1),
                nn.Linear(256, CONFIG["out_dim"]),
            )
            for i in range(CONFIG["n_levels"])])

    def forward(self, x):
        feat = self.encoder.encoder(x.squeeze(0).unsqueeze(1))
        # feat, _ = self.encoder.lstm(feat)
        feat, _ = self.lstm(feat)
        # feat[0] is for the CLS embedding
        return torch.stack([head(feat[0]) for head in self.heads], dim=0).unsqueeze(0)


class VIT_Model(nn.Module):
    def __init__(self, backbone, pretrained=False):
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
    def __init__(self, out_dim):
        super(NormMLPClassifierHead, self).__init__()

        self.out_dim = out_dim
        self.head = nn.Sequential(
            nn.LayerNorm(576, eps=1e-05, elementwise_affine=True),
            # nn.Flatten(start_dim=1, end_dim=-1),
            nn.Dropout(p=CONFIG["drop_rate_last"], inplace=True),
            nn.Linear(in_features=576, out_features=15, bias=True),
            # nn.Softmax()
        )

    def forward(self, x):
        return self.head(x)


class VIT_Model_25D(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(VIT_Model_25D, self).__init__()

        self.num_classes = CONFIG["out_dim"] * CONFIG["n_levels"]
        self.encoder = timm.create_model(
            backbone,
            num_classes=self.num_classes,
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained
        )
        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'vit' in backbone:
            hdim = 576
            self.encoder.head.fc = nn.Identity()
        self.attention_layer = nn.Sequential(
            # !TODO: Need to figure this one out
            nn.LayerNorm(hdim, eps=1e-05, elementwise_affine=True),
            nn.Dropout(p=CONFIG["drop_rate"], inplace=True),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=576, nhead=8, batch_first=True), num_layers=1),
        )
        # self.attention_layer = nn.Identity()
        self.head = NormMLPClassifierHead(self.num_classes)

    def forward(self, x):
        feat = self.encoder(x.squeeze(0))
        feat = self.attention_layer(feat.unsqueeze(0))
        # BERT-like approach
        feat = self.head(feat[:, 0])

        # !TODO: This is likely incorrect
        return feat.reshape((-1, CONFIG["n_levels"], CONFIG["out_dim"]))


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
        A.ToRGB()
    ])

    transform_val = A.Compose([
        A.Resize(*CONFIG["img_size"]),
        A.Normalize(mean=0.5, std=0.5),
        A.ToRGB()
    ])

    (trainloader, valloader, testloader,
     trainset, valset, testset) = create_datasets_and_loaders(training_data,
                                                              data_subset_label,
                                                              transform_train,
                                                              transform_val,
                                                              num_workers=24,
                                                              split_factor=0.3,
                                                              batch_size=8)

    NUM_EPOCHS = CONFIG["epochs"]

    model = VIT_Model(backbone=CONFIG["backbone"]).to(device)

    optimizers = [
        torch.optim.Adam(model.encoder.parameters(), lr=1e-4),
        # torch.optim.Adam(model.lstm.parameters(), lr=5e-4),
    ]

    # head_optimizers = [torch.optim.Adam(head.parameters(), lr=1e-3) for head in model.heads]
    # optimizers.extend(head_optimizers)

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=1e-6),
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], NUM_EPOCHS, eta_min=5e-5),
    ]
    # schedulers.extend([
    #     torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, NUM_EPOCHS, eta_min=1e-4) for head_optimizer in
    #     head_optimizers
    # ])

    criteria = [
        FocalLoss(alpha=0.2, gamma=3).to(device),
        FocalLoss(alpha=0.2, gamma=3).to(device),
        FocalLoss(alpha=0.2, gamma=3).to(device),
        FocalLoss(alpha=0.2, gamma=3).to(device),
        FocalLoss(alpha=0.2, gamma=3).to(device),
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
    ])

    transform_val = A.Compose([
        A.Resize(*CONFIG["img_size"]),
        A.Normalize(mean=0.5, std=0.5),
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_series_level_datasets_and_loaders(training_data,
                                                                           data_subset_label,
                                                                           transform_train,
                                                                           transform_val,
                                                                           base_path=os.path.join(
                                                                               data_basepath,
                                                                               "train_images"),
                                                                           num_workers=24,
                                                                           split_factor=0.3,
                                                                           batch_size=1)

    NUM_EPOCHS = CONFIG["epochs"]

    model_per_image = torch.load("./models/efficientnetv2b0_lstm_t2stir/efficientnetv2b0_lstm_t2stir_20.pt")
    model = CNN_LSTM_Model_Series(backbone=model_per_image).to(device)
    optimizers = [
        torch.optim.Adam(model.encoder.encoder.parameters(), lr=5e-5),
        # torch.optim.Adam(model.encoder.lstm.parameters(), lr=1e-4),
        torch.optim.Adam(model.lstm.parameters(), lr=5e-4),
    ]

    head_optimizers = [torch.optim.Adam(head.parameters(), lr=1e-3) for head in model.heads]
    optimizers.extend(head_optimizers)

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=1e-6),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], NUM_EPOCHS, eta_min=5e-4),
        # torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[2], NUM_EPOCHS, eta_min=1e-5),
    ]
    schedulers.extend([
        torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, NUM_EPOCHS, eta_min=1e-4) for head_optimizer in
        head_optimizers
    ])

    criteria = [
        FocalLoss(alpha=0.2).to(device),
        FocalLoss(alpha=0.2).to(device),
        FocalLoss(alpha=0.2).to(device),
        FocalLoss(alpha=0.2).to(device),
        FocalLoss(alpha=0.2).to(device),
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


def train():
    model_t2stir = train_model_for_series_per_image("Sagittal T2/STIR", "tiny_vit_21m_512_t2stir")
    # model_t1 = train_model_for_series("Sagittal T1", "efficientnet_b0_lstm_t1")
    # model_t2 = train_model_for_series("Axial T2", "efficientnet_b0_lstm_t2")


if __name__ == '__main__':
    train()
