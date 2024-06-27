import timm
import torchvision

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)
torchvision.disable_beta_transforms_warning()


CONFIG = dict(
    project_name="PL-RSNA-2024-Lumbar-Spine-Classification",
    artifact_name="rsnaEffNetModel",
    load_kernel=None,
    load_last=True,
    n_folds=5,
    n_levels=5,
    backbone="tiny_vit_21m_384",
    img_size=(384, 384),
    n_slice_per_c=16,
    in_chans=1,
    drop_rate=0.05,
    drop_rate_last=0.3,
    drop_path_rate=0.,
    p_mixup=0.5,
    p_rand_order_v1=0.2,
    lr=1e-3,
    out_dim=3,
    epochs=200,
    batch_size=8,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)


class CNN_LSTM_Model(nn.Module):
    def __init__(self, backbone, pretrained=False):
        super(CNN_LSTM_Model, self).__init__()

        self.encoder = timm.create_model(
            backbone,
            num_classes=CONFIG["out_dim"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            pretrained=pretrained
        )

        if 'efficient' in backbone:
            hdim = self.encoder.conv_head.out_channels
            self.encoder.classifier = nn.Identity()
        elif 'convnext' in backbone:
            hdim = self.encoder.head.fc.in_features
            self.encoder.head.fc = nn.Identity()
        elif 'vit' in backbone:
            hdim = self.encoder.head.in_features
            self.encoder.head.fc = nn.Identity()

        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=CONFIG["drop_rate"], bidirectional=True, batch_first=True)
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.Dropout(CONFIG["drop_rate_last"]),
                nn.LeakyReLU(0.1),
                nn.Linear(256, CONFIG["out_dim"]),
                # nn.Softmax(),
            )
            for i in range(CONFIG["n_levels"])])

    def forward(self, x):
        feat = self.encoder(x.squeeze(0))
        feat, _ = self.lstm(feat)
        # !TODO: This is probably incorrect
        return torch.mean(torch.stack([head(feat) for head in self.heads], dim=1), dim=0).unsqueeze(0)


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
        return self.encoder(x).reshape((-1, CONFIG["n_levels"], CONFIG["out_dim"]))


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

    transform_train = TrainingTransform(image_size=CONFIG["img_size"], num_channels=3)
    transform_val = ValidationTransform(image_size=CONFIG["img_size"], num_channels=3)

    trainloader, valloader, len_train, len_val = create_datasets_and_loaders(training_data,
                                                                             data_subset_label,
                                                                             transform_train,
                                                                             transform_val,
                                                                             num_workers=0,
                                                                             split_factor=0.05,
                                                                             batch_size=8)

    NUM_EPOCHS = CONFIG["epochs"]

    # model = CNN_LSTM_Model(backbone=CONFIG["backbone"]).to(device)
    model = VIT_Model(backbone=CONFIG["backbone"]).to(device)
    optimizers = [
        torch.optim.Adam(model.encoder.parameters(), lr=1e-3),
    ]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=1e-6),
    ]

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


def train_model_for_series(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_training_data(data_basepath)

    transform_train = TrainingTransform(image_size=CONFIG["img_size"], num_channels=3)
    transform_val = ValidationTransform(image_size=CONFIG["img_size"], num_channels=3)

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
                                                                                        batch_size=1)

    NUM_EPOCHS = CONFIG["epochs"]

    model = VIT_Model_25D(backbone=CONFIG["backbone"]).to(device)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
        #torch.optim.Adam(model.attention_layer.parameters(), lr=1e-3),
        #torch.optim.Adam(model.head.parameters(), lr=1e-3),
    ]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=1e-5),
        #torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], NUM_EPOCHS, eta_min=1e-5),
        #torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[2], NUM_EPOCHS, eta_min=1e-5),
    ]

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
    model_t2stir = train_model_for_series("Sagittal T2/STIR", "tiny_vit_21m_384_transformer_t2stir")
    # model_t1 = train_model_for_series("Sagittal T1", "efficientnet_b0_lstm_t1")
    # model_t2 = train_model_for_series("Axial T2", "efficientnet_b0_lstm_t2")


if __name__ == '__main__':
    train()
