import timm

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)

CONFIG = dict(
    project_name="PL-RSNA-2024-Lumbar-Spine-Classification",
    artifact_name="rsnaEffNetModel",
    load_kernel=None,
    load_last=True,
    n_folds=5,
    n_levels=5,
    backbone="tiny_vit_21m_512",
    img_size=(512, 512),
    n_slice_per_c=16,
    in_chans=1,

    drop_rate=0.,
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



def train_model_for_series(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_training_data(data_basepath)

    transform_train = TrainingTransform(image_size=CONFIG["img_size"], num_channels=3)
    transform_val = ValidationTransform(image_size=CONFIG["img_size"], num_channels=3)

    trainloader, valloader, len_train, len_val = create_datasets_and_loaders(training_data,
                                                                             data_subset_label,
                                                                             transform_train,
                                                                             transform_val,
                                                                             num_workers=12,
                                                                             split_factor=0.1,
                                                                             batch_size=8)

    NUM_EPOCHS = CONFIG["epochs"]

    # model = CNN_LSTM_Model(backbone=CONFIG["backbone"]).to(device)
    model = VIT_Model(backbone=CONFIG["backbone"]).to(device)
    optimizers = [
        torch.optim.Adam(model.encoder.parameters(), lr=1e-4),
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


def train():
    model_t2stir = train_model_for_series("Sagittal T2/STIR", "tiny_vit_21m_512_t2stir")
    # model_t1 = train_model_for_series("Sagittal T1", "efficientnet_b0_lstm_t1")
    # model_t2 = train_model_for_series("Axial T2", "efficientnet_b0_lstm_t2")


if __name__ == '__main__':
    train()