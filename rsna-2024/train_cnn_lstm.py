from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResnetBackbone(nn.Module):
    pretrained_path = "./models/resnet18_brainmri_model.safetensors"

    def __init__(self, out_features=512, pretrained_weights=None):
        super(ResnetBackbone, self).__init__()
        # self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model = AutoModelForImageClassification.from_pretrained("BehradG/resnet-18-MRI-Brain", torchscript=True)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        self.model.classifier = nn.Identity()

    def forward(self, x):
        return self.model(x)


class FCHead(nn.Module):
    def __init__(self, drop_rate=0.1, num_classes=3):
        super(FCHead, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(256, 128),
            # nn.BatchNorm1d(256),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 128),
            nn.Dropout(drop_rate),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.model(x)


class RNSAModel2_5D(nn.Module):
    hidden_size = 256
    num_layers = 3
    def __init__(self, num_classes=2, num_levels=5, drop_rate=0.2, resnet_weights=None):
        super(RNSAModel2_5D, self).__init__()
        self.backbone = ResnetBackbone(pretrained_weights=resnet_weights)
        self.temporal = nn.LSTM(input_size=512, hidden_size=self.hidden_size, dropout=drop_rate, num_layers=self.num_layers,
                                batch_first=True,
                                bidirectional=True)
        self.head = FCHead(num_classes=num_levels * num_classes)
        self.activation = nn.Sigmoid()

    def forward(self, x_3d):
        hidden = None

        # Iterate over each frame of a video in a video of batch * frames * channels * height * width
        for t in range(x_3d.size(1)):
            x = self.backbone(x_3d[:, t])
            # Pass latent representation of frame through lstm and update hidden state
            out, hidden = self.temporal(x.unsqueeze(0), hidden)

            # Get the last hidden state (hidden is a tuple with both hidden and cell state in it)

        x = self.head(hidden[0][-1])
        return self.activation(x)

def train_model_for_series(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_training_data(data_basepath)

    transform_train = transforms.Compose([
        transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        # transforms.RandomHorizontalFlip(p=0.3),
        # transforms.RandomVerticalFlip(p=0.3),
        # transforms.RandomRotation([0, 90]),
        transforms.RandomChoice([
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
            v2.Identity(),
        ], p=[0.25, 0.75]),
        v2.RandomPhotometricDistort(p=0.3),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    transform_val = transforms.Compose([
        transforms.Lambda(lambda x: (x * 255).astype(np.uint8)),  # Convert back to uint8 for PIL
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ])

    trainloader, valloader, len_train, len_val = create_series_level_datasets_and_loaders(training_data,
                                                                                          data_subset_label,
                                                                                          transform_val,
                                                                                          # Try overfitting first
                                                                                          transform_val,
                                                                                          data_basepath + "train_images",
                                                                                          num_workers=4, batch_size=1)

    NUM_EPOCHS = 40

    model = RNSAModel2_5D().to(device)
    optimizers = [torch.optim.Adam(model.head.parameters(), lr=1e-3),
                  torch.optim.Adam(model.temporal.parameters(), lr=5e-4),
                  torch.optim.Adam(model.backbone.parameters(), lr=1e-4)]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=5e-5),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[1], NUM_EPOCHS, eta_min=2e-6),
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[2], NUM_EPOCHS, eta_min=5e-7),
        ]

    criterion = nn.BCELoss()

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criterion,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {data_subset_label}",
                                epochs=NUM_EPOCHS)

    return model


def train():
    model_t2stir = train_model_for_series("Sagittal T2/STIR", "resnet18_lstm_t2stir")
    # model_t1 = train_model_for_series("Sagittal T1", "resnet18_lstm_t1")
    # model_t2 = train_model_for_series("Axial T2", "resnet18_lstm_t2")


if __name__ == '__main__':
    train()
