from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ResnetCoordinateDetector(nn.Module):
    pretrained_path = "./models/resnet18_brainmri_model.safetensors"

    def __init__(self, out_features=10, pretrained_weights=None):
        super(ResnetCoordinateDetector, self).__init__()
        self.model = AutoModelForImageClassification.from_pretrained("BehradG/resnet-18-MRI-Brain", torchscript=True)
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        self.model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=out_features),
        )

    def forward(self, x):
        return self.model(x.squeeze(0))[0]


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

    trainloader, valloader, len_train, len_val = create_series_level_coordinate_datasets_and_loaders(training_data,
                                                                                                     data_subset_label,
                                                                                                     transform_val,
                                                                                                     # Try overfitting first
                                                                                                     transform_val,
                                                                                                     data_basepath + "train_images",
                                                                                                     num_workers=0,
                                                                                                     batch_size=1)

    NUM_EPOCHS = 40

    model = ResnetCoordinateDetector().to(device)
    optimizers = [torch.optim.Adam(model.parameters(), lr=1e-3)]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=5e-5),
    ]

    criterion = nn.MSELoss()

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
    model_t2stir = train_model_for_series("Sagittal T2/STIR", "resnet18_coordinates_t2stir")


if __name__ == "__main__":
    train()
