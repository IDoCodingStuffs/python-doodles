from training_utils import *
from rsna_dataloader import *

from torchvision.ops import complete_box_iou_loss

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CoordinateDetector2D(nn.Module):
    def __init__(self, out_features=10, pretrained_weights=None):
        super(CoordinateDetector2D, self).__init__()
        # self.model = AutoModelForImageClassification.from_pretrained("BehradG/resnet-18-MRI-Brain", torchscript=True)
        self.model = models.resnet34()
        if pretrained_weights:
            self.model.load_state_dict(torch.load(pretrained_weights))
        # self.model.classifier = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(in_features=512, out_features=out_features),
        # )
        self.model.fc = nn.Linear(in_features=512, out_features=out_features)

    def forward(self, x):
        return self.model(x)  # [0]


class CompleteBoxIOULoss(nn.Module):
    def forward(self, inferred, target):
        inferred_boxes = get_bounding_boxes_for_label(inferred[0]).reshape(-1, 4)
        target_boxes = get_bounding_boxes_for_label(target[0]).reshape(-1, 4)

        return complete_box_iou_loss(inferred_boxes, target_boxes, reduction='mean')


def train_model_per_image(data_subset_label: str, model_label: str):
    data_basepath = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
    training_data = retrieve_coordinate_training_data(data_basepath)

    transform_train = TrainingTransform()
    transform_val = ValidationTransform()

    trainloader, valloader, len_train, len_val = create_coordinate_datasets_and_loaders(training_data,
                                                                                        data_subset_label,
                                                                                        transform_train,
                                                                                        transform_val,
                                                                                        data_basepath + "train_images",
                                                                                        num_workers=4,
                                                                                        batch_size=1)

    NUM_EPOCHS = 40

    model = CoordinateDetector2D().to(device)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]

    schedulers = [
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizers[0], NUM_EPOCHS, eta_min=5e-6),
    ]

    criteria = [
        #nn.HuberLoss().to("cuda"),
        CompleteBoxIOULoss().to("cuda")
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
    model_t2stir = train_model_per_image("Sagittal T2/STIR", "resnet34_cnn_coordinates_t2stir")


if __name__ == "__main__":
    train()
