import torch.nn.functional as F
from torch import Tensor
import torchvision.transforms as transforms
import segmentation_models_pytorch_3d as smp3d
from torchvision.transforms import v2

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)

CONFIG = dict(
    n_levels=5,
    interpolation="bspline",
    backbone="efficientnet-b5",
    vol_size=(96, 96, 96),
    img_size=(512, 512),
    num_workers=16,
    drop_rate=0.5,
    drop_rate_last=0.1,
    drop_path_rate=0.5,
    aug_prob=0.7,
    out_dim=3,
    epochs=50,
    batch_size=8,
    device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
    seed=2024
)
DATA_BASEPATH = "./data/spine_segmentation_nnunet_v2/"
# TRAINING_DATA = retrieve_segmentation_training_data(DATA_BASEPATH)

# region segment_loss
class SegmentationLoss(nn.Module):
    def __init__(self, multiclass=False):
        super().__init__()
        self.multiclass = multiclass
        # self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, input, target):
        if self.multiclass:
            ce_loss = F.cross_entropy(input, target)
        else:
            ce_loss = F.binary_cross_entropy_with_logits(input, target.unsqueeze(1).to(torch.float))
        dice_loss = self.dice_loss(F.softmax(input, dim=1), target if self.multiclass else target.unsqueeze(1))
        return (ce_loss + dice_loss) / 2

    def dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
        # Average of Dice coefficient for all batches, or for a single mask
        assert input.size() == target.size()
        sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

        inter = 2 * (input * target).sum(dim=sum_dim)
        sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
        sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

        dice = (inter + epsilon) / (sets_sum + epsilon)
        return dice.mean()

    def multiclass_dice_coeff(self, input: Tensor, target: Tensor, reduce_batch_first: bool = False,
                              epsilon: float = 1e-6):
        # Average of Dice coefficient for all classes
        # !TODO: Configurable num classes
        return self.dice_coeff(input.flatten(0, 1),
                               F.one_hot(target, num_classes=26).permute(0, -1, 1, 2, 3).flatten(0, 1),
                               reduce_batch_first, epsilon)

    def dice_loss(self, input: Tensor, target: Tensor):
        # Dice loss (objective to minimize) between 0 and 1
        fn = self.multiclass_dice_coeff if self.multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)


# endregion

def train_segmentation_model_3d(model_label: str):
    transform_3d_train = tio.Compose([
        tio.RandomAffine(p=CONFIG["aug_prob"]),
        tio.RandomFlip(axes=0, p=CONFIG["aug_prob"] / 3),
        tio.RandomFlip(axes=1, p=CONFIG["aug_prob"] / 3),
        tio.RandomFlip(axes=2, p=CONFIG["aug_prob"] / 3),
        tio.OneOf([
            tio.Resize(CONFIG["vol_size"], image_interpolation=CONFIG["interpolation"]),
            tio.Compose([
                tio.Resize(tuple(e * 2 for e in CONFIG["vol_size"]), image_interpolation=CONFIG["interpolation"]),
                tio.CropOrPad(CONFIG["vol_size"])
            ])
        ]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RandomBlur(p=CONFIG["aug_prob"]),
        tio.RandomAnisotropy(p=CONFIG["aug_prob"]),
        tio.RandomSpike(p=CONFIG["aug_prob"]),
        tio.RandomGamma(p=CONFIG["aug_prob"]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.Resize(CONFIG["vol_size"], image_interpolation=CONFIG["interpolation"]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_segmentation_datasets_and_loaders(
        transform_3d_train=transform_3d_train,
        transform_3d_val=transform_3d_val,
        base_path=DATA_BASEPATH,
        num_workers=CONFIG[
            "num_workers"],
        split_factor=0.3,
        batch_size=CONFIG[
            "batch_size"],
        pin_memory=False
    )

    NUM_EPOCHS = CONFIG["epochs"]

    model = smp3d.Unet(
        encoder_name=CONFIG["backbone"],  # choose encoder, e.g. resnet34
        in_channels=1,  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
    ).to(device)

    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]

    schedulers = [
    ]

    criteria = {
        "train": [
            SegmentationLoss()
        ],
        "val": [
            nn.CrossEntropyLoss()
        ]
    }

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {model_label}",
                                epochs=NUM_EPOCHS,
                                freeze_backbone_initial_epochs=0,
                                )


def train_segmentation_model_2d(data_type: str, model_label: str):
    transform_2d_train = transforms.Compose([
        # transforms.Resize(CONFIG["img_size"]),
        # tio.RandomAffine(p=CONFIG["aug_prob"]),
        # transforms.RandomNoise(p=CONFIG["aug_prob"]),
        # transforms.RandomBlur(p=CONFIG["aug_prob"]),
        # transforms.RandomAnisotropy(p=CONFIG["aug_prob"]),
        # transforms.RandomSpike(p=CONFIG["aug_prob"]),
        # transforms.RandomGamma(p=CONFIG["aug_prob"]),
        # v2.AutoAugment(),
        v2.RandomChoice((
            v2.GaussianBlur(kernel_size=(3, 3)),
            v2.GaussianBlur(kernel_size=(5, 5)),
            v2.GaussianBlur(kernel_size=(7, 7)),
            v2.Identity()
        ),
            p=[0.2, 0.2, 0.2, 0.4]),
        v2.RandomChoice((
            v2.ColorJitter(),
            v2.Identity()
        ), p=[0.7, 0.3]),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    transform_2d_val = transforms.Compose([
        # transforms.Resize(CONFIG["img_size"]),
        transforms.Normalize(mean=0.5, std=0.5),
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_series_level_segmentation_datasets_and_loaders(TRAINING_DATA,
                                                                                        data_type=data_type,
                                                                                        transform_2d_train=transform_2d_train,
                                                                                        transform_2d_val=transform_2d_val,
                                                                                        base_path=os.path.join(
                                                                                            DATA_BASEPATH,
                                                                                            "train_images"),
                                                                                        num_workers=CONFIG[
                                                                                            "num_workers"],
                                                                                        split_factor=0.3,
                                                                                        batch_size=CONFIG[
                                                                                            "batch_size"],
                                                                                        pin_memory=False
                                                                                        )

    NUM_EPOCHS = CONFIG["epochs"]
    # model = UNet3D(n_channels=num_channels, n_classes=CONFIG["n_levels"]).to(device)

    model = smp.Unet(
        classes=CONFIG["n_levels"]
    ).to(device)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]

    schedulers = [
    ]

    criteria = {
        "train": [
            SegmentationLoss() for i in range(CONFIG["n_levels"])
        ],
        "val": [
            nn.BCEWithLogitsLoss() for i in range(CONFIG["n_levels"])
        ]
    }

    train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                criteria,
                                trainloader,
                                valloader,
                                model_desc=model_label,
                                train_loader_desc=f"Training {model_label}",
                                epochs=NUM_EPOCHS,
                                freeze_backbone_initial_epochs=0,
                                )


def train():
    model = train_segmentation_model_3d(f"{CONFIG['backbone']}_unet_segmentation_{CONFIG['vol_size'][0]}_3d_binary")
    # torch.multiprocessing.set_start_method('spawn')


if __name__ == '__main__':
    train()
