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

# region unet3d
"""Adapted from https://github.com/jphdotam/Unet3D/blob/main/unet3d.py"""


class UNet3D(nn.Module):
    def __init__(self, n_channels, n_classes, width_multiplier=1, trilinear=True, use_ds_conv=False):
        """A simple 3D Unet, adapted from a 2D Unet from https://github.com/milesial/Pytorch-UNet/tree/master/unet
        Arguments:
          n_channels = number of input channels; 3 for RGB, 1 for grayscale input
          n_classes = number of output channels/classes
          width_multiplier = how much 'wider' your UNet should be compared with a standard UNet
                  default is 1;, meaning 32 -> 64 -> 128 -> 256 -> 512 -> 256 -> 128 -> 64 -> 32
                  higher values increase the number of kernels pay layer, by that factor
          trilinear = use trilinear interpolation to upsample; if false, 3D convtranspose layers will be used instead
          use_ds_conv = if True, we use depthwise-separable convolutional layers. in my experience, this is of little help. This
                  appears to be because with 3D data, the vast vast majority of GPU RAM is the input data/labels, not the params, so little
                  VRAM is saved by using ds_conv, and yet performance suffers."""
        super(UNet3D, self).__init__()
        _channels = (32, 64, 128, 256, 512)
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.channels = [int(c * width_multiplier) for c in _channels]
        self.trilinear = trilinear
        self.convtype = DepthwiseSeparableConv3d if use_ds_conv else nn.Conv3d

        self.inc = DoubleConv3D(n_channels, self.channels[0], conv_type=self.convtype)
        self.down1 = Down3D(self.channels[0], self.channels[1], conv_type=self.convtype)
        self.down2 = Down3D(self.channels[1], self.channels[2], conv_type=self.convtype)
        self.down3 = Down3D(self.channels[2], self.channels[3], conv_type=self.convtype)
        factor = 2 if trilinear else 1
        self.down4 = Down3D(self.channels[3], self.channels[4] // factor, conv_type=self.convtype)
        self.up1 = Up3D(self.channels[4], self.channels[3] // factor, trilinear)
        self.up2 = Up3D(self.channels[3], self.channels[2] // factor, trilinear)
        self.up3 = Up3D(self.channels[2], self.channels[1] // factor, trilinear)
        self.up4 = Up3D(self.channels[1], self.channels[0], trilinear)
        self.outc = OutConv3D(self.channels[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            conv_type(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            conv_type(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_type=nn.Conv3d):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv3D(in_channels, out_channels, conv_type=conv_type)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, trilinear=True):
        super().__init__()

        # if trilinear, use the normal convolutions to reduce the number of channels
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = DoubleConv3D(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv3D(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, nin, nout, kernel_size, padding, kernels_per_layer=1):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.Conv3d(nin, nin * kernels_per_layer, kernel_size=kernel_size, padding=padding, groups=nin)
        self.pointwise = nn.Conv3d(nin * kernels_per_layer, nout, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out


# endregion

# region unet
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = (DoubleConv(n_channels, 64))
        self.down1 = (Down(64, 128))
        self.down2 = (Down(128, 256))
        self.down3 = (Down(256, 512))
        factor = 2 if bilinear else 1
        self.down4 = (Down(512, 1024 // factor))
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 = (Up(128, 64, bilinear))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


# endregion

# region segment_loss
class SegmentationLoss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.dice = smp.losses.DiceLoss(mode="multiclass")

    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target)
        dice_loss = self.dice_loss(F.softmax(input, dim=1), target, multiclass=True)
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

    def dice_loss(self, input: Tensor, target: Tensor, multiclass: bool = False):
        # Dice loss (objective to minimize) between 0 and 1
        fn = self.multiclass_dice_coeff if multiclass else self.dice_coeff
        return 1 - fn(input, target, reduce_batch_first=True)


# endregion

def train_segmentation_model_3d(model_label: str):
    transform_3d_train = tio.Compose([
        tio.Resize(CONFIG["vol_size"], image_interpolation=CONFIG["interpolation"]),
        tio.RandomAffine(p=CONFIG["aug_prob"]),
        tio.RandomFlip(axes=0, p=CONFIG["aug_prob"] / 3),
        tio.RandomFlip(axes=1, p=CONFIG["aug_prob"] / 3),
        tio.RandomFlip(axes=2, p=CONFIG["aug_prob"] / 3),
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

    # model = UNet3D(n_channels=1, n_classes=26).to(device)
    model = smp3d.Unet(
        encoder_name="efficientnet-b4",  # choose encoder, e.g. resnet34
        in_channels=1,  # model input channels (1 for gray-scale volumes, 3 for RGB, etc.)
        classes=26,  # model output channels (number of classes in your dataset)
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
    model = train_segmentation_model_3d(f"efficientnetb4_unet_segmentation_{CONFIG['vol_size'][0]}_3d")
    # torch.multiprocessing.set_start_method('spawn')


if __name__ == '__main__':
    train()
