import math

import timm
import timm_3d
import torchio as tio
from spacecutter.losses import CumulativeLinkLoss
from spacecutter.models import LogisticCumulativeLink

from training_utils import *
from rsna_dataloader import *

_logger = logging.getLogger(__name__)

CONFIG = dict(
    n_levels=5,
    num_classes=25,
    # backbone="efficientnet_b4",
    backbone="tf_efficientnetv2_m",
    interpolation="bspline",
    # interpolation="gaussian",
    img_size=(128, 128),
    vol_size=(144, 144, 144),
    num_workers=6,
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
DATA_BASEPATH = "./data/rsna-2024-lumbar-spine-degenerative-classification/"
TRAINING_DATA = retrieve_coordinate_training_data(DATA_BASEPATH)

CLASS_RELATIVE_WEIGHTS = torch.Tensor([[1., 29.34146341, 601.5, ],
                                       [1., 10.46296296, 141.25, ],
                                       [1., 3.6539924, 43.68181818, ],
                                       [1., 1.89223058, 8.20652174, ],
                                       [1., 2.31736527, 5.60869565, ],
                                       [1., 19.46666667, 64.88888889, ],
                                       [1., 6.30674847, 18.69090909, ],
                                       [1., 2.92041522, 7.46902655, ],
                                       [1., 1.5144357, 2.00347222, ],
                                       [1., 3.43076923, 9.4893617, ],
                                       [1., 27.11363636, 132.55555556, ],
                                       [1., 10.5, 283.5, ],
                                       [1., 3.65267176, 35.44444444, ],
                                       [1., 2.05277045, 8.74157303, ],
                                       [1., 2.75333333, 6.88333333, ],
                                       [1., 14.59493671, 82.35714286, ],
                                       [1., 6.32926829, 23.59090909, ],
                                       [1., 2.82828283, 7.70642202, ],
                                       [1., 1.43367347, 1.92465753, ],
                                       [1., 3.57429719, 8.31775701, ],
                                       [1., 29.04878049, 85.07142857, ],
                                       [1., 11.31632653, 28.43589744, ],
                                       [1., 7.16083916, 12.96202532, ],
                                       [1., 6.25675676, 5.38372093, ],
                                       [1., 44.66666667, 92.76923077],
                                       ]).to(CONFIG["device"])

CLASS_LOGN_RELATIVE_WEIGHTS = 1 + 2 * torch.log(CLASS_RELATIVE_WEIGHTS)

CLASS_NEG_VS_POS = torch.Tensor(
    [3.57439734e-02, 2.93902439e+01, 6.22000000e+02,
     1.02654867e-01, 1.05370370e+01, 1.54750000e+02,
     0.29656608, 3.73764259, 55.63636364,
     0.65033113, 2.12280702, 12.54347826,
     0.60981912, 2.73053892, 8.02898551,
     6.67808219e-02, 1.97666667e+01, 6.82222222e+01,
     0.21206226, 6.64417178, 21.65454545,
     0.47630332, 3.31141869, 10.02654867,
     1.15944541, 2.27034121, 3.32638889,
     0.39686099, 3.79230769, 12.25531915,
     4.44258173e-02, 2.73181818e+01, 1.37444444e+02,
     9.87654321e-02, 1.05370370e+01, 3.10500000e+02,
     0.30198537, 3.75572519, 45.14814815,
     0.60154242, 2.28759894, 13.,
     0.50847458, 3.15333333, 9.38333333,
     8.06591500e-02, 1.47721519e+01, 8.80000000e+01,
     0.20038536, 6.59756098, 27.31818182,
     0.48333333, 3.1952862, 10.43119266,
     1.21708185, 2.17857143, 3.26712329,
     0.4, 4.00401606, 10.64485981,
     4.61796809e-02, 2.93902439e+01, 8.80000000e+01,
     0.12353472, 11.71428571, 30.94871795,
     0.21679688, 7.71328671, 14.7721519,
     0.34557235, 7.41891892, 6.24418605,
     3.31674959e-02, 4.51481481e+01, 9.48461538e+01]
).to(CONFIG["device"])

COMP_WEIGHTS = torch.Tensor([1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             1, 2, 4,
                             ]).to(CONFIG["device"])

CONFIG["loss_weights"] = CLASS_LOGN_RELATIVE_WEIGHTS

class CNN_Model_3D_Multihead(nn.Module):
    def __init__(self, backbone="efficientnet_lite0", in_chans=1, out_classes=5, out_dim=3, pretrained=True):
        super(CNN_Model_3D_Multihead, self).__init__()
        self.out_classes = out_classes

        self.encoder = timm_3d.create_model(
            backbone,
            num_classes=out_classes * CONFIG["out_dim"],
            features_only=False,
            drop_rate=CONFIG["drop_rate"],
            drop_path_rate=CONFIG["drop_path_rate"],
            # drop_rate_last=CONFIG["drop_rate_last"],
            pretrained=pretrained,
            in_chans=in_chans,
        )
        head_in_dim = self.encoder.classifier.in_features
        self.encoder.classifier = nn.Identity()
        self.heads = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(head_in_dim, 1),
                LogisticCumulativeLink(CONFIG["out_dim"])
            ) for i in range(out_classes)]
        )

    def forward(self, x):
        feat = self.encoder(x)
        return torch.swapaxes(torch.stack([head(feat) for head in self.heads]), 0, 1)


def train_model_3d(backbone, model_label: str):
    transform_3d_train = tio.Compose([
        tio.Resize(CONFIG["vol_size"], image_interpolation=CONFIG["interpolation"]),
        tio.RandomAffine(p=CONFIG["aug_prob"]),
        # tio.OneOf({
        #     tio.RandomElasticDeformation(): 0.3,
        #     tio.RandomAffine(): 0.7,
        # }, p=CONFIG["aug_prob"]),
        tio.RandomNoise(p=CONFIG["aug_prob"]),
        tio.RandomBlur(p=CONFIG["aug_prob"]),
        tio.RandomAnisotropy(p=CONFIG["aug_prob"]),
        # tio.RandomBiasField(p=CONFIG["aug_prob"]),
        tio.RandomSpike(p=CONFIG["aug_prob"]),
        tio.RandomGamma(p=CONFIG["aug_prob"]),
        # tio.RandomSwap(p=CONFIG["aug_prob"]),
        # tio.RandomGhosting(p=CONFIG["aug_prob"]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])

    transform_3d_val = tio.Compose([
        tio.Resize(CONFIG["vol_size"], image_interpolation=CONFIG["interpolation"]),
        tio.RescaleIntensity(out_min_max=(0, 1)),
    ])

    (trainloader, valloader, test_loader,
     trainset, valset, testset) = create_subject_level_datasets_and_loaders(TRAINING_DATA,
                                                                            transform_3d_train=transform_3d_train,
                                                                            transform_3d_val=transform_3d_val,
                                                                            base_path=os.path.join(
                                                                                DATA_BASEPATH,
                                                                                "train_images"),
                                                                            num_workers=CONFIG["num_workers"],
                                                                            split_factor=0.3,
                                                                            batch_size=CONFIG["batch_size"],
                                                                            pin_memory=False
                                                                            )

    NUM_EPOCHS = CONFIG["epochs"]

    model = CNN_Model_3D_Multihead(backbone=backbone, in_chans=3, out_classes=CONFIG["num_classes"]).to(device)
    optimizers = [
        torch.optim.Adam(model.parameters(), lr=1e-3),
    ]
    schedulers = [
    ]
    criteria = {
        "train": [
            CumulativeLinkLoss(class_weights=CONFIG["loss_weights"][i]) for i in range(CONFIG["num_classes"])
        ],
        "val": [
            CumulativeLinkLoss() for i in range(CONFIG["num_classes"])
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
                                loss_weights=CONFIG["loss_weights"]
                                )

    return model


def train():
    model = train_model_3d(CONFIG['backbone'],
                           f"{CONFIG['backbone']}_{CONFIG['vol_size'][0]}_3d")


if __name__ == '__main__':
    train()
