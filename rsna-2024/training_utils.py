import os.path
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import pandas as pd
from tqdm import tqdm
import logging
import seaborn as sn
from itertools import chain
from transformers import AutoImageProcessor, AutoModelForImageClassification
from torch.profiler import profile, record_function, ProfilerActivity

from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# IMPLEMENTATION CREDIT: https://github.com/clcarwin/focal_loss_pytorch
class WeightedBCELoss(nn.Module):
    def __init__(self, device, alpha=None):
        super(WeightedBCELoss, self).__init__()
        if alpha is None:
            # alpha = torch.Tensor(
            #     [[1., 29.34146341, 601.5],
            #      [1., 10.46296296, 141.25],
            #      [1., 3.6539924, 43.68181818],
            #      [1., 1.89223058, 8.20652174],
            #      [1., 2.31736527, 5.60869565],
            #      [1., 19.46666667, 64.88888889],
            #      [1., 6.30674847, 18.69090909],
            #      [1., 2.92041522, 7.46902655],
            #      [1., 1.5144357, 2.00347222],
            #      [1., 3.43076923, 9.4893617],
            #      [1., 27.11363636, 132.55555556],
            #      [1., 10.5, 283.5],
            #      [1., 3.65267176, 35.44444444],
            #      [1., 2.05277045, 8.74157303],
            #      [1., 2.75333333, 6.88333333],
            #      [1., 14.59493671, 82.35714286],
            #      [1., 6.32926829, 23.59090909],
            #      [1., 2.82828283, 7.70642202],
            #      [1., 1.43367347, 1.92465753],
            #      [1., 3.57429719, 8.31775701],
            #      [1., 29.04878049, 85.07142857],
            #      [1., 11.31632653, 28.43589744],
            #      [1., 7.16083916, 12.96202532],
            #      [1., 6.25675676, 5.38372093],
            #      [1., 44.66666667, 92.76923077]
            #      ])
            alpha = torch.Tensor(
                [1., 29.34146341, 601.5,
                 1., 10.46296296, 141.25,
                 1., 3.6539924, 43.68181818,
                 1., 1.89223058, 8.20652174,
                 1., 2.31736527, 5.60869565,
                 1., 19.46666667, 64.88888889,
                 1., 6.30674847, 18.69090909,
                 1., 2.92041522, 7.46902655,
                 1., 1.5144357, 2.00347222,
                 1., 3.43076923, 9.4893617,
                 1., 27.11363636, 132.55555556,
                 1., 10.5, 283.5,
                 1., 3.65267176, 35.44444444,
                 1., 2.05277045, 8.74157303,
                 1., 2.75333333, 6.88333333,
                 1., 14.59493671, 82.35714286,
                 1., 6.32926829, 23.59090909,
                 1., 2.82828283, 7.70642202,
                 1., 1.43367347, 1.92465753,
                 1., 3.57429719, 8.31775701,
                 1., 29.04878049, 85.07142857,
                 1., 11.31632653, 28.43589744,
                 1., 7.16083916, 12.96202532,
                 1., 6.25675676, 5.38372093,
                 1., 44.66666667, 92.76923077
                 ]
            ).to(device)
        self.alpha = alpha
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, label):
        losses = self.bce_loss(pred, label)
        return torch.mean(self.alpha * losses)

# !TODO: Optional
def freeze_model_backbone(model: nn.Module):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_model_backbone(model: nn.Module):
    for param in model.backbone.parameters():
        param.requires_grad = True


def model_validation_loss(model, val_loader, loss_fns, epoch):
    total_loss = 0

    with torch.no_grad():
        model.eval()

        for images, label in tqdm(val_loader, desc=f"Validating epoch {epoch}"):
            # !TODO: Do this in the data loader
            label = label.to(device)

            output = model(images.to(device))
            for index, loss_fn in enumerate(loss_fns):
                # loss = loss_fn(output[:, index], label[:, index])
                loss = loss_fn(output, label)
                total_loss += loss.cpu().item()
                del loss

            del output
            torch.cuda.empty_cache()

        total_loss = total_loss / len(val_loader)

        return total_loss


def dump_plots_for_loss_and_acc(losses, val_losses, data_subset_label, model_label):
    plt.plot(np.log(losses), label="train")
    plt.plot(np.log(val_losses), label="val")
    plt.legend(loc="center right")
    plt.title(data_subset_label)
    plt.savefig(f'./figures/{model_label}_loss.png')
    plt.close()


def trace_handler(p):
    output = p.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10)
    print(output)
    p.export_chrome_trace("./traces/trace_" + str(p.step_num) + ".json")


def profile_to_use():
    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   profile_memory=True,
                   record_shapes=True,
                   schedule=torch.profiler.schedule(
                       wait=5,
                       warmup=2,
                       active=6,
                       repeat=5
                   ),
                   on_trace_ready=trace_handler,
                   with_stack=True,
                   )


def train_model_with_validation(model,
                                optimizers,
                                schedulers,
                                loss_fns,
                                train_loader,
                                val_loader,
                                train_loader_desc=None,
                                model_desc="my_model",
                                gradient_accumulation_per=1,
                                epochs=10,
                                freeze_backbone_initial_epochs=0,
                                empty_cache_every_n_iterations=0):
    epoch_losses = []
    epoch_validation_losses = []

    if freeze_backbone_initial_epochs > 0:
        freeze_model_backbone(model)

    for epoch in tqdm(range(epochs), desc=train_loader_desc):
        epoch_loss = 0
        model.train()

        if freeze_backbone_initial_epochs > 0 and epoch == freeze_backbone_initial_epochs:
            unfreeze_model_backbone(model)

        for index, val in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images, label = val
            label = label.to(device)

            output = model(images.to(device))

            for loss_index, loss_fn in enumerate(loss_fns):
                # loss = loss_fn(output[:, loss_index], label[:, loss_index]) / gradient_accumulation_per
                loss = loss_fn(output, label) / gradient_accumulation_per
                epoch_loss += loss.detach().cpu().item() * gradient_accumulation_per
                loss.backward(retain_graph=True)
                del loss

            del output

            # Per gradient accumulation batch or if the last iter
            if index % gradient_accumulation_per == 0 or index == len(train_loader) - 1:
                for optimizer in optimizers:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

            # prof.step()
            if empty_cache_every_n_iterations > 0 and index % empty_cache_every_n_iterations == 0:
                torch.cuda.empty_cache()

                # !TODO: Refactor
                while os.path.exists(".pause"):
                    pass

        epoch_loss = epoch_loss / len(train_loader)
        epoch_validation_loss = model_validation_loss(model, val_loader, loss_fns, epoch)

        for scheduler in schedulers:
            scheduler.step()

        if epoch % 5 == 0 or epoch < 10:
            os.makedirs(f'./models/{model_desc}', exist_ok=True)
            torch.save(model,
                       # torch.jit.script(model),
                       f'./models/{model_desc}/{model_desc}' + "_" + str(epoch) + ".pt")

        epoch_validation_losses.append(epoch_validation_loss)
        epoch_losses.append(epoch_loss)

        dump_plots_for_loss_and_acc(epoch_losses, epoch_validation_losses,
                                    train_loader_desc, model_desc)
        print(f"Training Loss for epoch {epoch}: {epoch_loss:.6f}")
        print(f"Validation Loss for epoch {epoch}: {epoch_validation_loss:.6f}")

    return epoch_losses, epoch_validation_losses
