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


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, outputs, targets):
        ce_loss = torch.nn.functional.cross_entropy(outputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        return focal_loss


# !TODO: Optional
def freeze_model_backbone(model: nn.Module):
    for param in model.backbone.model.parameters():
        param.requires_grad = False
    # for param in model.backbone.model.fc.parameters():
    #     param.requires_grad = True


# !TODO: Optional
def unfreeze_model_backbone(model: nn.Module):
    for param in model.backbone.model.parameters():
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
                loss = loss_fn(output[:, index], label[:, index])
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


def train_model_with_validation(model, optimizers, schedulers, loss_fns, train_loader, val_loader,
                                train_loader_desc=None,
                                model_desc="my_model", epochs=10):
    epoch_losses = []
    epoch_validation_losses = []

    # freeze_model_backbone(model)

    for epoch in tqdm(range(epochs), desc=train_loader_desc):
        epoch_loss = 0
        model.train()

        # if epoch >= 10:
        #     unfreeze_model_backbone(model)

        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #              profile_memory=True,
        #              record_shapes=True,
        #              schedule=torch.profiler.schedule(
        #                  wait=5,
        #                  warmup=2,
        #                  active=6,
        #                  repeat=5
        #              ),
        #              on_trace_ready=trace_handler,
        #              with_stack=True,
        #              ) as prof:
        for index, val in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images, label = val
            # !TODO: Do this in the data loader
            label = label.to(device)
            # label = label.type(torch.LongTensor).to(device)

            for optimizer in optimizers:
                optimizer.zero_grad(set_to_none=True)

            output = model(images.to(device))

            # !TODO: Refactor
            # !TODO: Track separately
            for loss_index, loss_fn in enumerate(loss_fns):
                loss = loss_fn(output[:, loss_index], label[:, loss_index])
                epoch_loss += loss.detach().cpu().item()
                loss.backward(retain_graph=True)
                del loss

            del output

            for optimizer in optimizers:
                optimizer.step()

            #prof.step()
            if index % 20 == 0:
                torch.cuda.empty_cache()

        epoch_loss = epoch_loss / len(train_loader)
        epoch_validation_loss = model_validation_loss(model, val_loader, loss_fns, epoch)

        for scheduler in schedulers:
            scheduler.step()


        if (epoch + 1) % 25 == 0 or ((epoch + 1) % 10 == 0 and (
                (not epoch_validation_losses) or epoch_validation_loss < min(epoch_validation_losses))):
            os.makedirs(f'./models/{model_desc}', exist_ok=True)
            torch.save(model, f'./models/{model_desc}/{model_desc}' + "_" + str(epoch) + ".pt")

        epoch_validation_losses.append(epoch_validation_loss)
        epoch_losses.append(epoch_loss)

        dump_plots_for_loss_and_acc(epoch_losses, epoch_validation_losses,
                                    train_loader_desc, model_desc)
        print(f"Training Loss for epoch {epoch}: {epoch_loss:.6f}")
        print(f"Validation Loss for epoch {epoch}: {epoch_validation_loss:.6f}")

    return epoch_losses, epoch_validation_losses
