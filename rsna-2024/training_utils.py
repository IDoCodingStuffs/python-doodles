import os.path

from tqdm import tqdm
import logging
from torch.profiler import profile, record_function, ProfilerActivity
from torch.cuda.amp import autocast, GradScaler

from rsna_dataloader import *

_logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# !TODO: Optional
def freeze_model_backbone(model: nn.Module):
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_model_backbone(model: nn.Module):
    for param in model.backbone.parameters():
        param.requires_grad = True


def model_validation_loss(model, val_loader, loss_fns, epoch, loss_weights=None):
    total_loss = 0
    weighted_loss = 0

    with torch.no_grad():
        model.eval()

        for images, label in tqdm(val_loader, desc=f"Validating epoch {epoch}"):
            label = label.to(device).unsqueeze(-1)

            with autocast(enabled=device != "cpu", dtype=torch.bfloat16):
                output = model(images.to(device))
                for index, loss_fn in enumerate(loss_fns["val"]):
                    loss = loss_fn(output[:, index], label[:, index]) / len(loss_fns["val"])
                    total_loss += loss.cpu().item()

                for index, loss_fn in enumerate(loss_fns["train"]):
                    loss = loss_fn(output[:, index], label[:, index]) / len(loss_fns["train"])
                    weighted_loss += loss.cpu().item()

            del output
            # torch.cuda.empty_cache()

        total_loss = total_loss / len(val_loader)
        weighted_loss = weighted_loss / len(val_loader)

        return total_loss, weighted_loss


def dump_plots_for_loss_and_acc(losses, val_losses, alt_val_losses, data_subset_label, model_label):
    plt.plot(np.log(losses), label="train")
    plt.plot(np.log(val_losses), label="val")
    plt.plot(np.log(alt_val_losses), label="alt_val")
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
                                empty_cache_every_n_iterations=0,
                                loss_weights=None,
                                callbacks=None):
    epoch_losses = []
    epoch_validation_losses = []
    alt_epoch_validation_losses = []

    scaler = GradScaler(init_scale=4096)

    if freeze_backbone_initial_epochs > 0:
        freeze_model_backbone(model)

    for epoch in tqdm(range(epochs), desc=train_loader_desc):
        epoch_loss = 0
        model.train()

        if freeze_backbone_initial_epochs > 0 and epoch == freeze_backbone_initial_epochs:
            unfreeze_model_backbone(model)

        for index, val in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            images, label = val
            label = label.to(device).unsqueeze(-1)

            with autocast(enabled=device != "cpu", dtype=torch.bfloat16):
                output = model(images.to(device))

                loss = sum([(loss_fn(output[:, loss_index], label[:, loss_index]) / gradient_accumulation_per) for
                            loss_index, loss_fn in enumerate(loss_fns["train"])])
                # loss = loss_fn(output, label) / gradient_accumulation_per
                epoch_loss += loss.detach().cpu().item() * gradient_accumulation_per / len(loss_fns["train"])

            scaler.scale(loss).backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)

            del output

            # Per gradient accumulation batch or if the last iter
            if index % gradient_accumulation_per == 0 or index == len(train_loader) - 1:
                for optimizer in optimizers:
                    scaler.step(optimizer)
                    optimizer.zero_grad(set_to_none=True)

            scaler.update()

            if callbacks:
                for callback in callbacks:
                    callback()

            # prof.step()
            if empty_cache_every_n_iterations > 0 and index % empty_cache_every_n_iterations == 0:
                torch.cuda.empty_cache()

            while os.path.exists(".pause"):
                pass

        epoch_loss = epoch_loss / len(train_loader)
        epoch_validation_loss, alt_epoch_validation_loss = model_validation_loss(model, val_loader, loss_fns, epoch,
                                                                                 loss_weights)

        for scheduler in schedulers:
            scheduler.step()

        if (epoch % 5 == 0
            or epoch_validation_loss < min(epoch_validation_losses)) \
                or alt_epoch_validation_loss < min(alt_epoch_validation_losses):
            os.makedirs(f'./models/{model_desc}', exist_ok=True)
            torch.save(model,
                       # torch.jit.script(model),
                       f'./models/{model_desc}/{model_desc}' + "_" + str(epoch) + ".pt")

        epoch_validation_losses.append(epoch_validation_loss)
        alt_epoch_validation_losses.append(alt_epoch_validation_loss)
        epoch_losses.append(epoch_loss)

        dump_plots_for_loss_and_acc(epoch_losses, epoch_validation_losses, alt_epoch_validation_losses,
                                    train_loader_desc, model_desc)
        print(f"Training Loss for epoch {epoch}: {epoch_loss:.6f}")
        print(f"Validation Loss for epoch {epoch}: {epoch_validation_loss:.6f}")
        print(f"Alt Validation Loss for epoch {epoch}: {alt_epoch_validation_loss:.6f}")

    return epoch_losses, epoch_validation_losses
