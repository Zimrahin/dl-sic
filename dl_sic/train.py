import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.ctdcr_net import CTDCR_net
from utils.misc import set_seed
from utils.dataset import DummyDataset, create_dataloaders
from utils.loss_functions import mse_loss_complex


def train_epoch(
    model: nn.Module,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: callable,
    device: torch.device,
    *,
    log_interval: int = 1,  # Log every log_interval batches
    writer: None | SummaryWriter = None,
) -> float:
    """Train model for one epoch"""
    # docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    non_blocking = device.type == "cuda"  # Asynchronous GPU transfers

    model.train()
    total_loss = 0.0

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Train Epoch {epoch}",
        mininterval=1.0,  # Update at most once per second
        leave=True,
    )
    for batch_idx, (mixture, target) in progress_bar:
        mixture = mixture.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # Forward pass
        output = model(mixture)
        loss: torch.Tensor = loss_function(output, target)
        # Backward pass
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        # Update epoch loss and log
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.set_postfix(loss=f"{avg_loss:.4f}")
        if writer and (batch_idx % log_interval == 0):
            step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Loss/train", loss.item(), step)

    return total_loss / len(train_loader)  # Average epoch loss


def validate_epoch(
    model: nn.Module,
    epoch: int,
    val_loader: torch.utils.data.DataLoader,
    loss_function: callable,
    device: torch.device,
    *,
    writer: None | SummaryWriter = None,
) -> float:
    """Validate model with forward pass"""
    # docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    non_blocking = device.type == "cuda"  # Asynchronous GPU transfers

    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Validation Epoch {epoch}",
        leave=True,  # Keep progress bar after completion
    )

    with torch.no_grad():
        for batch_idx, (mixture, target) in progress_bar:
            mixture = mixture.to(device, non_blocking=non_blocking)
            target = target.to(device, non_blocking=non_blocking)

            # Forward pass
            output = model(mixture)
            # Compute loss
            loss: torch.Tensor = loss_function(output, target)
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    avg_epoch_loss = total_loss / len(val_loader)
    if writer:
        writer.add_scalar("Loss/val", avg_epoch_loss, epoch)

    return avg_epoch_loss
