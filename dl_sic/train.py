import os
import argparse
import time
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
    optimiser: torch.optim.Optimizer,
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
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

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
            loss: torch.Tensor = loss_function(output, target)
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix(loss=f"{avg_loss:.4f}")

    avg_epoch_loss = total_loss / len(val_loader)
    if writer:
        writer.add_scalar("Loss/val", avg_epoch_loss, epoch)

    return avg_epoch_loss


def train_ctdcr_net(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    epochs: int,
    val_split: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    *,
    resume: bool = False,
) -> None:
    tensorboard = True
    log_interval = 10  # Default log interval for TensorBoard
    seed = 0  # For reproducibility
    checkpoints_dir = "./checkpoints"

    # discuss.pytorch.org/t/guidelines-for-assigning-num-workers-to-dataloader/813/25
    num_workers = 0  # Default DataLoader value
    M, N, U, H, V = 128, 32, 128, 32, 8  # CTDCR net parameters

    set_seed(seed)
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    writer = (
        SummaryWriter(log_dir=os.path.join(checkpoints_dir, "logs"))
        if tensorboard
        else None
    )

    # Initialise model, dataloaders, loss function, and optimiser
    model = CTDCR_net(M, N, U, H, V).to(device)
    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")
    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size,
        num_workers,
        pin_memory=(device.type == "cuda"),
        val_split=val_split,
        split_seed=seed,
    )
    loss_function = mse_loss_complex
    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode="min")

    start_epoch = 0
    best_val_loss = float("inf")

    if resume:
        checkpoint_path = os.path.join(checkpoints_dir, "last_checkpoint.pth")
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
            model.load_state_dict(checkpoint["model_state"])
            optimiser.load_state_dict(checkpoint["optimiser_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"Checkpoint loaded at epoch {checkpoint['epoch']}")
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    # Training loop
    for epoch in range(start_epoch, epochs):
        epoch_start = time.time()

        train_loss = train_epoch(
            model=model,
            epoch=epoch,
            train_loader=train_loader,
            optimiser=optimiser,
            loss_function=loss_function,
            device=device,
            log_interval=log_interval,
            writer=writer,
        )
        val_loss = validate_epoch(
            model=model,
            epoch=epoch,
            val_loader=val_loader,
            loss_function=loss_function,
            device=device,
            writer=writer,
        )
        scheduler.step(val_loss)  # Reduce learning rate on plateau

        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, "best_model_weights.pth"),
            )
            print(f"New best model saved with val loss: {best_val_loss:.6f}")

        torch.save(
            {
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "model_state": model.state_dict(),
                "optimiser_state": optimiser.state_dict(),
                "scheduler_state": scheduler.state_dict(),
            },
            os.path.join(checkpoints_dir, "last_checkpoint.pth"),
        )

        epoch_time = time.time() - epoch_start
        print(
            f"Epoch {epoch} completed in {epoch_time:.2f}s - "
            f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )

    if writer:
        writer.close()
    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTDCR Network Training")
    parser.add_argument(
        "--batch_size", type=int, default=4, help="Input batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    args = parser.parse_args()

    train_ctdcr_net(
        dataset=DummyDataset(num_signals=20, signal_length=2048),
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        resume=args.resume,
    )
