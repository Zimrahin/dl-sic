import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from model.complex_tdcr_net import ComplexTDCRnet
from model.real_tdcr_net import RealTDCRnet
from utils.training import set_seed, TrainingLogger
from utils.dataset import DummyDataset, LoadDataset, create_dataloaders
from utils.loss_functions import si_snr_loss_complex
from data.data_generator import SignalDatasetGenerator, SimulationConfig


def train_epoch(
    model: nn.Module,
    epoch: int,
    train_loader: torch.utils.data.DataLoader,
    optimiser: torch.optim.Optimizer,
    loss_function: callable,
    device: torch.device,
) -> float:
    """Train model for one epoch"""
    # docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    non_blocking = device.type == "cuda"  # Asynchronous GPU transfers

    model.train()
    total_loss = 0.0

    progress_bar = tqdm(
        enumerate(train_loader),
        total=len(train_loader),
        desc=f"Train epoch {epoch}",
        mininterval=1.0,  # Update at most once per second
        leave=True,
        disable=not sys.stdout.isatty(),  # Disable tqdm for Slurm jobs
        postfix={"loss": "N/A"},
    )
    for batch_idx, (mixture, target) in progress_bar:
        mixture = mixture.to(device, non_blocking=non_blocking)
        target = target.to(device, non_blocking=non_blocking)

        # Forward pass
        output = model(mixture)
        loss: torch.Tensor = loss_function(output, target)
        if torch.isnan(loss).any():
            print("NaN detected in loss!")
            break
        # Backward pass
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()

        # Update epoch loss and log
        total_loss += loss.item()
        avg_loss = total_loss / (batch_idx + 1)
        progress_bar.postfix = f"loss={avg_loss:.4f}"

    return total_loss / len(train_loader)  # Average epoch loss


def validate_epoch(
    model: nn.Module,
    epoch: int,
    val_loader: torch.utils.data.DataLoader,
    loss_function: callable,
    device: torch.device,
) -> float:
    """Validate model with forward pass"""
    # docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    non_blocking = device.type == "cuda"  # Asynchronous GPU transfers

    model.eval()
    total_loss = 0.0

    progress_bar = tqdm(
        enumerate(val_loader),
        total=len(val_loader),
        desc=f"Validation epoch {epoch}",
        leave=True,
        disable=not sys.stdout.isatty(),  # Disable tqdm for Slurm jobs
        postfix={"loss": "N/A"},
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
            progress_bar.postfix = f"loss={avg_loss:.4f}"

    avg_epoch_loss = total_loss / len(val_loader)

    return avg_epoch_loss


def train_tdcr_net(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    epochs: int,
    val_split: float = 0.2,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    *,
    model_type: str = "complex",  # Use complex arithmetic (complex convs, norms, etc.) or real (treat real/imag as independent channels)
    dtype: torch.dtype = torch.complex64,
    resume: bool = False,
    pretrained_weights: str | None = None,
    num_workers: int = 0,  # Default DataLoader value
    model_param_M: int = 128,
    model_param_N: int = 32,
    model_param_U: int = 128,
    model_param_V: int = 8,
    checkpoints_dir: str = "./checkpoints",
) -> None:
    if resume and pretrained_weights is not None:
        raise ValueError("Cannot use both --resume and --pretrained_weights")

    seed = 0  # For reproducibility
    logger = TrainingLogger(checkpoints_dir, resume=resume)

    M, N, U, V = (
        model_param_M,
        model_param_N,
        model_param_U,
        model_param_V,
    )  # TDCR net parameters

    set_seed(seed)
    os.makedirs(checkpoints_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialise model, dataloaders, loss function, and optimiser
    if model_type == "complex":
        model = ComplexTDCRnet(M, N, U, V, dtype=dtype).to(device)
        print("Using complex model (complex arithmetic)")
    else:
        if dtype in (torch.complex32, torch.complex64, torch.complex128):
            raise ValueError(f"Real model cannot use complex dtype {dtype}")
        model = RealTDCRnet(M, N, U, V, dtype=dtype).to(device)
        print("Using real model (independent real/imag channels)")

    total_params = sum(p.numel() for p in model.parameters())
    total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
    print(f"Trainable parameters: {total_params:,}, dtype: {dtype}")
    print(f"Total Size: {total_memory:,} bytes")

    train_loader, val_loader = create_dataloaders(
        dataset,
        batch_size,
        num_workers,
        pin_memory=(device.type == "cuda"),
        val_split=val_split,
        split_seed=seed,
    )

    loss_function = si_snr_loss_complex

    optimiser = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", patience=3
    )

    start_epoch = 0
    best_val_loss = float("inf")

    if resume:
        checkpoint_path = os.path.join(checkpoints_dir, "last_checkpoint.pth")
        if os.path.isfile(checkpoint_path):
            print(f"Loading checkpoint: '{checkpoint_path}'")
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint["epoch"]
            best_val_loss = checkpoint["best_val_loss"]
            model.load_state_dict(checkpoint["model_state"])
            optimiser.load_state_dict(checkpoint["optimiser_state"])
            scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"Checkpoint loaded at epoch {checkpoint['epoch']}")

            # Verify logger is at the right epoch
            last_logged_epoch = logger.get_last_epoch()
            if last_logged_epoch != start_epoch - 1:
                raise RuntimeError(
                    f"Log shows last epoch as {last_logged_epoch}, "
                    f"but checkpoint is for epoch {start_epoch}. "
                    "Log and checkpoint are out of sync."
                )
        else:
            raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")

    if pretrained_weights:
        if os.path.isfile(pretrained_weights):
            print(f"Loading pretrained weights from: '{pretrained_weights}'")
            model.load_state_dict(torch.load(pretrained_weights))
            print("Pretrained weights loaded. Starting training from epoch 0")
        else:
            raise FileNotFoundError(
                f"No pretrained weights found at {pretrained_weights}"
            )

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
        )
        val_loss = validate_epoch(
            model=model,
            epoch=epoch,
            val_loader=val_loader,
            loss_function=loss_function,
            device=device,
        )
        scheduler.step(val_loss)  # Reduce learning rate on plateau

        # Save checkpoints
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(checkpoints_dir, "best_weights.pth"),
            )
            print(f"New best model saved with val loss: {best_val_loss:.6f}")

        # Logs and checkpoints
        logger.log_epoch(epoch, train_loss, val_loss, best_val_loss)
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
            f"train loss: {train_loss:.6f}, val loss: {val_loss:.6f}"
        )

    print("Training completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Time-Domain Dilated Convolutional Recurrent Network Training"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["complex", "real"],
        default="complex",
        help="Type of model: complex (complex arithmetic) or real (independent channels)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["complex64", "float32", "float16", "bfloat16"],
        default="complex64",
        help="Data type for model parameters and operations",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Input batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of epochs to train"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument(
        "--val_split", type=float, default=0.2, help="Validation split ratio"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        default=None,
        help="Path to pretrained weights file",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1,
        help="Use BLE (1) or IEEE 802.15.4 (2) as target",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to dataset file (.pt) to load",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="Number of subprocesses for data loading",
    )
    parser.add_argument(
        "--model_param_M",
        type=int,
        default=128,
        help="Middle channels in the complex encoder",
    )
    parser.add_argument(
        "--model_param_N",
        type=int,
        default=32,
        help="Out channels of encoder and input to LSTM = H",
    )
    parser.add_argument(
        "--model_param_U",
        type=int,
        default=128,
        help="Middle channels in complex dilated convolution",
    )
    parser.add_argument(
        "--model_param_V",
        type=int,
        default=8,
        help="Dilated convolutions on each side of the LSTM",
    )
    parser.add_argument(
        "--checkpoints_dir",
        type=str,
        default="./checkpoints",
        help="Directory to save checkpoints and logs",
    )
    parser.add_argument(
        "--runtime_gen",
        action="store_true",
        help="Generate data on the fly instead of loading from file",
    )
    args = parser.parse_args()

    # Loads all the dataset in RAM. Must change later (this is what DataLoaders are used for)
    dataset = LoadDataset(
        target_idx=args.target,
        runtime_generation=args.runtime_gen,
        generator_class=SignalDatasetGenerator(SimulationConfig()),
        dataset_path=args.dataset_path,
    )
    # dataset = DummyDataset(num_signals=10, signal_length=1024)

    dtype_map: dict = {
        "complex64": torch.complex64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    train_tdcr_net(
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_split=args.val_split,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        model_type=args.model_type,
        dtype=dtype,
        resume=args.resume,
        pretrained_weights=args.pretrained_weights,
        num_workers=args.num_workers,
        model_param_M=args.model_param_M,
        model_param_N=args.model_param_N,
        model_param_U=args.model_param_U,
        model_param_V=args.model_param_V,
        checkpoints_dir=args.checkpoints_dir,
    )
