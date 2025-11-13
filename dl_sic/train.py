import os
import sys
import argparse
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from utils.training import set_seed, TrainingLogger, create_checkpoint_dir
from utils.dataset import DummyDataset, LoadDataset, create_dataloaders
from utils.loss_functions import si_snr_loss_complex

from data.generator import SignalDatasetGenerator, SimulationConfig

from model_factory import ModelFactory


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        dataset: torch.utils.data.Dataset,
        batch_size: int,
        epochs: int,
        learning_rate: float = 1e-3,
        weight_decay: float = 0.0,
        *,
        dtype: torch.dtype = torch.complex64,
        resume: bool = False,
        num_workers: int = 0,  # Default DataLoader value
        checkpoints_dir: str = "./checkpoints",
    ) -> None:

        self.model = model
        self.device = device

        self.dataset = dataset
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.dtype = dtype
        self.num_workers = num_workers
        self.checkpoints_dir = checkpoints_dir

        self.logger = TrainingLogger(checkpoints_dir, resume=resume)

        self.loss_function = si_snr_loss_complex

        self.optimiser = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimiser, mode="min", patience=3
        )

        self.start_epoch = 0
        self.best_val_loss = float("inf")

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load training state from checkpoint"""
        print(f"Loading checkpoint: '{checkpoint_path}'")
        checkpoint = torch.load(checkpoint_path)
        self.start_epoch = checkpoint["epoch"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.model.load_state_dict(checkpoint["model_state"])
        self.optimiser.load_state_dict(checkpoint["optimizer_state"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state"])

        # Verify logger consistency
        last_logged_epoch = self.logger.get_last_epoch()
        if last_logged_epoch != self.start_epoch - 1:
            raise RuntimeError(
                f"Log/checkpoint mismatch: log epoch {last_logged_epoch}, "
                f"checkpoint epoch {self.start_epoch}"
            )
        print(f"Checkpoint loaded at epoch {checkpoint['epoch']}")

    def load_trained_weights(self, weights_path: str) -> None:
        """Load only model weights from a file"""
        if os.path.isfile(weights_path):
            print(f"Loading trained weights from: '{weights_path}'")
            self.model.load_state_dict(torch.load(weights_path))
            print("Trained weights loaded successfully.")
        else:
            raise FileNotFoundError(f"No trained weights found at {weights_path}")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save training state"""
        torch.save(
            {
                "epoch": epoch + 1,
                "best_val_loss": self.best_val_loss,
                "model_state": self.model.state_dict(),
                "optimiser_state": self.optimiser.state_dict(),
                "scheduler_state": self.scheduler.state_dict(),
            },
            os.path.join(self.checkpoints_dir, "last_checkpoint.pth"),
        )
        if is_best:
            torch.save(
                self.model.state_dict(),
                os.path.join(self.checkpoints_dir, "best_weights.pth"),
            )

    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ):
        """Main training loop"""
        for epoch in range(self.start_epoch, self.epochs):
            epoch_start = time.time()

            train_loss = self.train_epoch(epoch, train_loader)
            val_loss = self.validate_epoch(epoch, val_loader)
            self.scheduler.step(val_loss)  # Reduce learning rate on plateau

            # Update best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Logs and checkpoints
            self.save_checkpoint(epoch, is_best)
            self.logger.log_epoch(epoch, train_loss, val_loss, self.best_val_loss)

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {epoch} - {epoch_time:.2f}s | "
                f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                f"Best: {self.best_val_loss:.6f}"
            )

    def train_epoch(self, epoch: int, train_loader) -> float:
        self.model.train()
        return self._run_epoch(epoch, train_loader, self.optimiser, training=True)

    def validate_epoch(self, epoch: int, val_loader) -> float:
        self.model.eval()
        return self._run_epoch(epoch, val_loader, None, training=False)

    def _run_epoch(
        self,
        epoch: int,
        data_loader: torch.utils.data.DataLoader,
        optimiser: torch.optim.Optimizer,
        training: bool,
    ):
        # docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
        non_blocking = self.device.type == "cuda"  # Asynchronous GPU transfers
        total_loss = 0.0

        progress_bar = tqdm(
            enumerate(data_loader),
            total=len(data_loader),
            desc=f"{'Train' if training else 'Validation'} epoch {epoch}",
            mininterval=1.0,  # Update at most once per second
            leave=True,
            disable=not sys.stdout.isatty(),  # Disable tqdm for Slurm jobs
            postfix={"loss": "N/A"},
        )
        with torch.set_grad_enabled(
            training
        ):  # False for faster inference (validation)
            for batch_idx, (mixture, target) in progress_bar:
                mixture = mixture.to(self.device, non_blocking=non_blocking)
                target = target.to(self.device, non_blocking=non_blocking)

                # Forward pass
                output = self.model(mixture)
                loss: torch.Tensor = self.loss_function(output, target)
                if torch.isnan(loss).any():
                    print("NaN detected in loss!")
                    break
                if training:
                    # Backward pass
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    optimiser.step()

                total_loss += loss.item()
                avg_loss = total_loss / (batch_idx + 1)
                progress_bar.postfix = f"loss={avg_loss:.4f}"

        return total_loss / len(data_loader)  # Average epoch loss


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

    # Arguments
    if args.resume and args.pretrained_weights is not None:
        raise ValueError("Cannot use both --resume and --pretrained_weights")
    dtype_map: dict = {
        "complex64": torch.complex64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    # Setup
    seed = 0  # For reproducibility
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    checkpoints_dir = create_checkpoint_dir(args.checkpoints_dir, args)

    # Data
    dataset = LoadDataset(
        target_idx=args.target,
        runtime_generation=args.runtime_gen,
        generator_class=SignalDatasetGenerator(SimulationConfig()),
        dataset_path=args.dataset_path,
        return_real=(args.dtype not in ("complex32", "complex64", "complex128")),
    )
    # dataset = DummyDataset(
    #     num_signals=10,
    #     signal_length=1024,
    #     return_real=(args.dtype not in ("complex32", "complex64", "complex128")),
    # )
    train_loader, val_loader = create_dataloaders(
        dataset,
        args.batch_size,
        args.num_workers,
        pin_memory=(device.type == "cuda"),
        val_split=args.val_split,
        split_seed=seed,
    )

    # Model
    model_params = {
        "M": args.model_param_M,
        "N": args.model_param_N,
        "U": args.model_param_U,
        "V": args.model_param_V,
    }
    model = ModelFactory.create_model(args.model_type, model_params, dtype, device)
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
    print(f"Trainable parameters: {total_params:,}, dtype: {dtype}")
    print(f"Total Size: {total_memory:,} bytes")

    trainer = Trainer(
        model=model,
        device=device,
        dataset=dataset,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dtype=dtype,
        resume=args.resume,
        num_workers=args.num_workers,
        checkpoints_dir=checkpoints_dir,
    )

    if args.resume:
        trainer.load_checkpoint(os.path.join(checkpoints_dir, "last_checkpoint.pth"))
    if args.pretrained_weights:
        trainer.load_trained_weights(args.pretrained_weights)

    # Start training
    trainer.train(train_loader, val_loader)
    print("Training completed!")
