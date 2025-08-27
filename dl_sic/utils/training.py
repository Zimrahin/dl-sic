import random
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime


def set_seed(seed: int) -> None:
    """Set seed for reproducibility"""
    # docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TrainingLogger:
    def __init__(self, log_path: str, resume: bool = False, read_only: bool = False):
        self.log_path = Path(log_path) / "training_log.json"
        self.log_path.parent.mkdir(exist_ok=True)

        if read_only:
            if self.log_path.exists():
                self._load_log()
                return
            else:
                raise FileNotFoundError(f"No training log found at {self.log_path}.")

        if resume:
            if self.log_path.exists():
                print(f"Resuming training log: {self.log_path}")
                self._load_log()
                return
            else:
                raise FileNotFoundError(
                    f"Log file {self.log_path} does not exist. "
                    "Cannot resume training without log file."
                )

        # Create new log only if not resuming
        self.log_data = {
            "start_time": datetime.now().strftime("%Y%m%d_%H%M%S"),
            "epochs": [],
            "train_loss": [],
            "val_loss": [],
            "timestamps": [],
            "best_val_loss": float("inf"),
        }
        print(f"Created new training log: {self.log_path}")

    def _load_log(self):
        with open(self.log_path, "r") as f:
            self.log_data = json.load(f)

    def _save_log(self):
        with open(self.log_path, "w") as f:
            json.dump(self.log_data, f, indent=2)

    def log_epoch(
        self, epoch: int, train_loss: float, val_loss: float, best_val_loss: float
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_data["epochs"].append(epoch)
        self.log_data["train_loss"].append(float(train_loss))
        self.log_data["val_loss"].append(float(val_loss))
        self.log_data["timestamps"].append(timestamp)
        self.log_data["best_val_loss"] = float(best_val_loss)
        self._save_log()

    def plot_losses(self, save_path: str | None = None):
        if len(self.log_data["epochs"]) == 0:
            print("No data to plot")
            return

        plt.figure(figsize=(10, 6))
        plt.plot(
            self.log_data["epochs"],
            self.log_data["train_loss"],
            label="Training Loss",
            marker="o",
            markersize=3,
            alpha=0.8,
        )
        plt.plot(
            self.log_data["epochs"],
            self.log_data["val_loss"],
            label="Validation Loss",
            marker="s",
            markersize=3,
            alpha=0.8,
        )

        # Mark the best validation loss
        best_idx = self.log_data["val_loss"].index(min(self.log_data["val_loss"]))
        best_epoch = self.log_data["epochs"][best_idx]
        best_loss = self.log_data["val_loss"][best_idx]
        plt.axvline(
            x=best_epoch,
            color="red",
            linestyle="--",
            alpha=0.7,
            label=f"Best Epoch: {best_epoch}, Validation Loss: {best_loss:.4f}",
        )

        plt.title("Training and Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

        plt.show()

    def get_last_epoch(self):
        """Get the last logged epoch number"""
        if self.log_data["epochs"]:
            return self.log_data["epochs"][-1]
        return -1
