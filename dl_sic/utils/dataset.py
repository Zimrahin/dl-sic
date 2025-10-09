import torch
import numpy as np


class DummyDataset(torch.utils.data.Dataset):
    def __init__(self, num_signals: int, signal_length: int = 1024, seed: int = 0):
        torch.manual_seed(seed)
        self.mixture = torch.randn(num_signals, 1, signal_length, dtype=torch.complex64)
        self.target = torch.randn(num_signals, 1, signal_length, dtype=torch.complex64)

    def __len__(self):
        return self.mixture.shape[0]

    def __getitem__(self, idx: int):
        return self.mixture[idx], self.target[idx]


def create_dataloaders(
    dataset: torch.utils.data.Dataset,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,  # docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html
    val_split: float,
    *,
    split_seed: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation dataloaders"""
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(split_seed),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


class LoadDataset(torch.utils.data.Dataset):
    """
    Dataset loader. Uses BLE (1) or IEEE 802.15.4 (2) as target
    """

    def __init__(self, file_path: str, target_idx: int):
        dataset: torch.utils.data.TensorDataset = torch.load(
            file_path, weights_only=False
        )
        all_tensors = dataset.tensors
        if target_idx < 1 or target_idx >= len(all_tensors):
            raise ValueError(f"Target index must be between 1 and {len(all_tensors)-1}")

        # The first tensor is the input (mixture), the rest are targets
        self.mixtures = all_tensors[0]
        # Store only the selected target tensor
        self.target_tensor = all_tensors[target_idx]

        if self.mixtures.dim() == 2:  # (num_signals, signal_length)
            self.mixtures = self.mixtures.unsqueeze(
                1
            )  # (num_signals, 1, signal_length)
        if self.target_tensor.dim() == 2:  # (num_signals, signal_length)
            self.target_tensor = self.target_tensor.unsqueeze(
                1
            )  # (num_signals, 1, signal_length)

        del all_tensors
        del dataset

    def __len__(self):
        return len(self.mixtures)

    def __getitem__(self, idx):
        return self.mixtures[idx], self.target_tensor[idx]
