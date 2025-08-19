import torch
import numpy as np


class DummyDataset(torch.utils.data.Dataset):
    """
    Dummy random dataset generator for forward pass verification
    """

    def __init__(self, num_signals: int, signal_length: int = 1024, seed: int = 0):
        self.num_signals = num_signals
        self.signal_length = signal_length
        self.seed = seed

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.data = self._generate_data()

    def __len__(self):
        return self.num_signals

    def __getitem__(self, idx: int):
        return self.data[idx]

    def _generate_data(self) -> list:
        data = [None] * self.num_signals
        for i in range(self.num_signals):
            mixture = torch.randn(self.signal_length, dtype=torch.complex64)
            target = torch.randn(self.signal_length, dtype=torch.complex64)
            data[i] = (mixture, target)
        return data


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
        shuffle=True,
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


# Later here:
# - Define a function/class that generates and stores actual dataset (actual BLE and IEEE 802.15.4 mixtures)
# - Define a torch.utils.data.Dataset class to load the stored dataset (pickle)
