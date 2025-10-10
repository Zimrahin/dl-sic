import torch
from data.data_generator import SignalDatasetGenerator


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

    def __init__(
        self,
        target_idx: int,
        runtime_generation: bool = True,  # Generate data on the fly
        generator_class: SignalDatasetGenerator | None = None,
        dataset_path: str | None = None,
    ):
        self.target_idx = target_idx
        self.runtime_generation = runtime_generation
        self.generator_class = generator_class

        if runtime_generation:
            if generator_class is None:
                raise ValueError(
                    "For runtime generation, generator_class must be provided."
                )
            # Signals per epoch for runtime generation
            self.num_signals = generator_class.cfg.num_signals
            return

        # Load dataset otherwise
        if dataset_path is None:
            raise ValueError(
                "For loading pregenerated data, dataset_path must be provided."
            )
        dataset = torch.load(dataset_path, weights_only=False)
        all_tensors = dataset.tensors

        # Assume index 0 is the mixture, the rest are targets
        if target_idx < 1 or target_idx >= len(all_tensors):
            raise ValueError(f"Target index must be between 1 and {len(all_tensors)-1}")
        self.mixtures = all_tensors[0]
        self.target_tensor = all_tensors[target_idx]

        self.num_signals = len(self.mixtures)

        del all_tensors
        del dataset

    def __len__(self):
        return self.num_signals

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.runtime_generation:
            mixture = self.mixtures[idx].unsqueeze(0)
            target = self.target_tensor[idx].unsqueeze(0)
        else:
            mixture, *target = self.generator_class.generate_mixture()
            target = target[self.target_idx - 1]

            mixture = torch.from_numpy(mixture).unsqueeze(0)
            target = torch.from_numpy(target).unsqueeze(0)

        return mixture, target
