import torch


def mse_loss_complex(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complex-valued MSE loss function"""
    difference = pred - target
    return torch.mean((difference * difference.conj()).real)


# Later define SI-SNR loss (Guo et al., 2024.)
