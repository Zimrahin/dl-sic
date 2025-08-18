import torch


def mse_loss_complex(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Complex-valued MSE loss function"""
    return torch.mean(torch.abs(pred - target) ** 2)


# Later define SI-SNR loss (Guo et al., 2024.)
