import torch


def mse_loss_complex(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """Complex-valued MSE loss function"""
    if pred.shape != target.shape:
        raise RuntimeError(
            f"Dimention mismatch while computing MSE, {pred.shape} vs {target.shape}"
        )
    difference = pred - target
    mse = torch.mean(
        (difference * difference.conj()).real, dim=-1, keepdim=False
    )  # Shape (batch,)
    # return 10 * torch.log10(mse + eps)
    return mse


def si_snr_loss_complex(
    pred: torch.Tensor, target: torch.Tensor, zero_mean: bool = True, eps: float = 1e-8
) -> torch.Tensor:
    """Scale-invariant signal-to-noise ratio (SI-SNR) loss function"""
    # Reference: github.com/JusperLee/Conv-TasNet/blob/master/Conv_TasNet_Pytorch/SI_SNR.py
    if pred.shape != target.shape:
        raise RuntimeError(
            f"Dimention mismatch while computing SI-SNR, {pred.shape} vs {target.shape}"
        )
    if zero_mean:
        pred = pred - torch.mean(pred, dim=-1, keepdim=True)  # _s
        target = target - torch.mean(target, dim=-1, keepdim=True)  # s

    # Complex projection: s_target = (⟨_s, s⟩ / ||s||²) * s
    inner_prod = torch.sum(target * pred.conj(), dim=-1, keepdim=True)  # ⟨_s, s⟩
    s_norm = torch.sum(target * target.conj(), dim=-1, keepdim=True)  # ||s||²
    s_target = inner_prod / (s_norm + eps) * target  # Same shape as input (batch, time)

    e_noise = pred - s_target

    # Compute squared norms
    s_target_norm = (
        torch.sum(s_target * s_target.conj(), dim=-1, keepdim=False).real + eps
    )
    e_noise_norm = torch.sum(e_noise * e_noise.conj(), dim=-1, keepdim=False).real + eps

    return -10 * torch.log10(s_target_norm / e_noise_norm)  # Shape (batch,)
