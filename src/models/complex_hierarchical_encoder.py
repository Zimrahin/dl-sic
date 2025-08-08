import torch
import torch.nn as nn
import complextorch


class ComplexHierarchicalEncoder(nn.Module):
    """
    Complex Hierarchical Encoder (CHE) as described in Guo et al., 2024.
    """

    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 128,  # M in the paper
        out_channels: int = 32,  # N in the paper
        *,
        kernel_size: int = 2,  # J in the paper
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=torch.complex64,
        )

        self.layer_norm = complextorch.nn.LayerNorm(normalized_shape=mid_channels)

        self.conv2 = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=torch.complex64,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        - x: **complex** tensor with shape (B, 1, T)
        - returns: **complex** tensor y with shape (B, out_channels, T2)
        """
        if not torch.is_complex(x):
            raise TypeError("CHE expects a complex tensor")

        z = self.conv1(x)  # (B, mid_channels, T1)
        z = self.layer_norm(z)
        y = self.conv2(z)  # (B, out_channels, T2)
        return y
