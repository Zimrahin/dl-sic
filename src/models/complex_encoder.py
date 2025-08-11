import torch
import torch.nn as nn
import complextorch


class ComplexEncoder(nn.Module):
    """
    Complex Encoder (CHE) from Guo et al., 2024.
    """

    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 128,  # M in paper
        out_channels: int = 32,  # N in paper
        *,
        kernel_size: int = 2,  # J in paper
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=torch.complex64,
        )

        self.layer_norm = complextorch.nn.LayerNorm(normalized_shape=mid_channels)

        self.conv_out = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=torch.complex64,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (B, in_channels, T) complex tensor
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexEncoder expects a complex tensor")

        z = self.conv_in(x)  # (B, mid_channels, T)
        z = self.layer_norm(z)
        y = self.conv_out(z)  # (B, out_channels, T)
        return y
