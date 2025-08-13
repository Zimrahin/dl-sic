import torch
import torch.nn as nn
import complextorch


class ComplexDecoder(nn.Module):
    """
    Complex Decoder from Guo et al., 2024.
    Explained better at Luo et al., 2019, Conv-TasNet (Fig. 1.B)
    """

    def __init__(
        self,
        in_channels: int = 128,  # M in paper
        out_channels: int = 1,
        *,
        kernel_size: int = 2,  # J in paper
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
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
            raise TypeError("ComplexDecoder expects a complex tensor")

        return self.conv_in(x)  # (B, 1, T)
