import torch
import torch.nn as nn
from .utils import ComplexConv1d


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
        kernel_size: int = 3,  # J in paper, changed from 2 to 3 (even -> odd)
        dtype: type = torch.complex64,
    ) -> None:
        super().__init__()

        self.dtype_is_complex = dtype in (
            torch.complex32,
            torch.complex64,
            torch.complex128,
        )

        self.conv_in = ComplexConv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:
        - For complex: (batch, in_channels, T) complex tensor
        - For real: (2, batch, in_channels, T) real tensor (first dim: 0=real, 1=imag)
        """

        return self.conv_in(x)  # ((2), batch, 1, T)


def test_model():
    in_channels = 128
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.complex64, torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = ComplexDecoder(in_channels, dtype=dtype)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        if dtype in (torch.complex32, torch.complex64, torch.complex128):
            # Complex input: (batch, in_channels, T)
            input = torch.rand((batch_size, in_channels, signal_length), dtype=dtype)
        else:
            # Real input: (2, batch, in_channels, T)
            input = torch.rand((2, batch_size, in_channels, signal_length), dtype=dtype)
        output = model(input)  # Forward pass

        print("Input shape:", input.shape)
        print("Output shape:", output.shape)
        print("")


if __name__ == "__main__":
    test_model()
