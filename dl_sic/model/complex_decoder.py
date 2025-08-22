import torch
import torch.nn as nn


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
        padding = (kernel_size - 1) // 2

        self.conv_in = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, in_channels, T) complex tensor
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexDecoder expects a complex tensor")

        return self.conv_in(x)  # (batch, 1, T)


def test_model():
    in_channels = 128
    batch_size = 4
    signal_length = 2048  # T
    dtype = torch.complex64
    model = ComplexDecoder(in_channels, dtype=dtype)

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    input = torch.rand((batch_size, in_channels, signal_length), dtype=dtype)
    output = model(input)  # Forward pass

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
