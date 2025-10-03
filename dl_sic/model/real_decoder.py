import torch
import torch.nn as nn


class RealDecoder(nn.Module):
    """
    Real Decoder that treats real/imaginary as independent channels
    """

    def __init__(
        self,
        in_channels: int = 128,  # M in paper
        out_channels: int = 2,
        *,
        kernel_size: int = 3,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, in_channels, T)
        Output shape: (batch, 2, T) - channel 0: real, channel 1: imaginary
        """
        return self.conv_in(x)


def test_model():
    in_channels = 128
    batch_size = 4
    signal_length = 2048  # T

    dtypes = [torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = RealDecoder(in_channels, dtype=dtype)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        # Real input: (batch, in_channels, T)
        input = torch.rand((batch_size, in_channels, signal_length), dtype=dtype)
        output = model(input)  # Forward pass

        print("Input shape:", input.shape)
        print("Output shape:", output.shape)
        print("")


if __name__ == "__main__":
    test_model()
