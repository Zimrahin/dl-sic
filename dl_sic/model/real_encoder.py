import torch
import torch.nn as nn


class RealEncoder(nn.Module):
    """
    Real Encoder that treats real/imaginary as independent channels
    """

    def __init__(
        self,
        in_channels: int = 2,
        mid_channels: int = 128,  # M in paper
        out_channels: int = 32,  # N in paper
        *,
        kernel_size: int = 3,  # J in paper
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding="same",
            dtype=dtype,
        )

        self.layer_norm = nn.LayerNorm(
            normalized_shape=mid_channels,
            dtype=dtype,
        )

        self.conv_out = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, in_channels, T)
        """
        z = self.conv_in(x)  # (batch, mid_channels, T)

        z_ln = z.transpose(1, 2)  # (batch, T, mid_channels)
        z_ln = self.layer_norm(z_ln)
        z_ln = z_ln.transpose(1, 2)  # (batch, mid_channels, T)

        y = self.conv_out(z_ln)  # (batch, out_channels, T)

        return y, z


def test_model():
    in_channels = 2
    batch_size = 4
    signal_length = 2048  # T

    dtypes = [torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = RealEncoder(in_channels, dtype=dtype)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        # Real input: (batch, 2, T)
        input = torch.rand((batch_size, in_channels, signal_length), dtype=dtype)
        output, middle_layer = model(input)  # Forward pass

        print("Input shape:", input.shape)
        print("Middle shape:", middle_layer.shape)
        print("Output shape:", output.shape)
        print("")


if __name__ == "__main__":
    test_model()
