import torch
import torch.nn as nn
import complextorch


class ComplexEncoder(nn.Module):
    """
    Complex Encoder from Guo et al., 2024.
    """

    def __init__(
        self,
        in_channels: int = 1,
        mid_channels: int = 128,  # M in paper
        out_channels: int = 32,  # N in paper
        *,
        kernel_size: int = 3,  # J in paper, changed from 2 to 3 (even -> odd)
    ) -> None:
        super().__init__()

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding="same",
            dtype=torch.complex64,
        )

        self.layer_norm = complextorch.nn.LayerNorm(normalized_shape=mid_channels)

        self.conv_out = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding="same",
            dtype=torch.complex64,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, in_channels, T) complex tensor
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexEncoder expects a complex tensor")

        z: torch.Tensor = self.conv_in(x)  # (batch, mid_channels, T)

        # Prepare for LayerNorm: permute to (batch, T, mid_channels)
        z = z.permute(0, 2, 1)  # (batch, T, mid_channels)
        z = self.layer_norm(z)
        z = z.permute(0, 2, 1)  # (batch, mid_channels, T)

        y = self.conv_out(z)  # (batch, out_channels, T)
        return y, z


def test_model():
    in_channels = 1
    batch_size = 4
    signal_length = 2048  # T
    model = ComplexEncoder(in_channels)

    print(f"\nTotal Parameters: {sum(p.numel() for p in model.parameters()):,}")

    input = torch.rand((batch_size, in_channels, signal_length), dtype=torch.complex64)
    output, middle_layer = model(input)  # Forward pass

    print("Input shape:", input.shape)
    print("Middle shape:", middle_layer.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
