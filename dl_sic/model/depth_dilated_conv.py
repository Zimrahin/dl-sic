import torch
import torch.nn as nn


class DepthDilatedConv(nn.Module):
    """
    Depth-Dilated-Convolution Module from Conv-TasNet (Luo et al., 2019).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int,
        *,
        kernel_size: int = 3,
        dilation: int = 1,
        negative_slope: float = 0.25,  # For PReLU
        number_dconvs: int = 1,  # Hidden depth-dilated conv layers
        skip_connection: bool = True,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.skip = skip_connection

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,  # Pointwise 1x1-conv
            padding="same",
            dtype=dtype,
        )

        self.prelu_in = nn.PReLU(init=negative_slope, dtype=dtype)
        self.layer_norm_in = nn.GroupNorm(
            num_groups=1,
            num_channels=mid_channels,
            dtype=dtype,
            eps=1e-08,
        )

        self.dconvs = nn.ModuleList(
            nn.Conv1d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
                groups=mid_channels,  # Depthwise convolution
                dtype=dtype,
            )
            for _ in range(number_dconvs)
        )

        self.prelu_out = nn.PReLU(init=negative_slope, dtype=dtype)
        self.layer_norm_out = nn.GroupNorm(
            num_groups=1,
            num_channels=mid_channels,
            dtype=dtype,
            eps=1e-08,
        )

        self.conv_out = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=in_channels,  # Back to input size
            kernel_size=1,  # Pointwise 1x1-conv
            padding="same",
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, in_channels, T)
        Output shape: (batch, in_channels, T)
        """
        # Bottleneck, from in_channels to mid_channels
        y = self.conv_in(x)  # (batch, mid_channels, T)
        y = self.prelu_in(y)  # (batch, mid_channels, T)
        y = self.layer_norm_in(y)

        for dconv in self.dconvs:
            y = dconv(y)

        # Bottleneck, from mid_channels to in_channels
        y = self.prelu_out(y)  # (batch, mid_channels, T)
        y = self.layer_norm_out(y)
        y = self.conv_out(y)  # (batch, in_channels, T)

        if not self.skip:
            return y

        return y + x  # Skip connection


def test_model():
    in_channels = 32
    mid_channels = 128
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = DepthDilatedConv(
            in_channels,
            mid_channels,
            number_dconvs=2,
            dtype=dtype,
        )

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
