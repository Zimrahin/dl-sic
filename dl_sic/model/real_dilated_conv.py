import torch
import torch.nn as nn


class RealDilatedConv(nn.Module):
    """
    Real Dilated Convolution Module that treats real/imaginary as independent channels
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 128,  # U in paper
        *,
        kernel_size: int = 3,  # S in paper
        dilation: int = 1,  # 2**(V-1) for each added CDC
        negative_slope: float = 0.01,  # For PReLU (Leaky ReLU with learnt slope)
        number_dconvs: int = 2,  # Number of dilated conv layers
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

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

        return y + x  # Skip connection


def test_model():
    in_channels = 32
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = RealDilatedConv(in_channels, dtype=dtype)

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
