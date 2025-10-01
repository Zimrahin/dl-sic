import torch
import torch.nn as nn

# import complextorch
from .complex_operations import ComplexLayerNorm
from .activation_functions import ComplexPReLU


class ComplexDilatedConv(nn.Module):
    """
    Complex Dilated Convolution Module (CDC) from Guo et al., 2024,
    originally based on Conv-TasNet, Luo et al., 2019 (Fig. 1.C).
    """

    def __init__(
        self,
        in_channels: int,
        mid_channels: int = 128,  # U in paper (channels in dilated conv)
        *,
        kernel_size: int = 3,  # S in paper
        dilation: int = 1,  # 2**(V-1) for each added CDC
        negative_slope: float = 0.01,  # For PReLU (Leaky ReLU with learnt slope)
        number_dconvs: int = 2,  # Number of dilated conv layers
    ) -> None:
        super().__init__()

        self.conv_in = nn.Conv1d(
            in_channels=in_channels,
            out_channels=mid_channels,  # Expand channels to mid_channels to match dilated conv size
            kernel_size=1,  # Assume pointwise 1x1-conv (based on Conv-TasNet, Luo et al., 2019)
            padding="same",
            dtype=torch.complex64,
        )

        self.prelu_in = ComplexPReLU(init=negative_slope)
        self.layer_norm_in = ComplexLayerNorm(normalized_shape=mid_channels)

        self.dconvs = nn.ModuleList(
            nn.Conv1d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
                groups=mid_channels,  # Depthwise convolution (Conv-TasNet, Luo et al., 2019))
                dtype=torch.complex64,
            )
            for _ in range(number_dconvs)
        )

        self.prelu_out = ComplexPReLU(init=negative_slope)
        self.layer_norm_out = ComplexLayerNorm(normalized_shape=mid_channels)

        self.conv_out = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=in_channels,  # Back to input size
            kernel_size=1,  # Assume pointwise 1x1-conv (based on Conv-TasNet, Luo et al., 2019)
            padding="same",
            dtype=torch.complex64,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, in_channels, T) complex tensor
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexDilatedConv expects a complex tensor.")

        # Bottleneck, from in_channels to mid_channels
        y = self.conv_in(x)  # (batch, mid_channels, T)
        y = self.prelu_in(y)  # (batch, mid_channels, T)
        y = y.permute(0, 2, 1)  # (batch, T, mid_channels)
        y = self.layer_norm_in(y)
        y = y.permute(0, 2, 1)  # (batch, mid_channels, T)

        for dconv in self.dconvs:
            y = dconv(y)

        # Bottleneck, from mid_channels to in_channels
        y = self.prelu_out(y)  # (batch, mid_channels, T)
        y = y.permute(0, 2, 1)  # (batch, T, mid_channels)
        y = self.layer_norm_out(y)
        y = y.permute(0, 2, 1)  # (batch, mid_channels, T)
        y = self.conv_out(y)  # (batch, in_channels, T)

        return y + x  # Skip connection


def test_model():
    in_channels = 32
    batch_size = 4
    signal_length = 2048  # T
    model = ComplexDilatedConv(in_channels)

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    input = torch.rand((batch_size, in_channels, signal_length), dtype=torch.complex64)
    output: torch.Tensor = model(input)  # Forward pass

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
