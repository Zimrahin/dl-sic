import torch
import torch.nn as nn

from .complex_operations import ComplexLayerNorm, ComplexConv1d
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
        dtype: torch.dtype = torch.complex64,
    ) -> None:
        super().__init__()

        self.dtype_is_complex = dtype in (
            torch.complex32,
            torch.complex64,
            torch.complex128,
        )

        self.conv_in = ComplexConv1d(
            in_channels=in_channels,
            out_channels=mid_channels,  # Expand channels to mid_channels to match dilated conv size
            kernel_size=1,  # Pointwise 1x1-conv (based on Conv-TasNet, Luo et al., 2019)
            padding="same",
            dtype=dtype,
        )

        self.prelu_in = ComplexPReLU(init=negative_slope, dtype=dtype)
        self.layer_norm_in = ComplexLayerNorm(
            num_channels=mid_channels,
            complex_input_output=self.dtype_is_complex,
            dtype=dtype,
        )

        self.dconvs = nn.ModuleList(
            ComplexConv1d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
                groups=mid_channels,  # Depthwise convolution (Conv-TasNet, Luo et al., 2019))
                dtype=dtype,
            )
            for _ in range(number_dconvs)
        )

        self.prelu_out = ComplexPReLU(init=negative_slope, dtype=dtype)
        self.layer_norm_out = ComplexLayerNorm(
            num_channels=mid_channels,
            complex_input_output=self.dtype_is_complex,
            dtype=dtype,
        )

        self.conv_out = ComplexConv1d(
            in_channels=mid_channels,
            out_channels=in_channels,  # Back to input size
            kernel_size=1,  # Pointwise 1x1-conv (based on Conv-TasNet, Luo et al., 2019)
            padding="same",
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:
        - For complex: (batch, in_channels, T) complex tensor
        - For real: (2, batch, in_channels, T) real tensor (first dim: 0=real, 1=imag)
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
    dtypes = [torch.complex64, torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = ComplexDilatedConv(in_channels, dtype=dtype)

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
