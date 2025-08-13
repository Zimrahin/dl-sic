import torch
import torch.nn as nn
import complextorch
from activation_functions import ComplexPReLU


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
        self.layer_norm_in = complextorch.nn.LayerNorm(normalized_shape=mid_channels)

        self.dconvs = nn.ModuleList(
            nn.Conv1d(
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=kernel_size,
                padding="same",
                dilation=dilation,
                dtype=torch.complex64,
            )
            for _ in range(number_dconvs)
        )

        self.prelu_out = ComplexPReLU(init=negative_slope)
        self.layer_norm_out = complextorch.nn.LayerNorm(normalized_shape=mid_channels)

        self.conv_out = nn.Conv1d(
            in_channels=mid_channels,
            out_channels=in_channels,  # Back to input size
            kernel_size=1,  # Assume pointwise 1x1-conv (based on Conv-TasNet, Luo et al., 2019)
            padding="same",
            dtype=torch.complex64,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (B, in_channels, T) complex tensor
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexDilatedConv expects a complex tensor.")

        # Bottleneck, from in_channels to mid_channels
        y = self.conv_in(x)  # (B, mid_channels, T)
        y = self.prelu_in(y)  # (B, mid_channels, T)
        y = self.layer_norm_in(y)  # (B, mid_channels, T)

        for dconv in self.dconvs:
            y = dconv(y)

        # Bottleneck, from mid_channels to in_channels
        y = self.prelu_out(y)  # (B, mid_channels, T)
        y = self.layer_norm_out(y)  # (B, mid_channels, T)
        y = self.conv_out(x)  # (B, in_channels, T)

        return y + x  # Skip connection
