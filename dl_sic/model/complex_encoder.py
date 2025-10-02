import torch
import torch.nn as nn
from .utils import ComplexLayerNorm, ComplexConv1d


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
            out_channels=mid_channels,
            kernel_size=kernel_size,
            padding="same",
            dtype=dtype,
        )

        self.layer_norm = ComplexLayerNorm(
            normalized_shape=mid_channels,
            complex_input_output=self.dtype_is_complex,
            dtype=dtype,
        )

        self.conv_out = ComplexConv1d(
            in_channels=mid_channels,
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

        # For complex dtype, input shape is (batch, in_channels, T)
        # For real dtype, input shape is (2, batch, in_channels, T)
        z = self.conv_in(x)  # ((2), batch, mid_channels, T)

        # Normalise only in features (mid_channels) and not in time
        # This is to be time-length agnostic.
        # Layer Norm learns the affine transformation parameters with normalised shape
        z_ln = z.transpose(-1, -2)  # ((2), batch, T, mid_channels)
        z_ln = self.layer_norm(z_ln)
        z_ln = z_ln.transpose(-1, -2)  # ((2), batch, mid_channels, T)

        y = self.conv_out(z_ln)  # ((2), batch, out_channels, T)

        return y, z


def test_model():
    in_channels = 1
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.complex64, torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = ComplexEncoder(in_channels, dtype=dtype)

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
        output, middle_layer = model(input)  # Forward pass

        print("Input shape:", input.shape)
        print("Middle shape:", middle_layer.shape)
        print("Output shape:", output.shape)
        print("")


if __name__ == "__main__":
    test_model()
