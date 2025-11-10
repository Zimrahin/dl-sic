import torch
import torch.nn as nn

from .depth_dilated_conv import DepthDilatedConv
from .real_lstm import RealLSTM


class RealTDCRNet(nn.Module):
    """
    Real Time-Domain Dilated Convolutional Recurrent Network
    Treats real/imaginary as independent channels
    """

    def __init__(
        self,
        M: int = 128,  # Middle channels in the real encoder
        N: int = 32,  # Out channels of encoder and input to LSTM = H
        U: int = 128,  # Middle channels in real dilated convolution
        V: int = 8,  # Dilated convolutions on each side of the LSTM
        *,
        encoder_kernel_size: int = 3,
        dtype=torch.float32,
    ) -> None:
        super().__init__()

        self.dtype = dtype

        self.encoder = nn.Conv1d(
            in_channels=2,
            out_channels=M,
            kernel_size=encoder_kernel_size,
            padding="same",
            dtype=dtype,
        )
        self.layer_norm_in = nn.GroupNorm(
            num_groups=1,
            num_channels=M,
            dtype=dtype,
        )
        self.decoder = nn.Conv1d(
            in_channels=M,
            out_channels=2,
            kernel_size=encoder_kernel_size,
            padding="same",
            dtype=dtype,
        )

        self.conv_in = nn.Conv1d(
            in_channels=M,
            out_channels=N,
            kernel_size=encoder_kernel_size,  # Same as encoder in Guo et al., 2024, but pointwise in Conv-TasNet, Luo et al., 2019
            padding="same",
            dtype=dtype,
        )
        self.conv_out = nn.Conv1d(
            in_channels=N,
            out_channels=M,
            kernel_size=1,  # Pointwise 1x1-conv
            padding="same",
            dtype=dtype,
        )

        self.cdc_left = nn.ModuleList(
            [
                DepthDilatedConv(
                    in_channels=N,
                    mid_channels=U,
                    dilation=2**v,
                    number_dconvs=2,
                    dtype=dtype,
                )
                for v in range(V)
            ]
        )

        self.lstm = RealLSTM(
            input_size=N,
            hidden_size=N,
            dtype=dtype,
        )

        self.cdc_right = nn.ModuleList(
            [
                DepthDilatedConv(
                    in_channels=N,
                    mid_channels=U,
                    dilation=2**v,
                    number_dconvs=2,
                    dtype=dtype,
                )
                for v in range(V)
            ]
        )

        # Real-valued operations
        self.prelu_out = nn.PReLU(dtype=dtype)
        self.sigmoid_out = nn.Sigmoid()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Input shape:
        - Complex: (batch, 1, T)
        - Real: (batch, 2, T)
        Output: Same shape and type as input
        """
        assert (
            input.dim() == 3
        ), f"Expected 3D input (batch, channels, T), got {tuple(input.shape)}"

        input_is_complex = torch.is_complex(input)

        if input_is_complex:
            assert (
                input.size(1) == 1
            ), f"Expected 1 complex channel, got {input.size(1)} channels"
            # Complex 1D -> Real 2D: (batch, 1, T) -> (batch, 2, T)
            input = torch.stack([input.real, input.imag], dim=1)  # (batch, 2, 1, T)
            input = input.squeeze(2)  # (batch, 2, T)
        else:
            # Real input: (batch, 2, T)
            assert (
                input.size(1) == 2
            ), f"Expected 2 real channels, got {input.size(1)} channels"

        # Forward pass through real network
        z = self.encoder(input)  # (batch, M, T)
        y = self.conv_in(self.layer_norm_in(z))  # (batch, N, T)

        for cdc in self.cdc_left:
            y = cdc(y) + y  # (batch, N, T), residual connection

        y = self.lstm(y)  # (batch, N, T)

        for cdc in self.cdc_right:
            y = cdc(y) + y  # (batch, N, T), residual connection

        y = self.prelu_out(y)
        y = self.conv_out(y)  # (batch, M, T)
        y = self.sigmoid_out(y)  # (batch, M, T)

        s = y * z  # Elementwise product
        s = self.decoder(s)  # (batch, 2, T)

        if input_is_complex:
            # Convert back to complex: (batch, 2, T) -> (batch, 1, T)
            s = torch.complex(s[:, 0, :], s[:, 1, :])  # (batch, T)
            s = s.unsqueeze(1)  # (batch, 1, T)

        return s


def test_model():
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = RealTDCRNet(dtype=dtype, N=64)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        complex_dtype_map = {
            torch.float32: torch.complex64,
            torch.float16: torch.complex32,
        }
        complex_dtype = complex_dtype_map.get(dtype, torch.complex64)

        # Test with complex inputs (will be converted internally)
        print("Testing 3D complex input")
        input = torch.rand((batch_size, 1, signal_length), dtype=complex_dtype)
        output = model(input)
        print(f"Input shape: {input.shape}, dtype: {input.dtype}")
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")

        # Test with native real channel format
        print("Testing 3D real channel input (native format)")
        input = torch.rand((batch_size, 2, signal_length), dtype=dtype)
        output = model(input)
        print(f"Input shape: {input.shape}, dtype: {input.dtype}")
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")

        print("")


if __name__ == "__main__":
    test_model()
