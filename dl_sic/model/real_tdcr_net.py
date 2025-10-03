import torch
import torch.nn as nn

from .real_encoder import RealEncoder
from .real_dilated_conv import RealDilatedConv
from .real_lstm import RealLSTM
from .real_decoder import RealDecoder


class RealTDCRnet(nn.Module):
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
        decoder_kernel_size: int = 3,
        dtype=torch.float32,
    ) -> None:
        super().__init__()

        self.dtype = dtype

        self.encoder = RealEncoder(
            in_channels=2,  # Real+imaginary as separate channels
            mid_channels=M,
            out_channels=N,
            kernel_size=encoder_kernel_size,
            dtype=dtype,
        )

        self.cdc_left = nn.ModuleList(
            [
                RealDilatedConv(
                    in_channels=N,
                    mid_channels=U,
                    dilation=2**v,
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
                RealDilatedConv(
                    in_channels=N,
                    mid_channels=U,
                    dilation=2**v,
                    dtype=dtype,
                )
                for v in range(V)
            ]
        )

        # Real-valued operations
        self.prelu_out = nn.PReLU(dtype=dtype)
        self.conv_out = nn.Conv1d(
            in_channels=N,
            out_channels=M,
            kernel_size=1,
            padding="same",
            dtype=dtype,
        )
        self.sigmoid_out = nn.Sigmoid()

        self.decoder = RealDecoder(
            in_channels=M,
            out_channels=2,  # Output real+imaginary channels
            kernel_size=decoder_kernel_size,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, 2, T) where:
        - channel 0: real part
        - channel 1: imaginary part

        Output shape: (batch, 2, T)
        """

        original_dim = x.dim()
        # Convert complex input to real channel format
        if torch.is_complex(x):

            # Input shape handling
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, 1, T)
            if x.size(1) != 1:
                raise ValueError(f"Expected 1 input channel, got {x.size(1)} channels")

            # Complex input: (batch, 1, T) -> (batch, 2, T)
            x_channels = torch.stack([x.real, x.imag], dim=1)  # (batch, 2, T)
        else:
            # Already in real format, assume (batch, 2, T)
            if x.dim() == 3 and x.size(1) == 2:
                x_channels = x
            else:
                raise ValueError(f"Unexpected input format: {x.shape}")

        # Forward pass through real network
        x_channels = x_channels.to(self.dtype)
        y, z = self.encoder(x)  # (batch, N, T), (batch, M, T)

        for cdc in self.cdc_left:
            y = cdc(y)  # (batch, N, T)

        y = self.lstm(y)  # (batch, N, T)

        for cdc in self.cdc_right:
            y = cdc(y)  # (batch, N, T)

        y = self.prelu_out(y)
        y = self.conv_out(y)  # (batch, M, T)
        y = self.sigmoid_out(y)  # (batch, M, T)

        s = y * z  # Elementwise product
        s = self.decoder(s)  # (batch, 2, T)

        # Convert back to complex
        s_real = s[:, 0, :]  # (batch, T)
        s_imag = s[:, 1, :]  # (batch, T)
        s_complex = torch.complex(s_real, s_imag)  # (batch, T)
        s_complex = s_complex.unsqueeze(1)  # (batch, 1, T)

        # Restore original dimensions
        if original_dim == 2:
            s_complex = s_complex.squeeze(1)  # (batch, T)

        return s_complex


def test_model():
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = RealTDCRnet(dtype=dtype, N=64)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        # Real input: (batch, 2, T)
        input = torch.rand((batch_size, 2, signal_length), dtype=dtype)
        output = model(input)
        print(f"Input shape: {input.shape}, dtype: {input.dtype}")
        print(f"Output shape: {output.shape}, dtype: {output.dtype}")

        print("")


if __name__ == "__main__":
    test_model()
