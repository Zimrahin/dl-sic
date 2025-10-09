import torch
import torch.nn as nn

from .complex_encoder import ComplexEncoder
from .complex_dilated_conv import ComplexDilatedConv
from .complex_lstm import ComplexLSTM
from .complex_decoder import ComplexDecoder
from .activation_functions import ComplexPReLU, ComplexSigmoid
from .complex_operations import ComplexConv1d


class ComplexTDCRnet(nn.Module):
    """
    Complex Time-Domain Dilated Convolutional Recurrent Network from Guo et al., 2024.
    """

    def __init__(
        self,
        M: int = 128,  # Middle channels in the complex encoder
        N: int = 32,  # Out channels of encoder and input to LSTM = H
        U: int = 128,  # Middle channels in complex dilated convolution
        V: int = 8,  # Dilated convolutions on each side of the LSTM
        *,
        encoder_kernel_size: int = 3,
        decoder_kernel_size: int = 3,
        dtype=torch.complex64,
    ) -> None:
        super().__init__()

        self.dtype = dtype
        self.dtype_is_complex = dtype in (
            torch.complex32,
            torch.complex64,
            torch.complex128,
        )

        self.encoder = ComplexEncoder(
            in_channels=1,
            mid_channels=M,
            out_channels=N,
            kernel_size=encoder_kernel_size,
            dtype=dtype,
        )

        self.cdc_left = nn.ModuleList(
            ComplexDilatedConv(
                in_channels=N,
                mid_channels=U,
                dilation=2**v,
                dtype=dtype,
            )
            for v in range(V)
        )
        self.lstm = ComplexLSTM(
            input_size=N,
            hidden_size=N,
            dtype=dtype,
        )
        self.cdc_right = nn.ModuleList(
            ComplexDilatedConv(
                in_channels=N,
                mid_channels=U,
                dilation=2**v,
                dtype=dtype,
            )
            for v in range(V)
        )

        # Based on Conv-TasNet, Luo et al., 2019, Fig. 1.B
        self.prelu_out = ComplexPReLU(dtype=dtype)
        self.conv_out = ComplexConv1d(
            in_channels=N,
            out_channels=M,  # Expand channels to mid_channels to match z
            kernel_size=1,  # Pointwise 1x1-conv (based on Conv-TasNet, Luo et al., 2019)
            padding=0,
            dtype=dtype,
        )

        self.sigmoid_out = ComplexSigmoid()

        # From ConvTasNet (Luo et al., 2019):
        self.decoder = ComplexDecoder(
            in_channels=M,
            out_channels=1,
            kernel_size=decoder_kernel_size,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:
        - Complex: (batch, 1, T)
        - Real: (batch, 2, T)
        Output: Same shape and type as input
        """
        if x.dim() != 3:
            raise ValueError(
                f"Expected 3D input (batch, channels, T), got {tuple(x.shape)}D"
            )
        input_is_complex = torch.is_complex(x)

        if input_is_complex:
            # Complex input: (batch, 1, T)
            if x.size(1) != 1:
                raise ValueError(
                    f"Expected 1 complex channel, got {x.size(1)} channels"
                )
            # If network uses real parameters but got complex input, convert to real representation
            if not self.dtype_is_complex:
                x = torch.stack([x.real, x.imag], dim=0)  # (2, batch, 1, T)
        else:
            # Real input: (batch, 2, T)
            if x.size(1) != 2:
                raise ValueError(f"Expected 2 real channels, got {x.size(1)} channels")
            # Convert from (batch, 2, T) to (2, batch, 1, T) (complex modules expect this shape)
            x = x.transpose(0, 1).unsqueeze(2)  # (2, batch, 1, T)

        # Forward pass
        y, z = self.encoder(x)  # (batch, N, T), (batch, M, T)

        for cdc in self.cdc_left:
            y = cdc(y)  # (batch, N, T)

        y = self.lstm(y)  # (batch, N, T)

        for cdc in self.cdc_right:
            y = cdc(y)  # (batch, N, T)

        y = self.prelu_out(y)  # Based on Conv-TasNet, Luo et al., 2019, Fig. 1.B
        y = self.conv_out(y)  # (batch, M, T)
        y = self.sigmoid_out(y)  # (batch, M, T)

        s = y * z  # Elementwise (Hadamard) product
        s = self.decoder(s)

        if input_is_complex:
            if not self.dtype_is_complex:
                # Convert back to complex: (2, batch, 1, T) -> (batch, 1, T)
                s = torch.complex(s[0], s[1])
        else:
            # Convert back to real representation: (2, batch, 1, T) -> (batch, 2, T)
            s = s.squeeze(2).transpose(0, 1)

        return s


def test_model():
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.complex64, torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = ComplexTDCRnet(dtype=dtype, N=32)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        # Test 3D inputs for each dtype
        if dtype in (torch.complex32, torch.complex64, torch.complex128):
            print("Testing 3D complex input")
            input = torch.rand((batch_size, 1, signal_length), dtype=dtype)
            output = model(input)
            print(f"Input shape: {input.shape}, dtype: {input.dtype}")
            print(f"Output shape: {output.shape}, dtype: {output.dtype}")
        else:
            complex_dtype_map = {
                torch.float32: torch.complex64,
                torch.float16: torch.complex32,
                torch.float64: torch.complex128,
            }
            complex_dtype = complex_dtype_map.get(dtype, torch.complex64)
            print("Testing 3D complex input")
            input = torch.rand((batch_size, 1, signal_length), dtype=complex_dtype)
            output = model(input)
            print(f"Input shape: {input.shape}, dtype: {input.dtype}")
            print(f"Output shape: {output.shape}, dtype: {output.dtype}")

            print("Testing 3D real input")
            input = torch.rand((batch_size, 2, signal_length), dtype=dtype)
            output = model(input)
            print(f"Input shape: {input.shape}, dtype: {input.dtype}")
            print(f"Output shape: {output.shape}, dtype: {output.dtype}")

        print("")


if __name__ == "__main__":
    test_model()
