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
        N: int = 32,  # Out channels in the complex encoder
        U: int = 128,  # Middle channels in complex dilated convolution
        H: int = 32,  # Hidden size in complex LSTM
        V: int = 8,  # Dilated convolutions on each side of the LSTM
        *,
        encoder_kernel_size: int = 3,  # Changed from 2 to 3 (odd, keep same size)
        decoder_kernel_size: int = 3,  # Changed from 2 to 3 (odd, keep same size)
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
            hidden_size=H,
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
        Input shape handling for both complex and real dtypes:
        - For complex: (batch, T) or (batch, 1, T) complex tensor
        - For real: (2, batch, T) or (2, batch, 1, T) real tensor
        """
        # Input shape handling
        original_dim = x.dim()
        if self.dtype_is_complex:
            # Complex input: (batch, T) -> (batch, 1, T)
            if x.dim() == 2:
                x = x.unsqueeze(1)  # (batch, 1, T)
            if x.size(1) != 1:
                raise ValueError(f"Expected 1 input channel, got {x.size(1)} channels")
        else:
            # Real input: (2, batch, T) -> (2, batch, 1, T)
            if x.dim() == 3:
                x = x.unsqueeze(2)
            if x.size(2) != 1:
                raise ValueError(f"Expected 1 input channel, got {x.size(2)}")

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

        # Restore original dimensions
        if self.dtype_is_complex:
            if original_dim == 2:
                s = s.squeeze(1)  # (batch, T)
        else:
            if original_dim == 3:
                s = s.squeeze(2)  # (2, batch, T)

        return s


def test_model():
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.complex64, torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = ComplexTDCRnet(dtype=dtype)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        # Test both 2D and 3D inputs for each dtype
        if dtype in (torch.complex32, torch.complex64, torch.complex128):
            print("Testing 3D complex input")
            input = torch.rand((batch_size, 1, signal_length), dtype=dtype)
            output = model(input)
            print("Input shape:", input.shape)
            print("Output shape:", output.shape)

            print("Testing 2D complex input")
            input = torch.rand((batch_size, signal_length), dtype=dtype)
            output = model(input)
            print("Input shape:", input.shape)
            print("Output shape:", output.shape)
        else:
            print("Testing 4D real input")
            input = torch.rand((2, batch_size, 1, signal_length), dtype=dtype)
            output = model(input)
            print("Input shape:", input.shape)
            print("Output shape:", output.shape)

            print("Testing 3D real input")
            input = torch.rand((2, batch_size, signal_length), dtype=dtype)
            output = model(input)
            print("Input shape:", input.shape)
            print("Output shape:", output.shape)

        print("")


if __name__ == "__main__":
    test_model()
