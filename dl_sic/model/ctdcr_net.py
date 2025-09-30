import torch
import torch.nn as nn

from .complex_encoder import ComplexEncoder
from .complex_dilated_conv import ComplexDilatedConv
from .complex_lstm import ComplexLSTM
from .complex_decoder import ComplexDecoder
from .activation_functions import ComplexPReLU, ComplexSigmoid


class CTDCR_net(nn.Module):
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
    ) -> None:
        super().__init__()
        self.encoder = ComplexEncoder(
            in_channels=1,
            mid_channels=M,
            out_channels=N,
            kernel_size=encoder_kernel_size,
        )

        self.cdc_left = nn.ModuleList(
            ComplexDilatedConv(
                in_channels=N,
                mid_channels=U,
                dilation=2**v,
            )
            for v in range(V)
        )
        self.lstm = ComplexLSTM(
            input_size=N,
            hidden_size=H,
        )
        self.cdc_right = nn.ModuleList(
            ComplexDilatedConv(
                in_channels=N,
                mid_channels=U,
                dilation=2**v,
            )
            for v in range(V)
        )

        # Based on Conv-TasNet, Luo et al., 2019, Fig. 1.B
        self.prelu_out = ComplexPReLU()
        self.conv_out = nn.Conv1d(
            in_channels=N,
            out_channels=M,  # Expand channels to mid_channels to match z
            kernel_size=1,  # Assume pointwise 1x1-conv (based on Conv-TasNet, Luo et al., 2019)
            padding=0,
            dtype=torch.complex64,
        )

        self.sigmoid_out = ComplexSigmoid()

        # Guo et al., 2024:
        # "Each complex decoder recovers a source signal through contrary convolution operations of CHE"
        # But Luo et al., 2019: Encoder doesn't include LayerNorm and the second Conv layer
        # Also, input comes from z ∈ MxT in ComplexEncoder.
        self.decoder = ComplexDecoder(
            in_channels=M, out_channels=1, kernel_size=decoder_kernel_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, T) or (batch, 1, T) complex tensor
        Output shape: same as input shape
        """
        # Input shape handling
        original_dim = x.dim()
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, T)
        if x.size(1) != 1:
            raise ValueError(f"Expected 1 input channel, got {x.size(1)} channels")

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

        if original_dim == 2:
            s = s.squeeze(1)  # (batch, T)

        return s


def test_model():
    batch_size = 4
    signal_length = 2048  # T

    model = CTDCR_net()

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("Testing 3D input")
    input = torch.rand((batch_size, 1, signal_length), dtype=torch.complex64)
    output: torch.Tensor = model(input)  # Forward pass

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)

    print("Testing 2D input")
    input = torch.rand((batch_size, signal_length), dtype=torch.complex64)
    output: torch.Tensor = model(input)  # Forward pass

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
