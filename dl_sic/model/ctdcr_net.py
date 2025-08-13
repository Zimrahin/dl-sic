import torch
import torch.nn as nn

from complex_encoder import ComplexEncoder
from complex_dilated_conv import ComplexDilatedConv
from complex_lstm import ComplexLSTM
from complex_decoder import ComplexDecoder
from activation_functions import ComplexPReLU, ComplexSigmoid


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
    ):
        super().__init__()
        self.encoder = ComplexEncoder(in_channels=1, mid_channels=M, out_channels=N)

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
        self.prelu_out = ComplexPReLU
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
        # self.decoder = ComplexEncoder(in_channels=M, mid_channels=N, out_channels=1)
        self.decoder = ComplexDecoder(in_channels=M, out_channels=1)

    def forward(self, x):
        """
        Input shape: (B, 1, T) complex tensor
        """
        y, z = self.encoder(x)  # (B, N, T), (B, M, T)

        for cdc in self.cdc_left:
            y = cdc(y)  # (B, N, T)

        y = self.lstm(y)  # (B, N, T)

        for cdc in self.cdc_right:
            y = cdc(y)  # (B, N, T)

        y = self.prelu_out(y)  # Based on Conv-TasNet, Luo et al., 2019, Fig. 1.B
        y = self.conv_out(y)  # (B, M, T)
        y = self.sigmoid_out(y)  # (B, M, T)

        s = y * z  # Elementwise (Hadamard) product
        s = self.decoder(s)

        return s
