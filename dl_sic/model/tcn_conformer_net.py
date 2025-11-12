import torch
import torch.nn as nn
from torchaudio.models import Conformer

from .depth_dilated_conv import DepthDilatedConv


class TCNConformerNet(nn.Module):
    """
    TCN-Conformer for speech separation adapted from Sinha et al., 2022.
    Treats real/imaginary as independent channels
    """

    def __init__(
        self,
        M: int = 128,  # Mask channels
        N: int = 64,  # Bottleneck features, input to TCN and Conformer
        U: int = 128,  # Hidden channels in depth dilated convolution block (TCN)
        V: int = 3,  # Number of TCN/Conformer pairs
        *,
        encoder_kernel_size: int = 3,
        conformer_num_heads: int = 4,
        conformer_ffn_times_input: int = 2,  # ffn_dim = X * input_dim
        conformer_num_layers: int = 2,  # Stacked encode/decoder in conformer architecture
        conformer_conv_kernel_size: int = 15,  # 31 in reference paper
        conformer_dropout: float = 0.0,  # Dropout probability
    ) -> None:
        super().__init__()

        self.V = V

        self.encoder = nn.Conv1d(
            in_channels=2,
            out_channels=M,
            kernel_size=encoder_kernel_size,
            padding="same",
        )
        self.layer_norm_in = nn.GroupNorm(
            num_groups=1,
            num_channels=M,
        )
        self.decoder = nn.Conv1d(
            in_channels=M,
            out_channels=2,
            kernel_size=encoder_kernel_size,
            padding="same",
        )

        self.conv_in = nn.Conv1d(
            in_channels=M,
            out_channels=N,
            kernel_size=1,  # Pointwise 1x1-conv
            padding="same",
        )
        self.conv_out = nn.Conv1d(
            in_channels=N,
            out_channels=M,
            kernel_size=1,  # Pointwise 1x1-conv
            padding="same",
        )

        self.tcns = nn.ModuleList(
            [
                DepthDilatedConv(
                    in_channels=N,
                    hidden_channels=U,
                    dilation=2**v,
                    number_dconvs=1,
                )
                for v in range(V)
            ]
        )

        self.conformers = nn.ModuleList(
            [
                Conformer(
                    input_dim=N,
                    num_heads=conformer_num_heads,
                    ffn_dim=conformer_ffn_times_input * N,
                    num_layers=conformer_num_layers,
                    depthwise_conv_kernel_size=conformer_conv_kernel_size,
                    dropout=conformer_dropout,
                    use_group_norm=True,
                )
                for _ in range(V)
            ]
        )

        self.prelu_out = nn.PReLU()
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

        # Forward pass
        z = self.encoder(input)  # (batch, M, T)
        y = self.conv_in(self.layer_norm_in(z))  # (batch, N, T)

        for v in range(self.V):
            # TCN with skip connection
            y = self.tcns[v](y) + y  # (batch, N, T), residual connection
            # Conformer
            y = y.transpose(1, 2)  # (batch, T, N)
            cnf_len = torch.tensor(y.shape[0] * [y.shape[1]], device=y.device)  # B*[T]
            y, _ = self.conformers[v](y, lengths=cnf_len)
            y = y.transpose(1, 2)  # (batch, N, T)

        y = self.prelu_out(y)
        y = self.conv_out(y)  # (batch, M, T)
        y = self.sigmoid_out(y)  # (batch, M, T)

        s = y * z  # Elementwise product (batch, M, T)
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

        model = TCNConformerNet()

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
