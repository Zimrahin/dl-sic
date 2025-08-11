import torch
import torch.nn as nn


class ComplexLSTM(nn.Module):
    """
    Complex LSTM (CLSTM) module from Guo et al., 2024.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,  # H in paper
        *,
        num_layers: int = 1,  # L in paper
        batch_first: bool = True,  # Follow (B, in_channels, T) convention
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        self.lstm_real = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device,
            dtype,
        )
        self.lstm_imag = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first,
            dropout,
            bidirectional,
            proj_size,
            device,
            dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (B, input_size, T) complex tensor
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexLSTM expects a complex tensor")

        # Follow (B, in_channels, T) convention
        # Convert from (B, C, T) to (B, T, C)
        x_real = torch.permute(x.real, (0, 2, 1))
        x_imag = torch.permute(x.imag, (0, 2, 1))

        out_r1, _ = self.lstm_real(x_real)
        out_i1, _ = self.lstm_imag(x_imag)
        L_r = out_r1 - out_i1

        out_r2, _ = self.lstm_real(x_imag)
        out_i2, _ = self.lstm_imag(x_real)
        L_i = out_r2 + out_i2

        # Convert back from (B, T, C) to (B, C, T)
        return torch.complex(L_r, L_i).permute(0, 2, 1)
