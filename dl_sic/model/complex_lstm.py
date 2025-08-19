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
        batch_first: bool = True,  # Follow (batch, in_channels, T) convention
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()

        lstm_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": batch_first,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "proj_size": proj_size,
            "device": device,
            "dtype": dtype,
        }

        self.lstm_real = nn.LSTM(**lstm_kwargs)
        self.lstm_imag = nn.LSTM(**lstm_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, input_size, T) complex tensor
        """
        if not torch.is_complex(x):
            raise TypeError("ComplexLSTM expects a complex tensor")

        # Follow (batch, in_channels, T) convention
        # Convert from (batch, channels, T) to (batch, T, channels)
        x_real = x.real.permute(0, 2, 1)
        x_imag = x.imag.permute(0, 2, 1)

        out_r1, _ = self.lstm_real(x_real)
        out_i1, _ = self.lstm_imag(x_imag)
        L_r = out_r1 - out_i1

        out_r2, _ = self.lstm_real(x_imag)
        out_i2, _ = self.lstm_imag(x_real)
        L_i = out_r2 + out_i2

        # Convert back from (batch, T, channels) to (batch, channels, T)
        return torch.complex(L_r, L_i).permute(0, 2, 1)


def test_model():
    in_channels = 32  # N
    batch_size = 4
    signal_length = 2048  # T
    model = ComplexLSTM(in_channels)

    print(f"Total Parameters: {sum(p.numel() for p in model.parameters()):,}")

    input = torch.rand((batch_size, in_channels, signal_length), dtype=torch.complex64)
    output: torch.Tensor = model(input)  # Forward pass

    print("Input shape:", input.shape)
    print("Output shape:", output.shape)


if __name__ == "__main__":
    test_model()
