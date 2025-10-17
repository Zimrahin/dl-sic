import torch
import torch.nn as nn


class RealLSTM(nn.Module):
    """
    Wrapper for PyTorch LSTM with adapted default arguments
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,  # H in paper
        *,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()

        batch_first: bool = True

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
        self.lstm = nn.LSTM(**lstm_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape: (batch, input_size, T)
        Output shape: (batch, hidden_size, T)
        """
        x = x.transpose(1, 2)  # (batch, T, input_size)
        out_lstm, _ = self.lstm(x)  # (batch, T, hidden_size)
        out_lstm = out_lstm.transpose(1, 2)  # (batch, hidden_size, T)

        return out_lstm


def test_model():
    in_channels = 32  # N
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = RealLSTM(in_channels, dtype=dtype)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        # Real input: (batch, in_channels, T)
        input = torch.rand((batch_size, in_channels, signal_length), dtype=dtype)
        output = model(input)  # Forward pass

        print("Input shape:", input.shape)
        print("Output shape:", output.shape)
        print("")


if __name__ == "__main__":
    test_model()
