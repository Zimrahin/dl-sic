import torch
import torch.nn as nn


class ComplexLSTM(nn.Module):
    """
    Complex LSTM module that supports both native complex operations
    and real dtype with separate real/imaginary LSTMs.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 32,  # H in paper
        *,
        num_layers: int = 1,  # L in paper
        dropout: float = 0.0,
        bidirectional: bool = False,
        proj_size: int = 0,
        device=None,
        dtype: type = torch.complex64,
    ) -> None:
        super().__init__()

        batch_first: bool = True  # Follow (batch, in_channels, T) convention

        self.dtype = dtype
        complex_to_real = {
            torch.complex128: torch.float64,
            torch.complex64: torch.float32,
            torch.complex32: torch.float16,
        }
        lstm_dtype = complex_to_real.get(dtype, dtype)
        self.dtype_is_complex = dtype in complex_to_real

        lstm_kwargs = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "batch_first": batch_first,
            "bidirectional": bidirectional,
            "dropout": dropout,
            "proj_size": proj_size,
            "device": device,
            "dtype": lstm_dtype,
        }

        self.lstm_real = nn.LSTM(**lstm_kwargs)
        self.lstm_imag = nn.LSTM(**lstm_kwargs)

    def _validate_input_type(self, x: torch.Tensor) -> None:
        input_is_complex = torch.is_complex(x)
        if self.dtype_is_complex and not input_is_complex:
            raise ValueError(
                f"{self.__class__.__name__}: Model initialised with complex dtype {self.dtype}, "
                f"but received real input tensor. \nExpected complex input shape: (batch, in_channels, T)."
            )

        if not self.dtype_is_complex and input_is_complex:
            raise ValueError(
                f"{self.__class__.__name__}: Model initialised with real dtype {self.dtype}, "
                f"but received complex input tensor. \nExpected real input shape: (2, batch, in_channels, T)."
            )

    def _apply_complex_lstm(
        self, x_real: torch.Tensor, x_imag: torch.Tensor
    ) -> torch.Tensor:
        # Inputs shape: (batch, T, input_size)
        out_r1, _ = self.lstm_real(x_real)
        out_i1, _ = self.lstm_imag(x_imag)
        L_r = out_r1 - out_i1

        out_r2, _ = self.lstm_real(x_imag)
        out_i2, _ = self.lstm_imag(x_real)
        L_i = out_r2 + out_i2

        return torch.stack([L_r, L_i], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:
        - For complex: (batch, input_size, T) complex tensor
        - For real: (2, batch, input_size, T) real tensor (first dim: 0=real, 1=imag)
        """
        self._validate_input_type(x)
        input_is_complex = torch.is_complex(x)
        if input_is_complex:
            # Complex input: (batch, in_channels, T)
            if x.dim() != 3:
                raise ValueError(
                    f"{self.__class__.__name__}: Complex input must be 3D (batch, in_channels, T), got shape {x.shape}"
                )
            x = torch.stack((x.real, x.imag), dim=0)  # (B, F, ...) -> (2, B, F, ...)
        else:
            # Real representation: (2, batch, in_channels, T)
            if x.dim() != 4 or x.shape[0] != 2:
                raise ValueError(
                    f"{self.__class__.__name__}: Real input must be 4D (2, batch, in_channels, T), got shape {x.shape}"
                )

        # Convert from (batch, channels, T) to (batch, T, channels)
        x = x.transpose(-1, -2)
        out_lstm = self._apply_complex_lstm(x[0], x[1])  # (2, batch, T, channels)
        out_lstm = out_lstm.transpose(-1, -2)  # (batch, channels, T)

        if input_is_complex:
            return torch.complex(out_lstm[0], out_lstm[1])  # Complex shape (B, F, T)
        else:
            return out_lstm  # Real shape (2, batch, F, T)


def test_model():
    in_channels = 32  # N
    batch_size = 4
    signal_length = 2048  # T
    dtypes = [torch.complex64, torch.float32]

    for dtype in dtypes:
        print(f"Testing dtype: {dtype}")

        model = ComplexLSTM(in_channels, dtype=dtype)

        total_params = sum(p.numel() for p in model.parameters())
        total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())

        print(f"Total Parameters: {total_params:,}")
        print(f"Total Size: {total_memory:,} bytes")

        if dtype in (torch.complex32, torch.complex64, torch.complex128):
            # Complex input: (batch, in_channels, T)
            input = torch.rand((batch_size, in_channels, signal_length), dtype=dtype)
        else:
            # Real input: (2, batch, in_channels, T)
            input = torch.rand((2, batch_size, in_channels, signal_length), dtype=dtype)
        output = model(input)  # Forward pass

        print("Input shape:", input.shape)
        print("Output shape:", output.shape)
        print("")


if __name__ == "__main__":
    test_model()
