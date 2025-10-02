import torch
import torch.nn as nn


class ComplexPReLU(nn.Module):
    """
    Complex PReLU (Parametric Rectified Linear Activation Unit) activation function.
    Applies sigmoid separately to the real and imaginary parts.
    """

    def __init__(
        self, num_parameters: int = 1, init: float = 0.01, device=None, dtype=None
    ) -> None:
        super().__init__()
        self.act = nn.PReLU(num_parameters, init, device, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            # Complex input: (batch, channels, T)
            return torch.complex(self.act(x.real), self.act(x.imag))
        else:
            # Real representation: (2, batch, channels, T)
            if x.dim() != 4 or x.shape[0] != 2:
                raise ValueError(
                    f"ComplexPReLU: Real input must be 4D (2, batch, channels, T), got shape {x.shape}"
                )
            real_part = self.act(x[0])
            imag_part = self.act(x[1])
            return torch.stack([real_part, imag_part], dim=0)


class ComplexSigmoid(nn.Module):
    """
    Complex Sigmoid activation function.
    Applies sigmoid separately to the real and imaginary parts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.is_complex(x):
            # Complex input: (batch, channels, T)
            return torch.complex(self.act(x.real), self.act(x.imag))
        else:
            # Real representation: (2, batch, channels, T)
            if x.dim() != 4 or x.shape[0] != 2:
                raise ValueError(
                    f"ComplexSigmoid: Real input must be 4D (2, batch, channels, T), got shape {x.shape}"
                )
            # Apply activation to both real and imaginary parts
            real_part = self.act(x[0])
            imag_part = self.act(x[1])
            return torch.stack([real_part, imag_part], dim=0)
