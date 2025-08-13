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
        if not torch.is_complex(x):
            raise TypeError("ComplexPReLU expects a complex tensor")
        return torch.complex(self.act(x.real), self.act(x.imag))


class ComplexSigmoid(nn.Module):
    """
    Complex Sigmoid activation function.
    Applies sigmoid separately to the real and imaginary parts.
    """

    def __init__(self) -> None:
        super().__init__()
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(x):
            raise TypeError("ComplexSigmoid expects a complex tensor")
        return torch.complex(self.act(x.real), self.act(x.imag))
