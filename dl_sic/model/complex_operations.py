import torch
import torch.nn as nn


def complex_convolution(
    x_real: torch.Tensor,
    x_imag: torch.Tensor,
    conv_real: nn.Module,
    conv_imag: nn.Module,
) -> torch.Tensor:
    """Complex convolution real-valued tensors"""

    z_real = conv_real(x_real) - conv_imag(x_imag)
    z_imag = conv_real(x_imag) + conv_imag(x_real)

    return torch.stack([z_real, z_imag], dim=0)


class ComplexConv1d(nn.Module):
    """
    Complex 1D Convolution that supports both native complex operations
    and real dtype with separate real/imaginary convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: str = "same",
        dilation: int = 1,
        groups: int = 1,
        dtype: torch.dtype = torch.complex64,
    ):
        super().__init__()

        self.dtype = dtype
        self.dtype_is_complex = dtype in (
            torch.complex32,
            torch.complex64,
            torch.complex128,
        )

        if self.dtype_is_complex:
            # Use native PyTorch complex convolution
            self.conv = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                dtype=dtype,
            )
        else:
            # Use separate real and imaginary convolutions
            self.conv_real = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                dtype=dtype,
            )
            self.conv_imag = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=padding,
                dilation=dilation,
                groups=groups,
                dtype=dtype,
            )

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:
        - For complex: (batch, in_channels, T) complex tensor
        - For real: (2, batch, in_channels, T) real tensor (first dim: 0=real, 1=imag)
        """
        self._validate_input_type(x)
        if torch.is_complex(x):
            if x.dim() != 3:
                raise ValueError(
                    f"{self.__class__.__name__}: Input must be complex of shape (batch, in_channels, T), got shape {x.shape}. "
                )
            return self.conv(x)
        else:
            if x.dim() != 4 or x.shape[0] != 2:
                raise ValueError(
                    f"{self.__class__.__name__}: Real input must be 4D (2, batch, in_channels, T), got shape {x.shape}. "
                )
            # For real dtype, apply complex convolution using separate real/imag parts
            return complex_convolution(x[0], x[1], self.conv_real, self.conv_imag)


def _inv_sqrt_2x2(
    a11: torch.Tensor,
    a12: torch.Tensor,
    a21: torch.Tensor,
    a22: torch.Tensor,
    symmetric: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Inverse Squareroot of 2x2 Matrix adapted from complextorch library
    Following: https://en.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
    """
    if symmetric:
        det = a11 * a22 - a12 * a12  # Ignore a21 = a12
    else:
        det = a11 * a22 - a12 * a21
    trace = a11 + a22

    s = torch.sqrt(det)
    t = torch.sqrt(trace + 2 * s)
    coeff = 1 / (s * t)

    b11, b22 = coeff * (a22 + s), coeff * (a11 + s)

    if symmetric:
        b12, b21 = -coeff * a12, None  # None for optimisation
    else:
        b12, b21 = -coeff * a12, -coeff * a21

    return b11, b12, b21, b22


def _whiten2x2_layer_norm(
    x: torch.Tensor,  # Real(!) input tensor of shape (2, batch, features/channels, ...)
    normalized_shape: list[int],  # (features/channels, ...)
    eps: float = 1e-5,  # Ridge coefficient to stabilise the estimate of the covariance
) -> torch.Tensor:  # Returns real(!) stacked tensor
    """
    Layer Norm Whitening (centring and scaling) adapted from complextorch library
    """
    assert x.dim() >= 3  # Assume tensor has shape (2, B, F, ...)

    # Axes over which to compute mean and covariance
    axes = [-(i + 1) for i in range(len(normalized_shape))]
    mean = x.mean(dim=axes, keepdim=True)
    x -= mean
    var = (x * x).mean(dim=axes) + eps
    v_rr, v_ii = var[0], var[1]
    v_ir = (x[0] * x[1]).mean(dim=axes)  # Ignore v_ri (symmetric matrix)

    # Compute inverse matrix square root for ZCA whitening
    b11, b12, _, b22 = _inv_sqrt_2x2(v_rr, v_ir, None, v_ii, symmetric=True)

    head = mean.shape[1:]  # Head shape for broadcasting
    # output = var^(-1/2)*(x - mean) ((2,2) x (2,1) matrix multiplication)
    return torch.stack(
        [
            x[0] * b11.view(head) + x[1] * b12.view(head),  # Real component
            x[0] * b12.view(head) + x[1] * b22.view(head),  # Imaginary component
        ],
        dim=0,
    )


def complex_layer_norm(
    x: torch.Tensor,  # Complex (B, F, ...) or Real (2, B, F, ...) input tensor
    normalized_shape: list[int],  # shape (features/channels, ...)
    weight: torch.Tensor | None = None,  # Real(!) tensor of shape (2, 2, features)
    bias: torch.Tensor | None = None,  # Real(!) tensor of shape (2, features)
    eps: float = 1e-5,  # Ridge coefficient to stabilise covariance estimate
    *,
    complex_input_output: bool = True,  # Is input complex (B, F, ...) or real (2, B, F, ...)
) -> torch.Tensor:
    """
    Complex-Valued Layer Normalization adapted from complextorch library
    Extending the work of (Trabelsi et al., 2018) for each channel across a batch.
    """
    if complex_input_output:
        x = torch.stack((x.real, x.imag), dim=0)  # Shape (B, F, ...) -> (2, B, F, ...)

    z = _whiten2x2_layer_norm(x, normalized_shape, eps=eps)

    # Apply affine transformation with learnable parameters Gamma and Beta (Trabelsi et al., 2018)
    if weight is not None and bias is not None:
        shape = *([1] * (x.dim() - 1 - len(normalized_shape))), *normalized_shape
        weight = weight.view(2, 2, *shape)
        z = torch.stack(
            [
                z[0] * weight[0, 0] + z[1] * weight[0, 1],
                z[0] * weight[1, 0] + z[1] * weight[1, 1],
            ],
            dim=0,
        ) + bias.view(2, *shape)

    if complex_input_output:
        return torch.complex(z[0], z[1])  # Complex shape (B, F, ...)
    else:
        return z  # Real shape (2, B, F, ...)


class ComplexLayerNorm(nn.Module):
    """
    Complex-Valued Layer Normalization from complextorch library

    Uses whitening transformation to ensure standard normal complex distribution
    with equal variance in both real and imaginary components.
    Extending the work of (Barrachina et al., 2023) for each channel across a batch.
    """

    def __init__(
        self,
        normalized_shape: int | list[int] | torch.Size,
        *,
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        complex_input_output: bool = True,
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()

        # Convert `normalized_shape` to `torch.Size`
        if isinstance(normalized_shape, int):
            normalized_shape = torch.Size([normalized_shape])
        elif isinstance(normalized_shape, list):
            normalized_shape = torch.Size(normalized_shape)
        assert isinstance(normalized_shape, torch.Size)

        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        self.complex_io = complex_input_output

        # Handle complex dtypes
        complex_to_real = {
            torch.complex128: torch.float64,
            torch.complex64: torch.float32,
            torch.complex32: torch.float16,
        }
        dtype = complex_to_real.get(dtype, dtype)

        # Create parameters for Gamma and Beta for weight and bias
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(2, 2, *normalized_shape, dtype=dtype))
            self.bias = nn.Parameter(torch.zeros(2, *normalized_shape, dtype=dtype))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.elementwise_affine:
            return
        # Initialise Gamma and Beta (weight and bias)
        self.weight.data.copy_(
            0.70710678118
            * torch.eye(2, dtype=self.weight.dtype).view(
                2, 2, *([1] * len(self.normalized_shape))
            )
        )
        torch.nn.init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # Sanity check to make sure the shapes match
        assert (
            self.normalized_shape == input.shape[-len(self.normalized_shape) :]
        ), "Expected normalized_shape to match last dimensions of input shape!"

        return complex_layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            complex_input_output=self.complex_io,
        )
