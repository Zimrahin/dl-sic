import numpy as np
import scipy


def fractional_delay_fir_filter(
    data: np.ndarray, delay: float, num_taps: int = 21, same_size: bool = True
) -> np.ndarray:
    """
    Applies a delay to the input data by first applying a fractional delay using an FIR filter,
    and then applying an integer delay via sample shifting with zero-padding.
    """
    # Separate delay into its integer and fractional parts
    integer_delay = int(np.floor(delay))
    fractional_delay = delay - integer_delay

    # Build the FIR filter taps for the fractional delay
    n = np.arange(-num_taps // 2, num_taps // 2)  # ...-3,-2,-1,0,1,2,3...
    fir_kernel = np.sinc(n - fractional_delay)  # Shifted sinc function
    # fir_kernel *= np.hamming(len(n))  # Hamming window (avoid spectral leakage)
    fir_kernel /= np.sum(fir_kernel)  # Normalise filter taps, unity gain
    frac_delayed = scipy.signal.convolve(data, fir_kernel, mode="full")  # Apply filter

    # Compensate for the intrinsic delay caused by convolution
    frac_delayed = np.roll(frac_delayed, -num_taps // 2)
    if same_size:
        frac_delayed = frac_delayed[: len(data)]
    else:
        frac_delayed = frac_delayed[: len(data) + num_taps // 2]

    # Integer delay and pad with zeros
    delayed_output = np.zeros_like(frac_delayed)
    if integer_delay < len(frac_delayed):
        delayed_output[integer_delay:] = frac_delayed[
            : len(frac_delayed) - integer_delay
        ]

    return delayed_output
