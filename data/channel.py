# Should include
# fading ✅, sdd randomness in
#   power_delay_profile (both in length and in its keys/values)
#   Rician K-factor: random[0,10] -> 0 is Rayleigh. K -> infty -> AWGN channel
# frequency offsets,
# awgn ✅,
# phase offsets,
# time delays,
# IQ imbalance ✅
#   add tiny randomness in imbalance
#       (0 - 0.3 dB)
#       (0 - 2 degrees)
#

from gnuradio import blocks, gr, channels
import numpy as np


def fading_model(
    input_samples: np.ndarray,
    sample_rate: float | int,
    *,
    rician_factor_k: float = 4.0,  # LOS / NLOS. If 0, equivalent to Rayleigh model
    max_doppler_speed: float = 20,  # (m/s)
    power_delay_profile: dict[float, float] = {
        0.0: 1.0,
        0.1: 0.99,
        1.3: 0.97,
    },  # keys: delays in samples; values: magnitudes
    num_sinusoids: int = 16,
    num_taps: int = 16  # FIR taps for fractional delays
) -> np.ndarray:
    """Frequency Selective Fading Model block from GNU Radio."""
    # Note regarding max_doppler
    # For a receiver moving at 5 m/s, the coherence time of the channel is much
    # larger than the maximum packet size (~10 ms) for BLE or IEEE 802.15.4
    c = 3e8  # Speed of light (m/s)
    fc = 2.4e9  # Approximate carrier frequency at ISM band
    max_doppler_freq = fc * (max_doppler_speed / c)
    norm_max_doppler_freq = max_doppler_freq / sample_rate

    rician_model = True  # Simply use Rician always. Rayleigh is just Rician with K = 0
    seed: int = np.random.randint(-10000, 10000)

    # For a flat fading model, use power_delay_profile = {0.0: 1.0}
    power_delay_profile = sorted(power_delay_profile.items())
    pdp_delays = tuple(delay for delay, _ in power_delay_profile)
    pdp_magnitudes = tuple(magnitude for _, magnitude in power_delay_profile)

    # Convert NumPy array to GNU Radio format
    src = blocks.vector_source_c(input_samples.tolist(), False, 1, [])

    # Instantiate the Frequency Selective Fading block
    block = channels.selective_fading_model(
        num_sinusoids,
        norm_max_doppler_freq,
        rician_model,
        rician_factor_k,
        seed,
        pdp_delays,
        pdp_magnitudes,
        num_taps,
    )

    # Sink to collect the output
    sink = blocks.vector_sink_c(1, 1024 * 4)

    # Connect blocks and run
    tb = gr.top_block()
    tb.connect(src, block, sink)
    tb.run()

    return np.array(sink.data())


def add_white_gaussian_noise(
    signal: np.ndarray, noise_power: float, noise_power_db: bool = True
) -> np.ndarray:
    """Adds white noise to a signal (complex or real)."""
    if noise_power_db:
        noise_power = 10 ** (noise_power / 10)
    if np.iscomplexobj(signal):
        # Half the power in I and Q components respectively
        noise = np.sqrt(noise_power) * (
            np.random.normal(0, np.sqrt(2) / 2, len(signal))
            + 1j * np.random.normal(0, np.sqrt(2) / 2, len(signal))
        )
    else:
        # White Gaussian noise power is equal to its variance
        noise = np.sqrt(noise_power) * np.random.normal(0, 1, len(signal))
    return signal + noise


def apply_iq_imbalance(
    signal: np.ndarray,
    magnitude: float = 0,
    phase: float = 0,
    tx_mode: bool = True,
    magnitude_in_db: bool = True,
    phase_in_degrees: bool = True,
) -> np.ndarray:
    """
    Applies IQ imbalance to a complex signal following GNU Radio's approach.
    """
    magnitude_linear = 10 ** (magnitude / 20.0) if magnitude_in_db else magnitude
    phase_rad = phase * np.pi / 180.0 if phase_in_degrees else phase

    I_in = np.real(signal)
    Q_in = np.imag(signal)

    if tx_mode:
        # Transmitter Mode
        I_mag = magnitude_linear * I_in
        I_out = np.cos(phase_rad) * I_mag
        Q_out = np.sin(phase_rad) * I_mag + Q_in
    else:
        # Receiver Mode
        I_processed = np.cos(phase_rad) * I_in + np.sin(phase_rad) * Q_in
        I_out = magnitude_linear * I_processed
        Q_out = Q_in

    return I_out + 1j * Q_out
