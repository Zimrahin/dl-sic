import numpy as np
import torch
from dataclasses import dataclass
from simulation.transmitter import TransmitterBLE, Transmitter802154
from simulation.channel import (
    frequency_selective_fading,
    add_white_gaussian_noise,
    fractional_delay_fir_filter,
    iq_imbalance,
    multiply_by_complex_exponential,
)


@dataclass
class SimulationConfig:
    sample_rate: float = 4e6
    num_signals: int = 1000
    signal_length: int = 20000  # Samples
    ble_payload_size_range: tuple[int, int] = (10, 240)
    ieee802154_payload_size_range: tuple[int, int] = (10, 60)
    # Channel/impairment ranges
    amplitude_range: tuple[float, float] = (0.2, 1.0)
    freq_offset_range: tuple[float, float] = (-30e3, 30e3)
    sample_delay_range: tuple[float, float] = (0.0, 2000.0)
    snr_low_db_range: tuple[float, float] = (-20.0, 0.0)  # SNR weaker signal
    iq_imb_phase_range: tuple[float, float] = (0, 2)  # degrees
    iq_imb_mag_range: tuple[float, float] = (0, 0.3)  # dB
    rician_factor_range: tuple[float, float] = (0, 10.0)
    fading_max_doppler_speed: float = 20.0
    fading_pdp_max_size: int = 5
    fading_pdp_delay_range: tuple[float, float] = (0.0, 2.0)
    fading_pdp_power_range: tuple[float, float] = (0.99, 0.1)
    seed: int | None = None


class SignalDatasetGenerator:
    def __init__(self, config: SimulationConfig):
        self.cfg = config
        self.ble_tx = TransmitterBLE(self.cfg.sample_rate)
        self.ieee802154_tx = Transmitter802154(self.cfg.sample_rate)
        if self.cfg.seed is not None:
            np.random.seed(self.cfg.seed)

    def _modulate_random_packet(self, protocol: str, signal_length: int) -> np.ndarray:
        """Generate a random packet and modulate it"""
        if protocol == "ble":
            payload = np.random.randint(
                0,
                256,
                size=np.random.randint(*self.cfg.ble_payload_size_range),
                dtype=np.uint8,
            )
            base_address = np.random.randint(0, 2**32, dtype=np.uint32)
            modulated_signal = self.ble_tx.modulate_from_payload(payload, base_address)
        else:
            payload = np.random.randint(
                0,
                256,
                size=np.random.randint(*self.cfg.ieee802154_payload_size_range),
                dtype=np.uint8,
            )
            modulated_signal = self.ieee802154_tx.modulate_from_payload(payload)

        if len(modulated_signal) > signal_length:
            raise ValueError(
                f"Modulated signal length ({len(modulated_signal)}) exceeds input signal_length ({signal_length}). "
            )

        # Zero-pad to the right to achieve the desired length
        padded_signal = np.pad(
            modulated_signal,
            (0, signal_length - len(modulated_signal)),
            mode="constant",
        )
        return padded_signal

    def _delay_and_offsets(self, signal: np.ndarray) -> np.ndarray:
        """Apply random delay, frequency/phase offset"""
        # Time delay
        delay = np.random.uniform(*self.cfg.sample_delay_range)
        signal = fractional_delay_fir_filter(signal, delay, same_size=True)

        # Frequency/phase offset and amplitude scaling
        freq_offset = np.random.uniform(*self.cfg.freq_offset_range)
        phase_offset = np.random.uniform(0, 2 * np.pi)

        signal = multiply_by_complex_exponential(
            input_signal=signal,
            sample_rate=self.cfg.sample_rate,
            freq=freq_offset,
            phase=phase_offset,
        )
        return signal

    def _random_power_delay_profile(
        self,
        max_size: int = 5,
        delay_range: tuple[float, float] = (0.0, 2.0),
        power_range: tuple[float, float] = (0.99, 0.1),
    ) -> dict[float, float]:
        """Generate a random power delay profile dict with flexible size (>=1)."""
        size = np.random.randint(1, max_size)
        if size == 1:
            return {0.0: 1.0}  # Equivalent to flat fading
        delays = sorted(
            np.random.uniform(delay_range[0], delay_range[1]) for _ in range(size - 1)
        )
        powers = sorted(
            (np.random.uniform(power_range[1], power_range[0]) for _ in range(size)),
            reverse=True,  # Decreasing powers
        )
        pdp = {0.0: 1.0}
        for delay, power in zip(delays, powers[1:]):
            pdp[round(delay, 2)] = round(power, 2)

        return pdp

    def _apply_fading_channel(self, signal: np.ndarray) -> np.ndarray:
        """Apply channel effects: fading"""
        # Fading (random/different channel each call)
        pdp = self._random_power_delay_profile(
            max_size=self.cfg.fading_pdp_max_size,
            delay_range=self.cfg.fading_pdp_delay_range,
            power_range=self.cfg.fading_pdp_power_range,
        )
        signal = frequency_selective_fading(
            signal,
            self.cfg.sample_rate,
            rician_factor_k=np.random.uniform(*self.cfg.rician_factor_range),
            max_doppler_speed=self.cfg.fading_max_doppler_speed,
            power_delay_profile=pdp,
        )
        return signal

    def _apply_iq_imbalance(self, signal: np.ndarray, tx_mode: bool) -> np.ndarray:
        """Apply IQ imbalance impairment (random/different each call)"""
        mag_imb = np.random.uniform(*self.cfg.iq_imb_mag_range)
        phase_imb = np.random.uniform(*self.cfg.iq_imb_phase_range)
        signal = iq_imbalance(signal, mag_imb, phase_imb, tx_mode)

        return signal

    def _generate_mixture(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a mixture of two signals with impairments and noise"""
        # Generate base signals (size = signal_length)
        s1 = self._modulate_random_packet("ble", self.cfg.signal_length)
        s2 = self._modulate_random_packet("802154", self.cfg.signal_length)

        # Amplitudes will be used to compute SNR to weaker signal
        amplitude1 = np.random.uniform(*self.cfg.amplitude_range)
        amplitude2 = np.random.uniform(*self.cfg.amplitude_range)
        s1_offset = self._delay_and_offsets(s1) * amplitude1
        s2_offset = self._delay_and_offsets(s2) * amplitude2

        # Apply fading channel + AWGN
        s1_out = self._apply_fading_channel(s1_offset)
        s2_out = self._apply_fading_channel(s2_offset)
        snr_db = np.random.uniform(*self.cfg.snr_low_db_range)  # SNR of weaker signal
        noise_power = min(amplitude1, amplitude2) ** 2 / (10 ** (snr_db / 10))
        mixture = add_white_gaussian_noise(
            s1_out + s2_out, noise_power, noise_power_db=False
        )

        # Apply IQ imbalance at receiver
        mixture = self._apply_iq_imbalance(mixture, tx_mode=False)

        return (
            torch.from_numpy(mixture).clone(),  # Use as NN inpu
            torch.from_numpy(s1_offset).clone(),  # Use as target
            torch.from_numpy(s2_offset).clone(),  # Use as target
        )

    def generate_dataset(self) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        dataset = [None] * self.cfg.num_signals
        for i in range(self.cfg.num_signals):
            dataset[i] = self._generate_mixture()

        return dataset
