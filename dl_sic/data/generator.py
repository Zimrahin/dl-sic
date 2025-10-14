import os
import sys
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt

from dataclasses import dataclass
from tqdm import tqdm

from .simulation.transmitter import TransmitterBLE, Transmitter802154
from .simulation.channel import (
    frequency_selective_fading,
    add_white_gaussian_noise,
    fractional_delay_fir_filter,
    iq_imbalance,
    multiply_by_complex_exponential,
)


@dataclass
class SimulationConfig:
    sample_rate: float = 4e6
    num_signals: int = 5000
    signal_length: int = 3000  # Samples
    ble_payload_size_range: tuple[int, int] = (4, 61)  # Max exclusive
    ieee802154_payload_size_range: tuple[int, int] = (4, 16)  # Max exclusive
    # Channel/impairment ranges
    amplitude_range: tuple[float, float] = (0.3, 1.0)
    freq_offset_range: tuple[float, float] = (-10e3, 10e3)
    sample_delay_range: tuple[float, float] = (0, 1000)
    snr_low_db_range: tuple[float, float] = (8.0, 25.0)  # SNR weaker signal
    iq_imb_phase_range: tuple[float, float] = (0, 2)  # degrees
    iq_imb_mag_range: tuple[float, float] = (0, 0.3)  # dB
    rician_factor_range: tuple[float, float] = (0.0, 20.0)
    fading_max_doppler_speed: float = 20.0
    fading_pdp_max_size: int = 3  # Maximum fading paths
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
        size = np.random.randint(1, max_size + 1)
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

    def generate_mixture(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate a mixture of two signals with impairments and noise"""
        # Generate base signals (size = signal_length)
        s1 = self._modulate_random_packet("ble", self.cfg.signal_length)
        s2 = self._modulate_random_packet("802154", self.cfg.signal_length)

        # Amplitudes will be used to compute SNR to weaker signal
        s1_out = self._delay_and_offsets(s1)
        s2_out = self._delay_and_offsets(s2)
        amplitude1 = np.random.uniform(*self.cfg.amplitude_range)
        amplitude2 = np.random.uniform(*self.cfg.amplitude_range)
        s1_out = amplitude1 * s1_out
        s2_out = amplitude2 * s2_out

        # Apply fading channel + AWGN
        s1_out = self._apply_fading_channel(s1_out)
        s2_out = self._apply_fading_channel(s2_out)
        s1_target = s1_out
        s2_target = s2_out
        snr_db = np.random.uniform(*self.cfg.snr_low_db_range)  # SNR of weaker signal
        noise_power = min(amplitude1, amplitude2) ** 2 / (10 ** (snr_db / 10))
        mixture = add_white_gaussian_noise(
            s1_out + s2_out, noise_power, noise_power_db=False
        )

        # Apply IQ imbalance at receiver
        mixture = self._apply_iq_imbalance(mixture, tx_mode=False)

        return (
            mixture.astype(np.complex64),  # NN Input
            s1_target.astype(np.complex64),  # Target
            s2_target.astype(np.complex64),  # Target
        )

    def generate_dataset(self) -> torch.utils.data.TensorDataset:
        # Pre-allocate tensors for entire dataset
        mixtures = torch.empty(
            (self.cfg.num_signals, self.cfg.signal_length), dtype=torch.complex64
        )
        s1_targets = torch.empty(
            (self.cfg.num_signals, self.cfg.signal_length), dtype=torch.complex64
        )
        s2_targets = torch.empty(
            (self.cfg.num_signals, self.cfg.signal_length), dtype=torch.complex64
        )
        for i in tqdm(
            range(self.cfg.num_signals),
            desc="Generating dataset",
            mininterval=1.0,
            disable=not sys.stdout.isatty(),  # Disable tqdm for Slurm jobs
        ):
            mixture_np, s1_np, s2_np = self.generate_mixture()
            mixtures[i] = torch.from_numpy(mixture_np)
            s1_targets[i] = torch.from_numpy(s1_np)
            s2_targets[i] = torch.from_numpy(s2_np)

        return torch.utils.data.TensorDataset(mixtures, s1_targets, s2_targets)


def plot_signals(
    mixture: torch.Tensor,
    target1: torch.Tensor,
    target2: torch.Tensor,
    sample_rate: float,
    index: int | None = None,
) -> None:
    """Plot mixture and targets in 3x1 subplot with shared x-axis"""
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))
    if torch.is_tensor(mixture):
        mixture = mixture.numpy()
        target1 = target1.numpy()
        target2 = target2.numpy()
    time_axis = np.arange(len(mixture)) / sample_rate * 1000

    ax1.plot(time_axis, np.real(mixture), alpha=0.7)
    ax1.plot(time_axis, np.imag(mixture), alpha=0.7)
    ax1.set_ylabel("Amplitude (-)")
    ax1.set_title(
        "Mixture Signal" if index is None else f"Mixture Signal (Index {index})"
    )
    ax1.grid(True)

    ax2.plot(time_axis, np.real(target1), alpha=0.7)
    ax2.plot(time_axis, np.imag(target1), alpha=0.7)
    ax2.set_ylabel("Amplitude (-)")
    ax2.set_title("Target 1 (BLE)")
    ax2.grid(True)

    ax3.plot(time_axis, np.real(target2), alpha=0.7)
    ax3.plot(time_axis, np.imag(target2), alpha=0.7)
    ax3.set_ylabel("Amplitude (-)")
    ax3.set_title("Target 2 (802.15.4)")
    ax3.set_xlabel("Time (ms)")
    ax3.grid(True)

    max_mixture = np.max(np.abs(mixture))
    ax1.set_ylim(-max_mixture - 0.1, max_mixture + 0.1)
    ax2.set_ylim(-max_mixture - 0.1, max_mixture + 0.1)
    ax3.set_ylim(-max_mixture - 0.1, max_mixture + 0.1)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    DEFAULT_DATASET_PATH = "simulated_dataset.pt"

    parser = argparse.ArgumentParser(description="Dataset Generator")
    parser.add_argument(
        "--test", action="store_true", help="Generate and plot single example"
    )
    parser.add_argument(
        "--save",
        nargs="?",
        const=DEFAULT_DATASET_PATH,
        default=None,
        type=str,
        help=f"Path to save dataset (default: {DEFAULT_DATASET_PATH})",
    )
    parser.add_argument(
        "--read",
        nargs="?",
        const=DEFAULT_DATASET_PATH,
        default=None,
        type=str,
        help=f"Path to load dataset for inspection (default: {DEFAULT_DATASET_PATH})",
    )
    args = parser.parse_args()

    config = SimulationConfig()
    generator = SignalDatasetGenerator(config)

    if args.test:
        # Generate and plot single example
        mixture, target1, target2 = generator.generate_mixture()
        plot_signals(mixture, target1, target2, config.sample_rate)

    elif args.save is not None:
        # Generate and save full dataset
        dataset = generator.generate_dataset()
        torch.save(dataset, args.save)
        print(f"Saved dataset with {len(dataset)} examples to {args.save}")

    elif args.read is not None:
        # Load dataset and start interactive inspection
        if not os.path.exists(args.read):
            raise FileNotFoundError(f"File {args.read} not found")

        # Load with weights_only=False for compatibility
        dataset = torch.load(args.read, weights_only=False)
        mixtures, s1_targets, s2_targets = dataset.tensors

        print(
            f"Loaded dataset with {len(mixtures)} examples, of {mixtures.shape[1]} samples each"
        )

        while True:
            try:
                idx = input("Enter index to plot (q to quit): ")
                if idx.lower() == "q":
                    break

                idx = int(idx)
                if 0 <= idx < len(mixtures):
                    mixture = mixtures[idx]
                    target1 = s1_targets[idx]
                    target2 = s2_targets[idx]
                    plot_signals(mixture, target1, target2, config.sample_rate, idx)
                else:
                    print(f"Index must be between 0 and {len(mixtures)-1}")

            except ValueError:
                print("Please enter a valid integer or 'q' to quit")
            except KeyboardInterrupt:
                break
