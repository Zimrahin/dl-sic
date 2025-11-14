import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import sys
from tqdm import tqdm

from utils.dataset import LoadDataset
from utils.loss_functions import si_snr_loss_complex
from data.generator import SignalDatasetGenerator, SimulationConfig
from data.tranceiver.receiver import ReceiverBLE, Receiver802154, Receiver

from model_factory import ModelFactory


def to_complex(x: torch.Tensor) -> torch.Tensor:
    """Helper to convert real 2-channel tensor to complex tensor"""
    if not x.is_complex():
        assert x.dim() == 2, f"Expected 2D input (channels, T), got {tuple(x.shape)}"
        assert x.size(0) == 2, f"Expected 2 real channels, got {x.size(1)} channels"
        return torch.complex(x[0, :], x[1, :])
    return x


def plot_test_signals(
    mixture: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
    index: int,
    loss_val: float,
    *,
    mixture_crc: str = "",
    target_crc: str = "",
    output_crc: str = "",
) -> None:
    """Plot mixture, target and output"""
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    sample_axis = np.arange(len(mixture))

    # Plot mixture
    mixture_title = f"Input Mixture (Index {index})"
    if mixture_crc:
        mixture_title += f", CRC: {mixture_crc}"
    ax1.plot(sample_axis, np.real(mixture), alpha=0.7, label="Real")
    ax1.plot(sample_axis, np.imag(mixture), alpha=0.7, label="Imag")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(mixture_title)
    ax1.legend()
    ax1.grid(True)

    # Plot target
    target_title = "Target Signal"
    if target_crc:
        target_title += f", CRC: {target_crc}"
    ax2.plot(sample_axis, np.real(target), alpha=0.7, label="Real")
    ax2.plot(sample_axis, np.imag(target), alpha=0.7, label="Imag")
    ax2.set_ylabel("Amplitude")
    ax2.set_title(target_title)
    ax2.legend()
    ax2.grid(True)

    # Plot output
    output_title = f"Model Output (Loss: {loss_val:.6f} dB)"
    if output_crc:
        output_title += f", CRC: {output_crc}"
    ax3.plot(sample_axis, np.real(output), alpha=0.7, label="Real")
    ax3.plot(sample_axis, np.imag(output), alpha=0.7, label="Imag")
    ax3.set_ylabel("Amplitude")
    ax3.set_title(output_title)
    ax3.set_xlabel("Sample Index")
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.show()


def check_crc(signal: np.ndarray, receiver: Receiver) -> str:
    """Check CRC for a given signal using the receiver"""
    try:
        packets = receiver.demodulate_to_packet(signal)
        if packets and len(packets) > 0:
            return "PASS" if packets[0]["crc_check"] else "FAIL"
        else:
            return "NO PACKET"
    except Exception as e:
        return "NO PACKET"


def helper_set_receiver(target_idx: int, sample_rate: float) -> Receiver:
    """Helper to set up receiver based on target index"""
    if target_idx == 1:
        return ReceiverBLE(sample_rate, transmission_rate=1e6)
    elif target_idx == 2:
        return Receiver802154(sample_rate, transmission_rate=2e6)
    else:
        raise ValueError("Invalid target_idx for CRC checking")


def test_model_interactive(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    loss_function: callable,
    *,
    demodulate: bool = False,
    sample_rate: float = 4e6,
    target_idx: int = 1,  # (1 for BLE, 2 for IEEE 802.15.4)
) -> None:
    """Interactive model testing loop"""
    model.eval()

    receiver = helper_set_receiver(target_idx, sample_rate) if demodulate else None

    while True:
        try:
            user_input = input("Enter index to test (q to quit, r to random): ").lower()

            if user_input == "q":
                break

            if user_input == "r":
                idx = torch.randint(0, len(dataset), (1,)).item()
            else:
                idx = int(user_input)

            if not 0 <= idx < len(dataset):
                print(f"Index must be between 0 and {len(dataset)-1}")
                continue

            # Get data
            mixture, target = dataset[idx]
            mixture = mixture.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)

            # Forward pass
            with torch.no_grad():
                start = time.perf_counter()
                output = model(mixture)
                end = time.perf_counter()
                loss = loss_function(output, target)

            print(f"\nIndex {idx}:")
            print(f"Inference time: {(end - start) * 1e3:.2f} ms")
            print(f"Loss: {loss.item():.6f}")

            mixture_np = to_complex(mixture.squeeze().cpu()).numpy()
            target_np = to_complex(target.squeeze().cpu()).numpy()
            output_np = to_complex(output.squeeze().cpu()).numpy()

            mixture_crc = None
            target_crc = None
            output_crc = None
            if demodulate and receiver:
                mixture_crc = check_crc(mixture_np, receiver)
                target_crc = check_crc(target_np, receiver)
                output_crc = check_crc(output_np, receiver)

                print(f"Mixture CRC: {mixture_crc}")
                print(f"Target CRC: {target_crc}")
                print(f"Output CRC: {output_crc}")

            plot_test_signals(
                mixture_np,
                target_np,
                output_np,
                idx,
                loss.item(),
                mixture_crc=mixture_crc,
                target_crc=target_crc,
                output_crc=output_crc,
            )

        except ValueError:
            print("Please enter a valid integer, 'r' for random, or 'q' to quit")
        except KeyboardInterrupt:
            break


def test_model_statistics(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    loss_function: callable,
    *,
    demodulate: bool = False,
    sample_rate: float = 4e6,
    target_idx: int = 1,  # (1 for BLE, 2 for IEEE 802.15.4)
) -> None:
    """Compute PDR, inference time and loss statistics"""
    model.eval()

    receiver = helper_set_receiver(target_idx, sample_rate) if demodulate else None

    # Track statistics
    inference_times = []
    losses = []
    crc_counts: dict = {"mixture": [], "target": [], "output": []}

    print(f"Testing on {len(dataset)} packets")
    for idx in tqdm(
        range(len(dataset)),
        total=len(dataset),
        mininterval=1.0,
        leave=True,
        disable=not sys.stdout.isatty(),  # Disable tqdm for Slurm jobs
    ):
        # Get data
        mixture, target = dataset[idx]
        mixture = mixture.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)

        # Forward pass
        with torch.no_grad():
            start_time = time.perf_counter()
            output = model(mixture)
            end_time = time.perf_counter()
            loss = loss_function(output, target)

        inference_times.append((end_time - start_time) * 1e3)  # ms
        losses.append(loss.item())

        # Check CRC
        if demodulate and receiver:
            mixture_np = to_complex(mixture.squeeze().cpu()).numpy()
            target_np = to_complex(target.squeeze().cpu()).numpy()
            output_np = to_complex(output.squeeze().cpu()).numpy()

            crc_counts["mixture"].append(check_crc(mixture_np, receiver))
            crc_counts["target"].append(check_crc(target_np, receiver))
            crc_counts["output"].append(check_crc(output_np, receiver))

    # Compute statistics
    print(
        f"\nAverage inference time: {np.mean(inference_times):.2f} ± {np.std(inference_times):.2f} ms"
    )
    print(f"Average SI-SNR loss: {np.mean(losses):.6f} ± {np.std(losses):.6f} dB")

    if demodulate:
        print(f"\nPacket Delivery Rates:")
        for signal_type in ["target", "mixture", "output"]:
            passes = crc_counts[signal_type].count("PASS")
            total = len(crc_counts[signal_type])
            print(
                f"  {signal_type.capitalize():8s}: {passes/total*100:6.2f}% ({passes}/{total})"
            )

        # Model performance where target passes
        valid_cases = 0  # Rx can demodulate target/groundtruth
        model_success = 0
        model_improvement = 0  # Output CRC passes but mixture fails
        successful_losses = []
        failed_losses = []

        for target_crc, mixture_crc, output_crc, val_loss in zip(
            crc_counts["target"], crc_counts["mixture"], crc_counts["output"], losses
        ):
            if target_crc == "PASS":
                valid_cases += 1
                if output_crc == "PASS":
                    model_success += 1
                    successful_losses.append(val_loss)
                else:
                    failed_losses.append(val_loss)
                if mixture_crc != "PASS" and output_crc == "PASS":
                    model_improvement += 1

        if valid_cases > 0:
            print(f"\nModel Performance (on {valid_cases} valid target packets):")
            print(f"  Success rate: {(model_success/valid_cases)*100:.2f}%")
            if successful_losses:
                print(
                    f"  Successful output loss: {np.mean(successful_losses):.6f} ± {np.std(successful_losses):.6f} dB"
                )
            if failed_losses:
                print(
                    f"  Failed output loss: {np.mean(failed_losses):.6f} ± {np.std(failed_losses):.6f} dB"
                )
            print(f"  Packets improved: {model_improvement/valid_cases*100:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TDCR Network Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["complextdcr", "tdcr", "tcnconformer", "tcnconformer2"],
        default="tdcr",
        help="Type of model to evaluate",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["complex64", "float32"],
        default="float32",
        help="Data type for model parameters and operations",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./data/simulated_dataset.pt",
        help="Path to dataset file (.pt)",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=1,
        help="Target index (1 for BLE, 2 for IEEE 802.15.4)",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./checkpoints/last_checkpoint.pth",
        help="Checkpoint file (.pth)",
    )
    parser.add_argument(
        "--model_param_M",
        type=int,
        default=128,
        help="Middle channels in the complex encoder",
    )
    parser.add_argument(
        "--model_param_N",
        type=int,
        default=32,
        help="Out channels of encoder and input to LSTM = H",
    )
    parser.add_argument(
        "--model_param_U",
        type=int,
        default=128,
        help="Middle channels in complex dilated convolution",
    )
    parser.add_argument(
        "--model_param_V",
        type=int,
        default=8,
        help="Dilated convolutions on each side of the LSTM",
    )
    parser.add_argument(
        "--model_param_conformer_num_heads",
        type=int,
        default=4,
        help="Conformer num heads for TCNConformer",
    )
    parser.add_argument(
        "--model_param_conformer_ffn_times_input",
        type=int,
        default=2,
        help="Conformer FFN multiplier for TCNConformer",
    )
    parser.add_argument(
        "--model_param_conformer_num_layers",
        type=int,
        default=2,
        help="Conformer num layers for TCNConformer",
    )
    parser.add_argument(
        "--model_param_conformer_conv_kernel_size",
        type=int,
        default=15,
        help="Conformer conv kernel size for TCNConformer",
    )
    parser.add_argument(
        "--runtime_gen",
        action="store_true",
        help="Generate data on the fly instead of loading from file",
    )
    parser.add_argument(
        "--demodulate",
        action="store_true",
        help="Demodulate and check CRC of each signal",
    )
    parser.add_argument(
        "--sample_rate",
        type=float,
        default=4e6,
        help="Sample rate for receiver demodulation",
    )
    parser.add_argument(
        "--statistics",
        action="store_true",
        help="Run statistics test instead of interactive mode",
    )
    args = parser.parse_args()

    dtype_map: dict = {
        "complex64": torch.complex64,
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.dtype]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset = LoadDataset(
        target_idx=args.target,
        runtime_generation=args.runtime_gen,
        generator_class=SignalDatasetGenerator(SimulationConfig()),
        dataset_path=args.dataset_path,
        return_real=(args.dtype not in ("complex32", "complex64", "complex128")),
    )
    print(f"Loaded dataset with {len(dataset)} examples")

    model_params = {
        k.replace("model_param_", ""): v
        for k, v in vars(args).items()
        if k.startswith("model_param_")
    }

    # Initialise model
    model = ModelFactory.create_model(args.model_type, model_params, dtype, device)
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = sum(p.element_size() * p.nelement() for p in model.parameters())
    print(f"Trainable parameters: {total_params:,}")
    print(f"Total Size: {total_memory:,} bytes")

    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            model.load_state_dict(checkpoint["model_state"])
            if "epoch" in checkpoint:
                print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
            else:
                print(f"Loaded checkpoint weights")
        else:
            print(f"Loaded pretrained weights")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    if args.statistics:
        test_model_statistics(
            model=model,
            dataset=dataset,
            device=device,
            loss_function=si_snr_loss_complex,
            demodulate=args.demodulate,
            sample_rate=args.sample_rate,
            target_idx=args.target,
        )
    else:
        test_model_interactive(
            model=model,
            dataset=dataset,
            device=device,
            loss_function=si_snr_loss_complex,
            demodulate=args.demodulate,
            sample_rate=args.sample_rate,
            target_idx=args.target,
        )
