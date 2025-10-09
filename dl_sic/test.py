import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from model.complex_tdcr_net import ComplexTDCRnet
from model.real_tdcr_net import RealTDCRnet
from utils.dataset import LoadDataset
from utils.loss_functions import si_snr_loss_complex


def plot_test_signals(
    mixture: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
    index: int,
    loss_val: float,
) -> None:
    """Plot mixture, target and output"""
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(8, 6))

    # Convert to numpy if they're tensors
    if torch.is_tensor(mixture):
        mixture = mixture.cpu().numpy()
        target = target.cpu().numpy()
        output = output.cpu().numpy()

    sample_axis = np.arange(len(mixture))

    # Plot mixture
    ax1.plot(sample_axis, np.real(mixture), alpha=0.7, label="Real")
    ax1.plot(sample_axis, np.imag(mixture), alpha=0.7, label="Imag")
    ax1.set_ylabel("Amplitude")
    ax1.set_title(f"Input Mixture (Index {index})")
    ax1.legend()
    ax1.grid(True)

    # Plot target
    ax2.plot(sample_axis, np.real(target), alpha=0.7, label="Real")
    ax2.plot(sample_axis, np.imag(target), alpha=0.7, label="Imag")
    ax2.set_ylabel("Amplitude")
    ax2.set_title("Target Signal")
    ax2.legend()
    ax2.grid(True)

    # Plot output
    ax3.plot(sample_axis, np.real(output), alpha=0.7, label="Real")
    ax3.plot(sample_axis, np.imag(output), alpha=0.7, label="Imag")
    ax3.set_ylabel("Amplitude")
    ax3.set_title(f"Model Output (Loss: {loss_val:.6f} dB)")
    ax3.set_xlabel("Sample Index")
    ax3.legend()
    ax3.grid(True)

    # max_all = np.max(np.abs(np.concatenate([mixture, target, output])))
    # ax1.set_ylim(-max_all - 0.1, max_all + 0.1)
    # ax2.set_ylim(-max_all - 0.1, max_all + 0.1)
    # ax3.set_ylim(-max_all - 0.1, max_all + 0.1)

    plt.tight_layout()
    plt.show()


def test_model(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    loss_function: callable,
) -> None:
    """Interactive model testing loop"""
    model.eval()

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
                output = model(mixture)
                loss = loss_function(output, target)

            print(f"\nIndex {idx}:")
            print(f"Loss: {loss.item():.6f}")
            plot_test_signals(
                mixture.squeeze().cpu(),
                target.squeeze().cpu(),
                output.squeeze().cpu(),
                idx,
                loss.item(),
            )

        except ValueError:
            print("Please enter a valid integer, 'r' for random, or 'q' to quit")
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CTDCR Network Test")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["complex", "real"],
        default="complex",
        help="Type of model: complex (complex arithmetic) or real (independent channels)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["complex64", "float32", "float16", "bfloat16"],
        default="complex64",
        help="Data type for model parameters and operations",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="../data/simulated_dataset.pt",
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

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    dataset = LoadDataset(args.dataset_path, target_idx=args.target)
    print(f"Loaded dataset with {len(dataset)} examples")

    M, N, U, V = (
        args.model_param_M,
        args.model_param_N,
        args.model_param_U,
        args.model_param_V,
    )  # TDCR net parameters

    # Initialise model
    if args.model_type == "complex":
        model = ComplexTDCRnet(M, N, U, V, dtype=dtype).to(device)
        print("Using complex model (complex arithmetic)")
    else:
        if dtype in (torch.complex32, torch.complex64, torch.complex128):
            raise ValueError(f"Real model cannot use complex dtype {dtype}")
        model = RealTDCRnet(M, N, U, V, dtype=dtype).to(device)
        print("Using real model (independent real/imag channels)")
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

    # Start interactive testing
    test_model(
        model=model,
        dataset=dataset,
        device=device,
        loss_function=si_snr_loss_complex,
    )
