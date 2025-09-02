import os
import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

from model.ctdcr_net import CTDCR_net
from utils.dataset import LoadDataset
from utils.loss_functions import mse_loss_complex


def plot_test_signals(
    mixture: torch.Tensor,
    target: torch.Tensor,
    output: torch.Tensor,
    index: int,
    loss_val: float,
) -> None:
    """Plot mixture, target and output"""
    _, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 8))

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
    ax3.set_title(f"Model Output (Loss: {loss_val:.6f})")
    ax3.set_xlabel("Sample Index")
    ax3.legend()
    ax3.grid(True)

    max_all = np.max(np.abs(np.concatenate([mixture, target, output])))
    ax1.set_ylim(-max_all - 0.1, max_all + 0.1)
    ax2.set_ylim(-max_all - 0.1, max_all + 0.1)
    ax3.set_ylim(-max_all - 0.1, max_all + 0.1)

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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if not os.path.exists(args.dataset_path):
        raise FileNotFoundError(f"Dataset not found at {args.dataset_path}")
    dataset = LoadDataset(args.dataset_path, target_idx=args.target)
    print(f"Loaded dataset with {len(dataset)} examples")

    M, N, U, H, V = 128, 32, 128, 32, 8  # CTDCR net parameters
    model = CTDCR_net(M, N, U, H, V).to(device)  # Initialise model
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load checkpoint
    if os.path.exists(args.checkpoint_path):
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    else:
        raise FileNotFoundError(f"Checkpoint not found at {args.checkpoint_path}")

    # Start interactive testing
    test_model(
        model=model,
        dataset=dataset,
        device=device,
        loss_function=mse_loss_complex,
    )
