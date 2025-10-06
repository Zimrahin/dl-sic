import os
import argparse
from utils.training import TrainingLogger


def plot_training_log(log_path: str = "./checkpoints"):
    if not os.path.exists(log_path):
        print(f"No training log found at {log_path}")
        return

    logger = TrainingLogger(log_path, read_only=True)
    logger._load_log()
    logger.plot_losses()

    print(f"Training run started at: {logger.log_data['start_time']}")
    print(f"Total epochs: {len(logger.log_data['epochs'])}")
    if logger.log_data["epochs"]:
        print(f"Final training loss: {logger.log_data['train_loss'][-1]:.6f}")
        print(f"Final validation loss: {logger.log_data['val_loss'][-1]:.6f}")
        print(f"Best validation loss: {logger.log_data['best_val_loss']:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot Training Log")
    parser.add_argument(
        "--log_path",
        type=str,
        default="./checkpoints",
        help="Path to the training log directory",
    )
    args = parser.parse_args()
    plot_training_log(log_path=args.log_path)
