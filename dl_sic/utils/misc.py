import random
import numpy as np
import torch
import sys

sys.path.append("..")


def set_seed(seed: int) -> None:
    """Set seed for reproducibility"""
    # docs.pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
