import random

import torch
import monai
import numpy as np


def get_device(verbose: bool = True) -> str:
    """ TODO: Should probably be more complex for multiple GPUs etc. """
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if verbose:
        print(f"Training on {device}")
    return device


def set_seed(seed: int) -> None:
    """ Set the seed for determinism """ 
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    monai.utils.misc.set_determinism(seed, use_deterministic_algorithms=True)
    torch.use_deterministic_algorithms(True)


def count_parameters(model): 
    """ Get the number of params in a model. See: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)