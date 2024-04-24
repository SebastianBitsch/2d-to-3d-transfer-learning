import random

import torch
import monai
import numpy as np
import torch.nn as nn


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


def init_normal_per_module(from_module: nn.Module, to_module: nn.Module) -> None:
    """
    Initialize the weights of a model (to_module) using the normal distributions of the layers in another module (from_module).
    The function will take the weights in an entire module i.e. shape [128, 64, 3, 3] and compute a single number mean and std to 
    init the entire layer in another module.
    
    Args:
        from_module (nn.Module): Source module providing weights for initialization.
        to_module (nn.Module): Target module whose weights will be initialized.
    
    Note:
        The modules should be -very- similar as this functions makes the assumption the models have the same layers (either 2D or 3D) in the same order.
    """
    assert len(list(from_module.modules())) == len(list(to_module.modules())), "Error: Models should contain the 'same' layers"

    for from_mod, to_mod in zip(from_module.modules(), to_module.modules()):

        if isinstance(from_mod, (nn.Conv2d, nn.InstanceNorm2d, nn.ConvTranspose2d)):
            # Get distributions across the entire module, i.e. a single number for mean and std
            mean = torch.mean(from_mod.weight.data) 
            std  = torch.std(from_mod.weight.data)
            
            to_mod.weight.data.normal_(mean, std)

        elif isinstance(from_mod, nn.LeakyReLU):
            to_mod.negative_slope = from_mod.negative_slope
