import os
import random

import torch
import monai
import medpy.metric as metric
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig


def save_model(model:nn.Module, iteration_num:int, config:DictConfig, verbose:bool = True) -> None:
    """ Save a model checkpoint to a directory. Will save on the format /save_dir/exp1/exp1_1000iters.pt """
    dir_path = f"{config.base.save_location}/{config.base.experiment_name}/"
    file_path = dir_path + f"{config.base.experiment_name}_{iteration_num}iters.pt"

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if verbose:
        print(f"Saving model to {file_path}")
    
    torch.save(model, file_path)

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


def hd95(y:torch.Tensor, y_pred:torch.Tensor, config:DictConfig) -> float:
    if torch.all(y_pred == 0) or torch.all(y == 0):
        # Both arrays must contain binary objects, Runtime error if not
        return 0.0

    # TODO: Maybe there is a better way at handling 2D vs 3D
    if config.model.num_dimensions == 2:
        return metric.binary.hd95(y_pred, y, voxelspacing = config.data.voxel_dims) 
    else:
        return metric.binary.hd95(y_pred.squeeze(), y.squeeze(), voxelspacing = config.data.voxel_dims) # Shape y.squeeze(): [1, 256, 256, 32]

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
