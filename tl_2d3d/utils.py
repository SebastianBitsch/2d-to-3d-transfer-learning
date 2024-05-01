import os
import random

import torch
import monai
import medpy.metric as metric
import numpy as np
import torch.nn as nn
from omegaconf import DictConfig


def save_model(model:nn.Module, iteration_num:int, config:DictConfig, final:bool = False, verbose:bool = True) -> None:
    """ Save a model checkpoint to a directory. Will save on the format /save_dir/exp1_seed/exp1_1000iters.pt """
    dir_path = f"{config.base.save_location}/{config.base.experiment_name}_{config.hyperparameters.seed}/"
    file_path = dir_path + f"{config.base.experiment_name}_{iteration_num}iters.pt"
    if final:
        file_path = dir_path + f"{config.base.experiment_name}_final.pt"

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

    assert len(config.data.voxel_dims) == config.model.num_dimensions
    return metric.binary.hd95(y_pred.squeeze(), y.squeeze(), voxelspacing = config.data.voxel_dims) 

def count_parameters(model): 
    """ Get the number of params in a model. See: https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
