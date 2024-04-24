import torch
import torch.nn as nn
from monai.networks.nets import DynUNet
from omegaconf import DictConfig

def make_model(config: DictConfig, device: str) -> nn.Module:
    model = DynUNet(
        spatial_dims = config.model.num_dimensions,   # 2 for 2D convolutions, 3 for 3D convolutions
        in_channels  = 1,   # cant be bothered to put a config var for this
        out_channels = config.data.n_classes,
        kernel_size  = config.model.kernel_size,
        strides      = config.model.strides, 
        upsample_kernel_size = config.model.upsample_kernel_size
    )

    # Load weights
    if config.model.path_to_weights:
        print(f"Loading weights ({config.model.path_to_weights}) on to model")
        if device == 'cpu':
            model.load_state_dict(torch.load(config.model.path_to_weights, map_location=device))
        else:
            model.load_state_dict(torch.load(config.model.path_to_weights))
    else:
        print(f"No weights loaded, training from scratch")

    return model.to(device)