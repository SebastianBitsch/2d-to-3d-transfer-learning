import torch.nn as nn
from monai.networks.nets import DynUNet
from omegaconf import DictConfig

def make_model(config: DictConfig, device: str) -> nn.Module:
    model = DynUNet(
        spatial_dims = 2,   # 2 for 2D convolutions, 3 for 3D convolutions
        in_channels  = 1,   # Number of input channels/modalities (3 for RGB)
        out_channels = 4,   # Number of classes, including background
        kernel_size  = [3, 3, 3, 3, 3, 3], # Size of the filters
        strides      = [1, 2, 2, 2, 2, 2],
        upsample_kernel_size = [2, 2, 2, 2, 2]
    )

    return model.to(device)