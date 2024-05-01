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
    if config.model.continue_from: # This is so bad.. 
        # Terrible way of getting the name on the form: /work3/s204163/3dimaging_finalproject/weights/baseline2d_100/baseline2d_final.pt
        exp_name = config.model.continue_from.split("_")[0]
        full_path = f"{config.model.path_to_weights}{exp_name}_{config.hyperparameters.seed}/{config.model.continue_from}"
        print(f"Loading weights ({full_path}) on to model")

        saved_model = torch.load(full_path)
        if saved_model.spatial_dims == model.spatial_dims:
            model.load_state_dict(saved_model.state_dict())
        else:
            print(f"Loading weights from a {saved_model.spatial_dims}D model onto a {model.spatial_dims}D model")
            if config.model.use_modulewise_init:
                init_normal_per_module(from_module=saved_model, to_module=model)
            elif config.model.use_channelwise_init:
                init_normal_per_channel(from_module=saved_model, to_module=model)
            else:
                raise NotImplementedError("Error: Loading 2D weights onto a 3D model but method for interpolating weights was specified")

    else:
        print(f"No weights loaded, training from scratch")

    return model.to(device)


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


def init_normal_per_channel(from_module: nn.Module, to_module: nn.Module) -> None:
    """
    Initialize the weights of a model (to_module) using the normal distributions per conv channel in another module (from_module).
    The function will take the weights in an entire module i.e. shape [128, 64, 3, 3] and compute a single number mean and std with 
    shape [128, 64, 1], and tile it to shape [128, 64, 3, 3, 3]
    
    Args:
        from_module (nn.Module): Source module providing weights for initialization.
        to_module (nn.Module): Target module whose weights will be initialized.
    
    Note:
        The modules should be -very- similar as this functions makes the assumption the models have the same layers (either 2D or 3D) in the same order.
    """
    assert len(list(from_module.modules())) == len(list(to_module.modules())), "Error: Models should contain the 'same' layers"

    for from_mod, to_mod in zip(from_module.modules(), to_module.modules()):

        if isinstance(from_mod, (nn.Conv2d, nn.ConvTranspose2d)):

            # Handle the 'complicated' case where shapes dont match, for instance (512, 256, 3, 3) and (320, 256, 3, 3, 3).
            # TODO: We could try something more complicated, e.g. interpolation, random crop, tiling etc. for now we just take a simple crop
            if from_mod.weight.shape != to_mod.weight.shape[:4]:
                from_mod.weight.data = from_mod.weight.data[:to_mod.weight.shape[0], :to_mod.weight.shape[1]]

            # This ungodly line takes the mean of the 2D conv (3,3) and then expands the single number into a 3D convolution, i.e. [32, 64] -> [32, 64, 3, 3, 3]
            # TODO: I cant find a nicer way than the triple unsqueeze, though i am sure there is a nicer / more general way of doing it
            mean = torch.mean(from_mod.weight.data, axis = (-1, -2)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(to_mod.weight.data.shape)
            std  = torch.std(from_mod.weight.data, axis = (-1, -2), unbiased=False).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(to_mod.weight.data.shape) # Unbiased, see here: https://github.com/pytorch/pytorch/issues/29372
            
            to_mod.weight.data = torch.normal(mean, std = std)
                        
        elif isinstance(from_mod, nn.LeakyReLU):
            to_mod.negative_slope = from_mod.negative_slope

        elif isinstance(from_mod, nn.InstanceNorm2d):
            # TODO: This weight type is kinda weird, from_mod: torch.Size([32]) to_mod: torch.Size([32, 32, 3, 3, 3])
            # Dont know how to handle that ...
            pass