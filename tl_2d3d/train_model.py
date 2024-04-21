import time
import logging
from itertools import chain

import torch
import wandb
import hydra
import monai

from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid

from monai.data import DataLoader, Dataset
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms import BatchInverseTransform
from monai.networks.nets import DynUNet
import medpy.metric as metric

from tl_2d3d.data.make_dataset import make_dataloaders
from tl_2d3d.utils import get_device, set_seed
from tl_2d3d.models.model import make_model

@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    set_seed(seed = config.hyperparameters.seed)
    device = get_device()

    # Make model and dataloders
    model = make_model(config, device = device)
    train_dataloader, val_dataloader, test_dataloader = make_dataloaders(config)

    # 
    loss_fn = monai.losses.DiceLoss(softmax=True, to_onehot_y=False) # Apply "softmax" to the output of the network and don't convert to onehot because this is done already by the transforms.
    optimizer = torch.optim.Adam(model.parameters(), lr = config.hyperparameters.learning_rate)
    inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)

    # Initialize logging
    wandb.init(
        project = config.wandb.project_name,
        entity = config.wandb.team_name,
        name = config.wandb.run_name,
        config = {
            "architecture": model.__class__.__name__,
            "optimizer" : optimizer.__class__.__name__,
            "loss_fn" : loss_fn.__class__.__name__,
            "inferer" : inferer.__class__.__name__,
            "dataset": "KiTS",
            "learning_rate": config.hyperparameters.learning_rate,
            "max_iterations": config.hyperparameters.max_num_iterations,
            "epochs": config.hyperparameters.epochs,
            "batch_size" : config.hyperparameters.batch_size,
        },
        mode = config.wandb.mode
    )

    iteration_num = 0
    total_training_time = 0
    for epoch_num in range(config.hyperparameters.epochs):
        print(f"**** Epoch {epoch_num+1}/{config.hyperparameters.epochs} | Iterations {iteration_num + 1}/{config.hyperparameters.max_num_iterations} ****")

        epoch_start_time = time.time()
        training_loss = 0.0
        validation_loss = 0.0
        dice_score = 0.0

        # Train
        model.train()
        for batch_num, batch in enumerate(train_dataloader):
            x = batch['image'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error
            y = batch['label'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error
            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            training_loss += loss

            if (iteration_num % config.wandb.train_log_interval == config.wandb.train_log_interval - 1):
                print(f"{batch_num + 1}/{len(train_dataloader)} | loss: {loss:.3f}")
                wandb.log({
                    "epoch" : epoch_num,
                    "iteration" : iteration_num,
                    "batch/train" : batch_num,
                    "loss/train"  : training_loss.item() / (batch_num + 1),
                })
            
            iteration_num += 1

        total_training_time += time.time() - epoch_start_time
        
        # Check for termination criteria
        if config.hyperparameters.max_num_iterations <= iteration_num or config.hyperparameters.max_training_time <= total_training_time:
            break

        # Validate
        model.eval()
        for batch_num, batch in enumerate(val_dataloader):
            x = batch['image'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error
            y = batch['label'].to(device).squeeze(dim = -1) # TODO: This squeeze has to be here because monai.transforms.SqueezeDimd gives error

            with torch.no_grad():
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                validation_loss += loss

                print(y_pred.shape, y.shape)
                dice_score += metric.dc(y_pred.argmax(dim=1), y.argmax(dim=1)) # Not elegant, but ok

            if (batch_num % config.wandb.validation_log_interval == config.wandb.validation_log_interval - 1):
                print(f"{batch_num + 1}/{len(val_dataloader)} | val loss: {validation_loss.item() / (batch_num + 1):.3f} | dice: {dice_score / (batch_num + 1):.3f}")
                wandb.log({
                    "epoch" : epoch_num,
                    "batch/val" : batch_num,
                    "loss/val": validation_loss.item() / (batch_num + 1),
                    "score/dice": dice_score / (batch_num + 1),
                })


        # Check for termination criteria
        if config.hyperparameters.max_num_iterations <= iteration_num or config.hyperparameters.max_training_time <= total_training_time:
            break
        
        # Logging
    wandb.finish()


if __name__ == "__main__":
    train()