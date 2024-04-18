import torch
import wandb
import hydra
import logging
from itertools import chain

from omegaconf import DictConfig
from hydra.utils import to_absolute_path
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import make_grid

import monai
from monai.data import DataLoader, Dataset
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms import BatchInverseTransform
from monai.networks.nets import DynUNet


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def train(config: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Training on {device}")

    wandb.init(
        project = config.wandb.project_name,
        config = {
            "architecture": model.name,
            "dataset": "Custom-1",
            "learning_rate": config.hyperparameters.learning_rate,
            "epochs": config.hyperparameters.epochs,
            "batch_size" : config.hyperparameters.batch_size,
        },
        mode = config.wandb.mode
    )

    model = ...

    train_dataloader, val_dataloader, test_dataloader = ...

    loss_fn = monai.losses.DiceLoss(softmax=True, to_onehot_y=False) # Apply "softmax" to the output of the network and don't convert to onehot because this is done already by the transforms.
    optimizer = torch.optim.Adam(model.parameters(), lr = config.hyperparameters.learning_rate)
    inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)


    wandb.init(
        project = config.wandb.project_name,
        config = {
            "architecture": model.name,
            "dataset": "KiTS",
            "learning_rate": config.hyperparameters.learning_rate,
            "epochs": config.hyperparameters.epochs,
            "batch_size" : config.hyperparameters.batch_size,
        },
        mode = config.wandb.mode
    )

    iterations = 0
    for epoch in range(config.hyperparameters.epochs):
        print(f"**** Epoch {epoch+1}/{config.hyperparameters.epochs} ****")

        training_loss = 0.0
        validation_loss = 0.0

        # Train
        model.train()
        for batch_num, batch in enumerate(train_dataloader):
            x = batch['image'].to(device)
            y = batch['label'].to(device)

            optimizer.zero_grad()

            y_pred = model(x)

            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            
            training_loss += loss

            if (iterations % config.wandb.train_log_interval == config.wandb.train_log_interval - 1):
                print(f"{batch_num + 1}/{len(train_dataloader)} | loss: {loss:.3f}")
                wandb.log({
                    "epoch" : epoch,
                    "iteration" : iterations,
                    "batch/train" : batch_num,
                    "loss/train"  : training_loss.item() / (batch_num + 1),
                })
            
            iterations += 1


        
        # Validate
        model.eval()
        for batch_num, batch in enumerate(val_dataloader):
            x = batch['image'].to(device)
            y = batch['label'].to(device)

            with torch.no_grad():
                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                validation_loss += loss

            if (batch_num % config.wandb.validation_log_interval == config.wandb.validation_log_interval - 1):
                print(f"{batch_num + 1}/{len(val_dataloader)} | val loss: {loss.item():.3f}")
                wandb.log({
                    "epoch" : epoch,
                    "batch/val" : batch_num,
                    "loss/val": validation_loss.item() / (batch_num + 1),
                })


        
        # Logging
    wandb.finish()