import torch
import hydra
from omegaconf import DictConfig
import wandb

from tl_2d3d.data.make_dataset import make_dataloaders
from tl_2d3d.utils import get_device, set_seed
import medpy.metric as metric

from monai.transforms import BatchInverseTransform

def test(
    model_path: str,
    config: DictConfig
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """

    set_seed(seed = config.hyperparameters.seed)
    device = get_device()

    # Load test dataloader
    _, _, test_dataloader = make_dataloaders(config, use_dataset_a=config.data.use_dataset_a)

    # Load model
    model = model.load_state_dict(torch.load(model_path))
    model.to(device)

    inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)
    
    # Initialize logging
    wandb.init(
        project=config.wandb.project_name,
        entity=config.wandb.team_name,
        name=f"Test_{config.wandb.run_name}",
        config=config,
        mode=config.wandb.mode
    )

    # Metrics
    dice_score = 0.0
    hd95_score = 0.0

    # Evaluate
    model.eval()
    for batch_num, batch in enumerate(test_dataloader):
        x = batch['image'].to(device).squeeze(dim=-1)
        y = batch['label'].to(device).squeeze(dim=-1)

        with torch.no_grad():
            y_pred = model(x)
            # use inferer: prediction = prediction = inferer(inputs=x, network=model)
            # inverse the transformations as in week 3

            dice = metric.dc(y_pred.argmax(dim=1), y.argmax(dim=1))  # Dice score
            if torch.count_nonzero(y_pred.argmax(dim=1)) and torch.count_nonzero(y.argmax(dim=1)):
                hd95 = metric.binary.hd95(y_pred.argmax(dim=1), y.argmax(dim=1), voxelspacing=config.data.voxel_dims) 
            else:
                hd95 = 0
                
            dice_score += dice
            hd95_score += hd95
            
            # logging
            wandb.log({
                "batch": batch_num,
                "dice_score": dice,
                "hd95_score": hd95, 
            })
        
    # Average scores and log final metrics
    final_dice = dice_score / len(test_dataloader)
    final_hd95 = hd95_score / len(test_dataloader)

    wandb.log({
        "final_dice_score": final_dice,
        "final_hd95_score": final_hd95
    })
    
    print(f"Final Test Metrics | Dice Score: {final_dice:.3f} | HD95 Score: {final_hd95:.3f}")

    wandb.finish()


if __name__ == "__main__":
    test()
    