import torch
from omegaconf import DictConfig
import monai
import matplotlib.pyplot as plt

from tl_2d3d.data.make_dataset_test import make_dataloaders
from tl_2d3d.utils import get_device, set_seed
from tl_2d3d.models.model import make_model
import medpy.metric as metric

from monai.transforms import BatchInverseTransform
from monai import allow_missing_keys_mode

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
    _, _, test_dataloader, test_transforms = make_dataloaders(config, use_dataset_a=config.data.use_dataset_a)

    # Load model
    model = make_model(config, device = device)
    model = model.load_state_dict(torch.load(model_path))
    model.to(device)

    inferer = monai.inferers.SliceInferer(roi_size=[-1, -1], spatial_dim=2, sw_batch_size=1)

    # Evaluate
    model.eval()
    test_dices = []
    test_hd95 = []
    for batch_num, batch in enumerate(test_dataloader):
        x = batch['image'].to(device).squeeze(dim=-1)
        y = batch['label'].to(device).squeeze(dim=-1)

        with torch.no_grad():
            prediction = inferer(inputs=x, model=model)
        
        batch_inverter = BatchInverseTransform(test_transforms, test_dataloader)
        with allow_missing_keys_mode(test_transforms):
            inversed_prediction = batch_inverter({'label': prediction})
            inversed_targets = batch_inverter({'label': y})
        
        inversed_prediction = [monai.transforms.AsDiscrete(argmax=True)(pred['label']) for pred in inversed_prediction]
        
        for b in range(prediction.shape[0]):
            test_dices.append(metric.dc( 1*(inversed_prediction[b]),
                                       1*(inversed_targets[b]['label']) ) )
            test_hd95.append(metric.binary.hd95( 1*(inversed_prediction[b]),
                                       1*(inversed_targets[b]['label']), voxelspacing=config.data.voxel_dims))


        
    # Average scores and log final metrics
    final_dice = sum(test_dices) / len(test_dataloader)
    final_hd95 = sum(test_hd95) / len(test_dataloader)
    
    print(f"Final Test Metrics | Dice Score: {final_dice:.3f} | HD95 Score: {final_hd95:.3f}")

    fig, axes = plt.subplots(4, 4, figsize=(12, 12))

    slices = [1, 2, 3, 4]
    inputs_cpu = x.cpu().detach().numpy()
    inversed_prediction_cpu = inversed_prediction[0].cpu().detach().numpy()
    inversed_targets_cpu = inversed_targets[0]['label'].cpu().detach().numpy()

    for i, sl in enumerate(slices):
        # Image slices
        axes[0, i].imshow(inputs_cpu[0,0, :, :, sl], cmap="gray")
        axes[0, i].axis('off')
        # Predictions by our trained model
        axes[1, i].imshow(inversed_prediction_cpu[0, :, :, sl], cmap="gray")
        axes[1, i].axis('off')
        # Ground truth
        axes[2, i].imshow(inversed_targets_cpu[0, :, :, sl], cmap="gray")
        axes[2, i].axis('off')
        # Error maps
        axes[3, i].imshow(inversed_targets_cpu[0, :, :, sl] != inversed_prediction_cpu[0, :, :, sl], cmap="gray")
        axes[3, i].axis('off')
    fig.suptitle(f"Dice Score: {final_dice:.3f} | HD95 Score: {final_hd95:.3f}", fontsize=16)
    fig.savefig('predictions.png')    


if __name__ == "__main__":
    test()
    