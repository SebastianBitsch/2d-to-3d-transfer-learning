from pathlib import Path

from monai.data import DataLoader, Dataset
from torch.utils.data import random_split
from omegaconf import DictConfig
import monai
import torch

def make_dataloaders(config: DictConfig, use_dataset_a: bool) -> tuple[DataLoader]:
    """
    Create train,val,test dataloader. 
    The dataset is split into a set A and B and a test set. 'use_dataset_a' bool gives either set 'a' or 'b'
    The a or b set is then split into train and val set.
    """

    transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['image', 'label']),
        monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
        # monai.transforms.Spacingd(keys=['image', 'label'], pixdim=config.data.voxel_dims, mode=["bilinear", "nearest"]),
        monai.transforms.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=config.data.image_dims),
        monai.transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=[-1, -1, 1]),
    ])

    test_transforms = monai.transforms.Compose([
        monai.transforms.LoadImaged(keys=['image', 'label']),
        monai.transforms.EnsureChannelFirstd(keys=['image', 'label']),
        # monai.transforms.Spacingd(keys=['image', 'label'], pixdim=config.data.voxel_dims, mode=["bilinear", "nearest"]),
        monai.transforms.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=config.data.image_dims),
        monai.transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=[-1, -1, 1]),
        # monai.transforms.SqueezeDimd(keys=['image', 'label'], dim=-1), # TODO: This doesn't work for some reason idk why, should investigate

        # TODO: (Light) data augmentation - and check they look ok
        monai.transforms.RandRotated(keys=['image', 'label'], range_z = 180, prob = 0.5, mode='nearest'),
        monai.transforms.RandGaussianSmoothd(keys=['image'], sigma_x=(0.25, 1.5), sigma_y=(0.25, 1.5), sigma_z=(0.25, 1.5), approx='erf', prob=0.5),
        monai.transforms.RandGaussianNoised(keys=['image'], prob=0.5, mean=0.0, std=0.5),

        monai.transforms.AsDiscreted(keys=['label'], to_onehot=config.data.n_classes)
    ])
    
    # Read the dataset
    data = read_dataset(dataset_name=config.data.dataset_name, path=config.data.dataset_path)
    
    # Split the dataset into A and B datasets
    indices = torch.randperm(len(data))
    ab_dataset_size = 0.5 * (1.0 - config.data.pct_test_split)
    a_dataset = Dataset(data[indices[:ab_dataset_size]], transform=transforms)
    b_dataset = Dataset(data[indices[ab_dataset_size:ab_dataset_size*2]], transform=transforms)
    test_dataset = Dataset(data[indices[2*ab_dataset_size:]], transform=test_transforms)
    main_dataset = a_dataset if use_dataset_a else b_dataset

    # Train/Val split
    train_dataset, val_dataset = random_split(main_dataset, lengths = [config.data.pct_train_split, config.data.pct_val_split])

    # Dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size = config.hyperparameters.batch_size, shuffle = True)
    val_dataloader   = DataLoader(val_dataset,   batch_size = 1, shuffle = False) 
    test_dataloader  = DataLoader(test_dataset,  batch_size = 1, shuffle = False) 

    return train_dataloader, val_dataloader, test_dataloader, test_transforms


def read_dataset(dataset_name: str, path: str) -> list[dict]:
    """ TODO: Not a very cool implementation """
    if dataset_name == 'adc':
        return read_kits_data(path)
    elif dataset_name == "kits":
        return read_kits_data(path)
    else:
        raise NotImplementedError("Only adc and kits datasets are supported")


def read_acd_data(dataset_path: str = "/dtu/3d-imaging-center/courses/02510/data/ACDC17") -> list[dict]:
    """ Assumes the data is in patient00X folders etc. """
    return [
        {
            'image': str(path),                                 # the unlabeled volume file
            'label': str(path).replace(".nii.gz", "_gt.nii.gz"),# all the labeled data has _gt in the path
            'id': path.name.split(".")[0]                       # the file name without ending 
        }
        for path in sorted(Path(dataset_path).glob("*/*frame[0-9][0-9].nii.gz"))
    ]


def read_kits_data(dataset_path: str = "/dtu/3d-imaging-center/courses/02510/data/KiTS19") -> list[dict]:
    """ Assumes the data has quite specific structure. """
    return [
        {
            'image': str(path),                                                         # the unlabeled volume file, 'imaging.nii.gz'
            'label': str(path).replace("imaging.nii.gz", "segmentation_kidney.nii.gz"), # all the labeled data is named 'segmentation_kidney.nii.gz'
            'id': path.parent.name                                                      # the dir name, e.g. 'case_00000'
        }
        for path in sorted(Path(dataset_path).glob("*/imaging.nii.gz"))
    ]



if __name__ == '__main__':
    # Get the data and process it
    pass