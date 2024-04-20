from pathlib import Path

from monai.data import DataLoader, Dataset
from torch.utils.data import random_split
from omegaconf import DictConfig
from monai import transforms

def make_dataloaders(config: DictConfig) -> tuple[DataLoader]:
    """ Create train,val,test dataloader """

    transforms = transforms.Compose([
        transforms.LoadImaged(keys=['image', 'label']),
        transforms.EnsureChannelFirstd(keys=['image', 'label']),
        # transforms.Spacingd(keys=['image', 'label'], pixdim=config.data.voxel_dims, mode=["bilinear", "nearest"]),
        transforms.ResizeWithPadOrCropd(keys=['image', 'label'], spatial_size=config.data.image_dims),
        transforms.RandSpatialCropd(keys=['image', 'label'], roi_size=[-1, -1, 1]),
        # transforms.SqueezeDimd(keys=['image', 'label'], dim=-1), # TODO: This doesn't work for some reason idk why, should investigate

        # TODO: (Light) data augmentation

        # transforms.AsDiscreted(keys=['label'], to_onehot=2) # Convert "label" to onehot encoded.
    ])

    full_dataset = Dataset(
        data = read_dataset(dataset_name=config.data.dataset_name, path=config.data.dataset_path), 
        transform = transforms
    )
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, 
        lengths = [config.data.pct_train_split, config.data.pct_val_split, config.data.pct_test_split]
    )

    train_dataloader = DataLoader(train_dataset,batch_size = config.hyperparameters.batch_size, shuffle = True)
    val_dataloader   = DataLoader(val_dataset,  batch_size = 1, shuffle = False) 
    test_dataloader  = DataLoader(test_dataset, batch_size = 1, shuffle = False) 

    return train_dataloader, val_dataloader, test_dataloader


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


def make_dataloaders():
    pass

if __name__ == '__main__':
    # Get the data and process it
    pass