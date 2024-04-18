from pathlib import Path

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