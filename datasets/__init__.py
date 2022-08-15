from datasets import nyu_dataloader, floorplan3d_dataloader, structured3d_dataset, mirror3d

def register_dataset_specific_arguments(subparser):
    floorplan3d_dataloader.Floorplan3DDataset.add_dataset_specific_args(subparser)
    nyu_dataloader.NYUDataset.add_dataset_specific_args(subparser)
    structured3d_dataset.Structured3DDataset.add_dataset_specific_args(subparser)
    mirror3d.Mirror3DDataset.add_dataset_specific_args(subparser)

