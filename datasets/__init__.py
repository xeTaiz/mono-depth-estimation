from datasets import nyu_dataloader, floorplan3d_dataloader, structured3d_dataset, stdepth, stdepth_multi, stdepth_multi2

def register_dataset_specific_arguments(subparser):
    floorplan3d_dataloader.Floorplan3DDataset.add_dataset_specific_args(subparser)
    nyu_dataloader.NYUDataset.add_dataset_specific_args(subparser)
    structured3d_dataset.Structured3DDataset.add_dataset_specific_args(subparser)
    stdepth.SemiTransparentDepthDataset.add_dataset_specific_args(subparser)
    stdepth_multi.SemiTransparentMultiDepthDataset.add_dataset_specific_args(subparser)
    stdepth_multi2.SemiTransparentMulti2DepthDataset.add_dataset_specific_args(subparser)
