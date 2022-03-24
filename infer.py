import sys
from modules import get_module, register_module_specific_arguments
from datasets import register_dataset_specific_arguments
import torch
import pytorch_lightning as pl
from train import parse_args_into_namespaces
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import torchvision.transforms.functional as TF

import numpy as np

class NpyDataset(torch.utils.data.Dataset):
    def __init__(self, path):
        self.path = Path(path)
        self.items = self.path.rglob('*.npy')

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        item = np.load(self.items[i])
        rgb = TF.to_tensor(item['rgb'])
        gt = TF.to_tensor(item['gt'])
        return rgb, gt

if __name__ == "__main__":
    parser = ArgumentParser('Trains mono depth estimation models', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'log10', 'rmse'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')
    parser.add_argument('--inputs', type=str, default=None, help='Path to .npy files')

    commands = parser.add_subparsers(title='Commands')
    register_dataset_specific_arguments(commands)
    register_module_specific_arguments(commands)
    
    args = parse_args_into_namespaces(parser, commands)
    assert args.test, "Please provide test dataset"
    
    # windows safe
    if sys.platform in ["win32"]:
        args.globals.worker = 0

    use_gpu = not args.globals.gpus == 0

    pred_path = Path(str(Path(args.method.ckpt).parent).replace('checkpoints', 'predictions'))
    pred_path.mkdir(parents=True, exist_ok=True)


    trainer = pl.Trainer(
        gpus=args.globals.gpus,
        logger=pl.loggers.WandbLogger(project="stdepth", log_model=False, offline=True)
    )

    # Fit model
    module = get_module(args)
    module.pred_path = pred_path
    if args.inputs:
        ds = NpyDataset(args.inputs)
        dl = torch.utils.data.Dataloader(ds, batch_size=1)
        trainer.test(module, dl)
    else:
        result = trainer.test(module, verbose=False)
        for key, value in result[0].items():
            if "_epoch" in key:
                print("{}: {}".format(key.replace("_epoch", ""), round(value, 3)))
