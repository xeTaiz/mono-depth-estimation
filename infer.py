import sys
from modules import get_module, register_module_specific_arguments
from datasets import register_dataset_specific_arguments
import torch
import pytorch_lightning as pl
from train import parse_args_into_namespaces
from pathlib import Path
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

if __name__ == "__main__":
    parser = ArgumentParser('Trains mono depth estimation models', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'log10', 'rmse'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')

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
    result = trainer.test(module, verbose=False)
    for key, value in result[0].items():
        if "_epoch" in key:
            print("{}: {}".format(key.replace("_epoch", ""), round(value, 3)))
