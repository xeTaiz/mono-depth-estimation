from argparse import ArgumentParser
import random
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from module import DepthEstimation

if __name__=='__main__':
    parser = ArgumentParser('Trains Scene Graph GAN')
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--n_worker', default=1, type=int, help='Number of workers')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=50, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=150, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--cache', action='store_true', help='Cache images')
    parser.add_argument('--path', default="D:/Documents/data/Structured3D/Structured3D", type=str, help="Path to Structured3D dataset")
    parser = DepthEstimation.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)

    # Setup Model, Logger, Trainer
    model = DepthEstimation(hparams=args)

    trainer = pl.Trainer.from_argparse_args(args,
        log_gpu_memory=False,
        fast_dev_run=args.dev,
        profiler=True,
        gpus=1,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision,
        amp_level='O2',
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.TensorBoardLogger("logs", name="unet")
    )

    # Fit model
    trainer.fit(model)
