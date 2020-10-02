import pytorch_lightning as pl
from argparse import ArgumentParser
import random
from modules import midas

if __name__ == "__main__":
    parser = ArgumentParser('Trains mono depth estimation models')
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=50, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=150, type=int, help='Maximum number ob epochs to train')
    parser = midas.MidasModule.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # Manage Random Seed
    if args.seed is None: # Generate random seed if none is given
        args.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.seed)

    # Setup Model, Logger, Trainer
    model = midas.MidasModule(args)

    trainer = pl.Trainer.from_argparse_args(args,
        log_gpu_memory=False,
        fast_dev_run=args.dev,
        profiler=True,
        gpus=args.gpus,
        overfit_batches=1 if args.overfit else 0,
        precision=args.precision,
        amp_level='O2',
        min_epochs=args.min_epochs,
        max_epochs=args.max_epochs,
        logger=pl.loggers.TensorBoardLogger("result", name='midas')
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.seed,
            'gpu_name': torch.cuda.get_device_name(0),
            'gpu_capability': torch.cuda.get_device_capability(0)
            })
   
    if hasattr(trainer, 'logger'):
        trainer.logger.log_hyperparams(yaml) # Log random seed

    # Fit model
    trainer.fit(model)
