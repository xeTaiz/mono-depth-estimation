import sys
from modules import get_module, register_module_specific_arguments
from datasets import register_dataset_specific_arguments
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
import random
import torch

def parse_args_into_namespaces(parser, commands):        
    '''
    Split all command arguments (without prefix, like --) in
    own namespaces. Each command accepts extra options for
    configuration.
    Example: `add 2 mul 5 --repeat 3` could be used to a sequencial
                addition of 2, then multiply with 5 repeated 3 times.
    '''

    # Divide argv by commands
    split_argv = [[]]
    for c in sys.argv[1:]:
        if c in commands.choices:
            split_argv.append([c])
        else:
            split_argv[-1].append(c)

    # Globals arguments without commands
    args = Namespace()
    cmd, args_raw = 'globals', split_argv.pop(0)
    args_parsed = parser.parse_args(args_raw)
    setattr(args, cmd, args_parsed)

    # Split all commands to separate namespace
    pos = 0
    while len(split_argv):
        pos += 1
        cmd, *args_raw = split_argv.pop(0)
        assert cmd[0].isalpha(), 'Command must start with a letter.'
        args_parsed = commands.choices[cmd].parse_args(args_raw, namespace=Namespace())
        setattr(args, "method" if cmd in ['bts', 'eigen', 'vnl', 'dorn', 'midas', 'laina'] else cmd, args_parsed)
    assert hasattr(args, "method"), "Please provide the method you want to use: bts, eigen, vnl, dorn, midas, laina"
    assert any([hasattr(args, m) for m in ['nyu', 'structured3d', 'floorplan3d']]), "Please provide data set to use: nyu, floorplan3d, structured3d"
    # convert args in more usable format

    return args

if __name__ == "__main__":
    parser = ArgumentParser('Trains mono depth estimation models', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=50, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=150, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'log10', 'rmse'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers for data loader')



    commands = parser.add_subparsers(title='Commands')
    register_dataset_specific_arguments(commands)
    register_module_specific_arguments(commands)
    
    args = parse_args_into_namespaces(parser, commands)
    print(args.method)
    
    # windows safe
    if sys.platform in ["win32"]:
        args.globals.worker = 0

    # Manage Random Seed
    if args.globals.seed is None: # Generate random seed if none is given
        args.globals.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.globals.seed)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch}',
        verbose=True,
        save_weights_only=True,
        save_top_k=-1,
        monitor='val_checkpoint_on',
        mode='min'
    )

    trainer = pl.Trainer(
        log_gpu_memory=False,
        fast_dev_run=args.globals.dev,
        profiler=True,
        gpus=args.globals.gpus,
        overfit_batches=1 if args.globals.overfit else 0,
        precision=args.globals.precision if args.globals.gpus > 0 else 32,
        amp_level='O2' if args.globals.gpus > 0 else None,
        min_epochs=args.globals.min_epochs,
        max_epochs=args.globals.max_epochs,
        logger=pl.loggers.TensorBoardLogger("result", name=args.method.name),
        callbacks=[pl.callbacks.LearningRateLogger(), checkpoint_callback]
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.globals.seed,
            'gpu_name': torch.cuda.get_device_name(0) if args.globals.gpus > 0 else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if args.globals.gpus > 0 else None
            })
   
    if hasattr(trainer, 'logger'):
        trainer.logger.log_hyperparams(yaml) # Log random seed

    # Fit model
    module = get_module(args)
    trainer.fit(module)