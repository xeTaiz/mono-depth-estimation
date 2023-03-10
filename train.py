import sys
from modules import get_module, register_module_specific_arguments
from datasets import register_dataset_specific_arguments
import pytorch_lightning as pl
from argparse import ArgumentParser, Namespace, ArgumentDefaultsHelpFormatter
from pathlib import Path
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
    train_datasets = []
    val_datasets = []
    test_datasets = []
    while len(split_argv):
        pos += 1
        cmd, *args_raw = split_argv.pop(0)
        assert cmd[0].isalpha(), 'Command must start with a letter.'
        args_parsed = commands.choices[cmd].parse_args(args_raw, namespace=Namespace())
        if cmd in ["nyu", "structured3d", "floorplan3d", "stdepth", "stdepthmulti", "stdepthmulti2"]:
            if args_parsed.training: train_datasets.append((cmd, args_parsed))
            if args_parsed.validation: val_datasets.append((cmd, args_parsed))
            if args_parsed.test:      test_datasets.append((cmd, args_parsed))
        else:
            setattr(args, "method" if cmd in ['bts', 'eigen', 'vnl', 'dorn', 'midas', 'laina', 'my'] else cmd, args_parsed)
    setattr(args, "training" , train_datasets)
    setattr(args, "validation" , val_datasets)
    setattr(args, "test" , test_datasets)
    assert hasattr(args, "method"), "Please provide the method you want to use: bts, eigen, vnl, dorn, midas, laina"
    return args

if __name__ == "__main__":
    parser = ArgumentParser('Trains mono depth estimation models', formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--seed', default=None, type=int, help='Random Seed')
    parser.add_argument('--name', default=None, help='Name of the run')
    parser.add_argument('--precision', default=16,   type=int, help='16 to use Mixed precision (AMP O2), 32 for standard 32 bit float training')
    parser.add_argument('--gpus', type=int, default=1, help='Number of GPUs')
    parser.add_argument('--dev', action='store_true', help='Activate Lightning Fast Dev Run for debugging')
    parser.add_argument('--overfit', action='store_true', help='If this flag is set the network is overfit to 1 batch')
    parser.add_argument('--min_epochs', default=5, type=int, help='Minimum number of epochs.')
    parser.add_argument('--max_epochs', default=25, type=int, help='Maximum number ob epochs to train')
    parser.add_argument('--max-train-batches', default=1.0, type=float, help='Limit train dataset to percentage/amount')
    parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'log10', 'rmse', 'ssim'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--worker', default=8, type=int, help='Number of workers for data loader')
    parser.add_argument('--find_learning_rate', action='store_true', help="Finding learning rate.")
    parser.add_argument('--detect_anomaly', action='store_true', help='Enables pytorch anomaly detection')


    commands = parser.add_subparsers(title='Commands')
    register_dataset_specific_arguments(commands)
    register_module_specific_arguments(commands)

    args = parse_args_into_namespaces(parser, commands)
    if args.globals.name.startswith('VALIDATE'):
        VALIDATE_ONLY = True
    else:
        assert args.training and args.validation, "Please provide data training AND validation dataset"
        VALIDATE_ONLY = False

    args.ds_name = args.validation[0][1].path.split('/')[-1]
    args.depth_method = args.validation[0][1].depth_method

    if args.globals.detect_anomaly:
        print("Enabling anomaly detection")
        torch.autograd.set_detect_anomaly(True)

    # windows safe
    if sys.platform in ["win32"]:
        args.globals.worker = 0

    # Manage Random Seed
    if args.globals.seed is None: # Generate random seed if none is given
        args.globals.seed = random.randrange(4294967295) # Make sure it's logged
    pl.seed_everything(args.globals.seed)

    wandb_logger = pl.loggers.WandbLogger(project="stdepth", name=args.globals.name, log_model=True)
    
    if VALIDATE_ONLY:
        ckpt_dir = f'checkpoints/{args.globals.name.replace("VALIDATE", "")}'
    else:
        ckpt_dir = f'checkpoints/{args.globals.name}'
    ckpt_nam = '{epoch}-{val_loss}'
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        verbose=True,
        #save_weights_only=True,
        save_top_k=1,
        dirpath=ckpt_dir,
        filename=ckpt_nam,
        monitor='val_loss',
        mode='min'
    )

    ckpts = [(fn, float(fn.name.split('val_loss=')[-1][:-5])) for fn in Path(ckpt_dir).rglob('*.ckpt')]
    if len(ckpts) > 0: 
        best_ckpt = sorted(ckpts, key=lambda tup: tup[1], reverse=True)[0]
        print(f'Found existing checkpoint for this run with val_delta1={best_ckpt[1]:.2f}: {best_ckpt[0]}')
        args.method.ckpt = str(best_ckpt[0])


    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=5
    )

    use_gpu = not args.globals.gpus == 0

    trainer = pl.Trainer(
        log_gpu_memory=False,
        val_check_interval=0.2,
        limit_train_batches=args.globals.max_train_batches,
        fast_dev_run=args.globals.dev,
        gpus=args.globals.gpus,
        overfit_batches=1 if args.globals.overfit else 0,
        precision=args.globals.precision if use_gpu else 32,
        amp_level='O2' if use_gpu else None,
        min_epochs=args.globals.min_epochs,
        max_epochs=args.globals.max_epochs,
        logger=wandb_logger,
        callbacks=[pl.callbacks.lr_monitor.LearningRateMonitor(), checkpoint_callback, early_stop_callback]
    )

    yaml = args.__dict__
    yaml.update({
            'random_seed': args.globals.seed,
            'gpu_name': torch.cuda.get_device_name(0) if use_gpu else None,
            'gpu_capability': torch.cuda.get_device_capability(0) if use_gpu else None
            })

    #if hasattr(trainer, 'logger'):

    trainer.logger.log_hyperparams(args) # Log Hyper parameters
    torch.autograd.set_detect_anomaly(True)
    # Fit model
    module = get_module(args)
    if args.globals.find_learning_rate:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(module)
        suggested_lr = lr_finder.suggestion()
        print("Old learning rate: ", args.method.learning_rate)
        args.method.learning_rate = suggested_lr
        print("Suggested learning rate: ", args.method.learning_rate)
    else:
        if VALIDATE_ONLY:
            trainer.validate(module)
        trainer.fit(module)
        if args.test:
            trainer.test(module)
