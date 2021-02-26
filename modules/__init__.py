from argparse import Namespace
from pathlib import Path
from modules import eigen, bts, vnl, midas, laina, dorn, my

def register_module_specific_arguments(subparser):
    return [
        eigen.EigenModule.add_model_specific_args(subparser),
        bts.BtsModule.add_model_specific_args(subparser),
        dorn.DORNModule.add_model_specific_args(subparser),
        midas.MidasModule.add_model_specific_args(subparser),
        vnl.VNLModule.add_model_specific_args(subparser),
        laina.FCRNModule.add_model_specific_args(subparser),
        my.MyModule.add_model_specific_args(subparser)
    ]

def get_module(args):
    module = None
    if args.method.name == "eigen": module = eigen.EigenModule
    if args.method.name == "midas": module = midas.MidasModule
    if args.method.name == "vnl":   module = vnl.VNLModule
    if args.method.name == "dorn":  module = dorn.DORNModule
    if args.method.name == "laina": module = laina.FCRNModule
    if args.method.name == "bts":   module = bts.BtsModule
    if args.method.name == "my":    module = my.MyModule
    assert module, "Please select method!"
    if args.method.ckpt: 
        print("Loading checkpoint: {}".format(args.method.ckpt))
        hparams = Namespace()
        hparams.globals = args.globals
        hparams.method = args.method
        hparams.training = args.training
        hparams.validation = args.validation
        hparams.test = args.test
        return module.load_from_checkpoint(checkpoint_path=args.method.ckpt, hparams_file=(Path(args.method.ckpt).parents[1]/"hparams.yaml").as_posix(), hparams=hparams)
    else: return module(args)
