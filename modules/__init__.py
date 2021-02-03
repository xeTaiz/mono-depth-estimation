from modules import eigen, bts, vnl, midas, laina, dorn

def register_module_specific_arguments(subparser):
    return [
        eigen.EigenModule.add_model_specific_args(subparser),
        bts.BtsModule.add_model_specific_args(subparser),
        dorn.DORNModule.add_model_specific_args(subparser),
        midas.MidasModule.add_model_specific_args(subparser),
        vnl.VNLModule.add_model_specific_args(subparser),
        laina.FCRNModule.add_model_specific_args(subparser)
    ]

def get_module(args):
    module = None
    if args.method.name == "eigen": module = eigen.EigenModule
    if args.method.name == "midas": module = midas.MidasModule
    if args.method.name == "vnl":   module = vnl.VNLModule
    if args.method.name == "dorn":  module = dorn.DORNModule
    if args.method.name == "laina": module = laina.FCRNModule
    if args.method.name == "bts":   module = bts.BtsModule
    assert module, "Please select method!"
    if args.method.ckpt:
        return module.load_from_checkpoint(checkpoint_path=args.method.ckpt, hparams_file=args.method.ckpt)
    else:
        return module(args)
