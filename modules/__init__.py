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
    if args.method.name == "eigen": module = eigen.EigenModule(args)
    if args.method.name == "midas": module = midas.MidasModule(args)
    if args.method.name == "vnl":   module = vnl.VNLModule(args)
    if args.method.name == "dorn":  module = dorn.DORNModule(args)
    if args.method.name == "laina": module = laina.FCRNModule(args)
    if args.method.name == "bts":   module = bts.BtsModule(args)
    assert module, "Please select method!"
    return module