import argparse
from pathlib import Path
import torch
import modules
from modules.bts import BtsModule
from modules.vnl import VNLModule
from modules.dorn import DORNModule
from modules.midas import MidasModule
from modules.laina import FCRNModule
from modules.eigen import EigenModule
import pytorch_lightning as pl
import yaml

def get_checkpoint(version_path, min_epoch=1):
    if not (version_path and Path(version_path).exists()): return None
    if min_epoch == -1:
        checkpoints = [ckpt for ckpt in Path(version_path, "checkpoints").glob('*') if 'last' in ckpt.name]
        if len(checkpoints) == 1: return checkpoints[0]
    checkpoints = [ckpt for ckpt in Path(version_path, "checkpoints").glob('*') if not 'last' in ckpt.name and int(ckpt.name.replace("epoch=", "").replace(".ckpt", "")) >= min_epoch]
    checkpoints.sort(key=lambda x: int(x.name.replace("epoch=", "").replace(".ckpt", "")))
    if len(checkpoints):
        return checkpoints[-1]
    else:
        return None

def test_method(method, version_path, test_dataset, path, metrics, min_epoch, worker, mirrors_only, exclude_mirrors, save_images):
    hparams = Path(version_path, "hparams.yaml")
    checkpoint = get_checkpoint(version_path, min_epoch)
    trainer = pl.Trainer(gpus=1)
    if checkpoint:
        print("Testing {} {} {} on {}".format(method, version_path.name, checkpoint.name, test_dataset))
        save_dir = None
        if save_images: save_dir = checkpoint.parents[1]/"images"/test_dataset
        model = get_model(method, checkpoint.as_posix(), hparams.as_posix(), path, test_dataset, metrics, worker, save_dir,mirrors_only, exclude_mirrors)
        if model:
            result = trainer.test(model)
            return result[0], checkpoint
        else:
            print("Model unavailable: ", method)
    return None, None

def get_model(method, ckpt, hparams, path, test_dataset, metrics, worker, safe_dir, mirrors_only, exclude_mirrors):
    if   method == 'bts':   Module = BtsModule
    elif method == 'midas': Module = MidasModule
    elif method == 'laina': Module = FCRNModule
    elif method == 'vnl':   Module = VNLModule
    elif method == 'eigen': Module = EigenModule
    elif method == 'dorn':  Module = DORNModule
    else:return None
    return Module.load_from_checkpoint(
        checkpoint_path=ckpt, 
        hparams_file=hparams, 
        safe_dir=safe_dir,
        path=path, 
        metrics=metrics, 
        test_dataset=test_dataset, 
        worker=worker,
        n_images=-1,
        bn_no_track_stats=True,
        fix_first_conv_block=False,
        fix_first_conv_blocks=True,
        mirrors_only=mirrors_only,
        exclude_mirrors=exclude_mirrors,
        use_mat=0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=str, help='directory where snapshots are located.')
    parser.add_argument('--output', required=True, type=str, help='Text File to write test output to.')
    parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'rmse', 'rmsle', 'log10', 'absrel', 'sqrel'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--methods', default=['bts', 'vnl', 'laina', 'eigen', 'midas', 'dorn'], nargs='+', help='Methods to test')
    parser.add_argument('--path', required=True, type=str, help='Path to Floorplan3D')
    parser.add_argument('--test_dataset', default=['noreflection', 'isotropic', 'mirror'], nargs='+', help='test dataset(s)')
    parser.add_argument('--min_epoch', default=1, type=int, help='ignore checkpoints from less epochs.')
    parser.add_argument('--worker', default=6, type=int, help='Number of workers')
    parser.add_argument('--mirrors_only', action='store_true', help="Test mirrors only")
    parser.add_argument('--exclude_mirrors', action='store_true', help="Test while excluding mirror")
    parser.add_argument('--save_images', action='store_true', help="Saving images")
    parser.add_argument('--exp1', action='store_true', help="Do experiment 1")
    parser.add_argument('--exp2', action='store_true', help="Do experiment 2")

    args = parser.parse_args()
    results_directory = Path(args.results)
    if args.exp1:
        args.test_dataset = ['noreflection', 'isotropic', 'mirror']
        args.metrics = ['delta1', 'log10', 'rmse', 'absrel', 'sqrel']
        args.methods = ['vnl', 'midas', 'dorn', 'eigen', 'laina', 'bts']
    if args.exp2:
        args.test_dataset = ['nyu+exclude_mirrors', 'nyu+mirrors_only']
    assert results_directory.exists(), "{} does not exist!".format(results_directory.as_posix())
    output_file = Path(args.output).absolute()
    assert output_file.parent.exists(), "{} directory does not exist!".format(output_file.parent.as_posix())

    txt_file = open(output_file.as_posix(), "w")
    txt_file.write("version,epoch,method,loss,aug,train,test,{},\n".format(",".join(args.metrics)))

    for method in results_directory.glob('*'):
        if not method.name in args.methods:continue
        for version in method.glob('*'):
            for test_dataset in args.test_dataset:
                if "exclude_mirrors" in test_dataset: args.exclude_mirrors=True
                if "mirrors_only" in test_dataset: args.mirrors_only=True
                result, ckpt = test_method(method.name, version, test_dataset.split('+')[0], args.path, args.metrics, args.min_epoch, args.worker, args.mirrors_only, args.exclude_mirrors, args.save_images)
                if not result:continue
                with open(Path(version, "hparams.yaml").as_posix(), "r") as yamlf:
                    hparams = yaml.load(yamlf, Loader=yaml.FullLoader)
                line = "{},{},{},{},{},{},{},".format(version.name, ckpt.name, method.name, hparams['loss'], hparams['data_augmentation'], hparams['dataset'], test_dataset)
                for metric in args.metrics:
                    line += "{},".format(round(result[metric], 3))
                line += "\n"
                txt_file.write(line)
    txt_file.close()