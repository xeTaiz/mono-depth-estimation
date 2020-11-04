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

def get_checkpoint(version_path):
    checkpoints = [ckpt for ckpt in Path(version_path, "checkpoints").glob('*')]
    checkpoints.sort(key=lambda x: int(x.name.replace("epoch=", "").replace(".ckpt", "")))
    if len(checkpoints) > 0:
        return checkpoints[-1]
    else:
        return None

def test_method(method, version_path, test_dataset, path, metrics):
    hparams = Path(version_path, "hparams.yaml")
    checkpoint = get_checkpoint(version_path)
    trainer = pl.Trainer(gpus=1)
    if checkpoint:
        print("Testing {} {} {}".format(method, version_path.name, checkpoint.name))
        model = get_model(method, checkpoint.as_posix(), hparams.as_posix(), path, test_dataset, metrics)
        if model:
            result = trainer.test(model)
            return result[0]
        else:
            print("Model unavailable: ", method)
    return None

def get_model(method, ckpt, hparams, path, test_dataset, metrics):
    if   method == 'bts':   Module = BtsModule
    elif method == 'midas': Module = MidasModule
    elif method == 'laina': Module = FCRNModule
    elif method == 'vnl':   Module = VNLModule
    elif method == 'eigen': Module = EigenModule
    elif method == 'dorn':  Module = DORNModule
    else:return None
    return Module.load_from_checkpoint(checkpoint_path=ckpt, hparams_file=hparams, path=path, metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', required=True, type=str, help='directory where snapshots are located.')
    parser.add_argument('--output', required=True, type=str, help='Text File to write test output to.')
    parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'rmse', 'rmsle', 'log10', 'absrel', 'sqrel'], nargs='+', help='which metrics to evaluate')
    parser.add_argument('--methods', default=['bts', 'vnl', 'laina', 'eigen', 'midas', 'dorn'], nargs='+', help='Methods to test')
    parser.add_argument('--path', required=True, type=str, help='Path to Floorplan3D')
    parser.add_argument('--test_dataset', default=['noreflection', 'isotropic', 'mirror'], nargs='+', help='test dataset(s)')

    args = parser.parse_args()
    results_directory = Path(args.results)
    assert results_directory.exists(), "{} does not exist!".format(results_directory.as_posix())
    output_file = Path(args.output).absolute()
    assert output_file.parent.exists(), "{} directory does not exist!".format(output_file.parent.as_posix())

    txt_file = open(output_file.as_posix(), "w")
    txt_file.write("method,loss,aug,train,test,{},\n".format(",".join(args.metrics)))

    for method in results_directory.glob('*'):
        if not method.name in args.methods:continue
        for version in method.glob('*'):
            for test_dataset in args.test_dataset:
                result = test_method(method.name, version, test_dataset, args.path, args.metrics)
                if not result:continue
                with open(Path(version, "hparams.yaml").as_posix(), "r") as yamlf:
                    hparams = yaml.load(yamlf, Loader=yaml.FullLoader)
                line = "{},{},{},{},{},".format(method.name, hparams['loss'], hparams['data_augmentation'], hparams['dataset'], test_dataset)
                for metric in args.metrics:
                    line += "{},".format(round(result[metric], 3))
                line += "\n"
                txt_file.write(line)
    txt_file.close()