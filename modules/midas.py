import torch
import pytorch_lightning as pl
import criteria
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from datasets.structured3d_dataset import Structured3DDataset
from network import MiDaS
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger
import urllib.request
from pathlib import Path
from tqdm import tqdm

def get_dataset(path, split, dataset):
    if dataset == 'nyu':
        return NYUDataset(path, split=split, output_size=(384, 384), resize=400)
    elif dataset == 'noreflection':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.NO_REFLECTION, output_size=(384, 384), resize=400)
    elif dataset == 'isotropic':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(384, 384), resize=400)
    elif dataset == 'mirror':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(384, 384), resize=400)
    elif dataset == 'structured3d':
        return Structured3DDataset(path, split=split, dataset_type='perspective', output_size=(384, 384), resize=400)
    else:
        raise ValueError('unknown dataset {}'.format(dataset))

class MidasModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert self.args.loss in ['midas', 'eigen', 'laina']
        self.train_loader = torch.utils.data.DataLoader(get_dataset(self.args.path, 'train', self.args.dataset),
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(get_dataset(self.args.path, 'val', self.args.eval_dataset),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
        if self.args.pretrained: 
            weights_path = Path.cwd()/"model-f46da743.pt"
            if not weights_path.exists():                                                             
                self.download_weights(weights_path)
            weights_path = weights_path.resolve().as_posix()
            print("Using pretrained weights: ", weights_path)
        else:
            weights_path = None                   
                                       
        self.skip = len(self.val_loader) // 9
        print("=> creating Model")
        self.model = MiDaS.MidasNet(path=weights_path, features=256)
        print("=> model created.")
        if self.args.loss == 'midas':
            self.criterion = criteria.ScaleAndShiftInvariantLoss()
        elif self.args.loss == 'eigen':
            self.criterion = criteria.MaskedDepthLoss()
        elif self.args.loss == 'laina':
            self.criterion = criteria.MaskedL1Loss()
        self.metric_logger = MetricLogger(metrics=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'rmse', 'log10'])

    def download_weights(self, filename):
        def my_hook(t):
            last_b = [0]
            def update_to(b=1, bsize=1, tsize=None):
                if tsize is not None:
                    t.total = tsize
                t.update((b - last_b[0]) * bsize)
                last_b[0] = b
            return update_to
        # start download
        with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc="Downloading MiDaS weights: {}".format(filename.as_posix())) as t:
            urllib.request.urlretrieve("https://github.com/intel-isl/MiDaS/releases/download/v2/model-f46da743.pt", filename = filename, reporthook = my_hook(t), data = None)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        if batch_idx == 0:
            self.img_merge = visualize.merge_into_row(x, y, y_hat)
        elif (batch_idx < 8 * self.skip) and (batch_idx % self.skip == 0):
            row = visualize.merge_into_row(x, y, y_hat)
            self.img_merge = visualize.add_row(self.img_merge, row)
        elif batch_idx == 8 * self.skip:
            filename = "{}/{}/version_{}/epoch{}.jpg".format(self.logger.save_dir, self.logger.name, self.logger.version, self.current_epoch)
            visualize.save_image(self.img_merge, filename)
        return self.metric_logger.log_val(y_hat, y, checkpoint_on='mae')

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.pretrained.parameters(), 'lr': self.args.learning_rate * 0.1},
                        {'params': self.model.scratch.parameters(), 'lr': self.args.learning_rate}]

        optimizer = torch.optim.Adam(train_params, lr=self.args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.args.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateua': True,
            'monitor': 'val_checkpoint_on'
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning Rate')
        parser.add_argument('--batch_size',    default=8,     type=int,   help='Batch Size')
        parser.add_argument('--worker',        default=6,      type=int,   help='Number of workers for data loader')
        parser.add_argument('--path', required=True, type=str, help='Path to NYU')
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--pretrained', default=0, type=int, help="Use pretrained MiDaS")
        parser.add_argument('--features', default=256, type=int, help='Number of features')
        parser.add_argument('--loss', default='midas', type=str, help='loss function: [midas, eigen, laina]')
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror, structured3d]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror, structured3d]')
        return parser