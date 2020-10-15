import torch
import pytorch_lightning as pl
import criteria
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from network import SharpNet
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger

def get_dataset(path, split, dataset):
    if dataset == 'nyu':
        return NYUDataset(path, split=split, output_size=(240, 320), resize=250)
    elif dataset == 'noreflection':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.NO_REFLECTION, output_size=(240, 320), resize=250)
    elif dataset == 'isotropic':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(240, 320), resize=250)
    elif dataset == 'mirror':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(240, 320), resize=250)
    else:
        raise ValueError('unknown dataset {}'.format(dataset))

sharpnetloss = criteria.SharpNetLoss(lamb=1, mu=0.5, use_depth=True)
def SharpNetLoss(pred, target):
    mask = target > 0
    d_loss, grad_loss, n_loss, b_loss, geo_loss = sharpnetloss(mask, d_pred=pred, d_gt=target)
    return d_loss + grad_loss + n_loss + b_loss + geo_loss

class SharpNetModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert self.args.loss in ['berHuLoss', 'L1', 'SharpNetLoss']
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
        self.skip = len(self.val_loader) // 9
        print("=> creating Model")
        self.model = SharpNet.SharpNet(SharpNet.Bottleneck, [3, 4, 6, 3], [2, 2, 2, 2, 2], use_depth=True)
        print("=> model created.")
        if self.args.loss == 'berHuLoss':
            self.criterion = criteria.berHuLoss()
        elif self.args.loss == 'L1':
            self.criterion = criteria.MaskedL1Loss()
        elif self.args.loss == 'SharpNetLoss':
            self.criterion = SharpNetLoss
        self.metric_logger = MetricLogger(metrics=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'rmse', 'log10'])

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
        y /= 10.0
        loss = self.criterion(y_hat, y)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y /= 10.0
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
        train_params = SharpNet.get_params(self.model)

        optimizer = torch.optim.Adam(train_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
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
        parser.add_argument('--batch_size',    default=16,     type=int,   help='Batch Size')
        parser.add_argument('--worker',        default=6,      type=int,   help='Number of workers for data loader')
        parser.add_argument('--path', required=True, type=str, help='Path to NYU')
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--weight_decay', default=5e-5, type=float, help='Weight decay rate')
        parser.add_argument('--loss', default='SharpNetLoss', type=str, help='loss function: [berHuLoss, L1, SharpNetLoss]')
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror]')
        return parser