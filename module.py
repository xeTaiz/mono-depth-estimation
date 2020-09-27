import torch
import pytorch_lightning as pl
import criteria
from datasets import nyu_dataloader
from network import FCRN
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger

class FCRNModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_loader = torch.utils.data.DataLoader(nyu_dataloader.NYUDataset(args.path, split='train'),
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(nyu_dataloader.NYUDataset(args.path, split='val'),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
        self.skip = len(self.val_loader) // 9
        print("=> creating Model")
        self.model = FCRN.ResNet(output_size=self.train_loader.dataset.output_size)
        print("=> model created.")
        self.criterion = criteria.MaskedL1Loss()
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
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.args.learning_rate},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.args.learning_rate * 10}]

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
        parser.add_argument('--batch_size',    default=16,     type=int,   help='Batch Size')
        parser.add_argument('--worker',        default=6,      type=int,   help='Number of workers for data loader')
        parser.add_argument('--path', required=True, type=str, help='Path to NYU')
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        return parser