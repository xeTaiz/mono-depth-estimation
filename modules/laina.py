import torch
import criteria
from network import FCRN
from modules.base_module import BaseModule

class FCRNModule(BaseModule):
   
    def output_size(self):
        return (240, 320)

    def resize(self):
        return 250

    def setup_model(self):
        return FCRN.ResNet(output_size=self.output_size())

    def setup_criterion(self):
        return criteria.MaskedL1Loss()

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

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
        self.save_visualization(x, y, y_hat, batch_idx)
        return self.metric_logger.log_val(y_hat, y, checkpoint_on='mae')

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        x = torch.nn.functional.interpolate(x, (480, 640), mode='bilinear')
        y = torch.nn.functional.interpolate(y, (480, 640), mode='bilinear')
        y_hat = torch.nn.functional.interpolate(y_hat, (480, 640), mode='bilinear')
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.hparams.learning_rate},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.hparams.learning_rate * 10}]

        optimizer = torch.optim.Adam(train_params, lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.hparams.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'reduce_on_plateua': True,
            'monitor': 'val_checkpoint_on'
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('laina', help='Laina specific parameters')
        BaseModule.add_default_args(parser, name="laina", learning_rate=0.0001, batch_size=16)
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--data_augmentation', default='laina', type=str, help='Choose data Augmentation Strategy: laina or midas')
        parser.add_argument('--loss', default='laina', type=str, help='loss function: [laina]')
        return parser