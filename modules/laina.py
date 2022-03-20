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
        return FCRN.ResNet(output_size=self.output_size(), out_channels=self.method.out_channels)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        loss, pred_full = self.criterion(y_hat, y, x, return_composited=True)
        self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='train')
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        loss, pred_full = self.criterion(y_hat, y, x, return_composited=True)
        self.logger.experiment.log({'val_loss': loss.detach()})
        self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='val')
        return self.metric_logger.log_val(y_hat, y)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        loss, pred_full = self.criterion(y_hat, y, x, return_composited=True)
        self.logger.experiment.log({'test_loss': loss.detach()})
        if self.pred_path:
            self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='test', write_predictions=True)
            torch.save({'batch': batch, 'prediction': y_hat, 'composited': pred_full}, self.pred_path/f'pred_{batch_idx:04d}.pt')
        else:
            self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='test')
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.get_1x_lr_params(), 'lr': self.method.learning_rate},
                        {'params': self.model.get_10x_lr_params(), 'lr': self.method.learning_rate * 10}]

        optimizer = torch.optim.Adam(train_params, lr=self.method.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.method.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('laina', help='Laina specific parameters')
        BaseModule.add_default_args(parser, name="laina", learning_rate=0.0001, batch_size=16)
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--out-channels', default=20, type=int, help='Number of output channels.')
        parser.add_argument('--data_augmentation', default='laina', type=str, help='Choose data Augmentation Strategy: laina or midas')
        parser.add_argument('--loss', default='mae+composite', type=str, help='loss function: [laina]')
        parser.add_argument('--variance_focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
        return parser
