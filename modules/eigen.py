import torch
import criteria
from network import Eigen
from modules.base_module import BaseModule

class EigenModule(BaseModule):

    def setup_criterion(self):
        return criteria.MaskedDepthLoss()

    def setup_model(self):
        return Eigen.Eigen(scale1=self.hparams.backbone, pretrained=self.hparams.pretrained)

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def output_size(self):
        return (240, 320)

    def resize(self):
        return 250

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        (_, c, h, w) = y.size()
        # bilinerar upsample
        y_hat = torch.nn.functional.interpolate(y_hat, (h, w), mode='bilinear')
        loss = self.criterion(y_hat, y)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        (_, c, h, w) = y.size()
        # bilinerar upsample
        y_hat = torch.nn.functional.interpolate(y_hat, (h, w), mode='bilinear')
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
        train_params = [{'params': self.model.scale1.parameters(), 'lr': self.hparams.learning_rate},
                        {'params': self.model.scale2.parameters(), 'lr': self.hparams.learning_rate},
                        {'params': self.model.scale3.parameters(), 'lr': self.hparams.learning_rate}]

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
        parser = subparsers.add_parser('eigen', help='Eigen specific parameters')
        parser.add_argument('--method', default="eigen", type=str, help="Method for training.")
        parser.add_argument('--backbone', default='vgg', type=str, help="Backbone")
        parser.add_argument('--pretrained', default=1, type=int, help="Use pretrained backbone.")
        parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning Rate')
        parser.add_argument('--batch_size',    default=8,     type=int,   help='Batch Size')
        parser.add_argument('--worker',        default=6,      type=int,   help='Number of workers for data loader')
        parser.add_argument('--path', required=True, type=str, help='Path to NYU')
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--test_dataset', default='nyu', type=str, help='Dataset for Test [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--data_augmentation', default='laina', type=str, help='Choose data Augmentation Strategy: laina or eigen')
        parser.add_argument('--loss', default='eigen', type=str, help='loss function')
        parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'log10', 'rmse'], nargs='+', help='which metrics to evaluate')
        parser.add_argument('--mirrors_only', action='store_true', help="Test mirrors only")
        parser.add_argument('--exclude_mirrors', action='store_true', help="Test while excluding mirror")
        return parser