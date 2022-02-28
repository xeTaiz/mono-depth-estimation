import torch
import criteria
from network import Eigen
from modules.base_module import BaseModule

class EigenModule(BaseModule):

    def setup_criterion(self):
        return criteria.MaskedDepthLoss()

    def setup_model(self):
        return Eigen.Eigen(scale1=self.method.backbone, pretrained=self.method.pretrained)

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
        self.save_visualization(x, y, y_hat, batch_idx, nam='train')
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        (_, c, h, w) = y.size()
        # bilinerar upsample
        y_hat = torch.nn.functional.interpolate(y_hat, (h, w), mode='bilinear')
        self.save_visualization(x, y, y_hat, batch_idx, nam='val')
        return self.metric_logger.log_val(y_hat, y)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        x = torch.nn.functional.interpolate(x, (480, 640), mode='bilinear')
        y = torch.nn.functional.interpolate(y, (480, 640), mode='bilinear')
        y_hat = torch.nn.functional.interpolate(y_hat, (480, 640), mode='bilinear')
        self.save_visualization(x, y, y_hat, batch_idx, nam='test')
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.scale1.parameters(), 'lr': self.method.learning_rate},
                        {'params': self.model.scale2.parameters(), 'lr': self.method.learning_rate},
                        {'params': self.model.scale3.parameters(), 'lr': self.method.learning_rate}]

        optimizer = torch.optim.Adam(train_params, lr=self.method.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.method.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('eigen', help='Eigen specific parameters')
        BaseModule.add_default_args(parser, name="eigen", learning_rate=0.0001, batch_size=8)
        parser.add_argument('--backbone', default='vgg', type=str, help="Backbone")
        parser.add_argument('--pretrained', default=1, type=int, help="Use pretrained backbone.")
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--data_augmentation', default='laina', type=str, help='Choose data Augmentation Strategy: laina or eigen')
        parser.add_argument('--loss', default='eigen', type=str, help='loss function')
        return parser
