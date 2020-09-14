from argparse import ArgumentParser
import pytorch_lightning as pl
import torch.nn as nn
import torch
import cv2
import numpy as np
from datasets.structured3d import Structured3DLoader
from depth import BaseLine, DetailedDepth, UNet
from visualize import viz_depth_from_batch
from metrics import delta1, delta2, delta3, log10

class DepthEstimation(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.depth = UNet(n_channels=3)
        self.criterion = nn.MSELoss()
        self.training_loader = Structured3DLoader(path=self.hparams.path, 
                             split="train", 
                             batch_size=self.hparams.batch_size,
                             img_size=self.hparams.img_size, 
                             cache=self.hparams.cache,
                             n_workers=self.hparams.n_worker)
        self.validation_loader = Structured3DLoader(path=self.hparams.path, 
                             split="val", 
                             batch_size=1,
                             cache=self.hparams.cache,
                             img_size=self.hparams.img_size, 
                             n_workers=self.hparams.n_worker)
        self.visualization = [None] * 10

    def forward(self, data):
        img, _ = data
        return self.depth(img)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        _, y = batch
        loss = self.criterion(y_hat, y)
        l10 = log10(y_hat, y) 
        rmse = pl.metrics.functional.rmse(y_hat, y)
        result = pl.TrainResult(loss)
        result.log('mse', loss, logger=True, on_epoch=True, prog_bar=True)
        result.log('log10', l10, logger=True, on_epoch=True, prog_bar=True)
        result.log('rmse', rmse, logger=True, on_epoch=True, prog_bar=True)
        return result

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        _, y = batch
        mse = pl.metrics.functional.mse(y_hat, y)
        rmse = pl.metrics.functional.rmse(y_hat, y)
        ssim = pl.metrics.functional.ssim(y_hat, y)
        d1 = delta1(y_hat, y)
        d2 = delta2(y_hat, y)
        d3 = delta3(y_hat, y)
        l10 = log10(y_hat, y) 
        result = pl.EvalResult(checkpoint_on=mse)
        result.log('mse', mse, logger=True, on_epoch=True, prog_bar=True) 
        result.log('rmse', rmse, logger=True, on_epoch=True, prog_bar=True) 
        result.log('d1', d1, logger=True, on_epoch=True, prog_bar=True) 
        result.log('d2', d2, logger=True, on_epoch=True, prog_bar=True) 
        result.log('d3', d3, logger=True, on_epoch=True, prog_bar=True)     
        result.log('log10', l10, logger=True, on_epoch=True, prog_bar=True)  
        if batch_idx < len(self.visualization):
            viz = viz_depth_from_batch(batch, y_hat)
            self.visualization[batch_idx] = viz     
        elif batch_idx == len(self.visualization):
            cv2.imwrite("{}/{}/version_{}/epoch{}.jpg".format(self.logger.save_dir, self.logger.name, self.logger.version, self.current_epoch), np.vstack(self.visualization))
        return result

    def configure_optimizers(self):
        gen_params = [
            {'params': self.depth.parameters(), 'lr': self.hparams.learning_rate}
        ]
        return torch.optim.Adam(gen_params)

    def train_dataloader(self):
        return self.training_loader

    def val_dataloader(self):
        return self.validation_loader

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.001, type=float, help='Learning Rate')
        parser.add_argument('--batch_size',    default=8,     type=int,   help='Batch Size')
        parser.add_argument('--backbone_n_filter', default=32,  type=int,   help='Number of filters for latent_space')
        parser.add_argument('--pretrained', default=0, type=int, help='Use pretrained backbone')
        parser.add_argument('--latent_sz', default=128, type=int, help='Latent Space size after backbone')
        parser.add_argument('--im_feat_sz', default=128, type=int, help='Number image features')
        parser.add_argument('--img_size', default=(640, 360), type=tuple, help='Image size')

        return parser