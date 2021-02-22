import torch
import criteria
from network import MiDaS
import urllib.request
from tqdm import tqdm
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import cv2
from modules.base_module import BaseModule

midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform 

class MidasModule(BaseModule):
   
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

    def setup_criterion(self):
        if self.method.loss in ['ssil1', 'ssimse', 'l1', 'mse', 'trim']:
            return criteria.MidasLoss(alpha=self.method.alpha, loss=self.method.loss, reduction=self.method.reduction)
        elif self.method.loss == 'eigen':
            return criteria.MaskedDepthLoss()
        elif self.method.loss == 'laina':
            return criteria.MaskedL1Loss()
        elif self.method.loss == 'ssitrim':
            return criteria.TrimmedProcrustesLoss(alpha=self.method.alpha, reduction=self.method.reduction)

    def setup_model(self):
        if self.method.pretrained: return torch.hub.load("intel-isl/MiDaS", "MiDaS")
        else:                       return MiDaS.MidasNet(features=256)

    def output_size(self):
        return (384, 384)

    def resize(self):
        return 400

    def forward(self, x):
        y_hat = self.model(x)
        if y_hat.ndim < 4: y_hat = y_hat.unsqueeze(0)
        return y_hat.type(torch.float32)

    def scale_shift(self, pred, target):
        if pred.ndim == 4: pred = pred.squeeze(1)
        if target.ndim == 4: target = target.squeeze(1)
        scale, shift = criteria.compute_scale_and_shift(pred, target)
        pred = scale.view(-1, 1, 1) * pred + shift.view(-1, 1, 1)
        return pred.unsqueeze(1), target.unsqueeze(1)

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        if "ssi" in self.method.loss:
            y_hat, y = self.scale_shift(y_hat, y)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        if "ssi" in self.method.loss:
            y_hat, y = self.scale_shift(y_hat, y)
        self.save_visualization(x, y, y_hat, batch_idx)
        return self.metric_logger.log_val(y_hat, y)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        y = batch['depth']
        y_hat = self(batch['rgb'])
        if "ssi" in self.method.loss:
            y_hat, y = self.scale_shift(y_hat, y)
        y_hat = torch.nn.functional.interpolate(y_hat, (640, 640), mode='bilinear')
        y_hat = y_hat[..., 0:480, 0:640]
        return self.metric_logger.log_test(y_hat, batch['depth_raw'])

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.pretrained.parameters(), 'lr': self.method.learning_rate * 0.1},
                        {'params': self.model.scratch.parameters(), 'lr': self.method.learning_rate}]

        optimizer = torch.optim.Adam(train_params, lr=self.method.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.method.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    def train_preprocess(self, rgb, depth):
        if isinstance(rgb, np.ndarray):
            rgb = transforms.ToPILImage()(rgb)
        if isinstance(depth, np.ndarray):
            depth = transforms.ToPILImage()(depth)
        # Random resize
        size = np.random.randint(384, 720)
        resize = transforms.Resize(int(size))
        rgb = resize(rgb)
        depth = resize(depth)
        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(384,384))
        rgb = TF.crop(rgb, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)
        # Random horizontal flipping
        if np.random.uniform(0,1) > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
        # Transform to tensor
        rgb = midas_transform(np.array(rgb, dtype=np.uint8)).squeeze(0)#TF.to_tensor(np.array(rgb)) #
        depth = np.array(depth, dtype=np.float32)
        depth = TF.to_tensor(depth)
        #mask = depth > 0
        #depth[mask] = 10. / depth[mask]
        #depth[~mask] = 0.
        return rgb, depth

    def val_preprocess(self, rgb, depth):
        if isinstance(rgb, np.ndarray):
            rgb = transforms.ToPILImage()(rgb)
        if isinstance(depth, np.ndarray):
            depth = transforms.ToPILImage()(depth)
        # Resize
        resize = transforms.Resize(384)
        rgb = resize(rgb)
        depth = resize(depth)
        # center crop
        crop = transforms.CenterCrop((384, 384))
        rgb = crop(rgb)
        depth = crop(depth)
        # Transform to tensor
        rgb = midas_transform(np.array(rgb, dtype=np.uint8)).squeeze(0)#TF.to_tensor(np.array(rgb)) #
        depth = np.array(depth, dtype=np.float32)
        depth = TF.to_tensor(depth)
        #mask = depth > 0
        #depth[mask] = 10. / depth[mask]
        #depth[~mask] = 0.
        return rgb, depth

    def test_preprocess(self, rgb, depth):
        if isinstance(rgb, np.ndarray):
            rgb = transforms.ToPILImage()(rgb)
        if isinstance(depth, np.ndarray):
            depth = transforms.ToPILImage()(depth)
        # Resize
        resize = transforms.Resize(500)
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop((480, 640))
        rgb_raw = crop(rgb)
        depth_raw = crop(depth)
        # to cv2
        rgb_raw = np.array(rgb_raw, dtype=np.uint8)
        depth_raw = np.array(depth_raw, dtype=np.float32)
        # pad
        rgb = cv2.copyMakeBorder(rgb_raw, 0, 160, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
        depth = cv2.copyMakeBorder(depth_raw, 0, 160, 0, 0, cv2.BORDER_CONSTANT, value=[0])
        assert rgb.shape[0] == rgb.shape[1], "Not square!"
        # Resize
        rgb = cv2.resize(rgb, (384, 384))
        depth = cv2.resize(depth, (384, 384))
        # Transform to tensor
        rgb = midas_transform(rgb).squeeze(0)#TF.to_tensor(np.array(rgb)) #
        depth = TF.to_tensor(depth)
        rgb_raw = TF.to_tensor(rgb_raw)
        depth_raw = TF.to_tensor(depth_raw)
        
        return {'rgb_raw': rgb_raw, 'depth_raw': depth_raw, 'rgb': rgb, 'depth': depth}

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('midas', help='MiDaS specific parameters')
        BaseModule.add_default_args(parser, name="midas", learning_rate=0.0001, batch_size=8)
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--pretrained', default=0, type=int, help="Use pretrained MiDaS")
        parser.add_argument('--features', default=256, type=int, help='Number of features')
        parser.add_argument('--loss', default='ssitrim', type=str, help='loss function: [ssitrim, ssimse, ssil1, eigen, laina]')
        parser.add_argument('--data_augmentation', default='midas', type=str, help='Choose data Augmentation Strategy: laina or midas')
        parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
        parser.add_argument('--reduction', default='batch-based', type=str, help='reduction method')
        return parser
