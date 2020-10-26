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
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np

midas_transform = torch.hub.load("intel-isl/MiDaS", "transforms").default_transform 

def training_preprocess(rgb, depth):
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
    mask = depth > 0
    depth[mask] = 10. / depth[mask]
    depth[~mask] = 0.
    return rgb, depth

def validation_preprocess(rgb, depth):
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
    mask = depth > 0
    depth[mask] = 10. / depth[mask]
    depth[~mask] = 0.
    return rgb, depth

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
        assert self.args.loss in ['ssil1', 'ssitrim', 'ssimse', 'eigen', 'laina']
        self.train_dataset = get_dataset(self.args.path, 'train', self.args.dataset)
        self.val_dataset = get_dataset(self.args.path, 'val', self.args.eval_dataset)
        if self.args.data_augmentation == 'midas':
            self.train_dataset.transform = training_preprocess
            self.val_dataset.transform = validation_preprocess
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
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
        if self.args.loss == 'ssimse':
            self.criterion = criteria.MidasLoss(alpha=self.args.alpha, loss='mse', reduction=self.args.reduction)
        elif self.args.loss == 'ssitrim':
            self.criterion = criteria.MidasLoss(alpha=self.args.alpha, loss='trimmed', reduction=self.args.reduction)
        elif self.args.loss == 'ssil1':
            self.criterion = criteria.MidasLoss(alpha=self.args.alpha, loss='l1', reduction=self.args.reduction)
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
        return y_hat.type(torch.float32)

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def scale_and_shift(self, prediction, target):
        if prediction.ndim == 4:
            prediction = prediction.squeeze(1)
        if target.ndim == 4:
            target = target.squeeze(1)
        assert prediction.dim() == target.dim(), "inconsistent dimensions"
        mask = (target > 0).type(torch.float32)
        scale, shift = criteria.compute_scale_and_shift(prediction, target, mask)
        prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)
        target_ssi = scale.view(-1, 1, 1) * target + shift.view(-1, 1, 1)
        return prediction_ssi, target_ssi, mask

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        y_hat, y, mask = self.scale_and_shift(y_hat, y)
        loss = self.criterion(y_hat, y, mask)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        y_hat, y = self.scale_and_shift(y_hat, y)
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
        parser.add_argument('--loss', default='ssitrim', type=str, help='loss function: [ssitrim, ssimse, ssil1, eigen, laina]')
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror, structured3d]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror, structured3d]')
        parser.add_argument('--data_augmentation', default='midas', type=str, help='Choose data Augmentation Strategy: laina or midas')
        parser.add_argument('--alpha', default=0.5, type=float, help='alpha')
        parser.add_argument('--reduction', default='batch-based', type=str, help='reduction method')
        return parser