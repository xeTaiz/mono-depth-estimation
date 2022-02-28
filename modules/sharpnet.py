import torch
import pytorch_lightning as pl
import criteria
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from network import SharpNet
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np

RGB_PIXEL_MEANS = (0.485, 0.456, 0.406)  # (102.9801, 115.9465, 122.7717)
RGB_PIXEL_VARS = (0.229, 0.224, 0.225)  # (1, 1, 1)

def training_preprocess(rgb, depth):
    rgb = transforms.ToPILImage()(rgb)
    depth = transforms.ToPILImage()(depth)
    # Random resize
    size = np.random.randint(240, 720)
    resize = transforms.Resize(int(size))
    rgb = resize(rgb)
    depth = resize(depth)
    # Random crop
    i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(240, 320))
    rgb = TF.crop(rgb, i, j, h, w)
    depth = TF.crop(depth, i, j, h, w)
    # Random horizontal flipping
    if np.random.uniform(0,1) > 0.5:
        rgb = TF.hflip(rgb)
        depth = TF.hflip(depth)
    # random rotation
    angle = transforms.RandomRotation.get_params([-6,6])
    rgb = TF.rotate(rgb, angle)
    depth = TF.rotate(depth, angle)
    # Transform to tensor
    rgb = TF.to_tensor(np.array(rgb))
    depth = np.array(depth, dtype=np.float32)
    depth = TF.to_tensor(depth)
    # random scale
    s = np.random.uniform(0.5, 2)
    rgb /= s
    depth /= s
    # normalize color
    rgb = transforms.Normalize(RGB_PIXEL_MEANS, RGB_PIXEL_VARS)(rgb)
    return rgb, depth

def validation_preprocess(rgb, depth):
    rgb = transforms.ToPILImage()(rgb)
    depth = transforms.ToPILImage()(depth)
    # Resize
    resize = transforms.Resize(240)
    rgb = resize(rgb)
    depth = resize(depth)
    # center crop
    crop = transforms.CenterCrop((240, 320))
    rgb = crop(rgb)
    depth = crop(depth)
    # Transform to tensor
    rgb = TF.to_tensor(np.array(rgb))
    depth = np.array(depth, dtype=np.float32)
    depth = TF.to_tensor(depth)
    return rgb, depth

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


class SharpNetModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        assert self.args.loss in ['berHuLoss', 'L1', 'SharpNetLoss']
        self.train_dataset = get_dataset(self.args.path, 'train', self.args.dataset)
        self.val_dataset = get_dataset(self.args.path, 'val', self.args.eval_dataset)
        if self.args.data_augmentation == 'sharpnet':
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
        self.skip = len(self.val_loader) // 9
        print("=> creating Model")
        self.model = SharpNet.SharpNet(SharpNet.Bottleneck, [3, 4, 6, 3], [2, 2, 2, 2, 2], use_depth=True)
        print("=> model created.")
        if self.args.loss == 'berHuLoss':
            self.criterion = criteria.berHuLoss()
        elif self.args.loss == 'L1':
            self.criterion = criteria.MaskedL1Loss()
        elif self.args.loss == 'SharpNetLoss':
            self.criterion = criteria.LainaBerHuLoss()
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
        #y /= 10.0
        loss = self.criterion(y_hat, y)
        self.save_visualization(x, y, y_hat, batch_idx, nam='train')
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        #y /= 10.0
        y_hat = self(x)
        self.save_visualization(x, y, y_hat, batch_idx, nam='val')
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
        parser.add_argument('--data_augmentation', default='sharpnet', type=str, help='Choose data Augmentation Strategy: sharpnet or laina')
        return parser
