import torch
import pytorch_lightning as pl
import criteria
from datasets.dataset import ConcatDataset
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from datasets.structured3d_dataset import Structured3DDataset
from network import Bts
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger, MetricComputation
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

def augment_image(image):
    # gamma augmentation
    gamma = np.random.uniform(0.9, 1.1)
    image_aug = image ** gamma

    # brightness augmentation
    brightness = np.random.uniform(0.75, 1.25)
    image_aug = image_aug * brightness

    # color augmentation
    colors = np.random.uniform(0.9, 1.1, size=3)
    white = np.ones((image.shape[0], image.shape[1]))
    color_image = np.stack([white * colors[i] for i in range(3)], axis=2)
    image_aug *= color_image
    image_aug = np.clip(image_aug, 0, 1)

    return image_aug

def training_preprocess(rgb, depth):
    if isinstance(rgb, np.ndarray):
        rgb = transforms.ToPILImage()(rgb)
    if isinstance(depth, np.ndarray):
        depth = transforms.ToPILImage()(depth)
    
    height = rgb.height
    width = rgb.width
    left_margin = width * 0.05
    right_margin = width * (1.0 - 0.05)
    top_margin = height * 0.05
    bot_margin = height * (1.0 - 0.05)
    depth = depth.crop((left_margin, top_margin, right_margin, bot_margin))
    rgb     = rgb.crop((left_margin, top_margin, right_margin, bot_margin))

    # Random rotation
    angle = transforms.RandomRotation.get_params([-2.5, 2.5])
    rgb = TF.rotate(rgb, angle)
    depth = TF.rotate(depth, angle)

    # Resize
    h = int(np.random.choice([416, 452, 489, 507, 518, 550, 600, 650, 720]))
    resize = transforms.Resize(h)
    rgb = resize(rgb)
    depth = resize(depth)

    # Random Crop
    i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(416, 544))
    rgb = TF.crop(rgb, i, j, h, w)
    depth = TF.crop(depth, i, j, h, w)

    # Random flipping
    if np.random.uniform(0,1) > 0.5:
        rgb = TF.hflip(rgb)
        depth = TF.hflip(depth)

    rgb = np.asarray(rgb, dtype=np.float32) / 255.0
    depth = np.asarray(depth, dtype=np.float32)

    #depth = depth / 1000.0

    # Random gamma, brightness, color augmentation
    if np.random.uniform(0,1) > 0.5:
        rgb = augment_image(rgb)

    rgb = TF.to_tensor(np.array(rgb))
    depth = TF.to_tensor(np.array(depth))
    return rgb, depth

def validation_preprocess(rgb, depth):
    if isinstance(rgb, np.ndarray):
        rgb = transforms.ToPILImage()(rgb)
    if isinstance(depth, np.ndarray):
        depth = transforms.ToPILImage()(depth)
    # Resize
    resize = transforms.Resize(450)
    rgb = resize(rgb)
    depth = resize(depth)
    # Center crop
    crop = transforms.CenterCrop((416, 544))
    rgb = crop(rgb)
    depth = crop(depth)
    
    rgb = TF.to_tensor(np.array(rgb, dtype=np.float32))
    depth = TF.to_tensor(np.array(depth, dtype=np.float32))
    
    rgb /= 255.0
    #depth /= 1000.0
    return rgb, depth

def get_dataset(path, split, dataset):
    path = path.split('+')
    if dataset == 'nyu':
        return NYUDataset(path[0], split=split, output_size=(416, 544), resize=450)
    elif dataset == 'noreflection':
        return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.NO_REFLECTION, output_size=(416, 544), resize=450)
    elif dataset == 'isotropic':
        return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(416, 544), resize=450)
    elif dataset == 'mirror':
        return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(416, 544), resize=450)
    elif dataset == 'structured3d':
        return Structured3DDataset(path[0], split=split, dataset_type='perspective', output_size=(416, 544), resize=450)
    elif '+' in dataset:
        datasets = [get_dataset(p, split, d) for p, d in zip(path, dataset.split('+'))]
        return ConcatDataset(datasets)
    else:
        raise ValueError('unknown dataset {}'.format(dataset))

class BtsModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.train_dataset = get_dataset(self.hparams.path, 'train', self.hparams.dataset)
        self.val_dataset = get_dataset(self.hparams.path, 'val', self.hparams.eval_dataset)
        self.test_dataset = get_dataset(self.hparams.path, 'test', self.hparams.test_dataset)
        if self.hparams.data_augmentation == 'bts':
            self.train_dataset.transform = training_preprocess
            self.val_dataset.transform = validation_preprocess
            self.test_dataset.transform = validation_preprocess
        self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                    batch_size=self.hparams.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=self.hparams.worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=self.hparams.worker, 
                                                    pin_memory=True) 
        self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=self.hparams.worker, 
                                                    pin_memory=True)                                     
        self.skip = len(self.val_loader) // 9
        print("=> creating Model")
        self.model = Bts.BtsModel(max_depth=self.hparams.max_depth, bts_size=self.hparams.bts_size, encoder_version=self.hparams.encoder)
        print("=> model created.")
        if self.hparams.loss == 'bts':
            self.criterion = criteria.silog_loss(variance_focus=self.hparams.variance_focus)
        elif self.hparams.loss == 'laina':
            self.criterion = criteria.MaskedL1Loss()
        else:
            raise ValueError()
        self.metric_logger = MetricLogger(metrics=self.hparams.metrics)

    def forward(self, x):
        _, _, _, _, y_hat = self.model(x)
        return y_hat

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

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

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        train_param = [{'params': self.model.encoder.parameters(), 'weight_decay': self.hparams.weight_decay},
                       {'params': self.model.decoder.parameters(), 'weight_decay': 0}]
        # Training parameters
        optimizer = torch.optim.AdamW(train_param, lr=self.hparams.learning_rate, eps=self.hparams.adam_eps)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=self.hparams.lr_patience)
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
        parser.add_argument('--bts_size', type=int, default=512, help='initial num_filters in bts')
        parser.add_argument('--max_depth', type=int, default=10, help='Depth of decoder')
        parser.add_argument('--encoder', type=str, default='densenet161_bts', help='Type of encoder')
        parser.add_argument('--variance_focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
        parser.add_argument('--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-6)
        parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-2)
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--test_dataset', default='nyu', type=str, help='Dataset for Test [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--data_augmentation', default='bts', type=str, help='Choose data Augmentation Strategy: laina or bts')
        parser.add_argument('--loss', default='bts', type=str, help='loss function')
        parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'log10', 'rmse'], nargs='+', help='which metrics to evaluate')
        return parser

if __name__ == "__main__":
    import visualize
    val_dataset = get_dataset('D:/Documents/data/floorplan3d', 'train', 'noreflection')
    val_dataset.transform = training_preprocess
    for i in range(100):
        item = val_dataset.__getitem__(i)
        visualize.show_item(item)