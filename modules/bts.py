import torch
import pytorch_lightning as pl
import criteria
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from datasets.structured3d_dataset import Structured3DDataset
from network import Bts
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger
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
        image = transforms.ToPILImage()(rgb)
    if isinstance(depth, np.ndarray):
        depth_gt = transforms.ToPILImage()(depth)
    
    #height = image.height
    #width = image.width
    #top_margin = int(height - 400)
    #left_margin = int((width - 1216) / 2)
    #depth_gt = depth_gt.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
    #image       = image.crop((left_margin, top_margin, left_margin + 1216, top_margin + 352))
    
    # To avoid blank boundaries due to pixel registration
    depth_gt = depth_gt.crop((43, 45, 608, 472))
    image = image.crop((43, 45, 608, 472))

    # Random rotation
    angle = transforms.RandomRotation.get_params([-2.5, 2.5])
    image = TF.rotate(image, angle)
    depth_gt = TF.rotate(depth_gt, angle)

    # Random Crop
    i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(416, 544))
    image = TF.crop(image, i, j, h, w)
    depth_gt = TF.crop(depth_gt, i, j, h, w)

    # Random flipping
    if np.random.uniform(0,1) > 0.5:
        image = TF.hflip(image)
        depth_gt = TF.hflip(depth_gt)

    image = np.asarray(image, dtype=np.float32) / 255.0
    depth_gt = np.asarray(depth_gt, dtype=np.float32)

    #depth_gt = depth_gt / 1000.0

    # Random gamma, brightness, color augmentation
    if np.random.uniform(0,1) > 0.5:
        image = augment_image(image)

    image = TF.to_tensor(np.array(image))
    depth_gt = TF.to_tensor(np.array(depth_gt))
    return image, depth_gt

def validation_preprocess(rgb, depth):
    if isinstance(rgb, np.ndarray):
        rgb = transforms.ToPILImage()(rgb)
    if isinstance(depth, np.ndarray):
        depth = transforms.ToPILImage()(depth)
    # Resize
    resize = transforms.Resize(450)
    image = resize(rgb)
    depth_gt = resize(depth)
    # Center crop
    crop = transforms.CenterCrop((416, 544))
    image = crop(image)
    depth_gt = crop(depth_gt)

    #height = image.shape[0]
    #width = image.shape[1]
    #top_margin = int(height - 352)
    #left_margin = int((width - 1216) / 2)
    #image       = image[top_margin:top_margin + 352, left_margin:left_margin + 1216, :]
    #depth_gt = depth_gt[top_margin:top_margin + 352, left_margin:left_margin + 1216]
    
    image = TF.to_tensor(np.array(image, dtype=np.float32))
    depth_gt = TF.to_tensor(np.array(depth_gt, dtype=np.float32))
    
    image /= 255.0
    #depth_gt /= 1000.0
    return image, depth_gt


def get_dataset(path, split, dataset):
    if dataset == 'nyu':
        return NYUDataset(path, split=split, output_size=(416, 544), resize=450)
    elif dataset == 'noreflection':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.NO_REFLECTION, output_size=(416, 544), resize=450)
    elif dataset == 'isotropic':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(416, 544), resize=450)
    elif dataset == 'mirror':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(416, 544), resize=450)
    elif dataset == 'structured3d':
        return Structured3DDataset(path, split=split, dataset_type='perspective', output_size=(416, 544), resize=450)
    else:
        raise ValueError('unknown dataset {}'.format(dataset))

class BtsModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_dataset = get_dataset(self.args.path, 'train', self.args.dataset)
        self.val_dataset = get_dataset(self.args.path, 'val', self.args.eval_dataset)
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
        self.model = Bts.BtsModel(max_depth=self.args.max_depth, bts_size=self.args.bts_size, encoder_version=self.args.encoder)
        print("=> model created.")
        self.criterion = criteria.silog_loss(variance_focus=self.args.variance_focus)
        self.metric_logger = MetricLogger(metrics=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'rmse', 'log10'])

    def forward(self, x):
        _, _, _, _, y_hat = self.model(x)
        return y_hat

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x)
        mask = y > 0.1
        loss = self.criterion(y_hat, y, mask.bool())
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
        train_param = [{'params': self.model.encoder.parameters(), 'weight_decay': self.args.weight_decay},
                       {'params': self.model.decoder.parameters(), 'weight_decay': 0}]
        # Training parameters
        optimizer = torch.optim.AdamW(train_param, lr=self.args.learning_rate, eps=self.args.adam_eps)
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
        parser.add_argument('--bts_size', type=int, default=512, help='initial num_filters in bts')
        parser.add_argument('--max_depth', type=int, default=10, help='Depth of decoder')
        parser.add_argument('--encoder', type=str, default='densenet161_bts', help='Type of encoder')
        parser.add_argument('--variance_focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
        parser.add_argument('--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-6)
        parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-2)
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror]')
        return parser

if __name__ == "__main__":
    import visualize
    val_dataset = get_dataset('G:/data/nyudepthv2', 'train', 'nyu')
    val_dataset.transform = validation_preprocess
    for i in range(100):
        item = val_dataset.__getitem__(i)
        visualize.show_item(item)