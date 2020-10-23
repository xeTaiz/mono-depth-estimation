import torch
import pytorch_lightning as pl
import criteria
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from datasets.structured3d_dataset import Structured3DDataset
from network import VNL
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import cv2

RGB_PIXEL_MEANS = (0.485, 0.456, 0.406)  # (102.9801, 115.9465, 122.7717)
RGB_PIXEL_VARS = (0.229, 0.224, 0.225)  # (1, 1, 1)

def set_reshape_crop():
    raw_size = np.array([385, 416, 448, 480, 512, 544, 576, 608, 640])
    size_index = np.random.randint(0, 9)
    crop_size = raw_size[size_index]
    resize_ratio = float(385 / crop_size)
    return crop_size, resize_ratio

def scale_torch(img, scale):
    img = TF.to_tensor(np.array(img, dtype=np.float32))
    img /= scale
    if img.size(0) == 3:
        img = transforms.Normalize(RGB_PIXEL_MEANS, RGB_PIXEL_VARS)(img)
    else:
        img = transforms.Normalize((0,), (1,))(img)
    return img

def training_preprocess(rgb, depth):
    if isinstance(rgb, np.ndarray):
        rgb = transforms.ToPILImage()(rgb)
    if isinstance(depth, np.ndarray):
        depth = transforms.ToPILImage()(depth)
    crop_size, resize_ratio = set_reshape_crop()
    # Resize 
    resize = transforms.Resize(int(crop_size))
    rgb = resize(rgb)
    depth = resize(depth)
    # Random Crop
    i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=(385, 385))
    rgb = TF.crop(rgb, i, j, h, w)
    depth = TF.crop(depth, i, j, h, w)
    # Random flipping
    if np.random.uniform(0,1) > 0.5:
        rgb = TF.hflip(rgb)
        depth = TF.hflip(depth)

    rgb = scale_torch(rgb, 255.0)
    depth = scale_torch(depth, resize_ratio)
    return rgb, depth

def validation_preprocess(rgb, depth):
    if isinstance(rgb, np.ndarray):
        rgb = transforms.ToPILImage()(rgb)
    if isinstance(depth, np.ndarray):
        depth = transforms.ToPILImage()(depth)
    # Resize 
    resize = transforms.Resize(385)
    rgb = resize(rgb)
    depth = resize(depth)
    # Random Crop
    crop = transforms.CenterCrop((385, 385))
    rgb = crop(rgb)
    depth = crop(depth)

    rgb = scale_torch(rgb, 255.0)
    depth = scale_torch(depth, 1)
    return rgb, depth

def get_dataset(path, split, dataset):
    if dataset == 'nyu':
        return NYUDataset(path, split=split, output_size=(385, 385), resize=400)
    elif dataset == 'noreflection':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.NO_REFLECTION, output_size=(385, 385), resize=400)
    elif dataset == 'isotropic':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=(385, 385), resize=400)
    elif dataset == 'mirror':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=(385, 385), resize=400)
    elif dataset == 'structured3d':
        return Structured3DDataset(path, split=split, dataset_type='perspective', output_size=(385, 385), resize=400)
    else:
        raise ValueError('unknown dataset {}'.format(dataset))


class VNLModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args.depth_min_log = np.log10(self.args.depth_min)
        self.args.depth_bin_interval = (np.log10(self.args.depth_max) - np.log10(self.args.depth_min)) / self.args.dec_out_c
        self.args.wce_loss_weight = [[np.exp(-0.2 * (i - j) ** 2) for i in range(self.args.dec_out_c)] for j in np.arange(self.args.dec_out_c)]
        self.args.depth_bin_border = np.array([np.log10(self.args.depth_min) + self.args.depth_bin_interval * (i + 0.5) for i in range(self.args.dec_out_c)])
        self.train_dataset = get_dataset(self.args.path, 'train', self.args.dataset)
        self.val_dataset = get_dataset(self.args.path, 'val', self.args.eval_dataset)
        if self.args.data_augmentation == 'vnl':
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
        self.model = VNL.MetricDepthModel(self.args)
        print("=> model created.")
        self.criterion = criteria.ModelLoss(self.args)
        self.metric_logger = MetricLogger(metrics=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'rmse', 'log10'])

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth < 0.
        depth[depth < self.args.depth_min] = self.args.depth_min
        depth[depth > self.args.depth_max] = self.args.depth_max
        bins = ((torch.log10(depth) - self.args.depth_min_log) / self.args.depth_bin_interval).to(torch.int)
        bins[invalid_mask] = self.args.dec_out_c + 1
        bins[bins == self.args.dec_out_c] = self.args.dec_out_c - 1
        depth[invalid_mask] = -1.0
        return bins

    def bins_to_depth(self, depth_bin):
        """
        Transfer n-channel discrate depth bins to 1-channel conitnuous depth
        :param depth_bin: n-channel output of the network, [b, c, h, w]
        :return: 1-channel depth, [b, 1, h, w]
        """
        depth_bin = depth_bin.permute(0, 2, 3, 1) #[b, h, w, c]
        depth_bin_border = torch.tensor(self.args.depth_bin_border, dtype=torch.float32).cuda()
        depth = depth_bin * depth_bin_border
        depth = torch.sum(depth, dim=3, dtype=torch.float32, keepdim=True)
        depth = 10 ** depth
        depth = depth.permute(0, 3, 1, 2)  # [b, 1, h, w]
        return depth

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
        y /= 10.0
        pred_logits, pred_cls = self(x)
        loss = self.criterion(self.bins_to_depth(pred_cls), pred_logits, self.depth_to_bins(y), y)
        y_hat = self.predicted_depth_map(pred_logits, pred_cls)
        return self.metric_logger.log_train(y_hat, y, loss)

    def predicted_depth_map(self, logits, cls):
        if self.args.prediction_method == 'classification':
            pred_depth = self.bins_to_depth(cls)
        elif self.args.prediction_method == 'regression':
            pred_depth = torch.nn.functional.sigmoid(logits)
        else:
            raise ValueError("Unknown prediction methods")
        return pred_depth

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y /= 10.0
        pred_logits, pred_cls = self(x)
        y_hat = self.predicted_depth_map(pred_logits, pred_cls)
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
        encoder_params = []
        encoder_params_names = []
        decoder_params = []
        decoder_params_names = []
        nograd_param_names = []

        for key, value in dict(self.model.named_parameters()).items():
            if value.requires_grad:
                if 'res' in key:
                    encoder_params.append(value)
                    encoder_params_names.append(key)
                else:
                    decoder_params.append(value)
                    decoder_params_names.append(key)
            else:
                nograd_param_names.append(key)

        lr_encoder = self.args.learning_rate
        lr_decoder = self.args.learning_rate * self.args.scale_decoder_lr
        weight_decay = self.args.weight_decay

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'weight_decay': weight_decay},
            ]
        optimizer = torch.optim.SGD(net_params, momentum=0.9)
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
        parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay')
        parser.add_argument('--batch_size',    default=8,     type=int,   help='Batch Size')
        parser.add_argument('--worker',        default=6,      type=int,   help='Number of workers for data loader')
        parser.add_argument('--path', required=True, type=str, help='Path to NYU')
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--encoder', default='resnext50_32x4d_body_stride16', type=str, help='Encoder architecture')
        parser.add_argument('--init_type', default='xavier', type=str, help='Weight initialization')
        parser.add_argument('--pretrained', default=0, type=int, help='pretrained backbone')
        parser.add_argument('--enc_dim_in', nargs='+', default=[64, 256, 512, 1024, 2048], help='encoder input features')
        parser.add_argument('--enc_dim_out', nargs='+', default=[512, 256, 256, 256], help='encoder output features')
        parser.add_argument('--dec_dim_in', nargs='+', default=[512, 256, 256, 256, 256, 256], help='decoder input features')
        parser.add_argument('--dec_dim_out', nargs='+', default=[256, 256, 256, 256, 256], help='decoder output features')
        parser.add_argument('--dec_out_c', default=150, type=int, help='decoder output channels')
        parser.add_argument('--crop_size', default=(385, 385), help='Crop size for mobilenet')
        parser.add_argument('--scale_decoder_lr', default=0.1, type=float, help='Scaling of LR for decoder')
        parser.add_argument('--freeze_backbone', action='store_true', help='Freeze backbone')
        parser.add_argument('--depth_min', default=0.01, type=float, help='minimum depth')
        parser.add_argument('--depth_max', default=1.7, type=float, help='maximum depth')
        parser.add_argument('--focal_x', default=519.0, type=float, help='focal x')
        parser.add_argument('--focal_y', default=519.0, type=float, help='focal y')
        parser.add_argument('--diff_loss_weight', default=6, type=float, help='diff loss weight')
        parser.add_argument('--prediction_method', default='classification', type=str, help='type of prediction. classification or regression')
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--data_augmentation', default='laina', type=str, help='Choose data Augmentation Strategy: laina or vnl')
        return parser

if __name__ == "__main__":
    import visualize
    val_dataset = get_dataset('D:/Documents/data/floorplan3d', 'train', 'noreflection')
    val_dataset.transform = validation_preprocess
    for i in range(100):
        item = val_dataset.__getitem__(i)
        visualize.show_item(item)