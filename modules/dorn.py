import torch
import pytorch_lightning as pl
import criteria
from datasets.dataset import ConcatDataset
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from datasets.structured3d_dataset import Structured3DDataset
from network import Dorn
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np 


def get_dataset(path, split, dataset, img_size):
    path = path.split('+')
    if dataset == 'nyu':
        return NYUDataset(path[0], split=split, output_size=img_size, resize=img_size[0])
    elif dataset == 'noreflection':
        return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.NO_REFLECTION, output_size=img_size, resize=img_size[0])
    elif dataset == 'isotropic':
        return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=img_size, resize=img_size[0])
    elif dataset == 'mirror':
        return Floorplan3DDataset(path[0], split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=img_size, resize=img_size[0])
    elif dataset == 'structured3d':
        return Structured3DDataset(path[0], split=split, dataset_type='perspective', output_size=img_size, resize=img_size[0])
    elif '+' in dataset:
        datasets = [get_dataset(p, split, d, img_size) for p, d in zip(path, dataset.split('+'))]
        return ConcatDataset(datasets)
    else:
        raise ValueError('unknown dataset {}'.format(dataset))


def get_depth_sid(dataset, labels):
    if dataset == 'kitti':
        min = 0.001
        max = 80.0
        K = 71.0
    elif dataset == 'nyu':
        min = 0.02
        max = 10.0
        K = 68.0
    elif dataset == 'floorplan3d':
        min = 0.0552
        max = 10.0
        K = 68.0
    else:
        print('No Dataset named as ', dataset)

    alpha_ = torch.tensor(min).float()
    beta_ = torch.tensor(max).float()
    K_ = torch.tensor(K).float()

    # print('label size:', labels.size())
    if not alpha_ == 0.0:
        depth = torch.exp(torch.log(alpha_) + torch.log(beta_ / alpha_) * labels / K_)
    else:
        depth = torch.exp(torch.log(beta_) * labels / K_)
    # depth = alpha_ * (beta_ / alpha_) ** (labels.float() / K_)
    # print(depth.size())
    return depth.float()

def get_labels_sid(dataset, depth):
    if dataset == 'kitti':
        alpha = 0.001
        beta = 80.0
        K = 71.0
    elif dataset == 'nyu':
        alpha = 0.02
        beta = 10.0
        K = 68.0
    elif dataset == 'floorplan3d':
        alpha = 0.0552
        beta = 10.0
        K = 68.0
    else:
        print('No Dataset named as ', dataset)

    alpha = torch.tensor(alpha).float()
    beta = torch.tensor(beta).float()
    K = torch.tensor(K).float()

    if not alpha == 0.0:
        labels = K * torch.log(depth / alpha) / torch.log(beta / alpha)
    else:
        labels = K * torch.log(depth) / torch.log(beta)
    return labels.int()

class DORNModule(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.alpha = torch.tensor(self.hparams.alpha).float()
        self.beta = torch.tensor(self.hparams.beta).float()
        self.ord_num = torch.tensor(self.hparams.ord_num).int()
        self.train_dataset = get_dataset(self.hparams.path, 'train', self.hparams.dataset, self.hparams.input_size)
        self.val_dataset = get_dataset(self.hparams.path, 'val', self.hparams.eval_dataset, self.hparams.input_size)
        self.test_dataset = get_dataset(self.hparams.path, 'test', self.hparams.test_dataset, self.hparams.input_size)
        
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
        self.model = Dorn.DORN(self.hparams)
        print("=> model created.")
        self.criterion = criteria.ordLoss()
        self.metric_logger = MetricLogger(metrics=self.hparams.metrics)

    def label_to_depth(self, label):
        if self.hparams.discretization == "SID":
            depth = torch.exp(torch.log(self.alpha) + torch.log(self.beta / self.alpha) * label / self.ord_num)
        else:
            depth = self.alpha + (self.beta - self.alpha) * label / self.ord_num
        return depth

    def depth_to_label(self, depth):
        if self.hparams.discretization == "SID":
            label = self.ord_num * torch.log(depth / self.alpha) / torch.log(self.beta / self.alpha)
        else:
            label = self.ord_num * (depth - self.alpha) / (self.beta - self.alpha)
        return label

    def overlapping_window_method(self, image):
        def get_crop(x, size):
            (h,w) = x.shape[-2:]
            (height, width) = size
            h_diff = h - height
            w_diff = w - width
            assert h_diff >= 0, "wrong size"
            assert w_diff >= 0, "wrong size"
            i = np.random.randint(0, h_diff+1)
            j = np.random.randint(0, w_diff+1)
            return i,j,height,width
        s = np.random.uniform(1,1.5)
        input_size = image.shape[-2:]
        [height, width] = (np.array(input_size) * s).astype(int)
        pred_d, pred_ord = self(image)
        y_hat = self.label_to_depth(pred_d)

        resized = torch.nn.functional.interpolate(image, (height, width), mode='bilinear')
        y_hat = torch.nn.functional.interpolate(y_hat, (height, width), mode='bilinear')
        c = 20
        counts  = torch.ones((1, 1, height,width), device=image.device)

        batch = []
        params = []
        for q in range(c):
            i,j,h,w = get_crop(resized, input_size)
            batch.append(resized[:, :, i:i+h, j:j+w])
            params.append((i,j,h,w))
        batch = torch.cat(batch, dim=0)
        pred_d, pred_ord = self(batch)
        y_hat_crop = self.label_to_depth(pred_d)
        y_hat_crop = y_hat_crop * s
        
        for q in range(c):
            (i,j,h,w) = params[q]
            counts[..., i:i+h, j:j+w] += 1
            y_hat[..., i:i+h, j:j+w] += y_hat_crop[q]
            
        counts = counts.type(torch.float32)
        y_hat = y_hat.type(torch.float32)
        y_hat = y_hat / counts
        y_hat = torch.nn.functional.interpolate(y_hat, image.shape[-2:])
        return y_hat

    def forward(self, x):
        y_hat = self.model(x)
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
        pred_d, pred_ord = self(x)
        y_hat = self.label_to_depth(pred_d)
        y_sid = self.depth_to_label(y)
        loss = self.criterion(pred_ord, y_sid)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        pred_d, pred_ord = self(x)
        y_hat = self.label_to_depth(pred_d)
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
        self.skip = 1
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        x_ = torch.nn.functional.interpolate(x, self.hparams.input_size, mode='bilinear')
        pred_d, pred_ord = self(x_)
        y_hat = self.label_to_depth(pred_d)
        y_hat = torch.nn.functional.interpolate(y_hat, y.shape[-2:], mode='bilinear')
        if self.hparams.test_dataset == 'nyu':
            mask = (45, 471, 41, 601)
            x = x[..., mask[0]:mask[1], mask[2]:mask[3]]
            y = y[..., mask[0]:mask[1], mask[2]:mask[3]]
            y_hat = y_hat[..., mask[0]:mask[1], mask[2]:mask[3]] 
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.backbone.parameters(), 'lr': self.hparams.learning_rate},
                        {'params': self.model.SceneUnderstandingModule.parameters(), 'lr': self.hparams.learning_rate * 10}]

        optimizer = torch.optim.SGD(train_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
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
        parser.add_argument('--pretrained', default=1, type=int, help="Use pretrained backbone.")
        parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning Rate')
        parser.add_argument('--batch_size',    default=4,     type=int,   help='Batch Size')
        parser.add_argument('--worker',        default=6,      type=int,   help='Number of workers for data loader')
        parser.add_argument('--path', required=True, type=str, help='Path to NYU')
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay')
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--ord_num', default=68, type=float, help='ordinal number')
        parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
        parser.add_argument('--beta', default=80.0, type=float, help='beta')
        parser.add_argument('--input_size', default=(257, 353), help='image size')
        parser.add_argument('--kernel_size', default=16, type=int, help='kernel size')
        parser.add_argument('--pyramid', default=[4, 8, 12], nargs='+', help='pyramid')
        parser.add_argument('--batch_norm', default=0, type=int, help='Batch Normalization')
        parser.add_argument('--discretization', default="SID", type=str, help='Method for discretization')
        parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate.')
        parser.add_argument('--test_dataset', default='nyu', type=str, help='Dataset for Test [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--data_augmentation', default='laina', type=str, help='Choose data Augmentation Strategy: laina or bts')
        parser.add_argument('--loss', default='dorn', type=str, help='loss function')
        parser.add_argument('--metrics', default=['delta1', 'delta2', 'delta3', 'mse', 'mae', 'log10', 'rmse'], nargs='+', help='which metrics to evaluate')
        return parser


if __name__ == "__main__":

    def get_crop(x, size):
        (h,w) = x.shape[-2:]
        (height, width) = size
        h_diff = h - height
        w_diff = w - width
        assert h_diff >= 0, "wrong size"
        assert w_diff >= 0, "wrong size"
        i = np.random.randint(0, h_diff+1)
        j = np.random.randint(0, w_diff+1)
        return i,j,height,width

            
    def overlapping_window_method_2(image):
        s = np.random.uniform(1,1.5)
        input_size = image.shape[-2:]
        [h, w] = np.array(input_size) * 1.5
        height = int(h)
        width = int(w)

        resized = torch.nn.functional.interpolate(image, (height, width), mode='bilinear')
        y_hat = torch.zeros((1,1, height, width))
        c = 1000
        counts  = torch.ones((1, 1, height,width), device=x.device)

        for q in range(c):
            i,j,h,w = get_crop(resized, input_size)
            x_crop = resized[:, :, i:i+h, j:j+w]
            pred_d, pred_ord = self(crop)
            y_hat_crop = self.label_to_depth(pred_d)
            counts[..., i:i+h, j:j+w] += 1
            y_hat[..., i:i+h, j:j+w] += y_hat_crop
        counts = counts.type(torch.float32)
        y_hat = y_hat.type(torch.float32)
        y_hat = y_hat / counts
        y_hat = torch.nn.functional.interpolate(y_hat, image.shape[-2:])
        return y_hat
    from datasets.nyu_dataloader import NYUDataset
    from visualize import show_item
    dataset = NYUDataset(path="G:/data/nyudepthv2", split="test")
    img, depth = dataset.__getitem__(0)
    y_hat = overlapping_window_method_2(depth.cuda())
    
    show_item((img, y_hat.squeeze(1)))