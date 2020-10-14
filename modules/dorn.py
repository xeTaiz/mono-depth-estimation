import torch
import pytorch_lightning as pl
import criteria
from datasets.nyu_dataloader import NYUDataset
from datasets.floorplan3d_dataloader import Floorplan3DDataset, DatasetType
from datasets.structured3d_dataset import Structured3DDataset
from network import Dorn
from argparse import ArgumentParser
import visualize
from metrics import MetricLogger

def get_dataset(path, split, dataset, img_size):
    if dataset == 'nyu':
        return NYUDataset(path, split=split, output_size=img_size, resize=img_size[0])
    elif dataset == 'noreflection':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.NO_REFLECTION, output_size=img_size, resize=img_size[0])
    elif dataset == 'isotropic':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_MATERIAL, output_size=img_size, resize=img_size[0])
    elif dataset == 'mirror':
        return Floorplan3DDataset(path, split=split, datast_type=DatasetType.ISOTROPIC_PLANAR_SURFACES, output_size=img_size, resize=img_size[0])
    elif dataset == 'structured3d':
        return Structured3DDataset(path, split=split, dataset_type='perspective', output_size=img_size, resize=img_size[0])
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
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.train_loader = torch.utils.data.DataLoader(get_dataset(self.args.path, 'train', self.args.dataset, self.args.input_size),
                                                    batch_size=args.batch_size, 
                                                    shuffle=True, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(get_dataset(self.args.path, 'val', self.args.eval_dataset, self.args.input_size),
                                                    batch_size=1, 
                                                    shuffle=False, 
                                                    num_workers=args.worker, 
                                                    pin_memory=True)
        self.skip = len(self.val_loader) // 9
        print("=> creating Model")
        self.model = Dorn.DORN(self.args)
        print("=> model created.")
        self.criterion = criteria.OrdinalRegressionLoss(self.args.ord_num, self.args.beta, self.args.discretization)
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
        prob, y_hat = self(x)
        loss = self.criterion(prob, y)
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        _, y_hat = self(x)
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
        train_params = [{'params': self.model.backbone.parameters(), 'lr': self.args.learning_rate},
                        {'params': self.model.SceneUnderstandingModule.parameters(), 'lr': self.args.learning_rate * 10}]

        optimizer = torch.optim.SGD(train_params, lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
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
        parser.add_argument('--pretrained', default=1, type=int, help="Use pretrained backbone.")
        parser.add_argument('--learning_rate', default=0.0001, type=float, help='Learning Rate')
        parser.add_argument('--batch_size',    default=4,     type=int,   help='Batch Size')
        parser.add_argument('--worker',        default=6,      type=int,   help='Number of workers for data loader')
        parser.add_argument('--path', required=True, type=str, help='Path to NYU')
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay')
        parser.add_argument('--dataset', default='nyu', type=str, help='Dataset for Training [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--eval_dataset', default='nyu', type=str, help='Dataset for Validation [nyu, noreflection, isotropic, mirror]')
        parser.add_argument('--ord_num', default=90, type=float, help='ordinal number')
        parser.add_argument('--beta', default=80.0, type=float, help='beta')
        parser.add_argument('--gamma', default=1.0, type=float, help='gamma')
        parser.add_argument('--input_size', default=(257, 353), help='image size')
        parser.add_argument('--kernel_size', default=16, type=int, help='kernel size')
        parser.add_argument('--pyramid', default=[4, 8, 12], nargs='+', help='pyramid')
        parser.add_argument('--batch_norm', default=0, type=int, help='Batch Normalization')
        parser.add_argument('--discretization', default="SID", type=str, help='Method for discretization')
        parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate.')
        return parser