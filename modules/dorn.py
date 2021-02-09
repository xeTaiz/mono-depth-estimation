import torch
import pytorch_lightning as pl
import criteria
from network import Dorn
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np 
from modules.base_module import BaseModule

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

class DORNModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super(DORNModule, self).__init__(*args, **kwargs)
        self.alpha = torch.tensor(self.hparams.alpha).float()
        self.beta = torch.tensor(self.hparams.beta).float()
        self.ord_num = torch.tensor(self.hparams.ord_num).int()

    def output_size(self):
        return self.hparams.input_size

    def resize(self):
        return self.hparams.input_size[0]

    def setup_criterion(self):
        return criteria.ordLoss()

    def setup_model(self):
        return Dorn.DORN(self.hparams)
    
    def setup_model_from_ckpt(self):
        model = self.setup_model()
        state_dict = {}
        for key, value in torch.load(self.hparams.ckpt, map_location=self.device)["state_dict"].items():
            state_dict[key[6:]] = value
        model.load_state_dict(state_dict)
        return model

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
        self.save_visualization(x, y, y_hat, batch_idx)
        return self.metric_logger.log_val(y_hat, y)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        pred_d, pred_ord = self(x)
        y_hat = self.label_to_depth(pred_d)

        x = torch.nn.functional.interpolate(x, (480, 640), mode='bilinear')
        y = torch.nn.functional.interpolate(y, (480, 640), mode='bilinear')
        y_hat = torch.nn.functional.interpolate(y_hat, (480, 640), mode='bilinear')
        
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.backbone.parameters(), 'lr': self.hparams.learning_rate},
                        {'params': self.model.SceneUnderstandingModule.parameters(), 'lr': self.hparams.learning_rate * 10}]

        optimizer = torch.optim.SGD(train_params, lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.hparams.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('dorn', help='Dorn specific parameters')
        BaseModule.add_default_args(parser, name="dorn", learning_rate=0.0001, batch_size=4)
        parser.add_argument('--pretrained', default=1, type=int, help="Use pretrained backbone.")
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay')
        parser.add_argument('--ord_num', default=68, type=float, help='ordinal number')
        parser.add_argument('--alpha', default=1.0, type=float, help='alpha')
        parser.add_argument('--beta', default=80.0, type=float, help='beta')
        parser.add_argument('--input_size', default=(257, 353), help='image size')
        parser.add_argument('--kernel_size', default=16, type=int, help='kernel size')
        parser.add_argument('--pyramid', default=[4, 8, 12], nargs='+', help='pyramid')
        parser.add_argument('--batch_norm', default=0, type=int, help='Batch Normalization')
        parser.add_argument('--discretization', default="SID", type=str, help='Method for discretization')
        parser.add_argument('--dropout', default=0.5, type=float, help='Dropout rate.')
        parser.add_argument('--data_augmentation', default='laina', type=str, help='Choose data Augmentation Strategy: laina or bts')
        parser.add_argument('--loss', default='dorn', type=str, help='loss function')
        return parser