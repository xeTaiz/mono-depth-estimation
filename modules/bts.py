from datasets.dataset import BaseDataset
import torch
import criteria
from network import Bts
import numpy as np
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from torchvision.utils import save_image
from modules.base_module import BaseModule, freeze_params
import visualize
from stdepth_utils import depth_sort, composite_layers

def build_lr_optim_lambda(total_iters):
    def lr_optim_lambda(iter):
        return (1.0 - iter / (float(total_iters))) ** 0.9
    return lr_optim_lambda

def bn_init_as_tf(m):
    if isinstance(m, torch.nn.BatchNorm2d):
        m.track_running_stats = True  # These two lines enable using stats (moving mean and var) loaded from pretrained model
        m.eval()                      # or zero mean and variance of one if the batch norm layer has no pretrained values
        m.affine = True
        m.requires_grad = True


def weights_init_xavier(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

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

class BtsModule(BaseModule):

    def setup_model(self):
        model = Bts.BtsModel(max_depth=self.method.max_depth, bts_size=self.method.bts_size, encoder_version=self.method.encoder, out_channels=self.method.out_channels)
        model.decoder.apply(weights_init_xavier)
        if not('bn_no_track_stats' in self.method or 'fix_first_conv_blocks' in self.method): return model
        if self.method.bn_no_track_stats:
            print("Disabling tracking running stats in batch norm layers")
            model.apply(bn_init_as_tf)

        if self.method.fix_first_conv_blocks:
            if 'resne' in self.method.encoder:
                fixing_layers = ['base_model.conv1', 'base_model.layer1.0', 'base_model.layer1.1', '.bn']
            else:
                fixing_layers = ['conv0', 'denseblock1.denselayer1', 'denseblock1.denselayer2', 'norm']
            print("Fixing first two conv blocks")
        elif self.method.fix_first_conv_block:
            if 'resne' in self.method.encoder:
                fixing_layers = ['base_model.conv1', 'base_model.layer1.0', '.bn']
            else:
                fixing_layers = ['conv0', 'denseblock1.denselayer1', 'norm']
            print("Fixing first conv block")
        else:
            if 'resne' in self.method.encoder:
                fixing_layers = ['base_model.conv1', '.bn']
            else:
                fixing_layers = ['conv0', 'norm']
            print("Fixing first conv layer")

        for name, child in model.named_children():
            if not 'encoder' in name:
                continue
            for name2, parameters in child.named_parameters():
                # print(name, name2)
                if any(x in name2 for x in fixing_layers):
                    parameters.requires_grad = False
        return model

    def freeze_encoder(self):
        freeze_params(self.model.encoder)

    def output_size(self):
        return (512, 512)

    def resize(self):
        return 450

    def forward(self, x):
        _, _, _, _, y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x[:, :3])
        loss, pred_full = self.criterion(y_hat, y, x, return_composited=True)
        self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='train')
        return self.metric_logger.log_train(y_hat, y, loss)

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x[:, :3])
        loss, pred_full = self.criterion(y_hat, y, x, return_composited=True)
        self.logger.experiment.log({'val_loss': loss.detach()})
        self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='val')
        return self.metric_logger.log_val(y_hat, y)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        y_hat = self(x[:, :3])
        loss, pred_full = self.criterion(y_hat, y, x, return_composited=True)
        self.logger.experiment.log({'test_loss': loss.detach()})
        if self.pred_path:
            self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='test', write_predictions=True)
            torch.save({'batch': batch, 'prediction': y_hat, 'composited': pred_full}, self.pred_path/f'pred_{batch_idx:04d}.pt')
        else:
            self.save_visualization(x, y, y_hat, pred_full, batch_idx, nam='test')
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        train_param = [{'params': self.model.encoder.parameters(), 'weight_decay': self.method.weight_decay},
                       {'params': self.model.decoder.parameters(), 'weight_decay': 0}]
        # Training parameters
        optimizer = torch.optim.AdamW(train_param, lr=self.method.learning_rate, eps=self.method.adam_eps)
        #total_iters = (len(self.train_loader) // self.method.batch_size) * self.method.max_epochs
        #lr_optim = build_lr_optim_lambda(total_iters)
        #lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_optim)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.method.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    def train_preprocess(self, rgb, depth):
        rgb = transforms.ToPILImage()(rgb)
        depth = map(transforms.ToPILImage(), depth)

        height = rgb.height
        width = rgb.width
        left_margin = width * 0.05
        right_margin = width * (1.0 - 0.05)
        top_margin = height * 0.05
        bot_margin = height * (1.0 - 0.05)
        depth = map(lambda d: d.crop((left_margin, top_margin, right_margin, bot_margin)), depth)
        rgb     = rgb.crop((left_margin, top_margin, right_margin, bot_margin))


        # Random rotation
        angle = transforms.RandomRotation.get_params([-2.5, 2.5])
        rgb = TF.rotate(rgb, angle)
        depth = map(lambda d: TF.rotate(d, angle), depth)

        # Resize
        h = int(np.random.choice([512, 518, 550, 600, 650, 720]))
        resize = transforms.Resize(h)
        rgb = resize(rgb)
        depth = map(resize, depth)


        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=self.output_size())
        rgb = TF.crop(rgb, i, j, h, w)
        depth = map(lambda d: TF.crop(d, i, j, h, w), depth)

        # Random flipping
        if np.random.uniform(0,1) > 0.5:
            rgb = TF.hflip(rgb)
            depth = map(TF.hflip, depth)

        rgb = np.asarray(rgb, dtype=np.float32) / 255.0
        depth = map(lambda d: np.asarray(d, dtype=np.float32) / 255.0, depth)

        # Random gamma, brightness, color augmentation
        # if np.random.uniform(0,1) > 0.5:
        #     rgb = augment_image(rgb)

        rgb = TF.to_tensor(np.array(rgb))
        depth = map(lambda d: TF.to_tensor(np.array(d)), depth)

        return rgb, torch.cat(list(depth), dim=0)

    def val_preprocess(self, rgb, depth):
        rgb = transforms.ToPILImage()(rgb)
        depth = map(transforms.ToPILImage(), depth)
        # Resize
        resize = transforms.Resize(self.resize())
        rgb = resize(rgb)
        depth = map(resize, depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size())
        rgb = crop(rgb)
        depth = map(crop, depth)

        rgb = TF.to_tensor(np.array(rgb, dtype=np.float32))
        depth = map(lambda d: TF.to_tensor(np.array(d, dtype=np.float32)) / 255.0, depth)

        rgb /= 255.0
        return rgb, torch.cat(list(depth), dim=0)

    def test_preprocess(self, rgb, depth):
        return self.val_preprocess(rgb, depth)

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('bts', help='Bts specific parameters')
        BaseModule.add_default_args(parser, name="bts", learning_rate=0.0001, batch_size=8)
        parser.add_argument('--lr_patience', default=2, type=int, help='Patience of LR scheduler.')
        parser.add_argument('--bts_size', type=int, default=512, help='initial num_filters in bts')
        parser.add_argument('--out-channels', type=int, default=20, help='Number of out channels')
        parser.add_argument('--max_depth', type=int, default=1.0, help='Depth of decoder')
        parser.add_argument('--encoder', type=str, default='densenet161_bts', help='Type of encoder')
        parser.add_argument('--variance_focus', type=float, default=0.85, help='lambda in paper: [0, 1], higher value more focus on minimizing variance of error')
        parser.add_argument('--adam_eps', type=float, help='epsilon in Adam optimizer', default=1e-3)
        parser.add_argument('--weight_decay', type=float, help='weight decay factor for optimization', default=1e-2)
        parser.add_argument('--data_augmentation', default='bts', type=str, help='Choose data Augmentation Strategy: laina or bts')
        parser.add_argument('--loss', default='mae+composite', type=str, help='loss function')
        parser.add_argument('--fix_first_conv_blocks', help='if set, will fix the first two conv blocks', action='store_true')
        parser.add_argument('--fix_first_conv_block', help='if set, will fix the first conv block', action='store_true')
        parser.add_argument('--bn_no_track_stats', help='if set, will not track running stats in batch norm layers', action='store_true')
        return parser
