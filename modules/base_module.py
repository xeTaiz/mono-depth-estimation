import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from datasets.dataset import ConcatDataset
from datasets.nyu_dataloader import get_nyu_dataset
from datasets.floorplan3d_dataloader import get_floorplan3d_dataset
from datasets.structured3d_dataset import get_structured3d_dataset
from datasets.stdepth import get_stdepth_dataset
from datasets.stdepth_multi import get_stdepthmulti_dataset
from datasets.stdepth_multi2 import get_stdepthmulti2_dataset
from metrics import MetricLogger
import visualize
import criteria
from stdepth_utils import depth_sort, composite_layers, dssim2d
from torchvision import transforms
import torchvision.transforms.functional as TF
from torchvision.utils import save_image, make_grid
import matplotlib.pyplot as plt
import numpy as np
import wandb

NAME2FUNC = {
    "nyu": get_nyu_dataset,
    "structured3d": get_structured3d_dataset,
    "floorplan3d": get_floorplan3d_dataset,
    "stdepth": get_stdepth_dataset,
    "stdepthmulti": get_stdepthmulti_dataset,
    "stdepthmulti2": get_stdepthmulti2_dataset
}

def freeze_params(m):
    for p in m.parameters():
        p.requires_grad = False

class BaseModule(pl.LightningModule):
    def __init__(self, globals, training, validation, test, method=None, *args, **kwargs):
        super().__init__()
        self.img_merge = {}
        self.save_hyperparameters()
        if method is None:
            self.method = self.hparams.hparams.method if 'hparams' in self.hparams else self.hparams
        else:
            self.method = method
        self.globals = globals
        self.training = training
        self.validation = validation
        self.test = test
        self.train_dataset, self.val_dataset, self.test_dataset = self.get_dataset()
        if self.train_dataset:
            self.train_dataset.transform = self.train_preprocess
            self.train_loader = torch.utils.data.DataLoader(self.train_dataset,
                                                        batch_size=self.method.batch_size,
                                                        shuffle=True,
                                                        num_workers=self.globals.worker,
                                                        prefetch_factor=4,
                                                        pin_memory=True)
            self.single_layer = training.single_layer if 'single_layer' in training else True
        else: self.train_loader = None
        if self.val_dataset:
            self.val_dataset.transform = self.val_preprocess
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        num_workers=self.globals.worker,
                                                        prefetch_factor=4,
                                                        pin_memory=True)
            self.single_layer = validation.single_layer if 'single_layer' in validation else True
        else: self.val_loader = None
        if self.test_dataset:
            self.test_dataset.transform = self.test_preprocess
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=1,
                                                        shuffle=False,
                                                        prefetch_factor=4,
                                                        num_workers=self.globals.worker,
                                                        pin_memory=True)
            self.single_layer = test.single_layer if 'single_layer' in test else True
        else: self.test_loader = None
        print("=> creating Model")
        self.model = self.setup_model()
        print("=> model created.")
        self.criterion = self.setup_criterion()
        self.metric_logger = MetricLogger(metrics=self.globals.metrics, module=self)
        self.skip = {}
        if self.val_loader:   self.skip['val'] =   len(self.val_loader) // 9
        if self.train_loader: self.skip['train'] = len(self.train_loader) // 9
        if self.test_loader:  self.skip['test'] =  len(self.test_loader) // 9

        if 'freeze_encoder' in self.method and self.method.freeze_encoder:
            print("freezing encoder")
            self.freeze_encoder()

    def freeze_encoder(self):
        raise NotImplementedError()

    def output_size(self):
        raise NotImplementedError()

    def resize(self):
        raise NotImplementedError()

    def setup_model(self):
        raise NotImplementedError()

    def setup_model_from_ckpt(self):
        raise NotImplementedError()

    def setup_criterion(self):
        _silog_loss = criteria.silog_loss(variance_focus=self.method.variance_focus)
        def silog_loss(pred, targ):
            return torch.nan_to_num(_silog_loss(pred, targ))
        depth_w = self.method.depth_loss_weight
        comp_w = self.method.comp_loss_weight
        def _loss(pred, targ, rgba, return_composited=False, return_loss_dict=False):
            mask1 = targ[:, [9]] > 0.0 if self.single_layer else targ[:, [19]] > 0.0
            mask4 = mask1.expand(-1, 4, -1, -1)
            mask3 = mask1.expand(-1, 3, -1, -1)
            mask8 = mask1.expand(-1, 8, -1, -1)
            maskN = mask1.expand(-1, targ.size(1), -1, -1)
            depth_idx = (slice(None), slice(8,9)) if self.single_layer else (slice(None), slice(16, 19))
            maskD = targ[depth_idx] > 0.0
            loss = 0.0
            loss_dict = {}
            # Composite for vis (and possibly loss)
            if return_composited or 'composite' in self.method.loss:
                if self.single_layer:
                    targ_full = rgba
                    l1, back = pred[:, :4], pred[:, 4:8]
                    pred_full = composite_layers(torch.stack([l1, back], dim=1))
                else:
                    targ_full = torch.cat([rgba, targ[:, [19]]], dim=1)
                    l1 = torch.cat([pred[:,  :4],  pred[:, [16]]], dim=1)
                    l2 = torch.cat([pred[:, 4:8],  pred[:, [17]]], dim=1)
                    l3 = torch.cat([pred[:, 8:12], pred[:, [18]]], dim=1)
                    back = pred[:, 12:16].unsqueeze(1) # Add Layer dim for concat
                    sorted_layers = depth_sort(torch.stack([l1, l2, l3], dim=1))[:, :, :4] # Discard depth from here
                    pred_full = composite_layers(torch.cat([sorted_layers, back], dim=1))

            if 'silma' in self.method.loss:
                loss_dict['depth_silog'] = depth_w * torch.nan_to_num(silog_loss(pred[depth_idx][maskD], targ[depth_idx][maskD]))
                loss_dict['depth_mae']   = F.l1_loss(pred[:, :8][mask8], targ[:, :8][mask8])
            if 'mse' in self.method.loss:
                loss_dict['all_mse'] = F.mse_loss(pred[maskN], targ[maskN])
                loss_dict['all_mse'] += depth_w * F.mse_loss(pred[depth_idx][maskD], targ[depth_idx][maskD])
            if 'mae' in self.method.loss:
                loss_dict['all_mae'] = F.l1_loss(pred[maskN], targ[maskN])
                loss_dict['all_mae'] += depth_w * F.l1_loss(pred[depth_idx][maskD], targ[depth_idx][maskD])
            if 'allssim' in self.method.loss:
                loss_dict['all_ssim'] = dssim2d(
                    torch.clamp(pred, 0.0, 1.0),
                    torch.clamp(targ, 0.0, 1.0), reduction='none')[maskN].mean()
            if 'colorssim' in self.method.loss:
                mask8 = mask1.expand(-1, 8, -1, -1)
                loss_dict['color_ssim'] = dssim2d(
                    torch.clamp(pred[:, :8], 0.0, 1.0),
                    torch.clamp(targ[:, :8], 0.0, 1.0), reduction='none')[mask8].mean()
            if 'composite' in self.method.loss:
                comp_loss = comp_w * F.mse_loss(pred_full[mask4], targ_full[mask4], reduction='none')
                loss_dict['composite_loss'] = torch.mean(torch.nan_to_num(comp_loss))
            if 'fbdivergence':
                # implements cosine similarity between fronts and backs
                # "front pred back gt" / "front gt back pred"  magnitudes
                fpbg_mag = (torch.linalg.vector_norm(pred[:,  :3], dim=1, keepdim=True) * 
                            torch.linalg.vector_norm(targ[:, 4:7], dim=1, keepdim=True)) + 1e-3
                fgbp_mag = (torch.linalg.vector_norm(pred[:, 4:7], dim=1, keepdim=True) * 
                            torch.linalg.vector_norm(targ[:,  :3], dim=1, keepdim=True)) + 1e-3
                # manual dot products, divided by magnitude cos_sim = dot(A, B) / (|A|*|B|)
                fb_div_loss = ((pred[:,  :3] * targ[:, 4:7] / fpbg_mag).sum(dim=1) +
                               (pred[:, 4:7] * targ[:,  :3] / fgbp_mag).sum(dim=1))[mask1.squeeze(1)]
                loss_dict['fb_divergence'] = fb_div_loss.mean()

            loss = torch.stack(list(loss_dict.values())).sum()
            # Assemble tuple to return
            ret = [loss]
            if return_composited:
                ret.append(pred_full)
            if return_loss_dict:
                ret.append({k: v.detach() for k,v in loss_dict.items()})

            return tuple(ret)

        return _loss

    def forward(self, x):
        raise NotImplementedError()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def training_step(self, batch, batch_idx):
        raise NotImplementedError()

    def validation_step(self, batch, batch_idx):
        raise NotImplementedError()

    def test_step(self, batch, batch_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

    def train_preprocess(self, rgb, depth):
        s = np.random.uniform(1, 1.5)
        depth = map(lambda d: d / s, depth)

        rgb = transforms.ToPILImage()(rgb)
        depth = map(transforms.ToPILImage(), depth)
        # color jitter
        #rgb = transforms.ColorJitter(0.4, 0.4, 0.4)(rgb)
        # Resize
        resize = transforms.Resize(self.resize())
        rgb = resize(rgb)
        depth = map(resize, depth)
        # Random Rotation
        angle = np.random.uniform(-5,5)
        rgb = TF.rotate(rgb, angle)
        depth = map(lambda d: TF.rotate(d, angle), depth)
        # Resize
        resize = transforms.Resize(int(self.resize() * s))
        rgb = resize(rgb)
        depth = map(resize, depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size())
        rgb = crop(rgb)
        depth = map(crop, depth)
        # Random horizontal flipping
        if np.random.uniform(0,1) > 0.5:
            rgb = TF.hflip(rgb)
            depth = map(TF.hflip, depth)
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb, dtype=np.float32) / 255.0)
        depth = map(lambda d: TF.to_tensor(np.array(d, dtype=np.float32) / 255.0), depth)
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
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb, dtype=np.float32) / 255.0)
        depth = map(lambda d: TF.to_tensor(np.array(d, dtype=np.float32) / 255.0), depth)
        return rgb, torch.cat(list(depth), dim=0)

    def test_preprocess(self, rgb, depth):
        return self.val_preprocess(rgb, depth)

    def save_visualization(self, x, y, y_hat, pred_full, batch_idx, nam='val', write_predictions=False):
        x = x[0] if x.ndim == 4 else x
        y = y[0] if y.ndim == 4 else y
        y_hat = y_hat[0] if y_hat.ndim == 4 else y_hat
        pred_full = pred_full[0] if pred_full.ndim == 4 else y_hat

        if(batch_idx < 4 * self.skip[nam]) and (batch_idx % self.skip[nam] == 0) or write_predictions:
            if self.single_layer:
                fig = visualize.create_stdepth_plot_single(y_hat, y, x, pred_full)
            else:
                fig = visualize.create_stdepth_plot(y_hat, y, x, pred_full)
            if write_predictions:
                fig.savefig(self.pred_path/f'pred_{batch_idx:04d}.png')
            else:
                self.logger.experiment.log({f'{nam}_visualization_{batch_idx // self.skip[nam]}': fig})
            plt.close(fig)


    def get_dataset(self):
        training_dataset = []
        validation_dataset = []
        test_dataset = []
        for name, data_setargs in self.training:
            training_dataset.append(NAME2FUNC[name](data_setargs, 'train', self.output_size(), self.resize()))
        for name, data_setargs in self.validation:
            validation_dataset.append(NAME2FUNC[name](data_setargs, 'val', self.output_size(), self.resize()))
        for name, data_setargs in self.test:
            test_dataset.append(NAME2FUNC[name](data_setargs, 'test', self.output_size(), self.resize()))

        if len(training_dataset) > 1:   training_dataset = [ConcatDataset(training_dataset)]
        if len(validation_dataset) > 1: validation_dataset = [ConcatDataset(validation_dataset)]
        if len(test_dataset) > 1:       test_dataset = [ConcatDataset(test_dataset)]

        training_dataset = training_dataset[0] if training_dataset else None
        validation_dataset = validation_dataset[0] if validation_dataset else None
        test_dataset = test_dataset[0] if test_dataset else None
        return training_dataset, validation_dataset, test_dataset

    @staticmethod
    def add_default_args(parser, name, learning_rate, batch_size, ckpt=None):
        parser.add_argument('--name', default=name, type=str, help="Method for training.")
        parser.add_argument('--learning_rate', default=learning_rate, type=float, help='Learning Rate')
        parser.add_argument('--batch_size',    default=batch_size,     type=int,   help='Batch Size')
        parser.add_argument('--ckpt',    default=ckpt,     type=str,   help='Load checkpoint')
        parser.add_argument('--freeze_encoder', action='store_true', help='Freeze encoder')
        parser.add_argument('--depth-loss-weight', type=float, default=5.0, help='Extra loss weighting for depth layer(s)')
        parser.add_argument('--comp-loss-weight', type=float, default=10.0, help='Loss weighting for composite loss')
    @staticmethod
    def add_model_specific_args(parser):
        raise NotImplementedError()
