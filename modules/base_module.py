import torch
import pytorch_lightning as pl
from datasets.dataset import ConcatDataset
from datasets.nyu_dataloader import get_nyu_dataset
from datasets.floorplan3d_dataloader import get_floorplan3d_dataset
from datasets.structured3d_dataset import get_structured3d_dataset
from datasets.stdepth import get_stdepth_dataset
from metrics import MetricLogger
import visualize
from torchvision import transforms
import torchvision.transforms.functional as TF
import numpy as np

NAME2FUNC = {
    "nyu": get_nyu_dataset,
    "structured3d": get_structured3d_dataset,
    "floorplan3d": get_floorplan3d_dataset,
    "stdepth": get_stdepth_dataset
}

def freeze_params(m):
    for p in m.parameters():
        p.requires_grad = False

class BaseModule(pl.LightningModule):
    def __init__(self, globals, training, validation, test, method=None, *args, **kwargs):
        super().__init__()    
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
                                                        pin_memory=True)
        else: self.train_loader = None                                                          
        if self.val_dataset:                                                
            self.val_dataset.transform = self.val_preprocess 
            self.val_loader = torch.utils.data.DataLoader(self.val_dataset,
                                                        batch_size=1, 
                                                        shuffle=False, 
                                                        num_workers=self.globals.worker, 
                                                        pin_memory=True) 
        else: self.val_loader = None  
        if self.test_dataset: 
            self.test_dataset.transform = self.test_preprocess                                                
            self.test_loader = torch.utils.data.DataLoader(self.test_dataset,
                                                        batch_size=1, 
                                                        shuffle=False, 
                                                        num_workers=self.globals.worker, 
                                                        pin_memory=True)
        else: self.test_loader = None                                 
        print("=> creating Model")
        self.model = self.setup_model()
        print("=> model created.")
        self.criterion = self.setup_criterion()
        self.metric_logger = MetricLogger(metrics=self.globals.metrics, module=self)
        if self.val_loader: self.skip = len(self.val_loader) // 9
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
        raise NotImplementedError()

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
        depth = depth / s

        rgb = transforms.ToPILImage()(rgb)
        depth = transforms.ToPILImage()(depth)
        # color jitter
        rgb = transforms.ColorJitter(0.4, 0.4, 0.4)(rgb)
        # Resize
        resize = transforms.Resize(self.resize())
        rgb = resize(rgb)
        depth = resize(depth)
        # Random Rotation
        angle = np.random.uniform(-5,5)
        rgb = TF.rotate(rgb, angle)
        depth = TF.rotate(depth, angle)
        # Resize
        resize = transforms.Resize(int(self.resize() * s))
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size())
        rgb = crop(rgb)
        depth = crop(depth)
        # Random horizontal flipping
        if np.random.uniform(0,1) > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb))
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def val_preprocess(self, rgb, depth):
        rgb = transforms.ToPILImage()(rgb)
        depth = transforms.ToPILImage()(depth)
        # Resize
        resize = transforms.Resize(self.resize())
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size())
        rgb = crop(rgb)
        depth = crop(depth)
        # Transform to tensor
        rgb = TF.to_tensor(np.array(rgb))
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def test_preprocess(self, rgb, depth):
        return self.val_preprocess(rgb, depth)

    def save_visualization(self, x, y, y_hat, batch_idx):
        if batch_idx == 0:
            self.img_merge = visualize.merge_into_row(x, y, y_hat)
        elif (batch_idx < 8 * self.skip) and (batch_idx % self.skip == 0):
            row = visualize.merge_into_row(x, y, y_hat)
            self.img_merge = visualize.add_row(self.img_merge, row)
        elif batch_idx == 8 * self.skip:
            filename = "{}/{}/version_{}/epoch{}.jpg".format(self.logger.save_dir, self.logger.name, self.logger.version, self.current_epoch)
            visualize.save_image(self.img_merge, filename)
            self.logger.experiment.add_image('images', np.reshape(self.img_merge, (-1, *self.img_merge.shape[-3:]), self.global_step, dataformats='HWC')

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

    @staticmethod
    def add_model_specific_args(parser):
        raise NotImplementedError()
