import torch
import criteria
from network import MyNet
from modules.base_module import BaseModule
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np

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

class MyModule(BaseModule):
   
    def output_size(self):
        return (384, 384)

    def resize(self):
        return 400

    def setup_model(self):
        return MyNet.MyModel(self.output_size())

    def setup_criterion(self):
        return criteria.MidasLoss(alpha=0.5, loss='mse', reduction='batch-based')

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

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
        self.save_visualization(x, y, y_hat, batch_idx)
        return self.metric_logger.log_val(y_hat, y)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        x, y = batch
        x = torch.nn.functional.interpolate(x, self.output_size(), mode='bilinear')
        y_hat = self(x)
        y = torch.nn.functional.interpolate(y, (480, 640), mode='bilinear')
        y_hat = torch.nn.functional.interpolate(y_hat, (480, 640), mode='bilinear')
        return self.metric_logger.log_test(y_hat, y)

    def configure_optimizers(self):
        # different modules have different learning rate
        train_params = [{'params': self.model.encoder.parameters(), 'lr': self.method.learning_rate},
                        {'params': self.model.decoder.parameters(), 'lr': self.method.learning_rate * 10}]

        optimizer = torch.optim.Adam(train_params, lr=self.method.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.method.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    def train_preprocess(self, rgb, depth):
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
        i, j, h, w = transforms.RandomCrop.get_params(rgb, output_size=self.output_size())
        rgb = TF.crop(rgb, i, j, h, w)
        depth = TF.crop(depth, i, j, h, w)

        # Random flipping
        if np.random.uniform(0,1) > 0.5:
            rgb = TF.hflip(rgb)
            depth = TF.hflip(depth)

        rgb = np.asarray(rgb, dtype=np.float32) / 255.0
        depth = np.asarray(depth, dtype=np.float32)

        # Random gamma, brightness, color augmentation
        if np.random.uniform(0,1) > 0.5:
            rgb = augment_image(rgb)

        rgb = TF.to_tensor(np.array(rgb))
        depth = TF.to_tensor(np.array(depth))
        return rgb, depth

    def val_preprocess(self, rgb, depth):
        if isinstance(rgb, np.ndarray):
            rgb = transforms.ToPILImage()(rgb)
        if isinstance(depth, np.ndarray):
            depth = transforms.ToPILImage()(depth)
        # Resize
        resize = transforms.Resize(self.resize())
        rgb = resize(rgb)
        depth = resize(depth)
        # Center crop
        crop = transforms.CenterCrop(self.output_size())
        rgb = crop(rgb)
        depth = crop(depth)
        
        rgb = TF.to_tensor(np.array(rgb, dtype=np.float32))
        depth = TF.to_tensor(np.array(depth, dtype=np.float32))
        
        rgb /= 255.0
        return rgb, depth

    def test_preprocess(self, rgb, depth):        
        rgb = TF.to_tensor(np.array(rgb, dtype=np.float32))
        depth = TF.to_tensor(np.array(depth, dtype=np.float32))
        rgb /= 255.0
        return rgb, depth

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('my', help='MyModel specific parameters')
        BaseModule.add_default_args(parser, name="my", learning_rate=0.0001, batch_size=16)
        return parser