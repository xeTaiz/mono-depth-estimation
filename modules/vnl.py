import torch
import pytorch_lightning as pl
import criteria
from network import VNL
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import cv2
import visualize
from modules.base_module import BaseModule, freeze_params

RGB_PIXEL_MEANS = (0.485, 0.456, 0.406)  # (102.9801, 115.9465, 122.7717)
RGB_PIXEL_VARS = (0.229, 0.224, 0.225)  # (1, 1, 1)
CROP_SIZE = (385, 385)

def scale_torch(img, scale):
    """
    Scale the image and output it in torch.tensor.
    :param img: input image. [C, H, W]
    :param scale: the scale factor. float
    :return: img. [C, H, W
    """
    img = img.astype(np.float32)
    img /= scale
    img = torch.from_numpy(img.copy())
    if img.size(0) == 3:
        img = transforms.Normalize(RGB_PIXEL_MEANS, RGB_PIXEL_VARS)(img)
    else:
        img = transforms.Normalize((0,), (1,))(img)
    return img

def set_flip_pad_reshape_crop(phase, uniform_size):
    """
    Set flip, padding, reshaping, and cropping factors for the image.
    :return:
    """
    # flip
    flip_prob = np.random.uniform(0.0, 1.0)
    flip_flg = True if flip_prob > 0.5 and 'train' in phase else False

    raw_size = np.array([CROP_SIZE[1], 416, 448, 480, 512, 544, 576, 608, 640])
    size_index = np.random.randint(0, 9) if 'train' in phase else 8

    # pad
    pad_height = raw_size[size_index] - uniform_size[0] if raw_size[size_index] > uniform_size[0] else 0
    pad = [pad_height, 0, 0, 0]  # [up, down, left, right]

    # crop
    crop_height = raw_size[size_index]
    crop_width = raw_size[size_index]
    start_x = np.random.randint(0, int(uniform_size[1] - crop_width)+1)
    start_y = 0 if pad_height != 0 else np.random.randint(0, int(uniform_size[0] - crop_height) + 1)
    crop_size = [start_x, start_y, crop_height, crop_width]

    resize_ratio = float(CROP_SIZE[1] / crop_width)

    return flip_flg, crop_size, pad, resize_ratio

def flip_pad_reshape_crop(img, flip, crop_size, pad, pad_value=0):
    """
    Flip, pad, reshape, and crop the image.
    :param img: input image, [C, H, W]
    :param flip: flip flag
    :param crop_size: crop size for the image, [x, y, width, height]
    :param pad: pad the image, [up, down, left, right]
    :param pad_value: padding value
    :return:
    """
    # Flip
    if flip:
        img = np.flip(img, axis=1)

    # Pad the raw image
    if len(img.shape) == 3:
        img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3]), (0, 0)), 'constant',
                    constant_values=(pad_value, pad_value))
    else:
        img_pad = np.pad(img, ((pad[0], pad[1]), (pad[2], pad[3])), 'constant',
                            constant_values=(pad_value, pad_value))
    # Crop the resized image
    img_crop = img_pad[crop_size[1]:crop_size[1] + crop_size[3], crop_size[0]:crop_size[0] + crop_size[2]]

    # Resize the raw image
    img_resize = cv2.resize(img_crop, (CROP_SIZE[1], CROP_SIZE[0]), interpolation=cv2.INTER_LINEAR)
    return img_resize

def resize_image(img, size):
    if type(img).__module__ != np.__name__:
        img = img.cpu().numpy()
    img = cv2.resize(img, (size[1], size[0]))
    return img

def preprocess(A, B, phase):
    if B.shape[0] != 480:
        s = 480 / B.shape[0]
        A = cv2.resize(A, (0,0), fx=s, fy=s)
        B = cv2.resize(B, (0,0), fx=s, fy=s)
        w = B.shape[1] - 640
        if w > 0:
            w_off = w // 2
            A = A[:, w_off:-(w_off+1), :]
            B = B[:, w_off:-(w_off+1)]

    uniform_size = B.shape[0:2]
    flip_flg, crop_size, pad, resize_ratio = set_flip_pad_reshape_crop(phase, uniform_size)

    A_resize = flip_pad_reshape_crop(A, flip_flg, crop_size, pad, 128)
    B_resize = flip_pad_reshape_crop(B, flip_flg, crop_size, pad, -1)

    A_resize = A_resize.transpose((2, 0, 1))
    A = A.transpose((2, 0, 1))
    B_resize = B_resize[np.newaxis, :, :]

    # to torch, normalize
    A_resize = scale_torch(A_resize, 255.)
    B_resize = scale_torch(B_resize, resize_ratio)

    invalid_side = [int(pad[0] * resize_ratio), 0, 0, 0]

    data = {'A': A_resize, 'B': B_resize, 'A_raw': torch.from_numpy(A/255).type(torch.float32), 'B_raw': B, 'invalid_side': np.array(invalid_side), 'ratio': np.float32(1.0 / resize_ratio)}
    return data

def permute_image(im):
    if im.ndim > 3: im = im.squeeze(0)
    if im.size(0) in [1,3]:
        return im.permute(1,2,0).contiguous()
    else:
        return im
def training_preprocess(rgb, depth):
    print('Shapes before:')
    print(rgb.shape, depth.shape)
    rgb, depth = permute_image(rgb), permute_image(depth)
    print('Shapes after:')
    print(rgb.shape, depth.shape)
    A = np.array(rgb, dtype=np.uint8)
    B = np.array(depth, dtype=np.float32) / 10.0
    return preprocess(A, B, 'train')
    

def validation_preprocess(rgb, depth):
    print('Shapes before:')
    print(rgb.shape, depth.shape)
    rgb, depth = permute_image(rgb), permute_image(depth)
    print('Shapes after:')
    print(rgb.shape, depth.shape)
    A = np.array(rgb, dtype=np.uint8)
    B = np.array(depth, dtype=np.float32) / 10.0
    return preprocess(A, B, 'val')


class VNLModule(BaseModule):
    def __init__(self, *args, **kwargs):
        super(VNLModule, self).__init__(*args, **kwargs)
        self.params = pl.utilities.parsing.AttributeDict()
        self.params.depth_min = self.method.depth_min
        self.params.encoder = self.method.encoder
        self.params.enc_dim_in = self.method.enc_dim_in
        self.params.enc_dim_out = self.method.enc_dim_out
        self.params.pretrained = self.method.pretrained
        self.params.freeze_backbone = self.method.freeze_backbone
        self.params.init_type = self.method.init_type
        self.params.dec_dim_in = self.method.dec_dim_in
        self.params.dec_dim_out = self.method.dec_dim_out
        self.params.dec_out_c = self.method.dec_out_c        
        self.params.focal_x = self.method.focal_x
        self.params.focal_y = self.method.focal_y
        self.params.crop_size = self.method.crop_size
        self.params.diff_loss_weight = self.method.diff_loss_weight

        self.params.depth_min_log = np.log10(self.method.depth_min)
        self.params.depth_bin_interval = (np.log10(self.method.depth_max) - np.log10(self.method.depth_min)) / self.method.dec_out_c
        self.params.wce_loss_weight = [[np.exp(-0.2 * (i - j) ** 2) for i in range(self.method.dec_out_c)] for j in np.arange(self.method.dec_out_c)]
        self.params.depth_bin_border = np.array([np.log10(self.method.depth_min) + self.params.depth_bin_interval * (i + 0.5) for i in range(self.method.dec_out_c)])
        
        self.model = VNL.MetricDepthModel(self.params)
        self.criterion = criteria.ModelLoss(self.params)
        if 'finetune' in self.method and self.method.finetune in [-1, -2, -3, -4, -5]:
            freeze_params(self.model.depth_model.encoder_modules)
            layers = [
                self.model.depth_model.decoder_modules.top,
                self.model.depth_model.decoder_modules.topdown_fcn1,
                self.model.depth_model.decoder_modules.topdown_fcn2,
                self.model.depth_model.decoder_modules.topdown_fcn3,
                self.model.depth_model.decoder_modules.topdown_fcn4,
                self.model.depth_model.decoder_modules.topdown_fcn5,
                self.model.depth_model.decoder_modules.topdown_predict
            ]
            for layer in layers[0:self.method.finetune]:
                freeze_params(layer)
    
    def freeze_encoder(self):
        pass

    def setup_model(self):
        return None

    def setup_criterion(self):
        return None

    def output_size(self):
        return (385, 385)

    def resize(self):
        return 400

    def train_preprocess(self, rgb, depth):
        return training_preprocess(rgb, depth)

    def val_preprocess(self, rgb, depth):
        return validation_preprocess(rgb, depth)

    def depth_to_bins(self, depth):
        """
        Discretize depth into depth bins
        Mark invalid padding area as cfg.MODEL.DECODER_OUTPUT_C + 1
        :param depth: 1-channel depth, [1, h, w]
        :return: depth bins [1, h, w]
        """
        invalid_mask = depth < 0.
        depth[depth < self.method.depth_min] = self.method.depth_min
        depth[depth > self.method.depth_max] = self.method.depth_max
        bins = ((torch.log10(depth) - self.params.depth_min_log) / self.params.depth_bin_interval).to(torch.int)
        bins[invalid_mask] = self.method.dec_out_c + 1
        bins[bins == self.method.dec_out_c] = self.method.dec_out_c - 1
        depth[invalid_mask] = -1.0
        return bins

    def bins_to_depth(self, depth_bin):
        """
        Transfer n-channel discrate depth bins to 1-channel conitnuous depth
        :param depth_bin: n-channel output of the network, [b, c, h, w]
        :return: 1-channel depth, [b, 1, h, w]
        """
        depth_bin = depth_bin.permute(0, 2, 3, 1) #[b, h, w, c]
        depth_bin_border = torch.tensor(self.params.depth_bin_border, dtype=torch.float32).to(self.device)
        depth = depth_bin * depth_bin_border
        depth = torch.sum(depth, dim=3, dtype=torch.float32, keepdim=True)
        depth = 10 ** depth
        depth = depth.permute(0, 3, 1, 2)  # [b, 1, h, w]
        return depth

    def restore_prediction(self, y_hat, data):
        invalid_side = data['invalid_side'][0]
        pred_depth = y_hat[0].squeeze(0).detach()
        pred_depth = pred_depth[invalid_side[0]:pred_depth.size(0) - invalid_side[1], :]
        pred_depth = pred_depth / data['ratio'][0].to(self.device)
        pred_depth = torch.from_numpy(resize_image(pred_depth, data['B_raw'][0].shape)).unsqueeze(0)
        targ_depth = data['B_raw'][0].unsqueeze(0)
        image = data['A_raw'][0].unsqueeze(0)
        x,y,y_hat= image.to(self.device), targ_depth.unsqueeze(0).to(self.device), pred_depth.unsqueeze(0).to(self.device)
        #if self.method.test_dataset == 'nyu':
        #mask = (45, 471, 41, 601)
        #x = x[..., mask[0]:mask[1], mask[2]:mask[3]]
        #y = y[..., mask[0]:mask[1], mask[2]:mask[3]]
        #y_hat = y_hat[..., mask[0]:mask[1], mask[2]:mask[3]]
        return x,y,y_hat

    def forward(self, x):
        y_hat = self.model(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        pred_logits, pred_cls = self(batch['A'])
        loss = self.criterion(self.bins_to_depth(pred_cls), pred_logits, self.depth_to_bins(batch['B']), batch['B'])
        y_hat = self.predicted_depth_map(pred_logits, pred_cls)
        y = batch['B']
        return self.metric_logger.log_train(y_hat, y, loss)

    def predicted_depth_map(self, logits, cls):
        if self.method.prediction_method == 'classification':
            pred_depth = self.bins_to_depth(cls)
        elif self.method.prediction_method == 'regression':
            pred_depth = torch.nn.functional.sigmoid(logits)
        else:
            raise ValueError("Unknown prediction methods")
        return pred_depth

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        pred_logits, pred_cls = self(batch['A'])
        y_hat = self.predicted_depth_map(pred_logits, pred_cls)
        x, y, y_hat = self.restore_prediction(y_hat, batch)
        self.save_visualization(x, y, y_hat, batch_idx)
        return self.metric_logger.log_val(y_hat, y)

    def test_step(self, batch, batch_idx):
        if batch_idx == 0: self.metric_logger.reset()
        pred_logits, pred_cls = self(batch['A'])
        y_hat = self.predicted_depth_map(pred_logits, pred_cls)
        x, y, y_hat = self.restore_prediction(y_hat, batch)
        filename = "{}/{}/version_{}/test_{}".format(self.logger.save_dir, self.logger.name, self.logger.version, batch_idx)
        visualize.save_images(filename, batch_idx, x, y, y_hat)
        return self.metric_logger.log_test(y_hat, y)

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

        lr_encoder = self.method.learning_rate
        lr_decoder = self.method.learning_rate * self.method.scale_decoder_lr
        weight_decay = self.method.weight_decay

        net_params = [
            {'params': encoder_params,
             'lr': lr_encoder,
             'weight_decay': weight_decay},
            {'params': decoder_params,
             'lr': lr_decoder,
             'weight_decay': weight_decay},
            ]
        optimizer = torch.optim.SGD(net_params, momentum=0.9)
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=self.method.lr_patience)
        scheduler = {
            'scheduler': lr_scheduler,
            'monitor': 'val_delta1'
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(subparsers):
        parser = subparsers.add_parser('vnl', help='VNL specific parameters')
        BaseModule.add_default_args(parser, name="vnl", learning_rate=0.0001, batch_size=8)
        parser.add_argument('--weight_decay', default=0.0005, type=float, help='Weight decay')
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
        parser.add_argument('--data_augmentation', default='vnl', type=str, help='Choose data Augmentation Strategy: laina or vnl')
        parser.add_argument('--loss', default='vnl', type=str, help='loss function')
        parser.add_argument('--finetune', default=0, type=int, help='freeze all layers except the last n ones')
        return parser
