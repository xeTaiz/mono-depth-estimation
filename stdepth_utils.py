import torch
import torch.nn.functional as F

def depth_sort(layers):
    ''' Sorts RGBAD layers by depth

    Args:
        layers (torch.Tensor): ([BS,] L, C, H, W) RGBAD image stack of L layers. Sorts by last component of C dimension

    Returns:
        torch.Tensor: ([BS,] L, C, H, W) RGBAD image stack, sorted by depth (last component of C)
    '''
    l_dim = layers.ndim - 4
    c_dim = layers.ndim - 3
    _, idx = torch.sort(layers[..., :, -1, :,:].detach(), dim=l_dim, stable=True)
    return torch.stack([layers[..., :, i, :,:].gather(dim=l_dim, index=idx) for i in range(layers.size(c_dim))], dim=c_dim)

def composite_layers(layers):
    ''' Composites SORTED (batch of) RBAD layers

    Args:
        layers (torch.Tensor): ([BS,] L, 4+, H, W) RGBAD image stack of L layers, sorted by depth, can have more than RGBA channels, but they are ignored

    Returns:
        torch.Tensor: ([BS,], 4, H, W) (batch of) RGBA composited render
    '''
    l_dim, c_dim = layers.ndim - 4, layers.ndim - 3
    l_size = layers.size(l_dim)
    devdtype = {'dtype': layers.dtype, 'device': layers.device}

    shap_rgb = (*layers.shape[:-3], 3, *layers.shape[-2:])
    shap_a = (*layers.shape[:-3], 1, *layers.shape[-2:])
    # Shapes are ([BS,] L, 3, H, W) and ([BS,], L, 1, H, W)
    acc_rgb, acc_a = torch.zeros(*shap_rgb, **devdtype), torch.zeros(*shap_a, **devdtype)
    acc_rgb[..., 0, :,:,:] = layers[..., 0, :3, :,:]
    acc_a[  ..., 0, :,:,:]   = layers[..., 0, [3], :,:]
    for i in range(1, l_size):
        #                                   old acc       +           (1 - a)              *          Alpha           *          Color
        acc_rgb[..., i, :,:,:] = acc_rgb[..., i-1, :,:,:] + (1.0 - acc_a[..., i-1, :,:,:]) * layers[..., i, [3], :,:] * layers[..., i, :3, :,:]
        acc_a[  ..., i, :,:,:]   = acc_a[..., i-1, :,:,:] + (1.0 - acc_a[..., i-1, :,:,:]) * layers[..., i, [3], :,:]
    acc_rgba = torch.cat([acc_rgb[..., -1, :,:,:], acc_a[..., -1, :,:,:]], dim=c_dim-1) # Layer dim is removed, thus c_dim -1
    return torch.clamp(acc_rgba, 0.0, 1.0)


def make_nd(t, n):
    '''  Prepends singleton dimensions to `t` until n-dimensional '''
    if n < t.ndim:
        raise Exception(
            f'make_nd cannot reduce cardinality. Your Tensor.ndim={t.ndim} > n={n}.'
        )
    elif n == t.ndim:
        return t
    else:
        nons = [None] * (n - t.ndim)
        return t[nons]
def gaussian3d(device=None, dtype=torch.float32):
    if device is None: device = torch.device('cpu')
    gauss2d = torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1.]],
                           device=device).to(dtype) / 16.0
    return torch.stack([gauss2d, 2 * gauss2d, gauss2d]) / 4.0


def get_gaussian1d(size, sigma, dtype=torch.float32):
    coords = torch.arange(size)
    coords -= size // 2

    gauss = torch.exp(-coords**2 / (2 * sigma**2))
    gauss /= gauss.sum()
    return gauss.to(dtype)


def filter_gaussian_separated(input, win, dim):
    win = win.to(input.dtype).to(input.device)
    g = input.size(1)
    p = win.size(-1) // 2
    if dim == 1:
        return F.conv1d(input, win, padding=p, groups=g)
    elif dim == 2:
        out = F.conv2d(input, win, groups=g, padding=(0, p))
        return F.conv2d(out, win.transpose(2, 3), groups=g, padding=(p, 0))
    elif dim == 3:
        out = F.conv3d(input, win, groups=g, padding=(0, 0, p))
        out = F.conv3d(out, win.transpose(3, 4), groups=g, padding=(0, p, 0))
        return F.conv3d(out, win.transpose(2, 4), groups=g, padding=(p, 0, 0))
    else:
        raise Exception('Invalid dim! Must be 1, 2 or 3')


def ssim(pred,
         targ,
         dim=None,
         data_range=1.0,
         win_size=11,
         sigma=1.5,
         nonnegative_ssim=True,
         reduction='mean'):
    K1, K2 = 0.01, 0.03
    C1, C2 = (K1 * data_range)**2, (K2 * data_range)**2
    if dim is None: dim = pred.ndim - 2

    # win = gaussian3d(device=pred.device, dtype=pred.dtype)[None,None]
    win = get_gaussian1d(win_size, sigma, dtype=pred.dtype).to(pred.device)
    win = make_nd(win, dim + 1).expand((pred.size(1), -1, *([-1] * dim)))
    mu1 = filter_gaussian_separated(pred, win, dim=dim)
    mu2 = filter_gaussian_separated(targ, win, dim=dim)

    mu1_sq, mu2_sq = mu1.pow(2), mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = filter_gaussian_separated(pred * pred, win, dim=dim) - mu1_sq
    sigma2_sq = filter_gaussian_separated(targ * targ, win, dim=dim) - mu2_sq
    sigma12 = filter_gaussian_separated(pred * targ, win, dim=dim) - mu1_mu2

    cs_map = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    if nonnegative_ssim: cs_map = F.relu(cs_map, inplace=True)
    ssim_map = ((2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)) * cs_map

    if reduction.lower() == 'mean': return ssim_map.mean()
    elif reduction.lower() == 'sum': return ssim_map.sum()
    else: return ssim_map


def ssim1d(*args, **kwargs):
    return ssim(*args, dim=1, **kwargs)


def ssim2d(*args, **kwargs):
    return ssim(*args, dim=2, **kwargs)


def ssim3d(*args, **kwargs):
    return ssim(*args, dim=3, **kwargs)


def dssim1d(*args, **kwargs):
    return 1.0 - ssim(*args, dim=1, **kwargs)


def dssim2d(*args, **kwargs):
    return 1.0 - ssim(*args, dim=2, **kwargs)


def dssim3d(*args, **kwargs):
    return 1.0 - ssim(*args, dim=3, **kwargs)
