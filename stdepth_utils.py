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
    
    return torch.cat([acc_rgb[..., -1, :,:,:], acc_a[..., -1, :,:,:]], dim=c_dim-1) # Layer dim is removed, thus c_dim -1
