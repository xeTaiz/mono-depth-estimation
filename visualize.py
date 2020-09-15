import numpy as np
import cv2

def to_img(array):
    img = np.transpose(array, (1,2,0)) # (c,h,w) -> (h,w,c)
    img *= 255
    return img.astype(np.uint8)

def cmap(depth):
    d_min = np.min(depth)
    d_max = np.max(depth)
    a, b = (depth - d_min), (d_max - d_min)
    depth_relative = np.divide(a, b, where=b!=0)
    depth_relative = 1 - depth_relative
    depth_relative *= 255
    return cv2.applyColorMap(depth_relative.astype(np.uint8), cv2.COLORMAP_JET)

def viz_depth_from_batch(batch, pred=None):
    imgs, gt = batch
    imgs = imgs[0:8].cpu().numpy()
    gt = gt[0:8].cpu().numpy()
    if pred is None:
        pred = np.zeros_like(gt)
    else:
        pred = pred[0:8].cpu().numpy()
    (c,h,w) = imgs[0].shape
    n = imgs.shape[0]

    viz = np.zeros((h * n, w * 3, 3), dtype=np.uint8)
    for i, (img, y, y_hat) in enumerate(zip(imgs, gt, pred)):
        viz[i*h:i*h+h, 0:w, :] = to_img(img)
        viz[i*h:i*h+h, w:w*2, :] = cmap(to_img(y))
        viz[i*h:i*h+h, w*2:w*3, :] = cmap(to_img(y_hat))
    return cv2.cvtColor(viz, cv2.COLOR_RGB2BGR)