import numpy as np
import cv2
from pathlib import Path
from metrics import MetricComputation

def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_relative *= 255
    depth_relative = depth_relative.astype(np.uint8)
    depth_mapped = cv2.applyColorMap(depth_relative, cv2.COLORMAP_JET)  # H, W, C
    return depth_mapped


def merge_into_row(input, depth_target, depth_pred):
    rgb = 255 * np.transpose(np.squeeze(input.cpu().numpy()), (1, 2, 0))  # H, W, C
    rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    depth_target_cpu = np.squeeze(depth_target.cpu().numpy())
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())

    d_min = min(np.min(depth_target_cpu), np.min(depth_pred_cpu))
    d_max = max(np.max(depth_target_cpu), np.max(depth_pred_cpu))
    depth_target_col = colored_depthmap(depth_target_cpu, d_min, d_max)
    depth_pred_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    img_merge = np.hstack([rgb, depth_target_col, depth_pred_col])

    return img_merge

def add_row(img_merge, row):
    return np.vstack([img_merge, row])


def save_image(img_merge, filename):
    img_merge = cv2.imwrite(filename, img_merge.astype('uint8'))

def show_item(item):
    img, depth = item
    if img.ndim == 4:
        img = img.squeeze(0)
    img = 255 * np.transpose(img.cpu().numpy(), (1, 2, 0))  # H, W, C
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if depth.ndim == 4:
        depth = depth.squeeze(0).squeeze(0)
    elif depth.ndim == 3:
        depth = depth.squeeze(0)
    depth = depth.cpu().numpy()
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth = colored_depthmap(depth, d_min, d_max)
    cv2.imshow("item", np.hstack([img, depth]).astype('uint8'))
    cv2.waitKey(0)

def save_images(path, idx, rgb=None, depth_gt=None, depth_pred=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    min_ = np.finfo(np.float16).max
    max_ = np.finfo(np.float16).min
    if not rgb is None:
        if rgb.ndim == 4: rgb = rgb.squeeze(0)
        rgb = 255 * np.transpose(rgb.cpu().numpy(), (1, 2, 0))  # H, W, C
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        save_image(rgb, "{}/{}.rgb.jpg".format(path, idx))
    if not depth_gt is None:
        if depth_gt.ndim == 4: depth_gt = depth_gt.squeeze(0)
        if depth_gt.ndim == 3: depth_gt = depth_gt.squeeze(0)
        depth_gt = depth_gt.cpu().numpy()
        min_, max_ = min(np.min(depth_gt), min_), max(np.max(depth_gt), max_)
        
    if not depth_pred is None:
        if depth_pred.ndim == 4: depth_pred = depth_pred.squeeze(0)
        if depth_pred.ndim == 3: depth_pred = depth_pred.squeeze(0)
        depth_pred = depth_pred.cpu().numpy()
        min_, max_ = min(np.min(depth_pred), min_), max(np.max(depth_pred), max_)
        
    if not depth_pred is None:
        metric = MetricComputation(['mae'])
        mae = metric.compute(depth_pred, depth_gt)[0]
        depth_pred = colored_depthmap(depth_pred, min_, max_)
        save_image(depth_pred, "{}/mae.{}.{}.pred.jpg".format(path, round(mae,3), idx))

    if not depth_gt is None:
        depth_gt = colored_depthmap(depth_gt, min_, max_)
        save_image(depth_gt, "{}/{}.gt.jpg".format(path, idx))
    
    
    



def show_pred(depth_pred):
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    d_min = np.min(depth_pred_cpu)
    d_max = np.max(depth_pred_cpu)
    depth_target_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    cv2.imshow("pred", depth_target_col.astype('uint8'))
    cv2.waitKey(0)