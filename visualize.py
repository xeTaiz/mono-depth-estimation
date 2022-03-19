import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from metrics import MetricComputation

def colored_depthmap(depth, d_min=None, d_max=None, do_mapping=True):
    if d_min is None:
        d_min = np.min(depth)
    if d_max is None:
        d_max = np.max(depth)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_relative *= 255
    depth_relative = depth_relative.astype(np.uint8)
    if do_mapping:return cv2.applyColorMap(depth_relative, cv2.COLORMAP_INFERNO)  # H, W, C
    return depth_relative


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
    if path is None:return
    
    path=Path(path)
    path.mkdir(parents=True, exist_ok=True)
    path = path.as_posix()
    min_ = np.finfo(np.float16).max
    max_ = np.finfo(np.float16).min
    if not rgb is None:
        if rgb.ndim == 4: rgb = rgb.squeeze(0)
        rgb = 255 * np.transpose(rgb.cpu().numpy(), (1, 2, 0))  # H, W, C
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        save_image(rgb, "{}/{}_rgb.jpg".format(path, idx))
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
        depth_pred = colored_depthmap(depth_pred, min_, max_)
        save_image(depth_pred, "{}/{}_pred.jpg".format(path, idx))

    if not depth_gt is None:
        depth_gt = colored_depthmap(depth_gt, min_, max_)
        save_image(depth_gt, "{}/{}_gt.jpg".format(path, idx))
    
    
def create_stdepth_plot_single(pred, targ, rgb, pred_full):
    with torch.no_grad():
        pred, targ = pred.cpu().float(), targ.cpu().float()
        rgb, pred_full = rgb.cpu().float(), pred_full.cpu().float()
    fig, ax = plt.subplot_mosaic(
        [['Color (Input)', 'L1 Color (Targ)', 'L1 Depth (Targ)',   'Back Color (Targ)'], 
        ['Alpha (Targ)',   'L1 Color (Pred)', 'L1 Depth (Pred)',   'Back Color (Pred)'],
        ['Alpha (Pred)',   'L1 Alpha (Pred)', 'Back Alpha (Pred)', 'Color (Pred)'],
        ['none',           'L1 Alpha (Targ)', 'Back Alpha (Targ)', 'Color (Targ)'],
    ], figsize=(20,20), tight_layout=True)
    for n in ax.keys():
        ax[n].set_title(n)
        ax[n].set_axis_off()
    ax['Color (Input)'].imshow(rgb.permute(1,2,0))
    ax['Color (Targ)'].imshow(rgb.permute(1,2,0))
    ax['Color (Pred)'].imshow(pred_full[:3].permute(1,2,0))
    ax['Back Color (Pred)'].imshow(pred[4:7].permute(1,2,0))
    ax['Back Color (Targ)'].imshow(targ[4:7].permute(1,2,0))

    ax['Alpha (Targ)'].imshow(targ[9], cmap='gray')
    ax['Alpha (Pred)'].imshow(pred[9], cmap='gray')
    ax['Back Alpha (Pred)'].imshow(pred[7], cmap='gray')
    ax['Back Alpha (Targ)'].imshow(targ[7], cmap='gray')

    ax['L1 Color (Pred)'].imshow(pred[ :3].permute(1,2,0))
    ax['L1 Color (Targ)'].imshow(targ[ :3].permute(1,2,0))

    ax['L1 Alpha (Pred)'].imshow(pred[3], cmap='gray')
    ax['L1 Alpha (Targ)'].imshow(targ[3], cmap='gray')

    ax['L1 Depth (Pred)'].imshow(pred[8], cmap='hot')
    ax['L1 Depth (Targ)'].imshow(targ[8], cmap='hot')


    return fig

def create_stdepth_plot(pred, targ, rgb, pred_full):
    with torch.no_grad():
        pred, targ = pred.cpu().float(), targ.cpu().float()
        rgb, pred_full = rgb.cpu().float(), pred_full.cpu().float()
    fig, ax = plt.subplot_mosaic(
        [['Color (Input)', 'L1 Color (Targ)', 'L2 Color (Targ)', 'L3 Color (Targ)', 'Back Color (Targ)'], 
        ['Alpha (Targ)',  'L1 Color (Pred)', 'L2 Color (Pred)', 'L3 Color (Pred)', 'Back Color (Pred)'],
        ['Alpha (Pred)',  'L1 Alpha (Pred)', 'L2 Alpha (Pred)', 'L3 Alpha (Pred)', 'Back Alpha (Pred)'],
        ['Color (Targ)', 'L1 Alpha (Targ)', 'L2 Alpha (Targ)', 'L3 Alpha (Targ)', 'Back Alpha (Targ)'],
        ['Color (Pred)',  'L1 Depth (Targ)', 'L2 Depth (Targ)', 'L3 Depth (Targ)', 'Front Color'],
        ['none1',         'L1 Depth (Pred)', 'L2 Depth (Pred)', 'L3 Depth (Pred)', 'Front Alpha']
    ], figsize=(25,30), tight_layout=True)
    for n in ax.keys():
        ax[n].set_title(n)
        ax[n].set_axis_off()
    ax['Color (Input)'].imshow(rgb.permute(1,2,0))
    ax['Color (Targ)'].imshow(rgb.permute(1,2,0))
    ax['Color (Pred)'].imshow(pred_full[:3].permute(1,2,0))
    ax['Back Color (Pred)'].imshow(pred[12:15].permute(1,2,0))
    ax['Back Color (Targ)'].imshow(targ[12:15].permute(1,2,0))

    ax['Alpha (Targ)'].imshow(targ[19], cmap='gray')
    ax['Alpha (Pred)'].imshow(pred[19], cmap='gray')
    ax['Back Alpha (Pred)'].imshow(pred[15], cmap='gray')
    ax['Back Alpha (Targ)'].imshow(targ[15], cmap='gray')

    ax['L1 Color (Pred)'].imshow(pred[ :3].permute(1,2,0))
    ax['L2 Color (Pred)'].imshow(pred[4:7].permute(1,2,0))
    ax['L3 Color (Pred)'].imshow(pred[8:11].permute(1,2,0))
    ax['L1 Color (Targ)'].imshow(targ[ :3].permute(1,2,0))
    ax['L2 Color (Targ)'].imshow(targ[4:7].permute(1,2,0))
    ax['L3 Color (Targ)'].imshow(targ[8:11].permute(1,2,0))

    ax['L1 Alpha (Pred)'].imshow(pred[3], cmap='gray')
    ax['L2 Alpha (Pred)'].imshow(pred[7], cmap='gray')
    ax['L3 Alpha (Pred)'].imshow(pred[11], cmap='gray')
    ax['L1 Alpha (Targ)'].imshow(targ[3], cmap='gray')
    ax['L2 Alpha (Targ)'].imshow(targ[7], cmap='gray')
    ax['L3 Alpha (Targ)'].imshow(targ[11], cmap='gray')

    ax['L1 Depth (Pred)'].imshow(pred[16], cmap='hot')
    ax['L2 Depth (Pred)'].imshow(pred[17], cmap='hot')
    ax['L3 Depth (Pred)'].imshow(pred[18], cmap='hot')
    ax['L1 Depth (Targ)'].imshow(targ[16], cmap='hot')
    ax['L2 Depth (Targ)'].imshow(targ[17], cmap='hot')
    ax['L3 Depth (Targ)'].imshow(targ[18], cmap='hot')

    # ax['Front Color'].imshow(targ[20:23].permute(1,2,0))
    # ax['Front Alpha'].imshow(targ[23], cmap='gray')


    return fig






        



def show_pred(depth_pred):
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    d_min = np.min(depth_pred_cpu)
    d_max = np.max(depth_pred_cpu)
    depth_target_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    cv2.imshow("pred", depth_target_col.astype('uint8'))
    cv2.waitKey(0)
