import numpy as np
import cv2

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
    img = 255 * np.transpose(np.squeeze(img.cpu().numpy()), (1, 2, 0))  # H, W, C
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    depth = depth.squeeze(0).cpu().numpy()
    d_min = np.min(depth)
    d_max = np.max(depth)
    depth = colored_depthmap(depth, d_min, d_max)
    cv2.imshow("item", np.hstack([img, depth]).astype('uint8'))
    cv2.waitKey(0)


def show_pred(depth_pred):
    depth_pred_cpu = np.squeeze(depth_pred.data.cpu().numpy())
    d_min = np.min(depth_pred_cpu)
    d_max = np.max(depth_pred_cpu)
    depth_target_col = colored_depthmap(depth_pred_cpu, d_min, d_max)
    cv2.imshow("pred", depth_target_col.astype('uint8'))
    cv2.waitKey(0)