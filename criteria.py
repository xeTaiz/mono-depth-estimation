# -*- coding: utf-8 -*-
# @Time    : 2018/10/23 20:04
# @Author  : Wang Xin
# @Email   : wangxin_buaa@163.com


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MaskedDepthLoss(nn.Module):
    def __init__(self):
        super(MaskedDepthLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        bsize = target.shape[0]
        npix = int(np.prod(target.shape[1:]))
        mask = (target > 0).detach().type(torch.float32)

        y0_target_vec = target.reshape((bsize, npix))
        y0_mask_vec = mask.reshape((bsize, npix))
        pred_vec = pred.reshape((bsize, npix))

        # avg l2 + scale inv loss + spatial grad cost
        
        p = pred_vec * y0_mask_vec
        t = y0_target_vec * y0_mask_vec

        d = (p - t)

        nvalid_pix = torch.sum(y0_mask_vec, axis=1)
        depth_error = (torch.sum(nvalid_pix * torch.sum(d**2, axis=1))
                         - 0.5*torch.sum(torch.sum(d, axis=1)**2)) \
                      / torch.sum(nvalid_pix**2)
        depth_cost = depth_error

        if pred.ndim == 4:
            pred = pred[:,0,:,:]
        if target.ndim == 4:
            target = target[:,0,:,:]
        if mask.ndim == 4:
            mask = mask[:,0,:,:]

        h = 1
        p_di = (pred[:,h:,:] - pred[:,:-h,:]) * (1 / np.float32(h))
        p_dj = (pred[:,:,h:] - pred[:,:,:-h]) * (1 / np.float32(h))
        t_di = (target[:,h:,:] - target[:,:-h,:]) * (1 / np.float32(h))
        t_dj = (target[:,:,h:] - target[:,:,:-h]) * (1 / np.float32(h))
        m_di = torch.logical_and(mask[:,h:,:], mask[:,:-h,:])
        m_dj = torch.logical_and(mask[:,:,h:], mask[:,:,:-h])

        grad_cost = torch.sum(m_di * (p_di - t_di)**2) / torch.sum(m_di) \
                  + torch.sum(m_dj * (p_dj - t_dj)**2) / torch.sum(m_dj)

        depth_error += grad_cost
        self.loss = depth_error
        return self.loss


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = (diff ** 2).mean()
        return self.loss


class MaskedL1Loss(nn.Module):
    def __init__(self):
        super(MaskedL1Loss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"
        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        self.loss = diff.abs().mean()
        return self.loss


class berHuLoss(nn.Module):
    def __init__(self):
        super(berHuLoss, self).__init__()

    def forward(self, pred, target):
        assert pred.dim() == target.dim(), "inconsistent dimensions"

        huber_c = torch.max(pred - target)
        huber_c = 0.2 * huber_c

        valid_mask = (target > 0).detach()
        diff = target - pred
        diff = diff[valid_mask]
        diff = diff.abs()

        huber_mask = (diff > huber_c).detach()

        diff2 = diff[huber_mask]
        diff2 = diff2 ** 2

        self.loss = torch.cat((diff, diff2)).mean()

        return self.loss


def compute_scale_and_shift(prediction, target, mask):
    # system matrix: A = [[a_00, a_01], [a_10, a_11]]
    a_00 = torch.sum(mask * prediction * prediction, (1, 2))
    a_01 = torch.sum(mask * prediction, (1, 2))
    a_11 = torch.sum(mask, (1, 2))

    # right hand side: b = [b_0, b_1]
    b_0 = torch.sum(mask * prediction * target, (1, 2))
    b_1 = torch.sum(mask * target, (1, 2))

    # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
    x_0 = torch.zeros_like(b_0)
    x_1 = torch.zeros_like(b_1)

    det = a_00 * a_11 - a_01 * a_01
    valid = torch.nonzero(det, as_tuple=True)

    x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
    x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

    return x_0, x_1


def reduction_batch_based(image_loss, M):
    # average of all valid pixels of the batch

    # avoid division by 0 (if sum(M) = sum(sum(mask)) = 0: sum(image_loss) = 0)
    divisor = torch.sum(M)

    if divisor == 0:
        return 0
    else:
        return torch.sum(image_loss) / divisor


def reduction_image_based(image_loss, M):
    # mean of average of valid pixels of an image

    # avoid division by 0 (if M = sum(mask) = 0: image_loss = 0)
    valid = M.nonzero()

    image_loss[valid] = image_loss[valid] / M[valid]

    return torch.mean(image_loss)


def mse_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))
    res = prediction - target
    image_loss = torch.sum(mask * res * res, (1, 2))

    return reduction(image_loss, 2 * M)


def gradient_loss(prediction, target, mask, reduction=reduction_batch_based):

    M = torch.sum(mask, (1, 2))

    diff = prediction - target
    diff = torch.mul(mask, diff)

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = torch.mul(mask[:, :, 1:], mask[:, :, :-1])
    grad_x = torch.mul(mask_x, grad_x)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = torch.mul(mask[:, 1:, :], mask[:, :-1, :])
    grad_y = torch.mul(mask_y, grad_y)

    image_loss = torch.sum(grad_x, (1, 2)) + torch.sum(grad_y, (1, 2))

    return reduction(image_loss, M)


class MSELoss(nn.Module):
    def __init__(self, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

    def forward(self, prediction, target, mask):
        return mse_loss(prediction, target, mask, reduction=self.__reduction)


class GradientLoss(nn.Module):
    def __init__(self, scales=4, reduction='batch-based'):
        super().__init__()

        if reduction == 'batch-based':
            self.__reduction = reduction_batch_based
        else:
            self.__reduction = reduction_image_based

        self.__scales = scales

    def forward(self, prediction, target, mask):
        total = 0

        for scale in range(self.__scales):
            step = pow(2, scale)

            total += gradient_loss(prediction[:, ::step, ::step], target[:, ::step, ::step],
                                   mask[:, ::step, ::step], reduction=self.__reduction)

        return total


class ScaleAndShiftInvariantLoss(nn.Module):
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        super().__init__()

        self.__data_loss = MSELoss(reduction=reduction)
        self.__regularization_loss = GradientLoss(scales=scales, reduction=reduction)
        self.__alpha = alpha

        self.__prediction_ssi = None

    def forward(self, prediction, target):
        prediction = prediction.squeeze(1)
        target = target.squeeze(1)
        assert prediction.dim() == target.dim(), "inconsistent dimensions"
        mask = torch.ones(target.size(), dtype=torch.bool, device=prediction.device)

        scale, shift = compute_scale_and_shift(prediction, target, mask)
        self.__prediction_ssi = scale.view(-1, 1, 1) * prediction + shift.view(-1, 1, 1)

        total = self.__data_loss(self.__prediction_ssi, target, mask)
        if self.__alpha > 0:
            total += self.__alpha * self.__regularization_loss(self.__prediction_ssi, target, mask)

        return total

    def __get_prediction_ssi(self):
        return self.__prediction_ssi

    prediction_ssi = property(__get_prediction_ssi)


if __name__ == "__main__":
    criteria = MaskedDepthLoss()
    #criteria = MaskedL1Loss()
    criteria = ScaleAndShiftInvariantLoss()

    target = torch.rand((16, 1, 240, 320))
    pred = torch.rand((16, 1,240, 320))

    loss = criteria(pred, target)
    print(loss)