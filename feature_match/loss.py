import torch

import torch.nn.functional as F

def huber_loss(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.where(abs_error <= delta, 0.5 * abs_error ** 2, delta * (abs_error - 0.5 * delta))
    return quadratic.mean()

def huber_loss_sum(y_pred, y_true, delta=1.0):
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.where(abs_error <= delta, 0.5 * abs_error ** 2, delta * (abs_error - 0.5 * delta))
    return quadratic.sum()  # Sum instead of mean

def huber_loss_norm(y_pred, y_true, delta=1.0, scale_factor=1.0):
    error = y_pred - y_true
    abs_error = torch.abs(error)
    quadratic = torch.where(abs_error <= delta, 0.5 * abs_error ** 2, delta * (abs_error - 0.5 * delta))
    # Normalize the loss by dividing it by a scale factor
    normalized_loss = quadratic.mean() / scale_factor
    return normalized_loss




def mse_loss_mask(prediction_rgb, target_rgb, target_d, config_missing, config_depth_truncs):

    # Get depth and rgb weights for loss
    valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < config_depth_truncs)
    rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
    rgb_weight[rgb_weight==0] = config_missing

    # Get render loss
    rgb_loss = F.mse_loss(prediction_rgb*rgb_weight, target_rgb*rgb_weight)


    return rgb_loss


def l1_loss_mask(prediction_rgb, target_rgb, target_d, config_missing, config_depth_truncs):

    # Get depth and rgb weights for loss
    valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < config_depth_truncs)
    rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
    rgb_weight[rgb_weight==0] = config_missing

    # Get render loss
    rgb_loss = F.l1_loss(prediction_rgb*rgb_weight, target_rgb*rgb_weight)


    return rgb_loss