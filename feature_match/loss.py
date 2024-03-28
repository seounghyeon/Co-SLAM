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


def get_masks(z_vals, target_d, truncation):
    '''
    Params:
        z_vals: torch.Tensor, (Bs, N_samples)
        target_d: torch.Tensor, (Bs,)
        truncation: float
    Return:
        front_mask: torch.Tensor, (Bs, N_samples)
        sdf_mask: torch.Tensor, (Bs, N_samples)
        fs_weight: float
        sdf_weight: float
    '''

    # before truncation
    front_mask = torch.where(z_vals < (target_d - truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # after truncation
    back_mask = torch.where(z_vals > (target_d + truncation), torch.ones_like(z_vals), torch.zeros_like(z_vals))
    # valid mask
    depth_mask = torch.where(target_d > 0.0, torch.ones_like(target_d), torch.zeros_like(target_d))
    # Valid sdf regionn
    sdf_mask = (1.0 - front_mask) * (1.0 - back_mask) * depth_mask

    num_fs_samples = torch.count_nonzero(front_mask)
    num_sdf_samples = torch.count_nonzero(sdf_mask)
    num_samples = num_sdf_samples + num_fs_samples
    fs_weight = 1.0 - num_fs_samples / num_samples
    sdf_weight = 1.0 - num_sdf_samples / num_samples

    return front_mask, sdf_mask, fs_weight, sdf_weight




def loss_sdf(predicted_sdf, prev_sdf, sdf_weight, truncation):
    # if loss_type == 'l2':
    #     return F.mse_loss(prediction, target)
    # elif loss_type == 'l1':
    #     return F.l1_loss(prediction, target)
    
    sdf_loss = F.l1_loss((predicted_sdf * truncation), prev_sdf) * sdf_weight


    return sdf_loss

def l1_loss_3D(pred_3D, target_3D):

    # Get render loss
    rgb_loss = F.l1_loss(pred_3D, target_3D)


    return rgb_loss
