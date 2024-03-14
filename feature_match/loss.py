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