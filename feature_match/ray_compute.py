
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def compute_cur_rays_and_targets(rays_o, rays_d, target_s, target_d, sample):
    rays_o_cur = rays_o[sample:]
    rays_d_cur = rays_d[sample:]
    target_s_cur = target_s[sample:]
    target_d_cur = target_d[sample:]
    
    return rays_o_cur, rays_d_cur, target_s_cur, target_d_cur

def compute_prev_rays_and_targets(prev_c2w, rays_d_cam_prev, rgb_prev, depth_prev, iH, iW, indice_h_prev, indice_w_prev, device):
    rays_o_prev = prev_c2w[...,:3, -1].repeat(indice_h_prev.numel(), 1)
    rays_d_prev = torch.sum(rays_d_cam_prev[..., None, :] * prev_c2w[:, :3, :3], -1)
    target_s_prev = rgb_prev.squeeze(0)[iH:-iH, iW:-iW, :][indice_h_prev, indice_w_prev, :].to(device) # Assuming `device` is defined
    target_d_prev = depth_prev.squeeze(0)[iH:-iH, iW:-iW][indice_h_prev, indice_w_prev].to(device).unsqueeze(-1) # Assuming `device` is defined
    
    return rays_o_prev, rays_d_prev, target_s_prev, target_d_prev