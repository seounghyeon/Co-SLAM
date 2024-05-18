import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86'

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse
import shutil
import json
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion
from feature_match.sift import SIFTMatcher
from feature_match.common_f import ray_to_3D, proj_3D_2D_cur, uv_to_index, uv_to_index, depth_from_3D, filter_rays, filter_3D_points, compare_depth
from feature_match.loss import mse_loss_mask, huber_loss, huber_loss_norm, huber_loss_sum, l1_loss_mask, loss_sdf, l1_loss_3D



class CoSLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = get_dataset(config)
        
        self.create_bounds()
        self.create_pose_data()
        self.get_pose_representation()
        self.keyframeDatabase = self.create_kf_database(config)
        self.model = JointEncoding(config, self.bounding_box).to(self.device)
    
        # camera intrinsics, hedge and wedge
        self.fx, self.fy =  cfg['cam']['fx']//cfg['data']['downsample'],\
             cfg['cam']['fy']//cfg['data']['downsample']
        self.cx, self.cy = cfg['cam']['cx']//cfg['data']['downsample'],\
             cfg['cam']['cy']//cfg['data']['downsample']
        self.crop_size = cfg['cam']['crop_edge'] if 'crop_edge' in cfg['cam'] else 0
        self.ignore_edge_W = cfg['tracking']['ignore_edge_W']
        self.ignore_edge_H = cfg['tracking']['ignore_edge_H']
        self.H, self.W =  cfg['cam']['H']//cfg['data']['downsample'],\
             cfg['cam']['W']//cfg['data']['downsample']


    def seed_everything(self, seed):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def get_pose_representation(self):
        '''
        Get the pose representation axis-angle or quaternion
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle
            self.matrix_from_tensor = at_to_transform_matrix
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        Create the pose data
        '''
        self.est_c2w_data = {}
        self.est_c2w_data_rel = {}
        self.uv_idx = {}
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        Get the pre-defined bounds for the scene
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(torch.float32).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(torch.float32).to(self.device)

    def create_kf_database(self, config):  
        '''
        Create the keyframe database
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)
    
    def load_gt_pose(self):
        '''
        Load the ground truth pose
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):
            self.pose_gt[i] = pose
 
    def save_state_dict(self, save_path):
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
    
    def save_ckpt(self, save_path):
        '''
        Save the model parameters and the estimated pose
        '''
        save_dict = {'pose': self.est_c2w_data,
                     'pose_rel': self.est_c2w_data_rel,
                     'model': self.model.state_dict()}
        torch.save(save_dict, save_path)
        print('Save the checkpoint')

    def load_ckpt(self, load_path):
        '''
        Load the model parameters and the estimated pose
        '''
        dict = torch.load(load_path)
        self.model.load_state_dict(dict['model'])
        self.est_c2w_data = dict['pose']
        self.est_c2w_data_rel = dict['pose_rel']

    def select_samples(self, H, W, samples):
        '''
        randomly select samples from the image
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples))
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        Get the training loss
        '''
        loss = 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(self.config['training']['smooth_pts'], 
                                                                                  self.config['training']['smooth_vox'], 
                                                                                  margin=self.config['training']['smooth_margin'])
        
        return loss             

    def first_frame_mapping(self, batch, sample_points, n_iters=100):
        '''
        First frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        self.model.train()

        # Training
        for i in range(n_iters):
            self.map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d, 0, False)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        
        # First frame will always be a keyframe
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)
        
        print('First frame mapping done')
        return ret, loss

    def current_frame_mapping(self, batch, sample_points, cur_frame_id):
        '''
        Current frame mapping
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
        Returns:
            ret: dict
            loss: float
        
        '''
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        print('Current frame mapping...')
        
        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)
            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o, rays_d, target_s, target_d, 0, False)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()
        
        
        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        Smoothness loss of feature grid
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0]

        grid_size = (sample_points-1) * voxel_size
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin

        offset = torch.rand(3).to(offset_max) * offset_max + margin
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume)
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset

        if self.config['grid']['tcnn_encoding']:
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True)
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum()
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum()
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3)

        return loss
    
    def get_pose_param_optim(self, poses, mapping=True):
        task = 'mapping' if mapping else 'tracking'
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))
        pose_optimizer = torch.optim.Adam([{"params": cur_rot, "lr": self.config[task]['lr_rot']},
                                               {"params": cur_trans, "lr": self.config[task]['lr_trans']}])
        
        return cur_rot, cur_trans, pose_optimizer
    





    def global_BA(self, batch, sample_points, cur_frame_id, index_full_cur, index_full_prev, prev_rays_ba, depth_prev):
        '''
        Global bundle adjustment that includes all the keyframes and the current frame
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        # print("index_full SIZES ", index_full_cur.shape)

        # for i in range(0, len(index_full_cur), 100):
        #     # Check if the elements to be printed exist
        #     if i + 2 < len(index_full_cur):
        #         print("index full cur every i", i, index_full_cur[i:i+3])

        # Set up optimizer
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()
        
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])
        # prev rays are prev_rays_ba




        # # number if iterations to store in
        # iters = index_full_cur // self.config['mapping']['min_pixels_cur']
        # remainder = index_full_cur % self.config['mapping']['min_pixels_cur']
        # if remainder > 0:
        #     iters += 1
        # add my rays to current rays
        min_pix = self.config['mapping']['min_pixels_cur']
        for i in range(self.config['mapping']['iters']):

            # Sample rays with real frame ids
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])



        
            #TODO: Checkpoint...
            # starts sampling max #sample and less and less until min mixels cur
            # add here samples from own feature matcher
            if i == 0:
                idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W), max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
                index_size = index_full_cur.size()[0]
                if index_size < min_pix:
                    idx_cur[:index_size] = index_full_cur[:index_size].tolist()
                else:
                    idx_cur[:min_pix] = index_full_cur[:min_pix].tolist()

            elif i == 1 and index_size > min_pix:
                # Ensure all elements from index_full_cur are included if index_size is larger than min_pixels_cur
                remaining_indices = index_full_cur[min_pix:]
                index_size_rem = remaining_indices.size()[0]
                idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W), max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
                if index_size_rem < min_pix:
                    idx_cur[:index_size_rem] = index_full_cur[min_pix:min_pix+index_size_rem].tolist()
                else:
                    idx_cur[:index_size_rem] = index_full_cur[min_pix:2*min_pix].tolist()
            elif i == 2 and index_size > 2 * min_pix:
                # Ensure all elements from index_full_cur are included if index_size is larger than 2 * min_pixels_cur
                remaining_indices = index_full_cur[2 * min_pix:]
                index_size_rem = remaining_indices.size()[0]
                idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W), max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
                idx_cur[:index_size_rem] = index_full_cur[2*min_pix:2*min_pix+index_size_rem].tolist()

            else:
                idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W), max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))


            # need prev rays from batch direction rgb depth
            current_rays_batch = current_rays[idx_cur, :]


            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            ret = self.model.forward(rays_o, rays_d, target_s, target_d, 0, False)

            loss = self.get_loss_from_ret(ret, smooth=True)


            # # current previous ray loss
            # #######################
            # cur_rays_sift = current_rays[index_full_cur, :]
            # prev_rays_sift = prev_rays_ba[index_full_prev, :]
            # cur_rays_d_cam = cur_rays_sift[..., :3].to(self.device)
            # cur_target_s = cur_rays_sift[..., 3:6].to(self.device)
            # cur_target_d = cur_rays_sift[..., 6:7].to(self.device)
            # prev_rays_d_cam = prev_rays_sift[..., :3].to(self.device)
            # prev_target_s = prev_rays_sift[..., 3:6].to(self.device)
            # prev_target_d = prev_rays_sift[..., 6:7].to(self.device)

            # prev_c2w = self.est_c2w_data[cur_frame_id-1].clone().detach().unsqueeze(0)
            # cur_c2w = self.est_c2w_data[cur_frame_id].unsqueeze(0)
            # rays_d_cur = torch.sum(cur_rays_d_cam[..., None, :] * cur_c2w[:, :3, :3], -1)
            # rays_o_cur = cur_c2w[...,:3, -1].repeat(index_full_cur.numel(), 1)
            # # previous rays and target color depth
            # rays_d_prev = torch.sum(prev_rays_d_cam[..., None, :] * prev_c2w[:, :3, :3], -1)
            # rays_o_prev = prev_c2w[...,:3, -1].repeat(index_full_prev.numel(), 1)
            # # uv prev in cur point 3D
            # point_3D_current = ray_to_3D(rays_o_cur, rays_d_cur, cur_target_d.squeeze(), batch['depth'].squeeze(0))
            # point_3D_prev = ray_to_3D(rays_o_prev, rays_d_prev, prev_target_d.squeeze(), depth_prev.squeeze(0))
            # # uv_prev_in_cur = proj_3D_2D_cur(point_3D_prev, W1, H0, Wedge, fx, fy, cx, cy, c2w_est, self.device)  # is float
            # # get previous color and sdf from point
            # prev_pts_flat = (point_3D_prev - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            # prev_point_eval = self.model.query_color_sdf(prev_pts_flat)
            # prev_sdf_eval = prev_point_eval[:, -1]
            # prev_rgb_eval = torch.sigmoid(prev_point_eval[:,:3])
            # prev_sdf_eval_c = prev_sdf_eval.clone().detach()
            # prev_rgb_eval_c = prev_rgb_eval.clone().detach()

            # # get current color and sdf from point
            # cur_pts_flat = (point_3D_current - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            # cur_point_eval = self.model.query_color_sdf(cur_pts_flat)
            # cur_sdf_eval = cur_point_eval[:, -1]
            # cur_rgb_eval = torch.sigmoid(cur_point_eval[:,:3])
 
            # loss_3d_color = l1_loss_mask(cur_rgb_eval, prev_rgb_eval_c, cur_target_d, self.config['training']['rgb_missing'], self.config['training']['trunc']) * 0.01 # 0.1=>500 0.00349 ysterday 0.01=>0.0049
            # loss_3d_distance = l1_loss_3D(point_3D_current, point_3D_prev)

            # loss_3d_sdf = loss_sdf(cur_sdf_eval, prev_sdf_eval_c, self.config['training']['sdf_weight'], self.config['training']['trunc']) * 0.0000001     # * 0.000005 not that useful 
            # loss_total = loss + loss_3d_color + loss_3d_sdf
            # #######################

            loss.backward(retain_graph=True)
            
            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # So current pose is always unchanged
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
        
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
 








    def predict_current_pose(self, frame_id, constant_speed=True):
        '''
        Predict current pose from previous pose using camera motion model
        '''
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev
            
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id-2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id-1].to(self.device)
            delta = c2w_est_prev@c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta@c2w_est_prev
        
        return self.est_c2w_data[frame_id]

    
    
    def tracking_render(self, batch, sample_points, frame_id, index_cur, index_prev, rgb_prev, depth_prev, direction_prev, W1, H0, Wedge, fx, fy, cx, cy, uv_cur):
        '''
        Tracking camera pose using of the current frame
        Params:
            batch['c2w']: Ground truth camera pose [B, 4, 4]
            batch['rgb']: RGB image [B, H, W, 3]
            batch['depth']: Depth image [B, H, W, 1]
            batch['direction']: Ray direction [B, H, W, 3]
            frame_id: Current frame id (int)
        '''

        c2w_gt = batch['c2w'][0].to(self.device)
        self.uv_idx[frame_id] = index_cur
        # Initialize current pose
        if self.config['tracking']['iter_point'] > 0:
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            # init prev and current c2w and indices
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])
            cur_uv_index = self.uv_idx[frame_id]

            prev_c2w = self.est_c2w_data[frame_id-1].clone().detach() 
            prev_c2w = prev_c2w.unsqueeze(0)

        indice = None
        best_sdf_loss = None
        thresh=0

        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)
        # Start tracking
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad()
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)
            # Note here we fix the sampled points for optimisation
            # Original and current rays and target 
            if indice is None:
                indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['sample'])
                indice = torch.cat((indice, index_cur))             # index added
                # Slicing                

                indice_w, indice_h = indice % (self.dataset.W - iW * 2), indice // (self.dataset.W - iW * 2)
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)


                indice_w_prev, indice_h_prev = index_prev % (self.dataset.W - iW * 2), index_prev // (self.dataset.W - iW * 2)
                rays_d_cam_prev = direction_prev.squeeze(0)[iH:-iH, iW:-iW, :][indice_h_prev, indice_w_prev, :].to(self.device)             
            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)
            rays_o = c2w_est[...,:3, -1].repeat(self.config['tracking']['sample']+ index_cur.numel(), 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            # sample rays and target color depth
            rays_o_sample = rays_o[:self.config['tracking']['sample']]
            rays_d_sample = rays_d[:self.config['tracking']['sample']]
            target_s_sample = target_s[:self.config['tracking']['sample']]
            target_d_sample = target_d[:self.config['tracking']['sample']]
            # current rays and target color depth
            rays_o_cur = rays_o[self.config['tracking']['sample']:]
            rays_d_cur = rays_d[self.config['tracking']['sample']:]
            target_s_cur = target_s[self.config['tracking']['sample']:]
            target_d_cur = target_d[self.config['tracking']['sample']:]

            # previous rays and target color depth
            rays_o_prev = prev_c2w[...,:3, -1].repeat(index_prev.numel(), 1)
            rays_d_prev = torch.sum(rays_d_cam_prev[..., None, :] * prev_c2w[:, :3, :3], -1)
            target_s_prev = rgb_prev.squeeze(0)[iH:-iH, iW:-iW, :][indice_h_prev, indice_w_prev, :].to(self.device)
            target_d_prev = depth_prev.squeeze(0)[iH:-iH, iW:-iW][indice_h_prev, indice_w_prev].to(self.device).unsqueeze(-1)

            # uv prev in cur point 3D
            point_3D_current = ray_to_3D(rays_o_cur, rays_d_cur, target_d_cur.squeeze(), batch['depth'].squeeze(0)[iH:-iH, iW:-iW])
            point_3D_prev = ray_to_3D(rays_o_prev, rays_d_prev, target_d_prev.squeeze(), depth_prev.squeeze(0)[iH:-iH, iW:-iW])
            uv_prev_in_cur = proj_3D_2D_cur(point_3D_prev, W1, H0, Wedge, fx, fy, cx, cy, c2w_est, self.device)  # is float

            # prev in cur rays and target color depth
            index_pic, pic_filtered = uv_to_index(uv_prev_in_cur, self.dataset.W - iW * 2, self.dataset.H-iH*2)
            indice_w_pic, indice_h_pic = index_pic % (self.dataset.W - iW * 2), index_pic // (self.dataset.W - iW * 2)
            rays_d_cam_pic = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h_pic, indice_w_pic, :].to(self.device)
            rays_o_pic = c2w_est[...,:3, -1].repeat(index_pic.numel(), 1)
            rays_d_pic = torch.sum(rays_d_cam_pic[..., None, :] * c2w_est[:, :3, :3], -1)
            target_s_pic = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h_pic, indice_w_pic, :].to(self.device)
            target_d_pic = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h_pic, indice_w_pic].to(self.device).unsqueeze(-1)

            # filter if uv re projected is not on new frame
            if pic_filtered.numel() != 0: # if to delete 
                point_3D_prev = filter_3D_points(pic_filtered, point_3D_prev)
                target_d_cur = filter_3D_points(pic_filtered, target_d_cur)
                point_3D_current = filter_3D_points(pic_filtered, point_3D_current)

            rays_o = torch.cat((rays_o, rays_o_pic))
            rays_d = torch.cat((rays_d, rays_d_pic))
            target_s = torch.cat((target_s, target_s_pic))
            target_d = torch.cat((target_d, target_d_pic))

            # model rgb depth 
            # target_s and d, rays o and d: original, current, prev_in_cur
            ret = self.model.forward(rays_o, rays_d, target_s, target_d, sample_points, True)


            loss = self.get_loss_from_ret(ret) * self.config['tracking']['sift3d_supervis']
            


            # get previous color and sdf from point
            prev_pts_flat = (point_3D_prev - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            prev_point_eval = self.model.query_color_sdf(prev_pts_flat)
            prev_sdf_eval = prev_point_eval[:, -1]
            prev_rgb_eval = torch.sigmoid(prev_point_eval[:,:3])
            prev_sdf_eval_c = prev_sdf_eval.clone().detach()
            prev_rgb_eval_c = prev_rgb_eval.clone().detach()



            # get current color and sdf from point
            cur_pts_flat = (point_3D_current - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
            cur_point_eval = self.model.query_color_sdf(cur_pts_flat)
            cur_sdf_eval = cur_point_eval[:, -1]
            cur_rgb_eval = torch.sigmoid(cur_point_eval[:,:3])


            loss_3d_color = l1_loss_mask(cur_rgb_eval, prev_rgb_eval_c, target_d_cur, self.config['training']['rgb_missing'], self.config['training']['trunc']) * self.config['tracking']['sift3d_color'] # 0.1=>500 0.00349 ysterday 0.01=>0.0049
            # loss_3d_distance = torch.abs(point_3D_current - point_3D_prev).mean() *0.02  #     * 0.2
            loss_3d_distance = l1_loss_3D(point_3D_current, point_3D_prev) * self.config['tracking']['sift3d_distance']
            # print("loss_3D_distance ", loss_3d_distance*1000)
            loss_3d_sdf = loss_sdf(cur_sdf_eval, prev_sdf_eval_c, self.config['training']['sdf_weight'], self.config['training']['trunc']) * self.config['tracking']['sift3d_sdf']   # * 0.000005 not that useful 
            # total_loss = loss + loss_3d_color + loss_3d_distance + loss_3d_sdf
            total_loss = loss + loss_3d_color + loss_3d_distance + loss_3d_sdf

            # print("color distance sdf loss ", loss_3d_color, loss_3d_distance, loss_3d_sdf, loss)


            if best_sdf_loss is None:
                best_sdf_loss = total_loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                if total_loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = total_loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh +=1
            
            if thresh >self.config['tracking']['wait_iters']:
                break

            total_loss.backward()
            pose_optimizer.step()
        
        if self.config['tracking']['best']:
            # Use the pose with smallest loss
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

       # Save relative pose of non-keyframes
        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta
        
        self.uv_idx[frame_id] = index_cur
        print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
    
    def convert_relative_pose(self):
        poses = {}
        for i in range(len(self.est_c2w_data)):
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                c2w_key = self.est_c2w_data[kf_frame_id]
                delta = self.est_c2w_data_rel[i] 
                poses[i] = delta @ c2w_key
        
        return poses

    def create_optimizer(self):
        '''
        Create optimizer for mapping
        '''
        # Optimizer for BA
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
    
        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
        # Optimizer for current frame mapping
        if self.config['mapping']['cur_frame_iters'] > 0:
            params_cur_mapping = [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
                 
            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))
        
    
    def save_mesh(self, i, voxel_size=0.05):
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_track{}.ply'.format(i))
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color
        else:
            color_func = self.model.query_color
        extract_mesh(self.model.query_sdf, 
                        self.config, 
                        self.bounding_box, 
                        color_func=color_func, 
                        marching_cube_bound=self.marching_cube_bound, 
                        voxel_size=voxel_size, 
                        mesh_savepath=mesh_savepath)      
        
    def run(self):
        sift_matcher = SIFTMatcher()  # Instantiate the class
        self.create_optimizer()
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers'])
        gt_color_prev = None        # WORKS
        gt_depth_prev = None
        rgb_prev = None
        depth_prev = None
        direction_prev = None
        # initialize configs
        H, W, fx, fy, cx, cy = self.H, self.W, self.fx, self.fy, self.cx, self.cy
        Wedge = self.ignore_edge_W
        Hedge = self.ignore_edge_H


        # print("H W ", self.dataset.H, self.dataset.W)

        H0 = Hedge 
        H1 = self.dataset.H-Hedge
        W0 = Wedge 
        W1 = self.dataset.W-Wedge 
        width_small = W-2*Wedge
        height_small = H-2*Hedge
        sample_points = self.config['tracking']['sample']

        # Start Co-SLAM!
        for i, batch in tqdm(enumerate(data_loader)):

            rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)             #current rgb frame
            raw_depth = batch["depth"]
            mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
            depth_colormap = colormap_image(batch["depth"])

            gt_depth = raw_depth.squeeze(0)
            gt_depth = torch.tensor(gt_depth, dtype=torch.float32)
            gt_depth = gt_depth.to(device='cuda:0')
            depth_colormap[:, mask] = 255.
            depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()

            # gt_color = rgb
            gt_color = torch.tensor(rgb, dtype=torch.float32)
            gt_color = gt_color.to(device='cuda:0')


            # Visualisation
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                image = np.hstack((rgb, depth_colormap))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            # First frame mapping
            if i == 0:
                self.first_frame_mapping(batch, sample_points, self.config['mapping']['first_iters'])
                rgb_prev = batch["rgb"]
                depth_prev = batch["depth"]
                direction_prev = batch["direction"]
                print("FIRST FRAME MAPPING")

            # Tracking + Mapping
            else:
                gt_color_prev_clone = gt_color_prev.clone().detach()
                gt_color_clone = gt_color.clone().detach()
                gt_color_prev_clone = gt_color_prev_clone[H0:H1, W0:W1]
                gt_color_clone = gt_color_clone[H0:H1, W0:W1]
                gt_color_clone2 = gt_color_clone.clone().detach()

                gt_color_clone2 = gt_color_clone2.reshape(-1,3)
                gt_depth_prev_clone = gt_depth_prev.clone().detach()
                gt_depth_prev_clone = gt_depth_prev_clone[H0:H1, W0:W1]

                gt_depth_clone = gt_depth.clone().detach()
                gt_depth_clone = gt_depth_clone[H0:H1, W0:W1]


                uv_prev, uv_cur, index_prev, index_cur, colors_cur, colors_prev, index_full_cur, index_full_prev = sift_matcher.match(gt_color_prev_clone, gt_color_clone, i, Hedge, Wedge, gt_color)

                if self.config['tracking']['iter_point'] > 0:
                    self.tracking_pc(batch, i)
                self.tracking_render(batch, sample_points, i, index_cur, index_prev, rgb_prev, depth_prev, direction_prev, W1, H0, Wedge, fx, fy, cx, cy, uv_cur)

                if i%self.config['mapping']['map_every']==0:
                    self.current_frame_mapping(batch, sample_points, i)

                    prev_rays_ba= torch.cat([direction_prev, rgb_prev, depth_prev[..., None]], dim=-1)
                    prev_rays_ba = prev_rays_ba.reshape(-1, prev_rays_ba.shape[-1])

                    self.global_BA(batch, sample_points, i, index_full_cur, index_full_prev, prev_rays_ba, depth_prev)
                    
                # Add keyframe
                if i % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('add keyframe:',i)

                if i % self.config['mesh']['vis']==0:
                    self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval'])
                    pose_relative = self.convert_relative_pose()
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
                    pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')

                    if cfg['mesh']['visualisation']:
                        cv2.namedWindow('Traj:'.format(i), cv2.WINDOW_AUTOSIZE)
                        traj_image = cv2.imread(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "pose_r_{}.png".format(i)))
                        # best_traj_image = cv2.imread(os.path.join(best_logdir_scene, "pose_r_{}.png".format(i)))
                        # image_show = np.hstack((traj_image, best_traj_image))
                        image_show = traj_image
                        cv2.imshow('Traj:'.format(i), image_show)
                        key = cv2.waitKey(1)

            if rgb is not None:
                gt_color_prev = gt_color.clone().detach()
                gt_depth_prev = gt_depth.clone().detach()
                rgb_prev = batch["rgb"]
                depth_prev = batch["depth"]
                direction_prev = batch["direction"]

        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i)) 
        
        self.save_ckpt(model_savepath)
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])
        
        pose_relative = self.convert_relative_pose()
        pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
        pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')

        #TODO: Evaluation of reconstruction


if __name__ == '__main__':
            
    print('Start running...')
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    parser.add_argument('--config', type=str, help='Path to config file.')
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    
    args = parser.parse_args()

    cfg = config.load_config(args.config)
    if args.output is not None:
        cfg['data']['output'] = args.output

    print("Saving config and script...")
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    shutil.copy("coslam.py", os.path.join(save_path, 'coslam.py'))

    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    slam = CoSLAM(cfg)

    slam.run()
