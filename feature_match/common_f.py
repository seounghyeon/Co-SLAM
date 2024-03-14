
import numpy as np
import torch
import torch.nn.functional as F
import cv2


def replace_zero_depth(depth_tensor, gt_depth_tensor):
    """
    Replace zero depth values in a 1D depth tensor with a maximum depth value from gt_depth

    Args:
        depth tensor: 1D tensor with the depth values
        max_depth_value (float): Maximum depth value to replace zero depth
        gt_depth_tensor (tensor): tensor of gt depth
    Returns:
        torch.Tensor: updated tensor 1D with max_depth
    """
    max_depth_value = torch.max(gt_depth_tensor)
    # print("max depth value is: ", max_depth_value)
    def_depth = max_depth_value
    # Create a mask for zero depth values
    zero_mask = (depth_tensor == 0)
    
    # Replace zero depths with the maximum depth value
    depth_tensor[zero_mask] = def_depth
    
    return depth_tensor



def ray_to_3D(batch_rays_o, batch_rays_d, batch_gt_depth, gt_depth):
    """
    changing 0 depth into max depth
    0 depth means no depth information 
    """
    s_depth = batch_gt_depth
    s_depth     = replace_zero_depth(batch_gt_depth, gt_depth)
    # 3D coordinates projected from the previous and current image
    point_3D    = batch_rays_o + batch_rays_d * s_depth.unsqueeze(1) # output size is [100,3] torch.float32

    return point_3D


def proj_3D_2D(points, W, H, fx, fy, cx, cy, c2w, device):
    """
    projects 3D points into 2D space at pose given by c2w
    input args:
        - points: torch tensor of 3D points Nx3
        - fx fy cx cy intrinsic camera params
        - c2w camera pose for the image
        - W is the cropped image size since x y are flipped
        - H is the hedge
    output: 
        - uv coordinates (N,2)
    """
    # Define the concatenation tensor for [0, 0, 0, 1]
    concat_tensor = torch.tensor([0, 0, 0, 1], device=device, dtype=c2w.dtype)      # is torch.float32
    # Clone c2w to ensure we don't modify the original tensor

    # Concatenate [0, 0, 0, 1] to the copied tensor
    c2w = torch.cat([c2w, concat_tensor.unsqueeze(0)], dim=0)

    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0

    # Calculate the world-to-camera transformation matrix w2c
    w2c = torch.inverse(c2w)

    # Camera intrinsic matrix K
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=c2w.dtype, device=device)

    # Convert points to homogeneous coordinates
    ones = torch.ones_like(points[:, 0], device=device).unsqueeze(1)
    homo_points = torch.cat([points, ones], dim=1).unsqueeze(2)

    # Transform points to camera coordinates
    cam_cord_homo = torch.matmul(w2c, homo_points)

    # Remove the homogeneous coordinate
    cam_cord = cam_cord_homo[:, :3, :]

    cam_cord[:, 0] *= -1

    # Project points to image plane
    uv = torch.matmul(K, cam_cord)

    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z


    # Apply the correct transformation to uv coordinates
    uv[:, 0] = W - uv[:, 0] - 1
    # uv[:, 1] = H - uv[:, 1]
    uv[:, 1] = uv[:, 1] - H

    # print("these are the points size: \n", points.size())
    # print("uv.size: ", uv.size())
    num_points = points.size(0)
    uv = uv.view(num_points, 2)
    
    return uv

def get_rays_from_uv(i, j, c2w, H, W, fx, fy, cx, cy, device):
    """
    Get corresponding rays from input uv.

    """
    c2w = c2w.view(4, 4)

    # print("size of c2w ", c2w.shape)
    if isinstance(c2w, np.ndarray):
        c2w = torch.from_numpy(c2w).to(device)

    c2w = torch.squeeze(c2w)
    # print("size of c2w after ", c2w.shape)

    dirs = torch.stack(
        [(i-cx)/fx, -(j-cy)/fy, -torch.ones_like(i)], -1).to(device)
    dirs = dirs.reshape(-1, 1, 3)
    # Rotate ray directions from camera frame to the world frame
    # dot product, equals to: [c2w.dot(dir) for dir in dirs]
    rays_d = torch.sum(dirs * c2w[:3, :3], -1)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d





def proj_3D_2D(points, W, H, fx, fy, cx, cy, c2w, device):
    """
    projects 3D points into 2D space at pose given by c2w
    input args:
        - points: torch tensor of 3D points Nx3
        - fx fy cx cy intrinsic camera params
        - c2w camera pose for the image
        - W is the cropped image size since x y are flipped
        - H is the hedge
    output: 
        - uv coordinates (N,2)
    """
    c2w = c2w.view(4, 4)


    # c2w_clone = c2w.clone.detach
    c2w[:3, 1] *= -1.0
    c2w[:3, 2] *= -1.0

    # Calculate the world-to-camera transformation matrix w2c
    w2c = torch.inverse(c2w)

    # Camera intrinsic matrix K
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=c2w.dtype, device=device)

    # Convert points to homogeneous coordinates
    ones = torch.ones_like(points[:, 0], device=device).unsqueeze(1)
    homo_points = torch.cat([points, ones], dim=1).unsqueeze(2)

    # Transform points to camera coordinates
    cam_cord_homo = torch.matmul(w2c, homo_points)

    # Remove the homogeneous coordinate
    cam_cord = cam_cord_homo[:, :3, :]

    cam_cord[:, 0] *= -1

    # Project points to image plane
    uv = torch.matmul(K, cam_cord)

    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z

    uv[:, 0] = W - uv[:, 0] - 2 
    # uv[:, 1] = H - uv[:, 1]
    uv[:, 1] = uv[:, 1] - H

    # Apply the correct transformation to uv coordinates
    # uv[:, 0] = W - uv[:, 0] - 1
    # # uv[:, 1] = H - uv[:, 1]
    # uv[:, 1] = uv[:, 1] - H

    # print("these are the points size: \n", points.size())
    # print("uv.size: ", uv.size())
    num_points = points.size(0)
    uv = uv.view(num_points, 2)
    
    return uv


def proj_3D_2D_cur(points, W, H, Wedge, fx, fy, cx, cy, c2w, device):
    """
    projects 3D points into 2D space at pose given by c2w
    input args:
        - points: torch tensor of 3D points Nx3
        - fx fy cx cy intrinsic camera params
        - c2w camera pose for the image
        - W is the cropped image size since x y are flipped
        - H is the hedge
    output: 
        - uv coordinates (N,2)
    """
    c2w = c2w.view(4, 4)


    # c2w_clone = c2w.clone.detach
    # c2w[:3, 1] *= -1.0
    # c2w[:3, 2] *= -1.0

    # Calculate the world-to-camera transformation matrix w2c
    w2c = torch.inverse(c2w)

    # Camera intrinsic matrix K
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=c2w.dtype, device=device)

    # Convert points to homogeneous coordinates
    ones = torch.ones_like(points[:, 0], device=device).unsqueeze(1)
    homo_points = torch.cat([points, ones], dim=1).unsqueeze(2)

    # Transform points to camera coordinates
    cam_cord_homo = torch.matmul(w2c, homo_points)

    # Remove the homogeneous coordinate
    cam_cord = cam_cord_homo[:, :3, :]

    cam_cord[:, 0] *= -1

    # Project points to image plane
    uv = torch.matmul(K, cam_cord)

    z = uv[:, -1:] + 1e-5
    uv = uv[:, :2] / z

    uv[:, 0] = uv[:, 0] - Wedge
    # uv[:, 1] = H - uv[:, 1]
    uv[:, 1] = uv[:, 1] - H

    # Apply the correct transformation to uv coordinates
    # uv[:, 0] = W - uv[:, 0] - 1
    # # uv[:, 1] = H - uv[:, 1]
    # uv[:, 1] = uv[:, 1] - H

    # print("these are the points size: \n", points.size())
    # print("uv.size: ", uv.size())
    num_points = points.size(0)
    uv = uv.view(num_points, 2)
    
    return uv



def img_pre(image1in):

    if image1in is None or image1in.numel() == 0:
        # print("\nTHIS IS NONE IN IMAGE1IN no previous image saved up\n\n")
        return None, None, None, None
    # detach input images and change them from tensor to cv2 format
    ############################################
    # Detach the tensor from the GPU
    image1in_cpu = image1in.cpu()

    # Convert to NumPy arrays
    np_img1 = image1in_cpu.numpy()

    # color is set from 0 to 1 to ensure range of intensity for the pixel is inside this valid range
    np_img1 = np.clip(np_img1, 0, 1)
    # Check if the tensor shape is CxHxW, and if so, transpose it to HxWxC
    # print("Shape of np_img2:", np_img2.shape)
    if np_img1.shape[0] == 3:
        np_img1 = np.transpose(np_img1, (1, 2, 0))
    # If color values are in [0,1], scale to [0,255]
    if np_img1.max() <= 1.0:
        np_img1 = (np_img1 * 255).astype(np.uint8)
    # Save the images
    # image1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    image1 = cv2.cvtColor(np_img1, cv2.COLOR_RGB2BGR)
    print("Size of image1:", image1.shape)
    return image1





def render_surface_color(self, rays_o, normal):
    '''
    Render the surface color of the points.
    Params:
        points: [N_rays, 1, 3]
        normal: [N_rays, 3]
    '''
    n_rays = rays_o.shape[0]
    trunc = self.config['training']['trunc']
    z_vals = torch.linspace(-trunc, trunc, steps=self.config['training']['n_range_d']).to(rays_o)
    z_vals = z_vals.repeat(n_rays, 1)
    # Run rendering pipeline
    print("NORMAL AAAAAAAAAAAAAAAA ", normal)
    pts = rays_o[...,:] + normal[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
    raw = self.run_network(pts)
    rgb, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])
    return rgb


def save_img(image1, frame_id):
    print("size of imageasdadasda ", image1.shape)

    # color is set from 0 to 1 to ensure range of intensity for the pixel is inside this valid range
    np_img1 = np.clip(image1, 0, 1)

    # Check if the tensor shape is CxHxW, and if so, anspose it to HxWxC
    # print("Shape of np_img2:", np_img2.shape)
    if np_img1.shape[0] == 3:
        np_img1 = np.transpose(np_img1, (1, 2, 0))
    if np_img1.max() <= 1.0:
        np_img1 = (np_img1 * 255).astype(np.uint8)

    image1 = np_img1
    cv2.imwrite(f'/home/shham/Pictures/rendered/render{frame_id}.jpg', image1)

def uv_to_ray(uv, gt_color, m_i, m_j, c2w, H, W, fx, fy, cx, cy, device):
    
    u_reshaped_1 = uv[:, 0, ...]  # Extract the first channel along the second dimension
    v_reshaped_1 = uv[:, 1, ...]  # Extract the second channel along the second dimension

    

    W1 = gt_color.shape[1]
    index = (v_reshaped_1 * W1) + u_reshaped_1

    i_t = m_i[index]  
    j_t = m_j[index]  
    
    rays_o, rays_d = get_rays_from_uv(i_t, j_t, c2w, H, W, fx, fy, cx, cy, device)

    return rays_o, rays_d, index