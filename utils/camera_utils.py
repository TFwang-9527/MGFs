#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
from PIL import Image
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
import numpy as np
import torch
from scipy.ndimage import zoom
WARNED = False
import torch.nn.functional as F
def get_img_grad_weight(img, beta=2.0):
    _, hd, wd = img.shape
    bottom_point = img[..., 2:hd,   1:wd-1]
    top_point    = img[..., 0:hd-2, 1:wd-1]
    right_point  = img[..., 1:hd-1, 2:wd]
    left_point   = img[..., 1:hd-1, 0:wd-2]
    grad_img_x = torch.mean(torch.abs(right_point - left_point), 0, keepdim=True)
    grad_img_y = torch.mean(torch.abs(top_point - bottom_point), 0, keepdim=True)
    grad_img = torch.cat((grad_img_x, grad_img_y), dim=0)
    grad_img, _ = torch.max(grad_img, dim=0)
    grad_img = (grad_img - grad_img.min()) / (grad_img.max() - grad_img.min())
    grad_img = torch.nn.functional.pad(grad_img[None,None], (1,1,1,1), mode='constant', value=1.0).squeeze()
    return grad_img
def resize_ndarray_to_tensor(ndarray, target_tensor):
    """
    将 ndarray 缩放为与目标张量相同的大小，并转换为 PyTorch 张量
    :param ndarray: 原始的 numpy ndarray
    :param target_tensor: 目标 PyTorch 张量
    :return: 缩放后的 PyTorch 张量
    """
    target_shape = target_tensor.shape
    zoom_factors = (target_shape[0] / ndarray.shape[0], target_shape[1] / ndarray.shape[1])
    resized_ndarray = zoom(ndarray, zoom_factors, order=0)  # 使用最近邻插值 (order=0)
    resized_ndarray = resized_ndarray.astype(np.int32)  # 转换为 int 类型
    resized_tensor = torch.tensor(resized_ndarray, dtype=torch.int32)
    return resized_tensor

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8, 16, 32, 64]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    if len(cam_info.image.split()) > 3:
        import torch
        resized_image_rgb = torch.cat([PILtoTorch(im, resolution) for im in cam_info.image.split()[:3]], dim=0)
        loaded_mask = PILtoTorch(cam_info.image.split()[3], resolution)
        gt_image = resized_image_rgb
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        loaded_mask = None
        gt_image = resized_image_rgb
    ##################################
    path_mask = args.source_path+'/mask/'+cam_info.image_name+'.JPG'
    image_mask = Image.open(path_mask)
    mask_tensor = PILtoTorch(image_mask, resolution)
    mask_tensor = mask_tensor.squeeze(0)
    ###################################
    path_white_mask = args.source_path + '/white/' + cam_info.image_name + '.JPG'
    white_mask = Image.open(path_white_mask)
    white_tensor = PILtoTorch(white_mask, resolution)
    white_tensor = white_tensor.squeeze(0)
    ###################################
    path_multi_mask = args.source_path + '/multi_mask/' + cam_info.image_name + '.png'
    multi_image_mask = Image.open(path_multi_mask)
    multi_mask_tensor = PILtoTorch(multi_image_mask, resolution)
    multi_mask_tensor = multi_mask_tensor.squeeze(0)

    # For RGBA images, we'll use only the RGB channels (assuming alpha is not needed)
    # Convert to grayscale by taking mean of RGB channels if needed
    if multi_mask_tensor.shape[0] == 4:  # RGBA
        multi_mask_tensor = multi_mask_tensor[:3].mean(dim=0, keepdim=True)  # Convert to grayscale
    elif multi_mask_tensor.shape[0] == 3:  # RGB
        multi_mask_tensor = multi_mask_tensor.mean(dim=0, keepdim=True)  # Convert to grayscale
    multi_mask_tensor = multi_mask_tensor.squeeze(0)  # Remove channel dimension
    ###################################

    #计算multi_tensor的边界作为mask_edge
    import torch
    import torch.nn.functional as F

    edge_mask_tensor = torch.zeros_like(multi_mask_tensor)
    kernel = torch.tensor([[0, 1, 0],
                           [1, -4, 1],
                           [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 形状为 (1, 1, 3, 3)
    boundary = F.conv2d(multi_mask_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=1)
    boundary = boundary.squeeze(0).squeeze(0)
    edge_mask_tensor[(boundary != 0)] = 1
    edge_mask_tensor[(boundary == 0) & (white_tensor != 0)] = 0.1
    edge_mask_tensor[white_tensor == 0] = 0.1

    gradients = (1.0 - get_img_grad_weight(gt_image))
    gradients = (gradients).clamp(0, 1).detach() ** 2
    unique_labels = torch.unique(multi_mask_tensor)
    avg_grads = torch.ones_like(unique_labels, dtype=gradients.dtype)

    # Calculate average gradient for each mask region
    for i, label in enumerate(unique_labels):
        mask = (multi_mask_tensor == label)
        avg_grads[i] = gradients[mask].mean() if mask.any() else 0

    # Create RGB_weight tensor and assign average gradient values to each region
    RGB_weight = torch.ones_like(multi_mask_tensor, dtype=avg_grads.dtype)
    for i, label in enumerate(unique_labels):
        mask = (multi_mask_tensor == label)
        RGB_weight[mask] = avg_grads[i]

    # Set background (label 0) to corresponding gradient values
    background_mask = (white_tensor == 0)
    RGB_weight[background_mask] = gradients[background_mask]  # This ensures shape matching

    mask_merge = cam_info.mask_merge
    resized_mask_merge_tensor = resize_ndarray_to_tensor(mask_merge, multi_mask_tensor)



    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY,
                  image=gt_image, mask=mask_tensor,multi_mask=multi_mask_tensor,white_mask=white_tensor,RGB_weight =RGB_weight,edge_mask=edge_mask_tensor,gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name,mask_merge=resized_mask_merge_tensor, uid=id, data_device=args.data_device)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
