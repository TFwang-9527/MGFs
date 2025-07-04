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

import torch
import lpips
def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))
def psnr_mask(img1, img2, mask):
    extended_mask = mask.repeat(3, 1, 1)
    diff = ((img1 - img2) * extended_mask)** 2
    mse_all = diff.view(img1.shape[0], -1).sum(dim=1)
    mse = mse_all/mask.sum()
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def psnr_mask2(img1, img2, mask):
    extended_mask = mask.repeat(3, 1, 1)

    # 步骤3: 应用掩码
    img1 = img1 * extended_mask
    img2 = img2 * extended_mask
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))



def calculate_lpips(img1, img2, mask, net_type='alex', use_gpu=True):
    extended_mask = mask.repeat(3, 1, 1)
    img1 = img1 * extended_mask
    img2 = img2 * extended_mask
    # Initialize the LPIPS model with the specified backbone network
    lpips_model = lpips.LPIPS(net=net_type, verbose=False)

    # Optionally use GPU for computation
    if use_gpu and torch.cuda.is_available():
        lpips_model.cuda()
        img1 = img1.cuda()
        img2 = img2.cuda()

    # Reshape images to [batch_size, channels, height, width] if necessary
    if img1.dim() == 3:
        img1 = img1.unsqueeze(0)
    if img2.dim() == 3:
        img2 = img2.unsqueeze(0)

    # Compute the LPIPS distance
    with torch.no_grad():
        distance = lpips_model(img1, img2)

    return distance.item()