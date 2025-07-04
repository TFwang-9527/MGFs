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
from sklearn.cluster import KMeans
import time
import os
import sys
from PIL import Image
from scipy.ndimage import label
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor, as_completed
class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    mask_merge: np.array


class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str


def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}


def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx + 1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model == "SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]

        if not os.path.exists(image_path) or "sky_mask" in image_path:
            print("skip =====", image_path)
            continue

        image = Image.open(image_path)
        #########
        mask_merge = np.zeros((image.height, image.width), dtype=np.int32)

        #########

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,mask_merge=mask_merge)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos


def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


import torch
import numpy as np


def fov2focal(fov, dim, device='cuda'):
    fov_tensor = torch.tensor(fov, dtype=torch.float32, device=device)
    dim_tensor = torch.tensor(dim, dtype=torch.float32, device=device)
    return dim_tensor / (2 * torch.tan(fov_tensor / 2))


def world_to_camera_point(points, R, T):
    R = torch.tensor(R, dtype=torch.float32, device=points.device)
    T = torch.tensor(T, dtype=torch.float32, device=points.device).view(3, 1)
    points = points.clone().detach()  # 使用 clone().detach() 复制 tensor
    points_camera = torch.matmul(R.t(), points.t()) + T
    points_camera = points_camera.t()
    return points_camera

# 其他代码保持不变



def project_to_image(points, cam_info, scale):
    fx = fov2focal(cam_info.FovX, cam_info.width)
    fy = fov2focal(cam_info.FovY, cam_info.height)
    cx, cy = cam_info.width / 2, cam_info.height / 2

    points_camera = world_to_camera_point(points, cam_info.R, cam_info.T)

    x = points_camera[:, 0]
    y = points_camera[:, 1]
    z = points_camera[:, 2]

    valid = z > 0
    u = fx * x / z + cx
    v = fy * y / z + cy

    # Apply scale factor to account for mask downsampling
    u = u * scale
    v = v * scale

    # 将无效点（即在相机后面的点）的 u 和 v 设置为 -1
    u[~valid] = -1
    v[~valid] = -1

    return u, v

def remove_outliers(points, z_threshold=2):
    median = torch.median(points, dim=0).values
    abs_diff = torch.abs(points - median)
    mad = torch.median(abs_diff, dim=0).values
    modified_z_scores = 0.6745 * abs_diff / mad
    mask = modified_z_scores < z_threshold
    return points[mask.all(dim=1)]


def apply_merged_labels_to_masks(masks, merged_labels):
    # 将所有掩码合并到一个大的 tensor 中进行处理
    all_masks = np.stack(masks)
    all_masks_tensor = torch.tensor(all_masks, dtype=torch.int32, device='cuda')

    # 初始化新的掩码 tensor
    all_new_masks = torch.zeros_like(all_masks_tensor, dtype=torch.int32)

    # 将合并后的标签应用到所有掩码
    for old_label, new_label in merged_labels.items():
        all_new_masks[all_masks_tensor == old_label] = new_label

    return all_new_masks.cpu().numpy()


def apply_mask_to_point_cloud(points, colors, masks, camera_infos, scale, path):
    points_tensor = torch.tensor(points, dtype=torch.float32, device='cuda')
    # 输出处理前的点数
    print("处理前的点数：", points_tensor.shape[0])

    # 预处理步骤，保留有效点
    labels_chunk = torch.zeros(points_tensor.shape[0], dtype=torch.int32, device='cuda')
    false_proj_counts = torch.zeros(points_tensor.shape[0], dtype=torch.int32, device='cuda')

    # 统计预处理步骤所用时间
    preproc_start_time = time.time()

    for cam_info, mask in zip(camera_infos, masks):
        torch.cuda.empty_cache()
        mask_tensor = torch.tensor(mask, dtype=torch.int32, device='cuda')  # 转换 mask 为 CUDA 张量
        u, v = project_to_image(points_tensor, cam_info, scale)
        u = u.long()
        v = v.long()
        valid_mask = (u >= 0) & (u < mask_tensor.shape[1]) & (v >= 0) & (v < mask_tensor.shape[0])
        valid_indices = valid_mask.nonzero(as_tuple=True)

        valid_labels = mask_tensor[v[valid_indices], u[valid_indices]]
        labels_chunk[valid_indices] = valid_labels

        false_proj = valid_labels < 1
        false_proj_counts[valid_indices] += false_proj.long()

    valid_points = (labels_chunk > 0) & (false_proj_counts <= 4)


    points_tensor = points_tensor[valid_points]
    colors = np.array(colors)[valid_points.cpu().numpy()]
    # 输出处理后的点数
    print("处理后的点数：", points_tensor.shape[0])

    preproc_end_time = time.time()
    print("预处理步骤所用时间：", preproc_end_time - preproc_start_time, "秒")

    unique_labels = 1
    label_list = []

    for i, mask in enumerate(masks):
        labeled_mask, num_labels = label(mask)

        # Create an array to count the number of pixels for each label
        label_counts = np.bincount(labeled_mask.ravel())

        # Create a mask for labels that have more than 300 pixels
        valid_labels = np.where(label_counts >= 300)[0]

        # Create a new labeled mask with only valid labels
        new_labeled_mask = np.zeros_like(labeled_mask)
        new_label = 1
        for label_num in valid_labels:
            if label_num == 0:
                continue  # skip the background label
            new_labeled_mask[labeled_mask == label_num] = new_label
            new_label += 1

        # Increment labels to maintain unique labeling across masks
        new_labeled_mask[new_labeled_mask > 0] += unique_labels
        masks[i] = new_labeled_mask
        unique_labels += new_label - 1
        label_list.append(new_labeled_mask)

    # 为每个标签分配颜色
    cmap = plt.get_cmap("hsv", unique_labels)
    label_colors = {label_id: np.array(cmap(label_id)[:3]) * 255 for label_id in range(1, unique_labels)}

    label_ranges = {}
    start_time = time.time()
    occu_point_labels = torch.zeros(points_tensor.shape[0], dtype=torch.int32, device='cuda')

    for i, (cam_info, mask) in enumerate(zip(camera_infos, label_list)):
        u, v = project_to_image(points_tensor, cam_info, scale)
        u = u.long()
        v = v.long()
        point_labels = torch.zeros(points_tensor.shape[0], dtype=torch.int32, device='cuda')
        valid_mask = (u >= 0) & (u < mask.shape[1]) & (v >= 0) & (v < mask.shape[0])
        valid_indices = valid_mask.nonzero(as_tuple=True)

        mask = torch.tensor(mask, dtype=torch.int32, device='cuda')  # 确保掩码在CUDA上
        point_labels[valid_indices[0]] = mask[v[valid_indices], u[valid_indices]]
        occu_point_labels[valid_indices[0]] = mask[v[valid_indices], u[valid_indices]]

        for label_id in torch.unique(point_labels):
            if label_id == 0:
                continue
            label_points = points_tensor[point_labels == label_id]

            # 保存原始标签点为 PLY 文件
            if label_points.shape[0] < 0:
                original_ply_output_path = os.path.join(path, f"original_label_{label_id.item()}_points.ply")
                storePly(original_ply_output_path, label_points.cpu().numpy(), np.full((label_points.shape[0], 3), 255))


            # 去除离群点
            filtered_label_points = remove_outliers(label_points)

            # 保存去除离群点后的标签点为 PLY 文件
            if filtered_label_points.shape[0] < 0:
                filtered_ply_output_path = os.path.join(path, f"filtered_label_{label_id.item()}_points.ply")
                storePly(filtered_ply_output_path, filtered_label_points.cpu().numpy(),
                         np.full((filtered_label_points.shape[0], 3), 255))


            if filtered_label_points.shape[0] > 0:  # 确保去除离群点后仍有点存在
                min_xyz = torch.min(filtered_label_points, dim=0)[0]
                max_xyz = torch.max(filtered_label_points, dim=0)[0]
                label_ranges[label_id.item()] = (min_xyz.cpu().numpy(), max_xyz.cpu().numpy())
            else:
                # 如果所有点都是离群点，删除该标签
                label_ranges.pop(label_id.item(), None)

    end_time = time.time()

    start_time = time.time()

    label_ids = np.array(list(label_ranges.keys()))
    min_xy_values = np.array([label_ranges[label_id][0][:2] for label_id in label_ids])
    max_xy_values = np.array([label_ranges[label_id][1][:2] for label_id in label_ids])

    num_labels = len(label_ids)
    overlaps = np.zeros((num_labels, num_labels))

    for i in range(num_labels):
        for j in range(i + 1, num_labels):
            if (min_xy_values[i][0] < max_xy_values[j][0] and max_xy_values[i][0] > min_xy_values[j][0]) and \
                    (min_xy_values[i][1] < max_xy_values[j][1] and max_xy_values[i][1] > min_xy_values[j][1]):
                overlaps[i, j] = 1
                overlaps[j, i] = 1
            elif (min_xy_values[j][0] < max_xy_values[i][0] and max_xy_values[j][0] > min_xy_values[i][0]) and \
                    (min_xy_values[j][1] < max_xy_values[i][1] and max_xy_values[j][1] > min_xy_values[i][1]):
                overlaps[i, j] = 1
                overlaps[j, i] = 1

    # 初始化合并后的标签
    merged_labels = {label_id: label_id for label_id in label_ids}

    # 更新标签
    for i in range(num_labels):
        overlapping_labels = np.where(overlaps[i] == 1)[0]
        if overlapping_labels.size > 0:
            new_label = min(label_ids[i], merged_labels[label_ids[i]], *label_ids[overlapping_labels])
            for label_id in overlapping_labels:
                merged_labels[label_ids[label_id]] = new_label
            merged_labels[label_ids[i]] = new_label

    end_time = time.time()

    final_labels = torch.clone(occu_point_labels)
    for old_label, new_label in merged_labels.items():
        final_labels[occu_point_labels == old_label] = new_label

    valid_indices = (final_labels > 0).nonzero(as_tuple=True)[0]
    filtered_points = points_tensor[valid_indices]
    filtered_colors = torch.tensor(colors, dtype=torch.float32, device='cuda')[valid_indices]

    new_masks = apply_merged_labels_to_masks(masks, merged_labels)

    # 更新 camera_infos 中的 mask_merge 参数
    for i, new_mask in enumerate(new_masks):
        camera_infos[i] = camera_infos[i]._replace(mask_merge=new_mask)

    # 将不同标签的三维点保存为不同颜色的点
    color_map = plt.cm.get_cmap("hsv", len(label_ids))  # 使用HSV色彩映射
    points_colors = np.zeros((points_tensor.shape[0], 3))
    for label_id in label_ids:
        color = color_map(label_id)[:3]  # 获取RGB颜色
        points_colors[final_labels.cpu().numpy() == label_id] = color

    points_colors = (points_colors * 255).astype(np.uint8)
    filtered_colors_multi = torch.tensor(points_colors, dtype=torch.float32, device='cuda')[valid_indices]

    # 将结果保存为PLY文件
    ply_output_path = os.path.join(path, "colored_points.ply")
    storePly(ply_output_path, filtered_points.cpu().numpy(), filtered_colors_multi.cpu().numpy())

    return filtered_points.cpu().numpy(), filtered_colors.cpu().numpy()


def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics,
                                           images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key=lambda x: x.image_name)

    mask_folder = os.path.join(path, "mask")
    masks = []
    for cam_info in cam_infos:
        mask_path = None
        for ext in ['.png', '.jpg', '.JPG']:
            test_path = os.path.join(mask_folder, f"{cam_info.image_name}{ext}")
            if os.path.exists(test_path):
                mask_path = test_path
                break

        if mask_path is None:
            raise FileNotFoundError(
                f"Could not find mask image for {cam_info.image_name} "
                f"in {mask_folder} (tried .png and .jpg)"
            )

        mask = Image.open(mask_path).convert("L")
        masks.append(np.array(mask))

    # Assume all masks have the same scale relative to original images
    scale = masks[0].shape[1] / cam_infos[0].width
    scale = round(scale * 4) * 0.25
    print("scale", scale)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")

    if os.path.exists(ply_path):
        print(f"{ply_path} already exists. Skipping point cloud processing.")
        print(f"Reading point cloud from {bin_path}")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        print(f"Applying masks to point cloud")
        start_time = time.time()
        filtered_points, filtered_colors = apply_mask_to_point_cloud(xyz, rgb, masks, cam_infos, scale, path)
        end_time = time.time()
        print("the time of filter points is ", end_time - start_time)

        if len(filtered_points) == 0:
            print("Warning: No points left after applying masks")

        print(f"Filtered points: {filtered_points.shape}, Filtered colors: {filtered_colors.shape}")

        # Store filtered point cloud to PLY
        storePly(ply_path, xyz, rgb)
        #storePly(ply_path, filtered_points, filtered_colors)
    else:
        print(f"Reading point cloud from {bin_path}")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)

        print(f"Applying masks to point cloud")
        start_time = time.time()
        filtered_points, filtered_colors = apply_mask_to_point_cloud(xyz, rgb, masks, cam_infos, scale, path)
        end_time = time.time()
        print("the time of filter points is ", end_time - start_time)

        if len(filtered_points) == 0:
            print("Warning: No points left after applying masks")

        print(f"Filtered points: {filtered_points.shape}, Filtered colors: {filtered_colors.shape}")

        # Store filtered point cloud to PLY
        storePly(ply_path, xyz, rgb)
        #storePly(ply_path, filtered_points, filtered_colors)

    try:
        pcd = fetchPly(ply_path)
    except Exception as e:
        print(f"Error fetching PLY file: {e}")
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                        image_path=image_path, image_name=image_name, width=image.size[0],
                                        height=image.size[1]))

    return cam_infos


def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


def readMultiScale(path, white_background, split, only_highres=False):
    cam_infos = []

    print("read split:", split)
    with open(os.path.join(path, 'metadata.json'), 'r') as fp:
        meta = json.load(fp)[split]

    meta = {k: np.array(meta[k]) for k in meta}

    # should now have ['pix2cam', 'cam2world', 'width', 'height'] in self.meta
    for idx, relative_path in enumerate(meta['file_path']):
        if only_highres and not relative_path.endswith("d0.png"):
            continue
        image_path = os.path.join(path, relative_path)
        image_name = Path(image_path).stem

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = meta["cam2world"][idx]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        image = Image.open(image_path)

        im_data = np.array(image.convert("RGBA"))

        bg = np.array([1, 1, 1]) if white_background else np.array([0, 0, 0])

        norm_data = im_data / 255.0
        arr = norm_data[:, :, :3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr * 255.0, dtype=np.byte), "RGB")

        fovx = focal2fov(meta["focal"][idx], image.size[0])
        fovy = focal2fov(meta["focal"][idx], image.size[1])
        FovY = fovy
        FovX = fovx

        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                    image_path=image_path, image_name=image_name, width=image.size[0],
                                    height=image.size[1]))
    return cam_infos


def readMultiScaleNerfSyntheticInfo(path, white_background, eval, load_allres=False):
    print("Reading train from metadata.json")
    train_cam_infos = readMultiScale(path, white_background, "train", only_highres=(not load_allres))
    print("number of training images:", len(train_cam_infos))
    print("Reading test from metadata.json")
    test_cam_infos = readMultiScale(path, white_background, "test", only_highres=False)
    print("number of testing images:", len(test_cam_infos))
    if not eval:
        print("adding test cameras to training")
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender": readNerfSyntheticInfo,
    "Multi-scale": readMultiScaleNerfSyntheticInfo,
}