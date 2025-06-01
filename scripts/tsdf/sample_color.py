from scipy.spatial.transform import Rotation as R
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
import os
from utils.pose_utils import PoseInterpolator, compose_transform
from tsdf.o3d_utils import unity_pose_to_open3d_extrinsic


def load_characteristics(characteristics_json):
    with open(characteristics_json, "r", encoding="utf-8") as f:
        camera_characteristics = json.load(f)

    array_size = camera_characteristics["sensor"]["activeArraySize"]
    width = array_size["right"] - array_size["left"]
    height = array_size["bottom"] - array_size["top"]

    intrinsics = camera_characteristics["intrinsics"]

    intrinsic = np.eye(3, dtype=np.float32)
    intrinsic[0, 0] = intrinsics["fx"]
    intrinsic[1, 1] = intrinsics["fy"]
    intrinsic[0, 2] = intrinsics["cx"]
    intrinsic[1, 2] = intrinsics["cy"]

    camera_pose = camera_characteristics["pose"]

    local_transl = camera_pose["translation"]
    if len(local_transl) < 3:
        local_transl = [0, 0, 0]

    local_quat = camera_pose["rotation"]
    if len(local_quat) >=4:
        qw = local_quat[0]
        qx = local_quat[1]
        qy = local_quat[2]
        qz = local_quat[3]
        local_quat = [qw, qx, qy, qz]
    else:
        local_quat = [0, 0, 0, 1]

    local_rot = R.from_quat(local_quat).inv()

    return intrinsic, width, height, local_transl, local_rot


def preprocess_color_dataset(
    characteristics_json,
    hmd_pose_csv,
    color_map_dir,
):
    hmd_pose_interpolator = PoseInterpolator(hmd_pose_csv)
    intrinsic, width, height, local_transl, local_rot = load_characteristics(characteristics_json)

    color_map_paths = []
    intrinsics = []
    extrinsics = []
    widths = []
    heights = []
    timestamps = []

    for filename in tqdm(os.listdir(color_map_dir)):
        if filename.endswith(".png"):
            try:
                timestamp = int(os.path.splitext(filename)[0])
                color_map_path = os.path.join(color_map_dir, filename)
            except ValueError:
                continue
        else:
            continue
        
        result = hmd_pose_interpolator.interpolate_pose(timestamp)
        if result is None:
            continue

        hmd_pos, hmd_rot = result
        position, rotation = compose_transform(
            hmd_pos, hmd_rot,
            local_transl, local_rot,
        )
        rotation *= R.from_quat([0, 1, 0, 0])

        extrinsic = unity_pose_to_open3d_extrinsic(
            position=position, 
            rotation=rotation
        )

        color_map_paths.append(color_map_path)
        extrinsics.append(extrinsic)
        timestamps.append(timestamp)
        widths.append(width)
        heights.append(height)
        intrinsics.append(intrinsic)

    return {
        "color_map_paths": np.array(color_map_paths),
        "extrinsics": np.array(extrinsics),          # shape: (N, 4, 4)
        "intrinsics": intrinsics,
        "widths": np.array(widths),
        "heights": np.array(heights),
        "timestamps": np.array(timestamps),
    }


def to_tensor_auto(arr, dtype):
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.tensor(arr, dtype=dtype, device=device)

def sample_color_from_color_maps(pcd_points_np, image_paths_np, intrinsics_np, extrinsics_np):
    import torch
    from torch.nn import functional as F

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pcd_points = to_tensor_auto(pcd_points_np, dtype=torch.float32)
    intrinsics = [to_tensor_auto(K, dtype=torch.float32) for K in intrinsics_np]
    extrinsics = [to_tensor_auto(E, dtype=torch.float32) for E in extrinsics_np]

    N = pcd_points.shape[0]
    M = image_paths_np.shape[0]
    points_h = torch.cat([pcd_points, torch.ones((N, 1), device=device)], dim=1)

    color_accum = torch.zeros((N, 3), dtype=torch.float32, device=device)
    weight_accum = torch.zeros((N,), dtype=torch.float32, device=device)

    for cam_idx in tqdm(range(0, M), desc="Sampling colors from color maps"):
        T = extrinsics[cam_idx]
        K = intrinsics[cam_idx]
        img = np.array(np.fliplr(Image.open(image_paths_np[cam_idx]).convert("RGB")))
        img = to_tensor_auto(img, torch.float32)

        H, W = img.shape[:2]

        points_cam = (T @ points_h.T).T[:, :3]
        z = points_cam[:, 2]
        in_front = z > 0

        proj = (K @ points_cam[in_front].T).T
        u = torch.round(proj[:, 0] / proj[:, 2]).long()
        v = torch.round(proj[:, 1] / proj[:, 2]).long()

        valid_uv = (u >= 0) & (u < W) & (v >= 0) & (v < H)
        indices = in_front.nonzero().flatten()[valid_uv]

        u = u[valid_uv]
        v = v[valid_uv]

        colors = img[v, u].float() / 255.0

        cam_pos = torch.linalg.inv(T)[:3, 3]
        # view_dirs = F.normalize(pcd_points[indices] - cam_pos, dim=1)
        # z_dir = points_cam[indices][:, 2]

        dist = torch.norm(pcd_points[indices] - cam_pos, dim=1)
        weight = 1.0 / (dist * dist + 1e-6)

        weight_accum[indices] += weight
        color_accum[indices] += colors * weight.unsqueeze(1)

    eps = 1e-6
    color_avg = color_accum / (weight_accum.unsqueeze(1) + eps)

    return color_avg.clamp(0.0, 1.0).cpu().numpy()