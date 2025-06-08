import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn.functional as F
from pathlib import Path
from utils.depth_utils import compute_depth_camera_params, convert_depth_to_linear


def pose_to_matrix(position: torch.Tensor, rotation: torch.Tensor) -> torch.Tensor:
    w, x, y, z = rotation
    R = torch.tensor([
        [1 - 2 * (y**2 + z**2),     2 * (x * y - z * w),     2 * (x * z + y * w)],
        [    2 * (x * y + z * w), 1 - 2 * (x**2 + z**2),     2 * (y * z - x * w)],
        [    2 * (x * z - y * w),     2 * (y * z + x * w), 1 - 2 * (x**2 + y**2)]
    ], device=position.device)

    T = torch.eye(4, device=position.device)
    T[:3, :3] = R
    T[:3, 3] = position
    return T


def reproject_depth_with_unity_poses(
    depth_map: torch.Tensor,                    # (H, W)
    depth_intrinsics: torch.Tensor,            # (3, 3)
    depth_position: torch.Tensor,              # (3,)
    depth_rotation: torch.Tensor,              # (4,) wxyz
    rgb_intrinsics: torch.Tensor,              # (3, 3)
    rgb_position: torch.Tensor,                # (3,)
    rgb_rotation: torch.Tensor,                # (4,) wxyz
    out_size: tuple[int, int]                  # (H_rgb, W_rgb)
) -> torch.Tensor:
    depth_extrinsics = pose_to_matrix(depth_position, depth_rotation)
    rgb_extrinsics = pose_to_matrix(rgb_position, rgb_rotation)

    return reproject_depth(
        depth_map=depth_map,
        depth_intrinsics=depth_intrinsics,
        depth_extrinsics=depth_extrinsics,
        rgb_intrinsics=rgb_intrinsics,
        rgb_extrinsics=rgb_extrinsics,
        out_size=out_size
    )


def reproject_depth(
    depth_map: torch.Tensor,               # (H_d, W_d)
    depth_intrinsics: torch.Tensor,       # (3, 3)
    depth_extrinsics: torch.Tensor,       # (4, 4)
    rgb_intrinsics: torch.Tensor,         # (3, 3)
    rgb_extrinsics: torch.Tensor,         # (4, 4)
    out_size: tuple                        # (H_rgb, W_rgb)
) -> torch.Tensor:
    H_d, W_d = depth_map.shape
    H_rgb, W_rgb = out_size
    device = depth_map.device

    yy_rgb, xx_rgb = torch.meshgrid(
        torch.arange(H_rgb, device=device),
        torch.arange(W_rgb, device=device),
        indexing='ij'
    )
    ones_rgb = torch.ones_like(xx_rgb)
    pix_rgb = torch.stack([xx_rgb, yy_rgb, ones_rgb], dim=0).float().reshape(3, -1)  # (3, N)

    K_rgb_inv = torch.inverse(rgb_intrinsics)
    rays_rgb = K_rgb_inv @ pix_rgb                         # (3, N)
    rays_rgb_hom = torch.cat([rays_rgb, ones_rgb.view(1, -1)], dim=0)  # (4, N)

    T = torch.inverse(depth_extrinsics) @ rgb_extrinsics
    rays_in_depth = T @ rays_rgb_hom                       # (4, N)
    rays_in_depth = rays_in_depth[:3, :]                   # (3, N)

    proj = depth_intrinsics @ rays_in_depth
    z = rays_in_depth[2, :].clamp(min=1e-6)
    u = proj[0, :] / z
    v = proj[1, :] / z

    u_norm = (u / (W_d - 1)) * 2 - 1
    v_norm = (v / (H_d - 1)) * 2 - 1
    grid = torch.stack([u_norm, v_norm], dim=1).view(H_rgb, W_rgb, 2)  # (H_rgb, W_rgb, 2)

    depth_map_4d = depth_map.view(1, 1, H_d, W_d)

    depth_reprojected = F.grid_sample(
        depth_map_4d, grid.unsqueeze(0), mode='bilinear', align_corners=True
    )  # (1, 1, H_rgb, W_rgb)

    return depth_reprojected[0, 0]  # (H_rgb, W_rgb)


class DepthReprojector:
    def __init__(
        self, 
        depth_map_dir: Path,
        descriptor_csv: Path,
    ):
        self.depth_map_dir = depth_map_dir
        
        df = pd.read_csv(descriptor_csv, on_bad_lines='skip').dropna()
        df = df.sort_values('timestamp_ms')
        df = df.reset_index(drop=True)

        self.df = df

    def find_nearest_row(self, timestamp: int, window_ms: int = 30):
        diff = abs(self.df['timestamp_ms'] - timestamp)
        within_window = self.df[diff <= window_ms]

        if within_window.empty:
            return None

        nearest_idx = diff[within_window.index].idxmin()
        return self.df.loc[nearest_idx]
    
    def reproject_depth(
        self,
        timestamp: int,
        color_camera_unity_position: np.ndarray,
        color_camera_unity_rotation: R,
        color_camera_intrinsics: np.ndarray,
        color_camera_size: tuple[int, int],
    ) -> np.ndarray:
        row = self.find_nearest_row(timestamp)

        if row is None:
            return None

        depth_timestamp = int(row['timestamp_ms'])
        depth_width = int(row['width'])
        depth_height = int(row['height'])
        depth_near = float(row['near_z'])
        depth_far = float(row['far_z'])
        depth_left = float(row['fov_left_angle_tangent'])
        depth_right = float(row['fov_right_angle_tangent'])
        depth_top = float(row['fov_top_angle_tangent'])
        depth_bottom = float(row['fov_down_angle_tangent'])        

        depth_map_path = self.depth_map_dir / f"{depth_timestamp}.raw"

        if not os.path.exists(depth_map_path):
            print(f"Depth map file not found: {depth_map_path}")
            return None

        with open(depth_map_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype='<f4')
            
        depth_buffer =  data.reshape((depth_height, depth_width))
        depth_map_np = convert_depth_to_linear(depth_buffer, depth_near, depth_far)

        depth_camera_position_np = np.array([
            row['create_pose_location_x'],
            row['create_pose_location_y'],
            row['create_pose_location_z'],
        ], dtype=np.float32)

        depth_camera_rotation_np = np.array([
            row['create_pose_rotation_w'],
            row['create_pose_rotation_x'],
            row['create_pose_rotation_y'],
            row['create_pose_rotation_z'],
        ], dtype=np.float32)

        fx, fy, cx, cy = compute_depth_camera_params(
            depth_left, depth_right, 
            depth_top, depth_bottom, 
            depth_width, depth_height
        )

        depth_camera_intrinsics_np = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,   1]
        ], dtype=np.float32)

        dtype = torch.float32
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        depth_map = torch.tensor(depth_map_np, dtype=dtype, device=device)
        depth_intrinsics = torch.tensor(depth_camera_intrinsics_np, dtype=dtype, device=device)
        depth_position = torch.tensor(depth_camera_position_np, dtype=dtype, device=device)
        depth_rotation = torch.tensor(depth_camera_rotation_np, dtype=dtype, device=device)

        rgb_intrinsics = torch.tensor(color_camera_intrinsics, dtype=dtype, device=device)
        rgb_position = torch.tensor(color_camera_unity_position, dtype=dtype, device=device)
        rgb_rotation = torch.tensor(color_camera_unity_rotation.as_quat(), dtype=dtype, device=device)
        rgb_rotation = torch.roll(rgb_rotation, 1)

        depth_rgb = reproject_depth_with_unity_poses(
            depth_map=depth_map,
            depth_intrinsics=depth_intrinsics,
            depth_position=depth_position,
            depth_rotation=depth_rotation,
            rgb_intrinsics=rgb_intrinsics,
            rgb_position=rgb_position,
            rgb_rotation=rgb_rotation,
            out_size=color_camera_size
        )

        return depth_rgb.cpu().numpy()