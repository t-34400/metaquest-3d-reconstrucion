import open3d as o3d
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from utils.depth_utils import compute_depth_camera_params, convert_depth_to_linear
from tsdf.o3d_utils import unity_pose_to_open3d_extrinsic


def preprocess_depth_dataset(
    descriptor_csv_path,
    depth_map_dir,
):
    df = pd.read_csv(descriptor_csv_path)

    depth_maps = []
    extrinsics = []
    intrinsics = []
    widths = []
    heights = []
    fx_list, fy_list = [], []
    cx_list, cy_list = [], []
    timestamps = []

    for _, row in df.iterrows():
        timestamp = int(row['timestamp_ms'])
        width = int(row['width'])
        height = int(row['height'])
        near = float(row['near_z'])
        far = float(row['far_z'])
        left = float(row['fov_left_angle_tangent'])
        right = float(row['fov_right_angle_tangent'])
        top = float(row['fov_top_angle_tangent'])
        bottom = float(row['fov_down_angle_tangent'])

        depth_map_path = os.path.join(depth_map_dir, f'{timestamp}.raw')
        if not os.path.exists(depth_map_path):
            print(f"Depth map file not found: {depth_map_path}")
            continue

        with open(depth_map_path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype='<f4')
            
        depth_buffer =  np.flipud(data.reshape((height, width)))
        depth_map = convert_depth_to_linear(depth_buffer, near, far)

        position = np.array([
            row['create_pose_location_x'],
            row['create_pose_location_y'],
            row['create_pose_location_z'],
        ])

        rotation = np.array([
            row['create_pose_rotation_x'],
            row['create_pose_rotation_y'],
            row['create_pose_rotation_z'],
            row['create_pose_rotation_w'],
        ])
        extrinsic = unity_pose_to_open3d_extrinsic(
            position=position, 
            quaternion=rotation
        )

        fx, fy, cx, cy = compute_depth_camera_params(
            left, right, top, bottom, width, height
        )

        K = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,   1]
        ], dtype=np.float32)

        depth_maps.append(depth_map)
        extrinsics.append(extrinsic)
        intrinsics.append(K)
        widths.append(width)
        heights.append(height)
        fx_list.append(fx)
        fy_list.append(fy)
        cx_list.append(cx)
        cy_list.append(cy)
        timestamps.append(timestamp)

    return {
        "depth_maps": np.array(depth_maps),          # shape: (N, H, W)
        "extrinsics": np.array(extrinsics),          # shape: (N, 4, 4)
        "intrinsics": intrinsics,
        "widths": np.array(widths),
        "heights": np.array(heights),
        "fx": np.array(fx_list),
        "fy": np.array(fy_list),
        "cx": np.array(cx_list),
        "cy": np.array(cy_list),
        "timestamps": np.array(timestamps),
    }


def integrate_depth_maps(
    dataset: dict,
    voxel_length: float = 0.01,
    sdf_trunc: float = 0.05,
) -> o3d.geometry.PointCloud:
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.NoColor
    )

    num_frames = len(dataset["depth_maps"])

    for i in tqdm(range(num_frames), desc="Integrating depth maps"):
        depth_np = dataset["depth_maps"][i].astype(np.float32)

        depth_uint16 = (depth_np * 1000).astype(np.uint16)

        sobelx = cv2.Sobel(depth_uint16, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(depth_uint16, cv2.CV_32F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)

        depth_np[grad_mag > 500] = 0
        depth_np[(depth_np < 0.2) | (depth_np > 1.5)] = 0

        depth_image = o3d.geometry.Image(depth_np)

        width=int(dataset["widths"][i])
        height=int(dataset["heights"][i])

        intrinsic = o3d.camera.PinholeCameraIntrinsic()
        intrinsic.set_intrinsics(
            width=width,
            height=height,
            fx=float(dataset["fx"][i]),
            fy=float(dataset["fy"][i]),
            cx=float(dataset["cx"][i]),
            cy=float(dataset["cy"][i])
        )

        extrinsic = dataset["extrinsics"][i]

        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.zeros((height, width, 3), dtype=np.uint8)),
            depth_image,
            depth_scale=1.0,
            depth_trunc=1.5,
            convert_rgb_to_intensity=True
        )

        volume.integrate(rgbd_image, intrinsic, extrinsic)

    return volume.extract_point_cloud()