import os
import shutil
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import constants
from utils.pose_utils import PoseInterpolator, compose_transform
from nerfstudio.transforms import Transforms, Frame
from nerfstudio.depth_reprojector import DepthReprojector


OUTPUT_TRANSFORMS_JSON = "transforms.json"
OUTPUT_IMAGE_DIR = "images"
OUTPUT_DEPTH_DIR = "depth"
OUTPUT_PCD_PLY = "dense_pc.ply"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory containing QRC data."
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=5,
        help="Interval for processing images (default: 5)."
    )
    args = parser.parse_args()

    return args


def load_frames(project_dir: Path, output_dir: Path, image_interval: int) -> tuple[dict, dict]:
    def load_camera_characteristics(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def extract_local_transform(pose: dict) -> tuple[np.ndarray, R]:
        t = pose.get("translation", [0, 0, 0])
        q = pose.get("rotation", [0, 0, 0, 1])
        if len(q) >= 4:
            quat = q
        else:
            quat = [0, 0, 0, 1]

        rot = R.from_quat(quat).inv()
        rot *= R.from_quat([1, 0, 0, 0])

        return np.array(t), rot

    def convert_unity_pose_to_transform_matrix(position: np.ndarray, rotation: R) -> tuple[np.ndarray, np.ndarray]:
        rotation *= R.from_rotvec(-np.pi / 2 * np.array([1, 0, 0]))
        rot_matrix = rotation.as_matrix()  # shape (3, 3)

        position[1], position[2] = position[2], position[1]
        M = np.array([
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ])
        rot_matrix = M @ rot_matrix @ M.T

        transform = np.eye(4, dtype=np.float32)
        transform[:3, :3] = rot_matrix
        transform[:3, 3] = position

        return transform.tolist()

    def process_eye_view(
        eye: str,
        image_interval: int,
        characteristics_json: Path,
        hmd_pose_csv: Path,
        color_map_dir: Path,
        depth_map_dir: Path,
        depth_descriptor_csv: Path,
        output_file_prefix: str,
        output_image_dir: Path,
        output_depth_dir: Path,
    ) -> list[Frame]:
        characteristics = load_camera_characteristics(characteristics_json)

        array_size = characteristics["sensor"]["activeArraySize"]
        width = array_size["right"] - array_size["left"]
        height = array_size["bottom"] - array_size["top"]

        intr = characteristics["intrinsics"]
        fx = intr["fx"]
        fy = intr["fy"]
        cx = intr["cx"]
        cy = intr["cy"]
        color_camera_intrinsics = np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,   1]
        ], dtype=np.float32)

        local_t, local_r = extract_local_transform(characteristics["pose"])
        hmd_interpolator = PoseInterpolator(hmd_pose_csv)
        reprojector = DepthReprojector(
            depth_map_dir=depth_map_dir,
            descriptor_csv=depth_descriptor_csv
        )

        frames = []

        files = sorted(os.listdir(color_map_dir))
        for filename in tqdm(files[::image_interval], desc=f"Processing {eye} images"):
            if not filename.endswith(".png"):
                continue
            try:
                timestamp = int(os.path.splitext(filename)[0])
            except ValueError:
                continue

            result = hmd_interpolator.interpolate_pose(timestamp)
            if result is None:
                continue

            hmd_pos, hmd_rot = result
            position, rotation = compose_transform(hmd_pos, hmd_rot, local_t, local_r)

            depth_map = reprojector.reproject_depth(
                timestamp=timestamp,
                color_camera_unity_position=position,
                color_camera_unity_rotation=rotation,
                color_camera_intrinsics=color_camera_intrinsics,
                color_camera_size=(height, width),
            )
            if depth_map is None:
                print(f"[Warning] Depth map not found for timestamp {timestamp}. Skipping image {filename}.")
                continue

            depth_map = depth_map * 1000.0  # Convert to mm

            transform_matrix = convert_unity_pose_to_transform_matrix(
                position=position,
                rotation=rotation,
            )
       
            src_image_path = color_map_dir / filename
            dst_image_name = output_file_prefix + filename
            dst_image_path = output_image_dir / dst_image_name
            shutil.copy(src_image_path, dst_image_path)

            dst_depth_name = output_file_prefix + os.path.splitext(filename)[0] + ".npy"
            dst_depth_path = output_depth_dir / dst_depth_name
            np.save(dst_depth_path, depth_map)

            frame = Frame(
                file_path=f"{OUTPUT_IMAGE_DIR}/{dst_image_name}",
                transform_matrix=transform_matrix,
                fl_x=fx,
                fl_y=fy,
                cx=cx,
                cy=cy,
                w=width,
                h=height,
                depth_file_path= f"{OUTPUT_DEPTH_DIR}/{dst_depth_name}",
            )
            frames.append(frame)

        return frames

    # --- Main logic ---
    frames = []
    hmd_pose_csv = project_dir / constants.HMD_POSE_CSV

    output_image_dir = output_dir / OUTPUT_IMAGE_DIR
    output_depth_dir = output_dir / OUTPUT_DEPTH_DIR
    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_depth_dir.mkdir(parents=True, exist_ok=True)

    for side in ["left", "right"]:
        characteristics_json = project_dir / getattr(constants, f"{side.upper()}_CAMERA_CHARACTERISTICS_JSON")
        color_map_dir = project_dir / getattr(constants, f"{side.upper()}_CAMERA_RGB_IMAGE_DIR")
        depth_map_dir = project_dir / getattr(constants, f"{side.upper()}_DEPTH_RAW_DATA_DIR")
        depth_descriptor_csv = project_dir / getattr(constants, f"{side.upper()}_DEPTH_DESCRIPTOR_CSV")
        output_image_prefix = f"{side}_"

        eye_frames = process_eye_view(
            eye=side,
            image_interval=image_interval,
            characteristics_json=characteristics_json,
            hmd_pose_csv=hmd_pose_csv,
            color_map_dir=color_map_dir,
            depth_map_dir=depth_map_dir,
            depth_descriptor_csv=depth_descriptor_csv,
            output_file_prefix=output_image_prefix,
            output_image_dir=output_image_dir,
            output_depth_dir=output_depth_dir,
        )

        frames.extend(eye_frames)

    return frames


def convert_pointcloud_open3d_to_opengl(src_ply_file: str, dst_ply_file: str):
    print(f"[Info] Loading point cloud from: {src_ply_file}")
    pcd = o3d.io.read_point_cloud(src_ply_file)
    print(f"[Info] Number of points: {len(pcd.points)}")

    R = np.array([
        [1, 0,  0],
        [0, 0, -1],
        [0, 1,  0]
    ])

    print("[Info] Transforming point coordinates...")
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(pcd.points) @ R.T
    )

    if pcd.has_normals():
        print("[Info] Normals detected: transforming normals...")
        pcd.normals = o3d.utility.Vector3dVector(
            np.asarray(pcd.normals) @ R.T
        )

    o3d.io.write_point_cloud(dst_ply_file, pcd)
    print(f"[Info] Saved transformed point cloud to: {dst_ply_file}")


def main(args):
    project_dir = args.project_dir
    print(f"[Info] Project path: {project_dir}")
    
    output_dir = args.output_dir
    print(f"[Info] Output Nerfstudio Project path: {output_dir}")
    
    transform_json = output_dir / OUTPUT_TRANSFORMS_JSON

    frames = load_frames(
        project_dir=project_dir,
        output_dir=output_dir,
        image_interval=args.interval
    )
    
    ply_file_name = None

    src_ply_file = project_dir / constants.COLORED_PCD
    if os.path.exists(src_ply_file):
        dst_ply_file = output_dir / OUTPUT_PCD_PLY
        convert_pointcloud_open3d_to_opengl(src_ply_file, dst_ply_file)

        ply_file_name = OUTPUT_PCD_PLY

    transforms = Transforms(
        camera_model="OPENCV", 
        k1=0.0,
        k2=0.0,
        k3=0.0,
        k4=0.0,
        p1=0.0,
        p2=0.0,
        frames=frames,
        ply_file_path=ply_file_name
    )

    transforms.to_json(str(transform_json))
    print(f"[Info] Transforms saved to: {transform_json}")


if __name__ == "__main__":
    args = parse_args()
    main(args)