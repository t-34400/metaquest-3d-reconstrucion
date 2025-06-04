import os
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from tqdm import tqdm
import constants
from utils.pose_utils import PoseInterpolator, compose_transform
from third_party.colmap.read_and_write_model import Camera, Image, Point3D, write_model


OUTPUT_IMAGE_DIR = "input"
OUTPUT_MODEL_DIR = "distorted/sparse/0"


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


def load_cameras_and_images(project_dir: Path, output_dir: Path, image_interval: int) -> tuple[dict, dict]:
    def load_camera_characteristics(path: Path) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def create_camera_from_characteristics(camera_id: int, characteristics: dict) -> Camera:
        array_size = characteristics["sensor"]["activeArraySize"]
        width = array_size["right"] - array_size["left"]
        height = array_size["bottom"] - array_size["top"]

        intr = characteristics["intrinsics"]
        params = np.array([intr["fx"], intr["fy"], intr["cx"], intr["cy"]])
        model = "PINHOLE"

        return Camera(camera_id, model, width, height, params)

    def extract_local_transform(pose: dict) -> tuple[np.ndarray, R]:
        t = pose.get("translation", [0, 0, 0])
        q = pose.get("rotation", [0, 0, 0, 1])
        if len(q) >= 4:
            quat = [q[0], q[1], q[2], q[3]]
        else:
            quat = [0, 0, 0, 1]
        return np.array(t), R.from_quat(quat).inv()

    def convert_unity_pose_to_colmap(position: np.ndarray, rotation: R) -> tuple[np.ndarray, np.ndarray]:
        rotation *= R.from_quat([1, 0, 0, 0])
        rotation = rotation.inv()

        tvec = -rotation.as_matrix() @ position
        qvec = rotation.as_quat()

        tvec[1] *= -1
        qvec = np.array((
            qvec[3],
            -qvec[0],
            qvec[1],
            -qvec[2],
        ))

        return tvec, qvec

    def process_eye_view(
        eye: str,
        camera_id: int,
        image_start_id: int,
        image_interval: int,
        characteristics_json: Path,
        hmd_pose_csv: Path,
        color_map_dir: Path,
        output_image_prefix: str,
        output_image_dir: Path,
        cameras: dict,
        images: dict,
    ) -> int:
        characteristics = load_camera_characteristics(characteristics_json)
        cameras[camera_id] = create_camera_from_characteristics(camera_id, characteristics)
        local_t, local_r = extract_local_transform(characteristics["pose"])
        hmd_interpolator = PoseInterpolator(hmd_pose_csv)

        image_id = image_start_id
        for filename in tqdm(os.listdir(color_map_dir)[::image_interval], desc=f"Processing {eye} images"):
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
            pos, rot = compose_transform(hmd_pos, hmd_rot, local_t, local_r)
            trans, quat = convert_unity_pose_to_colmap(pos, rot)

            src_image_path = color_map_dir / filename
            dst_image_name = output_image_prefix + filename
            dst_image_path = output_image_dir / dst_image_name
            shutil.copy(src_image_path, dst_image_path)

            images[image_id] = Image(
                id=image_id,
                qvec=quat,
                tvec=trans,
                camera_id=camera_id,
                name=dst_image_name,
                xys=np.empty((0, 2)),
                point3D_ids=np.empty((0,))
            )
            image_id += 1

        return image_id

    # --- Main logic ---
    cameras, images = {}, {}
    image_start_id = 0
    hmd_pose_csv = project_dir / constants.HMD_POSE_CSV
    output_image_dir = output_dir / OUTPUT_IMAGE_DIR
    output_image_dir.mkdir(parents=True, exist_ok=True)

    for side in ["left", "right"]:
        camera_id = 0 if side == "left" else 1
        characteristics_json = project_dir / getattr(constants, f"{side.upper()}_CAMERA_CHARACTERISTICS_JSON")
        color_map_dir = project_dir / getattr(constants, f"{side.upper()}_CAMERA_RGB_IMAGE_DIR")
        output_image_prefix = f"{side}_"

        image_start_id = process_eye_view(
            eye=side,
            camera_id=camera_id,
            image_start_id=image_start_id,
            image_interval=image_interval,
            characteristics_json=characteristics_json,
            hmd_pose_csv=hmd_pose_csv,
            color_map_dir=color_map_dir,
            output_image_prefix=output_image_prefix,
            output_image_dir=output_image_dir,
            cameras=cameras,
            images=images,
        )

    return cameras, images

            
def load_point_cloud_as_points3D(project_dir):
    pcd_file = project_dir / constants.COLORED_PCD

    pcd = o3d.io.read_point_cloud(pcd_file)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)

    # Convert Open3D point coordinates to COLMAP coordinate system
    points[:, 1] *= -1
    points[:, 2] *= -1
    colors = (colors * 255).round().astype(np.uint8)

    points3D = {}

    for i in tqdm(range(len(points)), desc="Converting to Point3D"):
        point = points[i]
        color = colors[i]

        point3D = Point3D(
            id=i,
            xyz=point,
            rgb=color,
            error=0,
            image_ids=np.array([], dtype=np.int64),
            point2D_idxs=np.array([], dtype=np.int64),
        )

        points3D[i] = point3D

    return points3D


def main(args):
    project_dir = args.project_dir
    print(f"[Info] Project path: {project_dir}")
    
    output_dir = args.output_dir
    print(f"[Info] Output COLMAP Project path: {output_dir}")
    
    model_dir = output_dir / OUTPUT_MODEL_DIR
    model_dir.mkdir(parents=True, exist_ok=True)

    cameras, images = load_cameras_and_images(project_dir, output_dir, args.interval)
    points3D = load_point_cloud_as_points3D(project_dir)

    write_model(
        cameras=cameras,
        images=images,
        points3D=points3D,
        path=model_dir,
        ext=".bin"
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)