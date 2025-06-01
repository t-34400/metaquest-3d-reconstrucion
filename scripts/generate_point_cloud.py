import open3d as o3d
import numpy as np
import os
import argparse
from pathlib import Path
import constants
from tsdf.integrate_depth_maps import preprocess_depth_dataset, integrate_depth_maps
from tsdf.sample_color import preprocess_color_dataset, sample_color_from_color_maps

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=Path,
        required=True,
        help="Path to the project directory"
    )
    parser.add_argument(
        "--voxel_length",
        type=float,
        default=0.01,
        help="Length of the voxel in meters for TSDF volume (default: 0.01 m)"
    )
    parser.add_argument(
        "--sdf_trunc",
        type=float,
        default=0.04,
        help="Truncation distance for the TSDF volume (default: 0.04 m)"
    )
    parser.add_argument(
        "--color",
        action='store_true',
        help="Sample volume color from the color dataset"
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Visualize the TSDF volume"
    )

    return parser.parse_args()


def load_or_create_depth_dataset(project_dir: Path) -> dict:
    output_path = project_dir / constants.DEPTH_DATASET_CACHE

    if output_path.exists():
        print(f"[Info] Found existing depth dataset: {output_path}")
        data = np.load(output_path)
        return {key: data[key] for key in data.files}
    
    print("[Info] No existing depth dataset found. Generating new dataset...")

    sides = ["left", "right"]
    combined_data = {}
    total_frames = 0

    for side in sides:
        if side == "left":
            descriptor_csv = project_dir / constants.LEFT_DEPTH_DESCRIPTOR_CSV
            depth_dir = project_dir / constants.LEFT_DEPTH_RAW_DATA_DIR
        else:
            descriptor_csv = project_dir / constants.RIGHT_DEPTH_DESCRIPTOR_CSV
            depth_dir = project_dir / constants.RIGHT_DEPTH_RAW_DATA_DIR

        print(f"[Info] Processing '{side}' side...")
        dataset = preprocess_depth_dataset(descriptor_csv, depth_dir)

        frame_count = len(dataset["depth_maps"])
        total_frames += frame_count
        print(f"[Info] {frame_count} frames processed for '{side}'")

        if not combined_data:
            combined_data = dataset
        else:
            for key in combined_data:
                combined_data[key] = np.concatenate([combined_data[key], dataset[key]], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **combined_data)
    print(f"[Info] Saved {total_frames} total frames to: {output_path}")

    return combined_data


def load_color_dataset(project_path: Path) -> dict:
    output_path = project_path / constants.COLOR_DATASET_CACHE

    if output_path.exists():
        print(f"[Info] Found existing color dataset: {output_path}")
        data = np.load(output_path)
        return {key: data[key] for key in data.files}
    
    print("[Info] No existing color dataset found. Generating new dataset...")

    hmd_pose_csv = project_path / constants.HMD_POSE_CSV

    sides = ["left", "right"]
    combined_data = {}
    total_frames = 0

    for side in sides:
        if side == "left":
            characteristics_json = project_path / constants.LEFT_CAMERA_CHARACTERISTICS_JSON
            color_map_dir = project_path / constants.LEFT_CAMERA_RGB_IMAGE_DIR
        else:
            characteristics_json = project_path / constants.RIGHT_CAMERA_CHARACTERISTICS_JSON
            color_map_dir = project_path / constants.RIGHT_CAMERA_RGB_IMAGE_DIR

        print(f"[Info] Processing '{side}' side...")
        dataset = preprocess_color_dataset(characteristics_json, hmd_pose_csv, color_map_dir)

        frame_count = len(dataset["color_map_paths"])
        total_frames += frame_count
        print(f"[Info] {frame_count} frames loaded for '{side}'")

        if not combined_data:
            combined_data = dataset
        else:
            for key in combined_data:
                combined_data[key] = np.concatenate([combined_data[key], dataset[key]], axis=0)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **combined_data)
    print(f"[Info] Saved {total_frames} total frames to: {output_path}")

    return combined_data


def main(args):
    project_dir = args.project_dir
    print(f"[Info] Project path: {project_dir}")

    no_color_pcd = project_dir / constants.NO_COLOR_PCD
    colored_pcd = project_dir / constants.COLORED_PCD

    if os.path.exists(colored_pcd):
        print(f"[Info] Found existing colored point cloud: {colored_pcd}")
        pcd = o3d.io.read_point_cloud(colored_pcd)
      
    else:
        if os.path.exists(no_color_pcd):
            print(f"[Info] Found existing no-color point cloud: {no_color_pcd}")
            pcd = o3d.io.read_point_cloud(no_color_pcd)

        else:
            depth_dataset = load_or_create_depth_dataset(project_dir)

            print("[Info] Integrating depth maps into TSDF volume...")
            pcd = integrate_depth_maps(depth_dataset, args.voxel_length, args.sdf_trunc)
            print("[Info] Removing non-finite points...")
            pcd.remove_non_finite_points()
            print("[Info] Downsampling point cloud using voxel grid...")
            pcd.voxel_down_sample(voxel_size=0.01)
            print("[Info] Removing statistical outliers...")
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
            print(f"[Info] Valid points: {len(ind)} / {len(pcd.points)} ({len(ind) / len(pcd.points) * 100:.2f} %)")
            pcd = pcd.select_by_index(ind)

            no_color_pcd.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(no_color_pcd, pcd)
            print(f"[Info] Saved {len(pcd.points)} total points to {no_color_pcd}")

        if args.color:
            print("[Info] Loading color dataset...")
            color_dataset = load_color_dataset(project_dir)

            pcd_colors = sample_color_from_color_maps(
                np.asarray(pcd.points),
                color_dataset["color_map_paths"],
                color_dataset["intrinsics"],
                color_dataset["extrinsics"]
            )
            pcd.colors = o3d.utility.Vector3dVector(pcd_colors)

            colored_pcd.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(colored_pcd, pcd)
            print(f"[Info] Saved colored point cloud with {len(pcd.colors)} colors to {colored_pcd}")
        
    if args.visualize:
        print("[Info] Visualizing point cloud...")
        axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.6, origin=[0, 0, 0])
        o3d.visualization.draw_geometries([pcd, axis])


if __name__ == "__main__":
    args = parse_args()
    main(args)