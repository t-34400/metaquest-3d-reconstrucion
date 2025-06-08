import open3d as o3d
import numpy as np
import os
import argparse
from pathlib import Path
import constants
from dataset_loader import load_or_create_depth_dataset, load_or_create_color_dataset
from tsdf.integrate_depth_maps import integrate_depth_maps
from tsdf.sample_color import sample_color_from_color_maps


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
        "--down_voxel_size",
        type=float,
        default=0.02,
        help="Voxel size for downsampling. Controls how much to reduce point density (default: 0.02)"
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
            pcd = pcd.remove_non_finite_points()
            print("[Info] Downsampling point cloud using voxel grid...")
            pcd = pcd.voxel_down_sample(voxel_size=args.down_voxel_size)
            print("[Info] Removing statistical outliers...")
            _, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2)
            print(f"[Info] Valid points: {len(ind)} / {len(pcd.points)} ({len(ind) / len(pcd.points) * 100:.2f} %)")
            pcd = pcd.select_by_index(ind)

            no_color_pcd.parent.mkdir(parents=True, exist_ok=True)
            o3d.io.write_point_cloud(no_color_pcd, pcd)
            print(f"[Info] Saved {len(pcd.points)} total points to {no_color_pcd}")

        if args.color:
            print("[Info] Loading color dataset...")
            color_dataset = load_or_create_color_dataset(project_dir)

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