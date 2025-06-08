import numpy as np
from pathlib import Path
import constants
from tsdf.integrate_depth_maps import preprocess_depth_dataset
from tsdf.sample_color import preprocess_color_dataset


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


def load_or_create_color_dataset(project_path: Path) -> dict:
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