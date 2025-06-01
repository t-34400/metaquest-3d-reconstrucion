from pathlib import Path
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os

from utils.depth_utils import convert_depth_to_linear
import constants

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=str,
        required=True,
        help="Path to the project directory containing QRC data."
    )
    parser.add_argument(
        "--near",
        type=float,
        default=0.1,
        help="Near clipping plane distance for depth conversion. Default: 0.1"
    )
    parser.add_argument(
        "--far",
        type=float,
        default=10.0,
        help="Far clipping plane distance for depth conversion. Default: 10.0"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.project_dir):
        parser.error(f"Input directory does not exist: {args.project_dir}")

    return args


def convert_depth_directory_to_linear(
    depth_dir: str,
    output_dir: str,
    near: float,
    far: float,
    desc_near: float,
    desc_far: float,
    width: int,
    height: int,
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in tqdm(os.listdir(depth_dir), desc="Converting depth images"):
        if filename.endswith(".raw"):
            filepath = os.path.join(depth_dir, filename)

            with open(filepath, 'rb') as f:
                data = np.frombuffer(f.read(), dtype='<f4')
                
            depth_buffer = data.reshape((height, width))

            depth_image = convert_depth_to_linear(depth_buffer, desc_near, desc_far)
            output_path = os.path.join(output_dir, filename.replace(".raw", ".png"))

            cv2.imwrite(output_path, (depth_image - near) / (far - near) * 255.0)


def main(args):
    depth_params = [
        {
            "camera": "left",
            "input_depth_dir": Path(os.path.join(args.project_dir, constants.LEFT_DEPTH_RAW_DATA_DIR)),
            "depth_descriptor_csv": Path(os.path.join(args.project_dir, constants.LEFT_DEPTH_DESCRIPTOR_CSV)),
            "output_depth_dir": Path(os.path.join(args.project_dir, constants.LEFT_DEPTH_GRAY_IMAGE_DIR)),
        },
        {
            "camera": "right",
            "input_depth_dir": Path(os.path.join(args.project_dir, constants.RIGHT_DEPTH_RAW_DATA_DIR)),
            "depth_descriptor_csv": Path(os.path.join(args.project_dir, constants.RIGHT_DEPTH_DESCRIPTOR_CSV)),
            "output_depth_dir": Path(os.path.join(args.project_dir, constants.RIGHT_DEPTH_GRAY_IMAGE_DIR)),
        },
    ]

    for params in depth_params:
        print(f"[Info] Converting {params["camera"]} depth images...")

        depth_dir = params["input_depth_dir"]
        depth_descriptor_csv = params["depth_descriptor_csv"]
        if not depth_dir.exists() or not depth_descriptor_csv.exists():
            print(f"[Error] Required files for {params['camera']} camera are missing.")
            continue

        try:
            df = pd.read_csv(depth_descriptor_csv)
            if len(df) == 0:
                continue

            row = df.iloc[0]
            desc_near = float(row['near_z'])
            desc_far = float(row['far_z'])
            width = int(row['width'])
            height = int(row['height'])
        except Exception as e:
            print(f"[Error] Failed to read depth descriptor CSV for {params['camera']} camera: {e}")
            continue
        
        print(f"[Info] Depth Descriptor - Near: {desc_near}, Far: {desc_far}, Width: {width}, Height: {height}")

        output_dir = params["output_depth_dir"]
        output_dir.mkdir(parents=True, exist_ok=True)

        convert_depth_directory_to_linear(
            depth_dir=str(depth_dir),
            output_dir=str(output_dir),
            near=args.near,
            far=args.far,
            desc_near=desc_near,
            desc_far=desc_far,
            width=width,
            height=height,
        )
        print(f"[Info] Converted depth images for {params['camera']} camera to linear format.")


if __name__ == "__main__":
    args = parse_args()

    print(f"[Info] Project Directory: {args.project_dir}")

    main(args)
