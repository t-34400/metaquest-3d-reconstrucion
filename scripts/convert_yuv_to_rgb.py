from typing import Callable, Union
import numpy as np
from tqdm import tqdm
import cv2
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import argparse
import os

import constants
from utils.image_utils import convert_yuv420_888_to_bgr, is_valid_image, ImageFormatInfo, ImagePlaneInfo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project_dir", "-p",
        type=str,
        required=True,
        help="Path to the project directory containing QRC data."
    )
    parser.add_argument(
        "--filter", action="store_true", default=False,
        help="Enable image quality filtering."
    )
    parser.add_argument(
        "--blur_threshold", type=float, default=50.0,
        help="Blur threshold (Laplacian variance). Lower means more blur. Default: 50.0"
    )
    parser.add_argument(
        "--exposure_threshold_low", type=float, default=0.1,
        help="Cumulative histogram threshold to detect underexposure. Default: 0.1"
    )
    parser.add_argument(
        "--exposure_threshold_high", type=float, default=0.1,
        help="Cumulative histogram threshold to detect overexposure. Default: 0.1"
    )
    args = parser.parse_args()

    if not os.path.isdir(args.project_dir):
        parser.error(f"Input directory does not exist: {args.project_dir}")

    return args


def load_image_format_info(fmt_path):
    with open(fmt_path) as f:
        format_info = json.load(f)

    width = format_info["width"]
    height = format_info["height"]

    planes = [
        ImagePlaneInfo(
            bufferSize=plane["bufferSize"],
            rowStride=plane["rowStride"],
            pixelStride=plane["pixelStride"]
        ) for plane in format_info["planes"]
    ]

    return ImageFormatInfo(width=width, height=height, planes=planes)


def process_file(
    yuv_file: Path,
    output_dir: Path,
    format_info: ImageFormatInfo
) -> bool:
    try:
        raw_data = np.fromfile(yuv_file, dtype=np.uint8)
        bgr_img = convert_yuv420_888_to_bgr(raw_data, format_info)

        if is_valid_image:
            if not is_valid_image(bgr_img):
                return False

        file_name = os.path.splitext(os.path.basename(yuv_file))[0]
        out_path = output_dir / f"{file_name}.png"

        cv2.imwrite(str(out_path), bgr_img)
        return True

    except Exception as e:
        return False


def convert_yuv_directory_to_png(
    input_dir: Path,
    output_dir: Path,
    format_info: ImageFormatInfo,
    is_valid_image: Union[Callable[[np.ndarray], bool], None] = None,
):
    yuv_files = sorted(input_dir.glob("*.yuv"))

    excluded_count = 0
    processed_count = 0

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, yuv_file, output_dir, format_info)
            for yuv_file in yuv_files
        ]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Converting YUV to PNG"):
            result = future.result()
            if result:
                processed_count += 1
            else:
                excluded_count += 1

    print(f"[Info] {processed_count} images written to {output_dir}")
    if is_valid_image:
        print(f"[Info] {excluded_count} images were excluded by filtering.")


def main(args):
    camera_path_params = [
        {
            "camera": "left",
            "input_image_dir": Path(os.path.join(args.project_dir, constants.LEFT_CAMERA_YUV_IMAGE_DIR)),
            "image_format_json": Path(os.path.join(args.project_dir, constants.LEFT_CAMERA_IMAGE_FORMAT_JSON)),
            "output_image_dir": Path(os.path.join(args.project_dir, constants.LEFT_CAMERA_RGB_IMAGE_DIR)),
        },
        {
            "camera": "right",
            "input_image_dir": Path(os.path.join(args.project_dir, constants.RIGHT_CAMERA_YUV_IMAGE_DIR)),
            "image_format_json": Path(os.path.join(args.project_dir, constants.RIGHT_CAMERA_IMAGE_FORMAT_JSON)),
            "output_image_dir": Path(os.path.join(args.project_dir, constants.RIGHT_CAMERA_RGB_IMAGE_DIR)),
        },
    ]

    for params in camera_path_params:
        print(f"[Info] Converting {params['camera']} camera images...")

        params["output_image_dir"].mkdir(parents=True, exist_ok=True)

        format_info = load_image_format_info(params["image_format_json"])

        filter = None
        if args.filter:
            def filter(bgr_img): return is_valid_image(
                bgr_img,
                blur_threshold=args.blur_threshold,
                exposure_threshold_low=args.exposure_threshold_low,
                exposure_threshold_high=args.exposure_threshold_high
            )

        convert_yuv_directory_to_png(
            input_dir=params["input_image_dir"],
            output_dir=params["output_image_dir"],
            format_info=format_info,
            is_valid_image=filter,
        )


if __name__ == "__main__":
    args = parse_args()

    print(f"[Info] Project Directory: {args.project_dir}")

    main(args)
