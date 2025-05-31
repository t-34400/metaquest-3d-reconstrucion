import cv2
import numpy as np
from dataclasses import dataclass

@dataclass
class ImagePlaneInfo:
    bufferSize: int
    rowStride: int
    pixelStride: int

@dataclass
class ImageFormatInfo:
    width: int
    height: int
    planes: list[ImagePlaneInfo]

def reconstruct_plane(data, offset, width, height, row_stride, pixel_stride):
    plane = np.frombuffer(data, dtype=np.uint8)
    output = np.empty((height, width), dtype=np.uint8)

    for row in range(height):
        start = offset + row * row_stride
        end = start + width * pixel_stride
        row_data = plane[start:end:pixel_stride]
        output[row, :] = row_data[:width]

    return output

def convert_yuv420_888_to_i420(
        raw_data: bytes, 
        format_info: ImageFormatInfo, 
        uv_order="NV12"
    ):
    width = format_info.width
    height = format_info.height
    planes = format_info.planes

    if len(planes) != 3:
        raise ValueError("Expected 3 planes for YUV420_888 format")

    y_plane = reconstruct_plane(raw_data, 0, width, height, planes[0].rowStride, planes[0].pixelStride)
    u_offset = planes[0].bufferSize
    chroma_width = width // 2
    chroma_height = height // 2
    pixel_stride_uv = planes[1].pixelStride
    row_stride_uv = planes[1].rowStride

    if pixel_stride_uv == 1:
        u_plane = reconstruct_plane(raw_data, u_offset, chroma_width, chroma_height, row_stride_uv, 1)
        v_offset = u_offset + planes[1].bufferSize
        v_plane = reconstruct_plane(raw_data, v_offset, chroma_width, chroma_height, planes[2].rowStride, 1)
    else:
        uv_interleaved = np.frombuffer(raw_data, dtype=np.uint8)
        u_plane = np.empty((chroma_height, chroma_width), dtype=np.uint8)
        v_plane = np.empty((chroma_height, chroma_width), dtype=np.uint8)

        for row in range(chroma_height):
            row_start = u_offset + row * row_stride_uv
            row_data = uv_interleaved[row_start : row_start + chroma_width * pixel_stride_uv]
            if uv_order == "NV21":
                v_plane[row, :] = row_data[::2][:chroma_width]
                u_plane[row, :] = row_data[1::2][:chroma_width]
            else:
                u_plane[row, :] = row_data[::2][:chroma_width]
                v_plane[row, :] = row_data[1::2][:chroma_width]

    return np.concatenate([y_plane.ravel(), u_plane.ravel(), v_plane.ravel()])

def convert_yuv420_888_to_bgr(
        raw_data: bytes, 
        format_info: ImageFormatInfo, 
        uv_order="NV12"
    ):
    height = format_info.height
    width = format_info.width

    i420 = convert_yuv420_888_to_i420(raw_data, format_info, uv_order).reshape((height * 3) // 2, width)
    bgr_img = cv2.cvtColor(i420, cv2.COLOR_YUV2BGR_I420)

    return bgr_img


def measure_blur_laplacian(img_gray):
    return cv2.Laplacian(img_gray, cv2.CV_64F).var()


def is_over_or_under_exposed(img_gray, low_thresh=0.02, high_thresh=0.98):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()
    cum = np.cumsum(hist)

    return cum[5] > low_thresh or (1 - cum[250]) > high_thresh


def is_valid_image(img_bgr, blur_threshold=50.0, exposure_threshold_low=0.02, exposure_threshold_high=0.98):
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur_score = measure_blur_laplacian(img_gray)

    if blur_score < blur_threshold:
        return False
    if is_over_or_under_exposed(img_gray, low_thresh=exposure_threshold_low, high_thresh=exposure_threshold_high):
        return False
    
    return True