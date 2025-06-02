# Meta Quest 3D Reconstruction

<p align="center">
  <img src="docs/overview.png" alt="QuestRealityCapture" width="480"/>
</p>

**A project for reconstructing 3D scenes using image and depth data captured via the [Quest Reality Capture (QRC)](https://github.com/t-34400/QuestRealityCapture/) application.**

---

## 🧭 Overview

This project converts image and depth data captured on Meta Quest (via a custom Reality Capture app) into formats suitable for 3D reconstruction using Open3D or COLMAP.

---

## 🚀 Setup

### Requirements

* Open3D
* NumPy
* SciPy
* Pillow
* OpenCV
* **PyTorch** (required **only** when using `--color` in `generate_point_cloud.py` to generate colored point clouds)

---

## 🔧 Processing Pipeline

### 1. Convert passthrough YUV images to RGB

```bash
python scripts/convert_yuv_to_rgb.py \
  --project_dir path/to/your/project
```

**Required:**

* `--project_dir` (`-p`): Path to the directory containing QRC-generated directory.

**Optional:**

* `--filter`: Enable image quality filtering (default: disabled).
* `--blur_threshold`: Threshold for blur detection using Laplacian variance (default: 50.0).
* `--exposure_threshold_low`: Underexposure detection threshold (default: 0.1).
* `--exposure_threshold_high`: Overexposure detection threshold (default: 0.1).

---

### 2. Convert raw depth to linear depth map

```bash
python scripts/convert_depth_to_linear_map.py \
  --project_dir path/to/your/project
```

**Required:**

* `--project_dir` (`-p`): Path to the directory containing QRC-generated directory.

**Optional:**

* `--near`: Near clipping plane distance in meters (default: 0.1).
* `--far`: Far clipping plane distance in meters (default: 10.0).

---

### 3. Generate 3D point cloud via TSDF (Open3D)

> ⚠️ **Using the `--color` option, you must first convert YUV images to RGB using `convert_yuv_to_rgb.py` and install PyTorch.**

```bash
python scripts/generate_point_cloud.py \
  --project_dir path/to/your/project \
  --color \
  --visualize
```

**Required:**

* `--project_dir` (`-p`): Path to the directory containing QRC-generated directory.

**Optional:**

* `--voxel_length`: TSDF voxel size in meters (default: 0.01).
* `--sdf_trunc`: Truncation distance in meters (default: 0.04).
* `--color`: Generate colored point cloud using RGB images.
* `--visualize`: Display the reconstructed TSDF volume in an Open3D viewer.

---

## 📝 License

This project is licensed under the MIT License.
See the [LICENSE](LICENSE) file for full license text.

---

## 📌 TODO

* [ ] Convert to COLMAP-compatible and RGB-D dataset formats
