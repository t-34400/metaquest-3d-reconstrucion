from scipy.spatial.transform import Rotation as R
import numpy as np


def unity_pose_to_open3d_extrinsic(
    position: np.ndarray, 
    rotation: R = None,
    quaternion: np.ndarray = None,
):
    if rotation is not None:
        rotation_matrix = rotation.as_matrix()
    elif quaternion is not None and len(quaternion) == 4:
        rotation_matrix = R.from_quat(quaternion).as_matrix()
    else:
        rotation_matrix = np.eye(3)

    T_world_to_cam = np.eye(4)
    T_world_to_cam[:3, :3] = rotation_matrix
    T_world_to_cam[:3, 3] = position

    convert = np.diag([1, 1, -1, 1])
    T_world_to_cam = convert @ T_world_to_cam

    extrinsic = np.linalg.inv(T_world_to_cam)
    return extrinsic