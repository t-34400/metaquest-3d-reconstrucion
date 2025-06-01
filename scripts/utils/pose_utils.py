import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp


def compose_transform(
    parent_pos: np.ndarray,
    parent_rot: R,
    local_pos: np.ndarray,
    local_rot: R
) -> tuple[np.ndarray, R]:
    rotated_local_pos = parent_rot.apply(local_pos)

    world_pos = parent_pos + rotated_local_pos
    world_rot = parent_rot * local_rot

    return world_pos, world_rot


class PoseInterpolator:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path, on_bad_lines='skip').dropna()
        self.df = self.df.sort_values('unix_time')
        self.df = self.df.reset_index(drop=True)

    def find_nearest_frames(self, timestamp: int, window_ms: int = 30):
        before = self.df[self.df['unix_time'] <= timestamp]
        after = self.df[self.df['unix_time'] >= timestamp]

        prev = before.iloc[-1] if not before.empty and timestamp - before.iloc[-1]['unix_time'] <= window_ms else None
        next = after.iloc[0] if not after.empty and after.iloc[0]['unix_time'] - timestamp <= window_ms else None

        return prev, next

    def interpolate_pose(self, timestamp: int):
        prev, next = self.find_nearest_frames(timestamp)
        if prev is None or next is None:
            return None

        t0 = prev['unix_time']
        t1 = next['unix_time']
        alpha = (timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0

        pos0 = np.array([prev['pos_x'], prev['pos_y'], prev['pos_z']])
        pos1 = np.array([next['pos_x'], next['pos_y'], next['pos_z']])
        pos_interp = (1 - alpha) * pos0 + alpha * pos1

        rots = R.from_quat( [
            [prev['rot_x'], prev['rot_y'], prev['rot_z'], prev['rot_w']],
            [next['rot_x'], next['rot_y'], next['rot_z'], next['rot_w']],
        ] )
        rot_interp = Slerp([0, 1], rots)(alpha)

        return pos_interp, rot_interp