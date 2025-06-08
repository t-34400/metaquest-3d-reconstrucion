from dataclasses import dataclass, asdict, field
from typing import Optional
import json


@dataclass
class Frame:
    file_path: str
    transform_matrix: list[list[float]]
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    w: int
    h: int
    depth_file_path: Optional[str] = None


@dataclass
class Transforms:
    camera_model: str  # e.g., "OPENCV" or "OPENCV_FISHEYE"
    k1: Optional[float] = None
    k2: Optional[float] = None
    k3: Optional[float] = None
    k4: Optional[float] = None
    p1: Optional[float] = None
    p2: Optional[float] = None
    frames: list[Frame] = field(default_factory=list)

    def to_json(self, path: str):
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
