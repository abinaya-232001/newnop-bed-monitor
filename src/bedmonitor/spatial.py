"""Where is the person relative to the bed? Uses a manually configured ROI."""
import math
from typing import List, Optional, Tuple

from .models import Observation

L_HIP, R_HIP = 11, 12
Point = Tuple[float, float]


def reference_point(obs: Observation, min_conf: float = 0.5) -> Optional[Point]:
    """The point used to decide 'where the person is'.

    Hip midpoint if the hips are confident (it sits at the body's centre of mass
    whether lying, sitting or standing); otherwise the centre of the person box.
    """
    if obs.box is None:
        return None
    if obs.keypoints:
        pts = [(x, y) for x, y, c in (obs.keypoints[L_HIP], obs.keypoints[R_HIP]) if c >= min_conf]
        if pts:
            return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
    x1, y1, x2, y2 = obs.box
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def distance_to_roi(p: Point, roi: List[float]) -> float:
    """0 if the point is inside the rectangle, else pixel distance to its nearest edge."""
    dx = max(roi[0] - p[0], 0, p[0] - roi[2])
    dy = max(roi[1] - p[1], 0, p[1] - roi[3])
    return math.hypot(dx, dy)


def locate(obs: Observation, bed_roi: List[float], near_margin_frac: float = 0.3,
           min_conf: float = 0.5) -> Tuple[str, Optional[float]]:
    """Return (ON_BED | NEAR_BED | AWAY | UNKNOWN, pixel distance to the bed ROI).

    near_margin_frac is a fraction of the bed ROI width, so it scales with the
    camera setup instead of being a fixed pixel count.
    """
    p = reference_point(obs, min_conf)
    if p is None:
        return "UNKNOWN", None
    d = distance_to_roi(p, bed_roi)
    if d == 0:
        return "ON_BED", 0.0
    if d <= near_margin_frac * (bed_roi[2] - bed_roi[0]):
        return "NEAR_BED", d
    return "AWAY", d