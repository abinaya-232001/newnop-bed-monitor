"""Posture features from pose keypoints, and a rule-based posture classifier.

Everything here is a pure function of one Observation, so it can be tested
without video. Keypoints below `min_conf` are treated as missing.
"""
import math
from typing import Dict, Optional, Tuple

from .models import Observation

Point = Tuple[float, float]

# COCO keypoint indices used by YOLO pose
L_SHO, R_SHO, L_HIP, R_HIP, L_KNEE, R_KNEE = 5, 6, 11, 12, 13, 14

# Engineering thresholds, NOT tuned values. Move to config.yaml later.
DEFAULTS = {
    "min_kp_conf": 0.5,        # keypoint must be at least this confident to be used
    "lying_torso_deg": 60,     # torso angle >= this -> lying
    "sitting_thigh_deg": 50,   # thigh angle >= this (torso upright) -> sitting
    "standing_thigh_deg": 30,  # thigh angle <= this (torso upright) -> standing
    "lying_box_aspect": 1.3,   # fallback: box width/height >= this -> probably lying
}


def _pt(kps, i: int, min_conf: float) -> Optional[Point]:
    x, y, c = kps[i]
    return (x, y) if c >= min_conf else None


def _mid(a: Optional[Point], b: Optional[Point]) -> Optional[Point]:
    """Midpoint of the points that exist; None if neither exists."""
    pts = [p for p in (a, b) if p is not None]
    if not pts:
        return None
    return (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))


def _angle_from_vertical(p: Point, q: Point) -> float:
    """Angle in degrees between the line p->q and the vertical (0 = vertical, 90 = horizontal)."""
    dx, dy = abs(q[0] - p[0]), abs(q[1] - p[1])
    return math.degrees(math.atan2(dx, dy))


def extract_features(obs: Observation, min_conf: float = DEFAULTS["min_kp_conf"]
                     ) -> Optional[Dict]:
    """Return posture features, or None if there is no usable person."""
    if obs.n_people == 0 or obs.keypoints is None or obs.box is None:
        return None
    k = obs.keypoints
    shoulder = _mid(_pt(k, L_SHO, min_conf), _pt(k, R_SHO, min_conf))
    hip = _mid(_pt(k, L_HIP, min_conf), _pt(k, R_HIP, min_conf))
    knee = _mid(_pt(k, L_KNEE, min_conf), _pt(k, R_KNEE, min_conf))
    x1, y1, x2, y2 = obs.box
    return {
        "box_aspect": (x2 - x1) / max(y2 - y1, 1e-6),
        "torso_angle": _angle_from_vertical(shoulder, hip) if shoulder and hip else None,
        "thigh_angle": _angle_from_vertical(hip, knee) if hip and knee else None,
    }


def classify_posture(f: Optional[Dict], cfg: Dict = DEFAULTS) -> Tuple[str, float]:
    """Map features to (LYING | SITTING | STANDING | UNKNOWN, heuristic_confidence).

    The confidence numbers are hand-set placeholders, NOT calibrated
    probabilities. They only rank evidence: full-skeleton rules > box-shape fallback.
    """
    if f is None:
        return "UNKNOWN", 0.0
    torso, thigh = f["torso_angle"], f["thigh_angle"]

    if torso is not None:
        if torso >= cfg["lying_torso_deg"]:
            return "LYING", 0.8
        if thigh is not None:                       # torso upright, legs visible
            if thigh >= cfg["sitting_thigh_deg"]:
                return "SITTING", 0.8
            if thigh <= cfg["standing_thigh_deg"]:
                return "STANDING", 0.8
        return "UNKNOWN", 0.3   # upright but legs unseen/ambiguous: sitting vs standing can't be told

    if f["box_aspect"] >= cfg["lying_box_aspect"]:  # no hips: wide box is weak evidence of lying
        return "LYING", 0.4
    return "UNKNOWN", 0.0