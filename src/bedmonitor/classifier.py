"""Combine posture + location + motion into ONE required state per sample.

This is still a per-sample guess. Temporal smoothing and transitions come later.
"""
import math
from dataclasses import dataclass
from typing import List, Optional

from .models import Observation

CFG = {
    "walking_speed": 0.25,   # body-sizes per second; placeholder, tune on real clips
    "motion_window_sec": 2.0,
}


@dataclass
class FrameState:
    t: float
    state: str          # one of the required states
    conf: float         # heuristic ranking value, NOT a calibrated probability
    reason: str         # human-readable evidence, kept for the agent trace
    location: Optional[str] = None


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def motion_speeds(observations: List[Observation],
                  window_sec: float = CFG["motion_window_sec"]) -> List[Optional[float]]:
    """Speed per sample in body-sizes per second, measured over a sliding window.

    Uses the SAME track id only. Dividing by body size (longer side of the box)
    makes the number independent of camera distance. None = not enough history.
    This is what separates STANDING from WALKING: movement over time, not a pose.
    """
    speeds: List[Optional[float]] = []
    for i, o in enumerate(observations):
        speed = None
        if o.box is not None and o.track_id is not None:
            oldest = None
            for j in range(i - 1, -1, -1):
                p = observations[j]
                if o.t - p.t > window_sec:
                    break
                if p.box is not None and p.track_id == o.track_id:
                    oldest = p
            if oldest is not None and (o.t - oldest.t) >= 0.5 * window_sec:
                (cx1, cy1), (cx2, cy2) = _center(oldest.box), _center(o.box)
                size = max(o.box[2] - o.box[0], o.box[3] - o.box[1])
                speed = math.hypot(cx2 - cx1, cy2 - cy1) / max(size, 1e-6) / (o.t - oldest.t)
        speeds.append(speed)
    return speeds


def classify_frame(obs: Observation, posture: str, posture_conf: float,
                   location: Optional[str], speed: Optional[float],
                   cfg: dict = CFG) -> FrameState:
    t = obs.t
    if obs.n_people == 0:
        return FrameState(t, "UNKNOWN", 0.0, "no_person_detected", None)

    conf = posture_conf * (0.7 if obs.n_people > 1 else 1.0)
    note = "+multi_person" if obs.n_people > 1 else ""

    if posture == "LYING":
        if location == "ON_BED":
            return FrameState(t, "LYING_IN_BED", conf, "lying_posture_inside_bed_roi" + note, location)
        # lying but not on the bed: floor? ROI wrong? Do not force a label.
        return FrameState(t, "UNKNOWN", 0.3, "lying_but_not_on_bed" + note, location)

    if posture == "SITTING":
        if location == "ON_BED":
            return FrameState(t, "SITTING_ON_BED", conf, "sitting_hips_inside_bed_roi" + note, location)
        if location == "NEAR_BED":
            return FrameState(t, "SITTING_OUTSIDE_BED", conf * 0.7,
                              "sitting_near_bed_edge_ambiguous" + note, location)
        return FrameState(t, "SITTING_OUTSIDE_BED", conf, "sitting_away_from_bed" + note, location)

    if posture == "STANDING":
        if speed is not None and speed >= cfg["walking_speed"]:
            return FrameState(t, "WALKING", conf, f"upright_and_moving speed={speed:.2f}" + note, location)
        if speed is None:
            return FrameState(t, "STANDING", conf * 0.7, "upright_no_motion_history" + note, location)
        return FrameState(t, "STANDING", conf, f"upright_and_still speed={speed:.2f}" + note, location)

    # posture UNKNOWN
    if location == "AWAY":
        return FrameState(t, "OUT_OF_BED", 0.3, "away_from_bed_posture_unclear" + note, location)
    return FrameState(t, "UNKNOWN", 0.0, "posture_unclear" + note, location)