"""Glue: cached Observations + bed ROI -> per-sample FrameStates."""
from typing import List

from .classifier import FrameState, classify_frame, motion_speeds
from .features import classify_posture, extract_features
from .models import Observation
from .spatial import locate


def classify_all(obs_list: List[Observation], bed_roi: List[float]) -> List[FrameState]:
    speeds = motion_speeds(obs_list)
    out = []
    for o, sp in zip(obs_list, speeds):
        posture, pconf = classify_posture(extract_features(o))
        loc, _ = locate(o, bed_roi)
        out.append(classify_frame(o, posture, pconf, loc, sp))
    return out