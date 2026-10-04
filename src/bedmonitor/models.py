"""Plain data containers shared by all stages."""
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Observation:
    """What the vision models saw at one sampled moment. No decisions here."""
    t: float                                  # seconds from video start
    n_people: int                             # how many persons were detected
    track_id: Optional[int] = None            # ByteTrack id of the primary person
    box: Optional[List[float]] = None         # [x1, y1, x2, y2] in pixels
    box_conf: Optional[float] = None          # detector confidence, 0..1
    keypoints: Optional[List[List[float]]] = None  # 17 x [x, y, conf], COCO order