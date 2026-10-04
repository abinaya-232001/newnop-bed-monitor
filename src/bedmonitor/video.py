"""Video reading and frame sampling with reliable timestamps."""
from dataclasses import dataclass
from typing import Iterator, Tuple

import cv2
import numpy as np


@dataclass
class VideoInfo:
    path: str
    fps: float
    n_frames: int
    width: int
    height: int
    duration_sec: float


def open_video(path: str) -> Tuple[cv2.VideoCapture, VideoInfo]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    info = VideoInfo(
        path=path,
        fps=fps,
        n_frames=n_frames,
        width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        duration_sec=n_frames / fps,
    )
    return cap, info


def sample_frames(cap: cv2.VideoCapture, fps: float, sample_fps: float
                  ) -> Iterator[Tuple[float, np.ndarray]]:
    """Yield (timestamp_sec, frame) roughly `sample_fps` times per second.

    grab() advances one frame without the expensive decode; retrieve() decodes
    only the frames we keep. On 1080p video this saves a lot of time.
    """
    step = max(1, round(fps / sample_fps))
    idx = 0
    while True:
        if not cap.grab():
            break
        if idx % step == 0:
            ok, frame = cap.retrieve()
            if ok:
                yield idx / fps, frame
        idx += 1