"""Run person detection + tracking + pose once and cache Observations to JSON."""
import json
import logging
import time
from dataclasses import asdict
from typing import Optional

from ultralytics import YOLO

from .models import Observation
from .video import open_video, sample_frames

log = logging.getLogger(__name__)


def _primary_observation(res, t: float) -> Observation:
    """Turn one YOLO result into an Observation.

    If several people are present, the one with the LARGEST box is treated as
    the primary person. This is a simple assumption and a known limitation
    (e.g. a caregiver closer to the camera can win).
    """
    boxes = res.boxes
    n = 0 if boxes is None else len(boxes)
    if n == 0:
        return Observation(t=t, n_people=0)

    xyxy = boxes.xyxy.cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(areas.argmax())

    track_id = None
    if boxes.id is not None:
        track_id = int(boxes.id[i])

    keypoints = None
    if res.keypoints is not None and res.keypoints.xy is not None:
        xy = res.keypoints.xy[i].cpu().numpy()
        kc = res.keypoints.conf
        conf = kc[i].cpu().numpy() if kc is not None else [1.0] * len(xy)
        keypoints = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)]
                     for (x, y), c in zip(xy, conf)]

    return Observation(
        t=round(t, 3),
        n_people=n,
        track_id=track_id,
        box=[round(float(v), 1) for v in xyxy[i]],
        box_conf=round(float(boxes.conf[i]), 3),
        keypoints=keypoints,
    )


def run_perception(video_path: str, out_path: str, sample_fps: float = 3.0,
                   model_name: str = "yolov8n-pose.pt", conf: float = 0.25,
                   device: str = "cpu", max_seconds: Optional[float] = None) -> dict:
    cap, info = open_video(video_path)
    model = YOLO(model_name)
    observations = []
    t_start = time.time()

    for t, frame in sample_frames(cap, info.fps, sample_fps):
        if max_seconds is not None and t > max_seconds:
            break
        res = model.track(frame, persist=True, tracker="bytetrack.yaml",
                          conf=conf, device=device, verbose=False)[0]
        observations.append(_primary_observation(res, t))
        if len(observations) % 50 == 0:
            log.info("processed %d samples (t=%.1fs)", len(observations), t)

    cap.release()
    elapsed = time.time() - t_start
    n = len(observations)
    log.info("done: %d samples in %.1fs (%.3f s/sample) on device=%s",
             n, elapsed, elapsed / max(n, 1), device)

    cache = {
        "video": info.path,
        "fps": info.fps,
        "width": info.width,
        "height": info.height,
        "duration_sec": info.duration_sec,
        "sample_fps": sample_fps,
        "model": model_name,
        "conf": conf,
        "samples": [asdict(o) for o in observations],
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(cache, f)
    return cache