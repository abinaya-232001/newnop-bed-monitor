"""Smoke test: video reading, timestamps, YOLO pose, ByteTrack.

Usage: python smoke_test.py path/to/video.mp4 [sample_fps] [max_samples]
"""
import sys
import time

import cv2
from ultralytics import YOLO


def main(path: str, sample_fps: float = 3.0, max_samples: int = 60) -> None:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"video: {w}x{h}, {fps:.2f} fps, {n_frames} frames, {n_frames / fps:.1f} s")

    step = max(1, round(fps / sample_fps))  # process every Nth frame
    model = YOLO("yolov8n-pose.pt")         # downloads weights on first run

    idx, done = 0, 0
    t_start = time.time()
    while done < max_samples:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            ts = idx / fps  # timestamp in seconds, from the frame index
            res = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            boxes = res.boxes
            n_people = 0 if boxes is None else len(boxes)
            ids = boxes.id.int().tolist() if (boxes is not None and boxes.id is not None) else []
            shape = ""
            if n_people:
                x1, y1, x2, y2 = boxes.xyxy[0].tolist()
                shape = f"box w/h={(x2 - x1) / max(y2 - y1, 1):.2f}"
            print(f"t={ts:7.2f}s people={n_people} ids={ids} {shape}")
            if done in (0, max_samples // 2):
                cv2.imwrite(f"debug_{done}.jpg", res.plot())
            done += 1
        idx += 1

    elapsed = time.time() - t_start
    if done:
        print(f"\n{done} samples in {elapsed:.1f}s ({elapsed / done:.2f} s/sample) on this machine")


if __name__ == "__main__":
    main(
        sys.argv[1],
        float(sys.argv[2]) if len(sys.argv) > 2 else 3.0,
        int(sys.argv[3]) if len(sys.argv) > 3 else 60,
    )