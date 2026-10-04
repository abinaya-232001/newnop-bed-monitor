import os
import sys

import cv2
import numpy as np

video, out_png = sys.argv[1], sys.argv[2]
step = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
cols = 6
thumb_w = 300

cap = cv2.VideoCapture(video)
if not cap.isOpened():
    sys.exit("Cannot open " + video)
fps = cap.get(cv2.CAP_PROP_FPS)
n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
dur = n / fps if fps else 0.0

tiles = []
t = 0.0
while t < dur:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
    ok, frame = cap.read()
    if ok:
        h, w = frame.shape[:2]
        thumb = cv2.resize(frame, (thumb_w, int(h * thumb_w / w)))
        label = "%gs" % t
        cv2.putText(thumb, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 4)
        cv2.putText(thumb, label, (6, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(thumb)
    t += step
cap.release()

if not tiles:
    sys.exit("No frames could be read")
n_real = len(tiles)
blank = np.zeros((tiles[0].shape[0], thumb_w, 3), dtype=np.uint8)
while len(tiles) % cols:
    tiles.append(blank.copy())
rows = [np.hstack(tiles[i:i + cols]) for i in range(0, len(tiles), cols)]
sheet = np.vstack(rows)

os.makedirs(os.path.dirname(out_png), exist_ok=True)
cv2.imwrite(out_png, sheet)
print("wrote", out_png, "frames:", n_real, "size:", sheet.shape[1], "x", sheet.shape[0])
