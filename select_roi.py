"""Usage: python select_roi.py VIDEO OUT_JSON
Shows the first frame scaled to fit the screen; drag a rectangle around the BED
(the mattress area where a person lies or sits), press ENTER. Saves the ROI in
full-frame pixel coordinates."""
import json
import sys

import cv2

video, out = sys.argv[1], sys.argv[2]
cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()
if not ok:
    sys.exit(f"Cannot read a frame from {video}")

h0, w0 = frame.shape[:2]
# fit inside 1200 x 700 so it works on a laptop screen, in portrait or landscape
scale = min(1200 / w0, 700 / h0, 1.0)
small = cv2.resize(frame, None, fx=scale, fy=scale)

x, y, w, h = cv2.selectROI("Drag around the BED, then press ENTER", small, showCrosshair=False)
cv2.destroyAllWindows()
if w == 0 or h == 0:
    sys.exit("No ROI selected")

roi = [round(x / scale), round(y / scale), round((x + w) / scale), round((y + h) / scale)]
with open(out, "w", encoding="utf-8") as f:
    json.dump({"bed_roi": roi, "video": video, "frame_size": [w0, h0]}, f)
print("bed_roi =", roi, "frame_size =", [w0, h0])