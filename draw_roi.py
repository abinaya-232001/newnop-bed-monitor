"""Usage: python draw_roi.py VIDEO ROI_JSON OUT_JPG"""
import json
import sys

import cv2

cap = cv2.VideoCapture(sys.argv[1])
ok, frame = cap.read()
cap.release()
x1, y1, x2, y2 = json.load(open(sys.argv[2], encoding="utf-8"))["bed_roi"]
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 4)
cv2.imwrite(sys.argv[3], frame)
print("saved", sys.argv[3])