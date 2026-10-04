import json
import sys
from pathlib import Path

sys.path.insert(0, r"C:\dev\newnop-bed-monitor\src")
from bedmonitor.models import Observation
from bedmonitor.pipeline import classify_all

cache_path, roi_path = sys.argv[1], sys.argv[2]
data = json.load(open(cache_path, encoding="utf-8"))
roi = json.load(open(roi_path, encoding="utf-8"))["bed_roi"]

obs_list = [Observation(**s) for s in data["samples"]]
states = classify_all(obs_list, roi)

print("roi:", roi, " frame:", data["width"], "x", data["height"])
for i, (o, st) in enumerate(zip(obs_list, states)):
    print("%2d t=%.2f n_people=%s box=%s state=%r" % (i, o.t, o.n_people, o.box, st))
