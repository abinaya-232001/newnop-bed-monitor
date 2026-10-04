"""Usage: python run_classify.py CACHE_JSON ROI_JSON
Per-sample posture -> location -> state, then a count of states.
Still frame-level only: no smoothing yet, so flicker here is expected."""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bedmonitor.classifier import classify_frame, motion_speeds  # noqa: E402
from bedmonitor.features import classify_posture, extract_features  # noqa: E402
from bedmonitor.models import Observation  # noqa: E402
from bedmonitor.spatial import locate  # noqa: E402

data = json.load(open(sys.argv[1], encoding="utf-8"))
roi = json.load(open(sys.argv[2], encoding="utf-8"))["bed_roi"]

obs_list = [Observation(**s) for s in data["samples"]]
speeds = motion_speeds(obs_list)

counts = Counter()
for o, sp in zip(obs_list, speeds):
    feats = extract_features(o)
    posture, pconf = classify_posture(feats)
    loc, _ = locate(o, roi)
    fs = classify_frame(o, posture, pconf, loc, sp)
    counts[fs.state] += 1
    torso = feats["torso_angle"] if feats else None
    torso_txt = "None" if torso is None else f"{torso:.0f}"
    print(f"t={o.t:6.1f} posture={posture:8s} loc={str(loc):8s} torso={torso_txt:>4s} "
          f"-> {fs.state:20s} ({fs.reason})")

print("\nstate counts:", dict(counts), "of", len(obs_list), "samples")