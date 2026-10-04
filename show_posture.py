"""Usage: python show_posture.py CACHE_JSON  - print posture guess per sample."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bedmonitor.features import classify_posture, extract_features  # noqa: E402
from bedmonitor.models import Observation  # noqa: E402

data = json.load(open(sys.argv[1], encoding="utf-8"))
for s in data["samples"]:
    obs = Observation(**s)
    feats = extract_features(obs)
    label, conf = classify_posture(feats)
    print(f"t={obs.t:6.2f}  {label:9s} conf={conf:.1f}  {feats}")