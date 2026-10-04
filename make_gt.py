"""Usage: python make_gt.py LABELS_TXT OUT_JSON
LABELS_TXT lines (comments start with #):
  duration 92.0
  START END STATE          (for example: 0 30 LYING_IN_BED)
  exit 38.0                (time they stood up to leave)
  return 70.0              (time they sat back on the bed)
  note free text
Validates the labels and writes the ground-truth JSON."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bedmonitor.evaluation import validate_gt  # noqa: E402

txt, out = sys.argv[1], sys.argv[2]
gt = {"video": Path(out).stem, "duration_sec": None, "segments": [], "bed_events": [], "notes": ""}
for raw in open(txt, encoding="utf-8"):
    line = raw.split("#")[0].strip()
    if not line:
        continue
    tok = line.split(None, 1)
    key = tok[0].lower()
    if key == "duration":
        gt["duration_sec"] = float(tok[1])
    elif key in ("exit", "return"):
        gt["bed_events"].append({"event": "bed_exit" if key == "exit" else "return_to_bed",
                                 "time": float(tok[1])})
    elif key == "note":
        gt["notes"] = tok[1]
    else:
        a, b, state = line.split()
        gt["segments"].append({"start": float(a), "end": float(b), "state": state})
if gt["duration_sec"] is None:
    sys.exit("missing 'duration' line")
validate_gt(gt)
with open(out, "w", encoding="utf-8") as f:
    json.dump(gt, f, indent=2)
print("wrote", out, "-", len(gt["segments"]), "segments,", len(gt["bed_events"]), "events")
