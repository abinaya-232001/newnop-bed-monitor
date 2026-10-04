"""Usage: python run_timeline.py CACHE_JSON ROI_JSON
Prints timeline, bed events, alerts, agent traces; writes data/output/<name>_report.json"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bedmonitor.agent import review  # noqa: E402
from bedmonitor.alerts import evaluate_alerts  # noqa: E402
from bedmonitor.events import detect_bed_events  # noqa: E402
from bedmonitor.models import Observation  # noqa: E402
from bedmonitor.pipeline import classify_all  # noqa: E402
from bedmonitor.report import build_report, timeline_text  # noqa: E402
from bedmonitor.temporal import smooth  # noqa: E402

cache_path, roi_path = sys.argv[1], sys.argv[2]
data = json.load(open(cache_path, encoding="utf-8"))
roi = json.load(open(roi_path, encoding="utf-8"))["bed_roi"]
dur = data["duration_sec"]

obs_list = [Observation(**s) for s in data["samples"]]
states = classify_all(obs_list, roi)
segments = smooth(states, dur)
bed = detect_bed_events(segments, dur)
alerts = evaluate_alerts(segments, bed, dur)      # also adds "decision" to each bed event
traces = review(bed, segments, states)            # NullVLM by default: no network needed
report = build_report(segments, dur, bed)
report["alerts"] = alerts["alerts"]
report["overall_decision"] = alerts["overall_decision"]
report["agent_traces"] = traces

print(timeline_text(segments))
print(json.dumps({k: v for k, v in report.items() if k != "timeline"}, indent=2))

out = Path(__file__).parent / "data" / "output" / (Path(cache_path).stem + "_report.json")
out.write_text(json.dumps(report, indent=2), encoding="utf-8")
print("wrote", out)
