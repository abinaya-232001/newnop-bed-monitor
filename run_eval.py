"""Usage: python run_eval.py GT_DIR REPORT_DIR [--tol 5] [--step 0.5]
For every GT_DIR/NAME.json, evaluates REPORT_DIR/NAME_report.json (written by run_timeline).
Prints per-video and pooled results and writes REPORT_DIR/eval_<GT_DIR name>.json"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bedmonitor.evaluation import evaluate, pool, summarize, summary_text  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("gt_dir")
p.add_argument("report_dir")
p.add_argument("--tol", type=float, default=5.0)
p.add_argument("--step", type=float, default=0.5)
a = p.parse_args()

results = []
for gt_path in sorted(Path(a.gt_dir).glob("*.json")):
    rep_path = Path(a.report_dir) / (gt_path.stem + "_report.json")
    if not rep_path.exists():
        print("MISSING report, skipped:", rep_path)
        continue
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)
    with open(rep_path, encoding="utf-8") as f:
        rep = json.load(f)
    r = evaluate(gt, rep, a.step, a.tol)
    results.append(r)
    print(summary_text(r))
    print()
if not results:
    sys.exit("nothing evaluated")
everything = list(results)
if len(results) > 1:
    pooled = pool(results)
    print(summary_text(pooled))
    everything.append(pooled)
out = Path(a.report_dir) / ("eval_" + Path(a.gt_dir).name + ".json")
with open(out, "w", encoding="utf-8") as f:
    json.dump([summarize(r) for r in everything], f, indent=2)
print("wrote", out)
