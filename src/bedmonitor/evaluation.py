"""Evaluation: compare a predicted timeline and bed events with hand-labelled ground truth.

Pure functions, so they are tested without video. Ground truth must be labelled by a
human watching the video, never copied from the system's own output.
"""
from .report import STATES

GT_EVENTS = ("bed_exit", "return_to_bed")
SHORT = {"LYING_IN_BED": "LYING", "SITTING_ON_BED": "SIT_BED", "SITTING_OUTSIDE_BED": "SIT_OUT",
         "STANDING": "STAND", "WALKING": "WALK", "OUT_OF_BED": "OUT", "UNKNOWN": "UNK"}


def validate_gt(gt, tol=0.05):
    segs = gt["segments"]
    if not segs:
        raise ValueError("ground truth has no segments")
    if abs(segs[0]["start"]) > tol:
        raise ValueError("first segment must start at 0")
    for s in segs:
        if s["state"] not in STATES:
            raise ValueError("unknown state: " + str(s["state"]))
        if s["end"] <= s["start"]:
            raise ValueError("empty or reversed segment at " + str(s["start"]))
    for a, b in zip(segs, segs[1:]):
        if abs(a["end"] - b["start"]) > tol:
            raise ValueError("gap or overlap between %s and %s" % (a["end"], b["start"]))
    if abs(segs[-1]["end"] - gt["duration_sec"]) > tol:
        raise ValueError("last segment must end at duration_sec")
    for e in gt.get("bed_events", []):
        if e["event"] not in GT_EVENTS:
            raise ValueError("unknown event: " + str(e["event"]))


def state_at(segs, t):
    for s in segs:
        if s["start"] <= t < s["end"]:
            return s["state"]
    return segs[-1]["state"]


def sample_pairs(gt_segs, pred_segs, duration, step=0.5):
    """Compare both timelines on a fixed time grid (midpoint of each step)."""
    n = int(duration / step)
    return [(state_at(gt_segs, (i + 0.5) * step), state_at(pred_segs, (i + 0.5) * step))
            for i in range(n)]


def confusion_matrix(pairs):
    m = {g: {p: 0 for p in STATES} for g in STATES}
    for g, p in pairs:
        m[g][p] += 1
    return m


def _ratio(a, b):
    return a / b if b else None


def matrix_metrics(m, step):
    total = sum(sum(row.values()) for row in m.values())
    correct = sum(m[s][s] for s in STATES)
    unk_pred = sum(m[g]["UNKNOWN"] for g in STATES)
    per_state = {}
    for s in STATES:
        tp = m[s][s]
        support = sum(m[s].values())
        predicted = sum(m[g][s] for g in STATES)
        p, r = _ratio(tp, predicted), _ratio(tp, support)
        f1 = _ratio(2 * p * r, p + r) if p is not None and r is not None else None
        per_state[s] = {"precision": p, "recall": r, "f1": f1,
                        "support_sec": support * step, "predicted_sec": predicted * step}
    return {"accuracy": _ratio(correct, total),
            "unknown_predicted_fraction": _ratio(unk_pred, total),
            "accuracy_excluding_unknown": _ratio(correct - m["UNKNOWN"]["UNKNOWN"], total - unk_pred),
            "per_state": per_state}


def state_seconds(segs, duration):
    d = {s: 0.0 for s in STATES}
    for s in segs:
        a, b = max(s["start"], 0.0), min(s["end"], duration)
        if b > a:
            d[s["state"]] += b - a
    return d


def duration_table(d):
    out = {}
    for s in STATES:
        g, p = d[s]["gt"], d[s]["pred"]
        out[s] = {"gt_sec": round(g, 1), "pred_sec": round(p, 1),
                  "abs_err_sec": round(abs(p - g), 1),
                  "pct_err": None if g == 0 else round(abs(p - g) / g * 100, 1)}
    return out


def match_events(gt_times, pred_times, tol):
    """One-to-one greedy matching, nearest pairs first, within +-tol seconds."""
    cand = sorted((abs(g - p), gi, pi) for gi, g in enumerate(gt_times)
                  for pi, p in enumerate(pred_times) if abs(g - p) <= tol)
    used_g, used_p, errs = set(), set(), []
    for d, gi, pi in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        errs.append(d)
    tp = len(errs)
    return {"tp": tp, "fp": len(pred_times) - tp, "fn": len(gt_times) - tp, "time_errors_sec": errs}


def event_scores(tp, fp, fn):
    return {"tp": tp, "false_detections": fp, "missed": fn,
            "precision": _ratio(tp, tp + fp), "recall": _ratio(tp, tp + fn)}


def evaluate(gt, report, step=0.5, tol_sec=5.0):
    """gt: ground-truth dict. report: the dict written by run_timeline (needs timeline, bed_events)."""
    validate_gt(gt)
    duration = min(gt["duration_sec"], report["observation_duration_sec"])
    pred_segs = report["timeline"]
    m = confusion_matrix(sample_pairs(gt["segments"], pred_segs, duration, step))
    gt_d, pred_d = state_seconds(gt["segments"], duration), state_seconds(pred_segs, duration)
    events = {}
    for name in GT_EVENTS:
        g = [e["time"] for e in gt.get("bed_events", []) if e["event"] == name]
        p = [e["start_sec"] for e in report["bed_events"] if e["event"] == name]
        events[name] = match_events(g, p, tol_sec)
    return {"video": gt.get("video"), "evaluated_sec": duration, "step_sec": step,
            "tolerance_sec": tol_sec, "matrix": m,
            "durations": {s: {"gt": gt_d[s], "pred": pred_d[s]} for s in STATES},
            "events": events}


def pool(results):
    """Sum raw counts over several videos, then recompute the metrics from the sums."""
    m = {g: {p: sum(r["matrix"][g][p] for r in results) for p in STATES} for g in STATES}
    d = {s: {"gt": sum(r["durations"][s]["gt"] for r in results),
             "pred": sum(r["durations"][s]["pred"] for r in results)} for s in STATES}
    ev = {n: {"tp": sum(r["events"][n]["tp"] for r in results),
              "fp": sum(r["events"][n]["fp"] for r in results),
              "fn": sum(r["events"][n]["fn"] for r in results),
              "time_errors_sec": [x for r in results for x in r["events"][n]["time_errors_sec"]]}
          for n in GT_EVENTS}
    return {"video": "POOLED (%d videos)" % len(results),
            "evaluated_sec": sum(r["evaluated_sec"] for r in results),
            "step_sec": results[0]["step_sec"], "tolerance_sec": results[0]["tolerance_sec"],
            "matrix": m, "durations": d, "events": ev}


def summarize(r):
    ev = {}
    for name, e in r["events"].items():
        errs = e["time_errors_sec"]
        ev[name] = {**event_scores(e["tp"], e["fp"], e["fn"]),
                    "mean_time_error_sec": round(sum(errs) / len(errs), 2) if errs else None}
    return {"video": r["video"], "evaluated_sec": round(r["evaluated_sec"], 1),
            "tolerance_sec": r["tolerance_sec"],
            "state": matrix_metrics(r["matrix"], r["step_sec"]),
            "confusion_matrix": r["matrix"],
            "durations": duration_table(r["durations"]),
            "events": ev}


def confusion_text(m):
    used = [s for s in STATES if any(m[s].values()) or any(m[g][s] for g in STATES)]
    if not used:
        return "(empty)"
    w = max(len(s) for s in used)
    lines = ["".ljust(w) + " " + " ".join(SHORT[s].rjust(8) for s in used)]
    for g in used:
        lines.append(g.ljust(w) + " " + " ".join(str(m[g][p]).rjust(8) for p in used))
    return "\n".join(lines)


def _f(x, nd=2):
    return "n/a" if x is None else "%.*f" % (nd, x)


def summary_text(r):
    s = summarize(r)
    st = s["state"]
    lines = ["=== %s  (evaluated %.1fs, grid %ss) ===" % (s["video"], s["evaluated_sec"], r["step_sec"]),
             "state accuracy: %s  (UNKNOWN predictions count as wrong)" % _f(st["accuracy"]),
             "share of time predicted UNKNOWN: %s   accuracy on non-UNKNOWN predictions: %s"
             % (_f(st["unknown_predicted_fraction"]), _f(st["accuracy_excluding_unknown"])),
             "", "confusion matrix (rows = ground truth, columns = predicted, in grid samples):",
             confusion_text(r["matrix"]), "", "per-state precision / recall / F1 (n/a = undefined):"]
    for name, v in st["per_state"].items():
        if v["support_sec"] or v["predicted_sec"]:
            lines.append("  %-20s P=%s R=%s F1=%s  gt=%.1fs pred=%.1fs" % (
                name, _f(v["precision"]), _f(v["recall"]), _f(v["f1"]),
                v["support_sec"], v["predicted_sec"]))
    lines += ["", "duration error (ground truth vs predicted):"]
    for name, v in s["durations"].items():
        if v["gt_sec"] or v["pred_sec"]:
            lines.append("  %-20s gt=%6.1fs pred=%6.1fs abs_err=%5.1fs pct_err=%s" % (
                name, v["gt_sec"], v["pred_sec"], v["abs_err_sec"], _f(v["pct_err"], 1)))
    lines += ["", "bed events (match tolerance +-%ss):" % s["tolerance_sec"]]
    for name, e in s["events"].items():
        lines.append("  %-14s TP=%d false=%d missed=%d precision=%s recall=%s mean_time_err=%ss" % (
            name, e["tp"], e["false_detections"], e["missed"], _f(e["precision"]),
            _f(e["recall"]), _f(e["mean_time_error_sec"])))
    return "\n".join(lines)
