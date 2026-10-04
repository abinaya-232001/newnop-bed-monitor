"""Configurable NORMAL / MONITOR / ALERT rules.

These are ENGINEERING RULES for this assignment, not medically validated thresholds.
There is no time-of-day input, so a night-time exit is treated like a daytime one.
"""
from typing import List

from .temporal import Segment

DEFAULT_RULES = {
    "prolonged_out_of_bed_sec": 300,   # out of bed this long, or still out at the end -> ALERT
    "prolonged_unknown_sec": 60,       # UNKNOWN this long -> MONITOR
    "prolonged_edge_sitting_sec": 120, # one continuous SITTING_ON_BED this long -> MONITOR
    "bed_exit_decision": "MONITOR",    # decision attached to every confirmed bed exit
}
SEVERITY = {"NORMAL": 0, "MONITOR": 1, "ALERT": 2}


def _alert(rule, decision, start, end, reason) -> dict:
    return {"rule": rule, "decision": decision, "start_sec": round(start, 1),
            "end_sec": round(end, 1), "reason": reason}


def evaluate_alerts(segments: List[Segment], bed: dict, duration_sec: float,
                    rules: dict = None) -> dict:
    """Return {"alerts": [...], "overall_decision": ...}; adds "decision" to each bed event."""
    r = {**DEFAULT_RULES, **(rules or {})}
    alerts: List[dict] = []

    # out-of-bed periods: from an exit to the next return (or to the end of the video)
    periods, open_exit = [], None
    for e in bed["events"]:
        if e["event"] == "bed_exit":
            e["decision"] = r["bed_exit_decision"]
            alerts.append(_alert("bed_exit", r["bed_exit_decision"], e["start_sec"], e["confirmed_sec"],
                                 "bed exit confirmed"))
            open_exit = e
        else:
            e["decision"] = "NORMAL"
            if open_exit is not None:
                periods.append((open_exit, open_exit["start_sec"], e["start_sec"], True))
                open_exit = None
    if open_exit is not None:
        periods.append((open_exit, open_exit["start_sec"], duration_sec, False))

    for exit_event, start, end, returned in periods:
        if end - start >= r["prolonged_out_of_bed_sec"]:
            exit_event["decision"] = "ALERT"
            why = (f"out of bed {end - start:.0f}s" if returned else
                   f"still out of bed at end of video after {end - start:.0f}s")
            alerts.append(_alert("prolonged_out_of_bed", "ALERT", start, end,
                                 f"{why} (threshold {r['prolonged_out_of_bed_sec']}s)"))

    for seg in segments:
        if seg.state == "UNKNOWN" and seg.duration >= r["prolonged_unknown_sec"]:
            alerts.append(_alert("prolonged_unknown", "MONITOR", seg.start, seg.end,
                                 f"activity undetermined for {seg.duration:.0f}s"))
        if seg.state == "SITTING_ON_BED" and seg.duration >= r["prolonged_edge_sitting_sec"]:
            alerts.append(_alert("prolonged_edge_sitting", "MONITOR", seg.start, seg.end,
                                 f"sitting on the bed for {seg.duration:.0f}s"))

    overall = max((a["decision"] for a in alerts), key=SEVERITY.get, default="NORMAL")
    return {"alerts": alerts, "overall_decision": overall}
