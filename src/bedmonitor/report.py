"""Aggregate segments and bed events into durations and a JSON-ready report."""
from typing import Dict, List, Optional

from .temporal import Segment

STATES = ["LYING_IN_BED", "SITTING_ON_BED", "SITTING_OUTSIDE_BED",
          "STANDING", "WALKING", "OUT_OF_BED", "UNKNOWN"]
# Simplification, documented in the README: "in bed" = lying or sitting on the bed;
# "out of bed" = every other known state. UNKNOWN is reported separately.
IN_BED = ["LYING_IN_BED", "SITTING_ON_BED"]
OUT_OF_BED = ["SITTING_OUTSIDE_BED", "STANDING", "WALKING", "OUT_OF_BED"]


def fmt_mmss(sec: float) -> str:
    m, s = divmod(int(round(sec)), 60)
    return f"{m:02d}:{s:02d}"


def durations(segments: List[Segment]) -> Dict[str, float]:
    d = {s: 0.0 for s in STATES}
    for seg in segments:
        d[seg.state] += seg.duration
    return d


def timeline_text(segments: List[Segment]) -> str:
    return "\n".join(f"{fmt_mmss(s.start)} - {fmt_mmss(s.end)}  {s.state}" for s in segments)


def build_report(segments: List[Segment], duration_sec: float,
                 bed: Optional[dict] = None) -> dict:
    d = durations(segments)
    bed = bed or {}
    return {
        "observation_duration_sec": round(duration_sec, 1),
        "activity_duration_sec": {k.lower(): round(v, 1) for k, v in d.items()},
        "bed_exit_count": bed.get("bed_exit_count", 0),
        "bed_return_count": bed.get("bed_return_count", 0),
        "total_in_bed_sec": round(sum(d[s] for s in IN_BED), 1),
        "total_out_of_bed_sec": round(sum(d[s] for s in OUT_OF_BED), 1),
        "unknown_sec": round(d["UNKNOWN"], 1),
        "longest_out_of_bed_period_sec": bed.get("longest_out_of_bed_period_sec", 0.0),
        "final_state": segments[-1].state.lower(),
        "bed_events": bed.get("events", []),
        "rejected_candidates": bed.get("rejected_candidates", []),
        "timeline": [{"start": round(s.start, 1), "end": round(s.end, 1), "state": s.state}
                     for s in segments],
    }