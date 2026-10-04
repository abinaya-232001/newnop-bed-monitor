"""Bed exit / return detection from the smoothed timeline (a small state machine).

Phases:  IN_BED -> EXITING -> OUT -> RETURNING -> IN_BED
  IN_BED     last known state is lying / sitting on bed
  EXITING    stood up, not yet away from the bed (candidate, may be rejected)
  OUT        exit confirmed (away state seen); waiting for a return
  RETURNING  sat on the bed after being out; needs LYING to confirm
UNKNOWN segments are skipped, but remembered so confidence can be lowered.
"""
from typing import Dict, List

from .temporal import Segment

IN_BED = {"LYING_IN_BED", "SITTING_ON_BED"}
AWAY = {"WALKING", "OUT_OF_BED", "SITTING_OUTSIDE_BED"}
# STANDING is transitional: neither in-bed nor away.


def hms(sec: float) -> str:
    s = int(round(sec))
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


def _event(name, start, confirmed, prev, cur, conf, **extra) -> dict:
    return {"event": name, "start_time": hms(start), "confirmed_time": hms(confirmed),
            "start_sec": round(start, 1), "confirmed_sec": round(confirmed, 1),
            "previous_state": prev.lower(), "current_state": cur.lower(),
            "confidence": conf, **extra}


def detect_bed_events(segments: List[Segment], duration_sec: float) -> Dict:
    events, rejected, out_periods = [], [], []
    phase = None
    prev_known = None
    unknown_seen = False
    exit_start = exit_prev = None
    exit_unknown = False
    out_start = None
    ret_start = ret_prev = None

    for seg in segments:
        s = seg.state
        if s == "UNKNOWN":
            unknown_seen = True
            continue

        if phase is None:                      # first known state decides the start phase
            phase = "IN_BED" if s in IN_BED else "OUT"
            out_start = 0.0 if phase == "OUT" else None
            prev_known, unknown_seen = s, False
            continue

        if phase == "IN_BED" and s not in IN_BED:
            phase, exit_start, exit_prev, exit_unknown = "EXITING", seg.start, prev_known, unknown_seen

        if phase == "EXITING":
            if s in IN_BED:                    # sat/lay back down: not an exit
                rejected.append({"type": "possible_bed_exit", "at": hms(exit_start),
                                 "reason": "returned_to_bed_before_moving_away"})
                phase = "IN_BED"
            elif s in AWAY:
                conf = 0.6 if exit_unknown else 0.9
                events.append(_event("bed_exit", exit_start, seg.start, exit_prev, s, conf,
                                     unknown_gap=exit_unknown))
                phase, out_start = "OUT", exit_start
        elif phase == "OUT":
            if s == "LYING_IN_BED":            # straight back to lying: weaker evidence
                events.append(_event("return_to_bed", seg.start, seg.start, prev_known, s, 0.7,
                                     via_sitting=False))
                out_periods.append((out_start, seg.start))
                phase = "IN_BED"
            elif s == "SITTING_ON_BED":
                phase, ret_start, ret_prev = "RETURNING", seg.start, prev_known
        elif phase == "RETURNING":
            if s == "LYING_IN_BED":
                events.append(_event("return_to_bed", ret_start, seg.start, ret_prev, s, 0.9,
                                     via_sitting=True))
                out_periods.append((out_start, ret_start))
                phase = "IN_BED"
            elif s != "SITTING_ON_BED":        # sat on the bed, then left again
                rejected.append({"type": "possible_return", "at": hms(ret_start),
                                 "reason": "sat_on_bed_but_did_not_lie_down"})
                phase = "OUT"

        prev_known, unknown_seen = s, False

    if phase == "EXITING":
        rejected.append({"type": "possible_bed_exit", "at": hms(exit_start),
                         "reason": "video_ended_before_confirmation"})
    if phase in ("OUT", "RETURNING"):
        out_periods.append((out_start, duration_sec))   # still out at the end

    longest = max((e - s for s, e in out_periods), default=0.0)
    return {
        "events": events,
        "rejected_candidates": rejected,
        "bed_exit_count": sum(e["event"] == "bed_exit" for e in events),
        "bed_return_count": sum(e["event"] == "return_to_bed" for e in events),
        "longest_out_of_bed_period_sec": round(longest, 1),
    }