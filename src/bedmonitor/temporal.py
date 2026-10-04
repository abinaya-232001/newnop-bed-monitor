"""Temporal smoothing: per-sample FrameStates -> stable timeline segments.

A new state is committed only after it has persisted for MIN_DURATION_SEC[state]
seconds (debouncing). The change is then backdated to when the state first
appeared. Short noise (one bad sample) never makes it into the timeline.
"""
from dataclasses import dataclass
from statistics import median
from typing import Dict, List, Optional

from .classifier import FrameState

# Engineering placeholders, NOT tuned values. Longer for UNKNOWN so a brief
# detection loss does not fragment the timeline.
MIN_DURATION_SEC: Dict[str, float] = {
    "LYING_IN_BED": 2.0,
    "SITTING_ON_BED": 2.0,
    "SITTING_OUTSIDE_BED": 2.0,
    "STANDING": 1.0,
    "WALKING": 1.0,
    "OUT_OF_BED": 2.0,
    "UNKNOWN": 3.0,
}


@dataclass
class Segment:
    start: float
    end: float
    state: str

    @property
    def duration(self) -> float:
        return self.end - self.start


def estimate_dt(states: List[FrameState]) -> float:
    """Typical gap between samples (median), default 1.0 s if it cannot be measured."""
    gaps = [b.t - a.t for a, b in zip(states, states[1:]) if b.t > a.t]
    return median(gaps) if gaps else 1.0


def smooth(states: List[FrameState], duration_sec: float,
           min_duration: Optional[Dict[str, float]] = None) -> List[Segment]:
    min_duration = min_duration or MIN_DURATION_SEC
    if not states:
        return [Segment(0.0, duration_sec, "UNKNOWN")]

    dt = estimate_dt(states)
    current = states[0].state
    changes = [(0.0, current)]           # (start_time, state) for each committed state
    candidate: Optional[str] = None      # state we are considering switching to
    cand_start = 0.0

    for s in states[1:]:
        if s.state == current:           # evidence for the current state: drop any candidate
            candidate = None
            continue
        if s.state != candidate:         # a new candidate starts its clock here
            candidate = s.state
            cand_start = s.t
        # +dt: a sample stands for one sampling interval, not an instant
        if s.t - cand_start + dt >= min_duration[candidate] - 1e-6:
            changes.append((cand_start, candidate))   # backdated commit
            current = candidate
            candidate = None

    segments: List[Segment] = []
    for i, (t0, st) in enumerate(changes):
        t1 = changes[i + 1][0] if i + 1 < len(changes) else duration_sec
        segments.append(Segment(t0, max(t1, t0), st))

    # A too-short first segment is start-up noise: absorb it into the next one.
    if len(segments) > 1 and segments[0].duration < min_duration[segments[0].state]:
        segments[1].start = 0.0
        segments.pop(0)
    return segments