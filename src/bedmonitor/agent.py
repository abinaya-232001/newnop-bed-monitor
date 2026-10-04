"""Agent layer: decides when the timeline needs more context, fetches it, records why.

Deterministic on purpose. The optional VLM is advisory only: its answer is validated
and stored in the trace, but it never changes the timeline or the events.
"""
from collections import Counter
from typing import Dict, List, Optional, Protocol

from .classifier import FrameState
from .events import AWAY, IN_BED, hms
from .temporal import Segment

CFG = {
    "unknown_review_sec": 5.0,   # UNKNOWN segments at least this long get reviewed
    "min_away_sec": 3.0,         # an away state must persist this long to confirm an exit
}

STATES_ALLOWED = {"LYING_IN_BED", "SITTING_ON_BED", "SITTING_OUTSIDE_BED", "STANDING",
                  "WALKING", "OUT_OF_BED", "UNKNOWN"}


class VLMClient(Protocol):
    def ask(self, t_start: float, t_end: float, question: str) -> Optional[dict]: ...


class NullVLM:
    """Default: no VLM configured, so the system is fully offline and reproducible."""
    def ask(self, t_start: float, t_end: float, question: str) -> Optional[dict]:
        return None


def validate_vlm(out) -> Optional[dict]:
    """Accept only {"state": <known state>, "confidence": 0..1}; anything else is ignored."""
    if not isinstance(out, dict) or out.get("state") not in STATES_ALLOWED:
        return None
    c = out.get("confidence")
    if not isinstance(c, (int, float)) or isinstance(c, bool) or not 0 <= c <= 1:
        return None
    return {"state": out["state"], "confidence": float(c)}


# ---- context-retrieval tools the agent can call ----------------------------------
def segment_before(segments: List[Segment], t: float) -> Optional[Segment]:
    prev = [s for s in segments if s.end <= t + 1e-6]
    return prev[-1] if prev else None


def segment_after(segments: List[Segment], t: float) -> Optional[Segment]:
    nxt = [s for s in segments if s.start >= t - 1e-6]
    return nxt[0] if nxt else None


def frame_reasons(states: List[FrameState], t0: float, t1: float) -> Dict[str, int]:
    """Count why frames in [t0, t1) got their label (e.g. no_person_detected)."""
    c: Counter = Counter()
    for s in states:
        if t0 <= s.t < t1:
            c[s.reason.split(" ")[0].split("+")[0]] += 1
            if "+multi_person" in s.reason:
                c["multi_person_frames"] += 1
    return dict(c)


def _trace(observation, evidence, uncertainty, requested, decision, reason, **extra) -> dict:
    return {"observation": observation, "evidence": evidence, "uncertainty": uncertainty,
            "requested_context": requested, "decision": decision, "reason": reason, **extra}


def review(bed: dict, segments: List[Segment], states: List[FrameState],
           vlm: Optional[VLMClient] = None, cfg: Optional[dict] = None) -> List[dict]:
    cfg = {**CFG, **(cfg or {})}
    vlm = vlm or NullVLM()
    traces: List[dict] = []

    # 1) bed exits: only ask for context when the evidence is weak
    for e in bed["events"]:
        if e["event"] != "bed_exit":
            continue
        obs = f"person left the in-bed states at {e['start_time']} (was {e['previous_state']})"
        if e["confidence"] >= 0.9:
            traces.append(_trace(obs, [f"{e['previous_state']} -> {e['current_state']}, no gaps"],
                                 "low", [], "CONFIRMED", "complete sequence, no extra context needed"))
            continue
        evidence = []
        gap = segment_before(segments, e["start_sec"])
        if gap is not None and gap.state == "UNKNOWN":
            evidence.append(f"UNKNOWN for {gap.duration:.0f}s just before the exit; "
                            f"frame reasons: {frame_reasons(states, gap.start, gap.end)}")
        after = segment_after(segments, e["confirmed_sec"])
        if after is not None and after.state in AWAY and after.duration >= cfg["min_away_sec"]:
            evidence.append(f"{after.state} persisted {after.duration:.0f}s after the exit")
            decision, reason = "CONFIRMED", "away state persisted after the uncertain gap"
        else:
            decision, reason = "UNRESOLVED", "away state too short or missing after the gap"
        traces.append(_trace(obs, evidence, "UNKNOWN gap before exit",
                             ["segment_before", "segment_after", "frame_reasons"], decision, reason))

    # 2) rejected candidates: recorded so the negative decision is auditable
    for r in bed["rejected_candidates"]:
        traces.append(_trace(f"{r['type']} at {r['at']}", [r["reason"]], "candidate only",
                             ["segment_after"], "REJECTED", r["reason"]))

    # 3) long UNKNOWN segments: gather context, optionally ask the VLM (advisory)
    for seg in segments:
        if seg.state != "UNKNOWN" or seg.duration < cfg["unknown_review_sec"]:
            continue
        before, after = segment_before(segments, seg.start), segment_after(segments, seg.end)
        b = before.state if before else None
        a = after.state if after else None
        evidence = [f"before: {b}", f"after: {a}",
                    f"frame reasons: {frame_reasons(states, seg.start, seg.end)}"]
        if b in IN_BED and a in IN_BED:
            uncertainty = "possible occlusion or detection loss while in bed"
        elif b in AWAY or a in AWAY:
            uncertainty = "possible time outside the camera view"
        else:
            uncertainty = "cannot tell from context"
        advisory = None
        try:
            advisory = validate_vlm(vlm.ask(seg.start, seg.end, "What is the person doing?"))
        except Exception as exc:                       # a VLM failure must never break the run
            evidence.append(f"vlm_error: {type(exc).__name__}")
        reason = ("timeline keeps UNKNOWN; VLM opinion stored as advisory only" if advisory
                  else "timeline keeps UNKNOWN; no valid VLM opinion available")
        traces.append(_trace(f"UNKNOWN from {hms(seg.start)} to {hms(seg.end)}", evidence, uncertainty,
                             ["segment_before", "segment_after", "frame_reasons", "vlm(optional)"],
                             "UNRESOLVED", reason, vlm_advisory=advisory))
    return traces
