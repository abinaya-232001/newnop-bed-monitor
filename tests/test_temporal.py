from bedmonitor.classifier import FrameState
from bedmonitor.report import build_report, durations
from bedmonitor.temporal import smooth


def fs(t, state):
    return FrameState(t=t, state=state, conf=0.8, reason="test")


def seq(spec, dt=0.5):
    """spec: [(state, n_samples), ...] -> (list of FrameState, total duration)"""
    out, t = [], 0.0
    for state, n in spec:
        for _ in range(n):
            out.append(fs(t, state))
            t += dt
    return out, t


def test_single_unknown_blip_is_absorbed():
    states, dur = seq([("LYING_IN_BED", 10), ("UNKNOWN", 1), ("LYING_IN_BED", 10)])
    assert [s.state for s in smooth(states, dur)] == ["LYING_IN_BED"]


def test_real_transition_is_committed_and_backdated():
    states, dur = seq([("LYING_IN_BED", 10), ("SITTING_ON_BED", 10)])
    segs = smooth(states, dur)
    assert [s.state for s in segs] == ["LYING_IN_BED", "SITTING_ON_BED"]
    assert segs[1].start == 5.0          # first sitting sample, not the commit moment


def test_brief_standing_is_ignored():
    states, dur = seq([("SITTING_ON_BED", 10), ("STANDING", 1), ("SITTING_ON_BED", 10)])
    assert [s.state for s in smooth(states, dur)] == ["SITTING_ON_BED"]


def test_durations_sum_to_observation_time():
    states, dur = seq([("LYING_IN_BED", 8), ("SITTING_ON_BED", 6), ("STANDING", 4),
                       ("WALKING", 6), ("OUT_OF_BED", 6)])
    segs = smooth(states, dur)
    assert abs(sum(durations(segs).values()) - dur) < 1e-6


def test_leading_blip_is_dropped():
    states, dur = seq([("UNKNOWN", 1), ("LYING_IN_BED", 10)])
    segs = smooth(states, dur)
    assert [s.state for s in segs] == ["LYING_IN_BED"] and segs[0].start == 0.0


def test_report_fields():
    states, dur = seq([("LYING_IN_BED", 10), ("SITTING_ON_BED", 10)])
    r = build_report(smooth(states, dur), dur)
    assert r["total_in_bed_sec"] == 10.0 and r["final_state"] == "sitting_on_bed"