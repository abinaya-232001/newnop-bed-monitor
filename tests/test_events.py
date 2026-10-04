from bedmonitor.events import detect_bed_events
from bedmonitor.temporal import Segment


def segs(spec):
    """spec: [(state, seconds), ...] -> (segments, total duration)"""
    out, t = [], 0.0
    for state, d in spec:
        out.append(Segment(t, t + d, state))
        t += d
    return out, t


def run(spec):
    s, dur = segs(spec)
    return detect_bed_events(s, dur)


def test_full_exit_sequence():
    r = run([("LYING_IN_BED", 60), ("SITTING_ON_BED", 10), ("STANDING", 5),
             ("WALKING", 20), ("SITTING_OUTSIDE_BED", 30)])
    assert r["bed_exit_count"] == 1
    e = r["events"][0]
    assert e["start_sec"] == 70.0 and e["confirmed_sec"] == 75.0
    assert e["previous_state"] == "sitting_on_bed" and e["current_state"] == "walking"


def test_sitting_up_is_not_an_exit():
    r = run([("LYING_IN_BED", 30), ("SITTING_ON_BED", 20), ("LYING_IN_BED", 30)])
    assert r["bed_exit_count"] == 0 and r["rejected_candidates"] == []


def test_stand_briefly_then_sit_back_is_rejected():
    r = run([("SITTING_ON_BED", 20), ("STANDING", 5), ("SITTING_ON_BED", 20)])
    assert r["bed_exit_count"] == 0
    assert r["rejected_candidates"][0]["reason"] == "returned_to_bed_before_moving_away"


def test_exit_then_return_via_sitting():
    r = run([("LYING_IN_BED", 30), ("STANDING", 5), ("WALKING", 20), ("SITTING_ON_BED", 8),
             ("LYING_IN_BED", 30)])
    assert r["bed_exit_count"] == 1 and r["bed_return_count"] == 1
    # out of bed from 30 s (stood up) until 55 s (sat back on the bed)
    assert r["longest_out_of_bed_period_sec"] == 25.0
    assert r["events"][1]["via_sitting"] is True


def test_standing_near_bed_is_not_a_return():
    r = run([("LYING_IN_BED", 30), ("STANDING", 5), ("WALKING", 20), ("STANDING", 10)])
    assert r["bed_return_count"] == 0


def test_sat_on_bed_then_left_is_not_a_return():
    r = run([("LYING_IN_BED", 30), ("STANDING", 5), ("WALKING", 20), ("SITTING_ON_BED", 8),
             ("WALKING", 20)])
    assert r["bed_return_count"] == 0
    assert r["rejected_candidates"][-1]["type"] == "possible_return"


def test_unknown_gap_lowers_confidence():
    r = run([("LYING_IN_BED", 30), ("UNKNOWN", 5), ("WALKING", 20)])
    assert r["bed_exit_count"] == 1 and r["events"][0]["confidence"] < 0.9


def test_video_starting_out_of_bed():
    r = run([("WALKING", 20), ("SITTING_ON_BED", 8), ("LYING_IN_BED", 30)])
    assert r["bed_exit_count"] == 0 and r["bed_return_count"] == 1


def test_still_out_at_end_counts_open_period():
    r = run([("LYING_IN_BED", 30), ("STANDING", 5), ("WALKING", 20), ("OUT_OF_BED", 100)])
    assert r["longest_out_of_bed_period_sec"] == 125.0
