import pytest

from bedmonitor.evaluation import evaluate, match_events, pool, summarize, validate_gt


def seg(a, b, s):
    return {"start": a, "end": b, "state": s}


GT = {"video": "x", "duration_sec": 20.0, "bed_events": [],
      "segments": [seg(0, 10, "LYING_IN_BED"), seg(10, 20, "SITTING_ON_BED")]}


def report(segs, events=None, dur=20.0):
    return {"observation_duration_sec": dur, "timeline": segs, "bed_events": events or []}


def test_perfect_prediction():
    s = summarize(evaluate(GT, report(GT["segments"]), step=1.0))
    assert s["state"]["accuracy"] == 1.0
    assert all(v["abs_err_sec"] == 0 for v in s["durations"].values())


def test_late_boundary_gives_known_errors():
    pred = [seg(0, 12, "LYING_IN_BED"), seg(12, 20, "SITTING_ON_BED")]
    r = evaluate(GT, report(pred), step=1.0)
    s = summarize(r)
    assert s["state"]["accuracy"] == pytest.approx(0.9)
    assert r["matrix"]["SITTING_ON_BED"]["LYING_IN_BED"] == 2
    assert s["durations"]["LYING_IN_BED"]["abs_err_sec"] == 2.0
    assert s["durations"]["LYING_IN_BED"]["pct_err"] == 20.0


def test_event_matching_tolerance_and_false_exit():
    m = match_events([10.0], [12.0, 50.0], 5.0)
    assert (m["tp"], m["fp"], m["fn"]) == (1, 1, 0) and m["time_errors_sec"] == [2.0]
    far = match_events([10.0], [20.0], 5.0)
    assert (far["tp"], far["fp"], far["fn"]) == (0, 1, 1)


def test_bad_ground_truth_is_rejected():
    bad = {"duration_sec": 20.0, "segments": [seg(0, 10, "LYING_IN_BED"), seg(12, 20, "WALKING")]}
    with pytest.raises(ValueError):
        validate_gt(bad)


def test_unknown_counts_as_wrong_and_pooling_sums_counts():
    r_unk = evaluate(GT, report([seg(0, 20, "UNKNOWN")]), step=1.0)
    s = summarize(r_unk)
    assert s["state"]["accuracy"] == 0.0 and s["state"]["unknown_predicted_fraction"] == 1.0
    assert s["state"]["accuracy_excluding_unknown"] is None
    r_ok = evaluate(GT, report(GT["segments"]), step=1.0)
    assert pool([r_ok, r_ok])["matrix"]["LYING_IN_BED"]["LYING_IN_BED"] == 20
