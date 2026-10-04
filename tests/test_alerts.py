from bedmonitor.alerts import evaluate_alerts
from bedmonitor.events import detect_bed_events
from bedmonitor.temporal import Segment


def run(spec):
    out, t = [], 0.0
    for state, d in spec:
        out.append(Segment(t, t + d, state))
        t += d
    bed = detect_bed_events(out, t)
    return bed, evaluate_alerts(out, bed, t)


def test_nothing_happens_is_normal():
    bed, res = run([("LYING_IN_BED", 100)])
    assert res["overall_decision"] == "NORMAL" and res["alerts"] == []


def test_short_excursion_is_monitor_not_alert():
    bed, res = run([("LYING_IN_BED", 30), ("STANDING", 5), ("WALKING", 20),
                    ("SITTING_ON_BED", 8), ("LYING_IN_BED", 30)])
    assert res["overall_decision"] == "MONITOR"
    assert [e["decision"] for e in bed["events"]] == ["MONITOR", "NORMAL"]


def test_long_absence_still_out_at_end_is_alert():
    bed, res = run([("LYING_IN_BED", 30), ("STANDING", 5), ("WALKING", 400)])
    assert res["overall_decision"] == "ALERT"
    assert bed["events"][0]["decision"] == "ALERT"
    assert any(a["rule"] == "prolonged_out_of_bed" for a in res["alerts"])


def test_long_edge_sitting_is_monitor():
    bed, res = run([("LYING_IN_BED", 30), ("SITTING_ON_BED", 130), ("LYING_IN_BED", 30)])
    assert res["overall_decision"] == "MONITOR"
    assert res["alerts"][0]["rule"] == "prolonged_edge_sitting"


def test_long_unknown_is_monitor():
    bed, res = run([("LYING_IN_BED", 30), ("UNKNOWN", 70), ("LYING_IN_BED", 30)])
    assert res["overall_decision"] == "MONITOR"
    assert res["alerts"][0]["rule"] == "prolonged_unknown"
