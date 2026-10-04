from bedmonitor.agent import NullVLM, review
from bedmonitor.classifier import FrameState
from bedmonitor.events import detect_bed_events
from bedmonitor.temporal import Segment


def segs(spec):
    out, t = [], 0.0
    for state, d in spec:
        out.append(Segment(t, t + d, state))
        t += d
    return out, t


def frames(t0, t1, state, reason, step=0.5):
    out, t = [], t0
    while t < t1:
        out.append(FrameState(t=t, state=state, conf=0.5, reason=reason))
        t += step
    return out


class FakeVLM:
    def ask(self, t_start, t_end, question):
        return {"state": "LYING_IN_BED", "confidence": 0.7}


class BadVLM:
    def ask(self, t_start, t_end, question):
        return {"state": "FLYING", "confidence": 5}


def test_confident_exit_needs_no_context():
    s, d = segs([("LYING_IN_BED", 30), ("STANDING", 5), ("WALKING", 20)])
    traces = review(detect_bed_events(s, d), s, [])
    assert traces[0]["decision"] == "CONFIRMED" and traces[0]["requested_context"] == []


def test_exit_with_unknown_gap_requests_context_and_confirms():
    s, d = segs([("LYING_IN_BED", 30), ("UNKNOWN", 5), ("WALKING", 20)])
    st = frames(30, 35, "UNKNOWN", "no_person_detected")
    t = review(detect_bed_events(s, d), s, st)[0]
    assert "segment_after" in t["requested_context"]
    assert t["decision"] == "CONFIRMED" and "no_person_detected" in t["evidence"][0]


def test_rejected_candidate_is_traced():
    s, d = segs([("SITTING_ON_BED", 20), ("STANDING", 5), ("SITTING_ON_BED", 20)])
    traces = review(detect_bed_events(s, d), s, [])
    assert traces[0]["decision"] == "REJECTED"


def test_long_unknown_in_bed_stays_unresolved_without_vlm():
    s, d = segs([("LYING_IN_BED", 30), ("UNKNOWN", 10), ("LYING_IN_BED", 30)])
    st = frames(30, 40, "UNKNOWN", "no_person_detected")
    t = review(detect_bed_events(s, d), s, st, vlm=NullVLM())[0]
    assert t["decision"] == "UNRESOLVED" and t["vlm_advisory"] is None
    assert "occlusion" in t["uncertainty"]


def test_vlm_answer_is_validated_and_advisory_only():
    s, d = segs([("LYING_IN_BED", 30), ("UNKNOWN", 10), ("LYING_IN_BED", 30)])
    st = frames(30, 40, "UNKNOWN", "no_person_detected")
    ok = review(detect_bed_events(s, d), s, st, vlm=FakeVLM())[0]
    bad = review(detect_bed_events(s, d), s, st, vlm=BadVLM())[0]
    assert ok["vlm_advisory"] == {"state": "LYING_IN_BED", "confidence": 0.7}
    assert bad["vlm_advisory"] is None
    assert [x.state for x in s] == ["LYING_IN_BED", "UNKNOWN", "LYING_IN_BED"]  # timeline untouched
