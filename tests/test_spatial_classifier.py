from bedmonitor.classifier import classify_frame, motion_speeds
from bedmonitor.models import Observation
from bedmonitor.spatial import locate

BED = [200, 300, 800, 600]   # x1, y1, x2, y2


def obs_with_hips(hip_x, hip_y, t=0.0, box=None, conf=0.9, n=1, tid=1):
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    kps[11] = [hip_x - 20, hip_y, conf]
    kps[12] = [hip_x + 20, hip_y, conf]
    return Observation(t=t, n_people=n, track_id=tid,
                       box=box or [hip_x - 100, hip_y - 100, hip_x + 100, hip_y + 100],
                       box_conf=0.9, keypoints=kps)


def test_locate_on_near_away():
    assert locate(obs_with_hips(500, 450), BED)[0] == "ON_BED"
    assert locate(obs_with_hips(900, 450), BED)[0] == "NEAR_BED"   # 100px right, margin = 180
    assert locate(obs_with_hips(1500, 450), BED)[0] == "AWAY"


def test_lying_on_bed_vs_lying_elsewhere():
    o = obs_with_hips(500, 450)
    assert classify_frame(o, "LYING", 0.8, "ON_BED", None).state == "LYING_IN_BED"
    assert classify_frame(o, "LYING", 0.8, "AWAY", None).state == "UNKNOWN"


def test_sitting_on_bed_vs_chair():
    o = obs_with_hips(500, 450)
    assert classify_frame(o, "SITTING", 0.8, "ON_BED", None).state == "SITTING_ON_BED"
    assert classify_frame(o, "SITTING", 0.8, "AWAY", None).state == "SITTING_OUTSIDE_BED"


def test_standing_vs_walking_uses_speed():
    o = obs_with_hips(500, 450)
    assert classify_frame(o, "STANDING", 0.8, "AWAY", 0.02).state == "STANDING"
    assert classify_frame(o, "STANDING", 0.8, "AWAY", 0.60).state == "WALKING"


def test_no_person_is_unknown():
    assert classify_frame(Observation(t=0.0, n_people=0), "UNKNOWN", 0.0, None, None).state == "UNKNOWN"


def test_motion_speed_moving_vs_still():
    moving = [obs_with_hips(500 + 100 * i, 450, t=0.5 * i) for i in range(5)]   # 100 px per 0.5 s
    still = [obs_with_hips(500, 450, t=0.5 * i) for i in range(5)]
    assert motion_speeds(moving)[-1] > 0.25
    assert motion_speeds(still)[-1] == 0.0