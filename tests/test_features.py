from bedmonitor.features import classify_posture, extract_features
from bedmonitor.models import Observation


def make_obs(points, box):
    """points: {keypoint_index: (x, y, conf)}; all other keypoints get conf 0."""
    kps = [[0.0, 0.0, 0.0] for _ in range(17)]
    for i, (x, y, c) in points.items():
        kps[i] = [x, y, c]
    return Observation(t=0.0, n_people=1, track_id=1, box=box, box_conf=0.9, keypoints=kps)


def label(obs):
    return classify_posture(extract_features(obs))[0]


def test_lying_horizontal_torso():
    obs = make_obs({5: (100, 300, .9), 6: (100, 340, .9), 11: (300, 310, .9), 12: (300, 350, .9)},
                   box=[80, 280, 500, 380])
    assert label(obs) == "LYING"


def test_standing_vertical_torso_and_legs():
    obs = make_obs({5: (100, 100, .9), 6: (140, 100, .9), 11: (105, 300, .9), 12: (135, 300, .9),
                    13: (105, 450, .9), 14: (135, 450, .9)}, box=[60, 50, 180, 620])
    assert label(obs) == "STANDING"


def test_sitting_horizontal_thighs():
    obs = make_obs({5: (100, 100, .9), 6: (140, 100, .9), 11: (105, 300, .9), 12: (135, 300, .9),
                    13: (260, 310, .9), 14: (290, 310, .9)}, box=[60, 50, 320, 450])
    assert label(obs) == "SITTING"


def test_low_confidence_hips_gives_unknown():
    # like your test1.mp4 frame: shoulders clear, hips/legs unreliable, tall narrow box
    obs = make_obs({5: (1130, 508, .98), 6: (651, 519, .99), 11: (1011, 1049, .30), 12: (696, 1069, .33)},
                   box=[515, 15, 1320, 1070])
    assert label(obs) == "UNKNOWN"


def test_no_person_gives_unknown():
    assert classify_posture(extract_features(Observation(t=0.0, n_people=0)))[0] == "UNKNOWN"