# Bed Monitor: pose-based monitoring of a person in a bedroom video

Author: Abinaya Rajasekara. Submission for the Newnop Associate AI/ML Engineer assignment.

## Honest summary

- A deterministic pipeline turns a bedroom video into a timeline of states (lying, sitting, standing, walking, out of bed, unknown), bed exit/return events, alerts (NORMAL / MONITOR / ALERT) and a JSON report. It runs on CPU only.
- It was evaluated against hand-labelled ground truth on 5 clips. Per-clip results: 1.00, 0.95, 0.90, 0.00, 0.00 state accuracy. Two clips fail completely; see "Failure cases".
- **Bed exit / return detection is implemented and unit-tested, but it has NOT been validated on real footage.** No clip I could use contains a confirmed exit or return, so precision, recall, false exits and exit-time error are "Not measured".
- **ALERT has never fired on a real clip.** The longest clip is shorter than the 300 s rule.
- Almost none of the footage shows an elderly person. The clips are third-party downloads, several of them edited. Provenance and licence status are in `data/sources.md`.
- Thresholds are untuned placeholders. There is no held-out split.
- The agent layer is deterministic. The optional VLM interface exists but no VLM backend was implemented or tested.

## Architecture

```mermaid
flowchart LR
  V[Video file] --> S[Frame sampler 3 fps]
  S --> P[YOLOv8n-pose + ByteTrack]
  P --> C[(Cached observations JSON)]
  C --> F[Posture features]
  F --> L[Spatial: bed ROI]
  L --> K[Per-sample classifier]
  K --> T[Temporal smoother]
  T --> G[Timeline segments]
  G --> E[Bed exit/return phase machine]
  E --> A[Agent: NullVLM default]
  A --> R[Alert rules]
  R --> J[JSON report]
  J --> EV[Evaluation vs ground truth]
```

Source files are in `src/bedmonitor/`: `video`, `perception`, `features`, `spatial`, `classifier`, `temporal`, `events`, `agent`, `alerts`, `report`, `pipeline`, `evaluation`, `models`.

## Models and design choices

- Pose and tracking: Ultralytics `yolov8n-pose.pt` with `bytetrack.yaml`, detection confidence 0.25, CPU. Ultralytics fetches the weights on first use; `*.pt` files are not committed. Chosen because it is small, pretrained (no training from scratch) and fast enough on CPU.
- Why a deterministic pipeline first: every state has a recorded reason string, so a wrong answer can be traced. A VLM would be advisory only.
- Perception is cached to JSON, so classification and evaluation can be re-run in seconds without re-running the model.
- When several people are visible, perception keeps the person with the LARGEST box (`perception.py`). There is no logic for choosing the monitored person. This caused failure case 2.
- Trade-off: simple rules over learned classifiers. Easy to explain and debug; weak on unusual camera angles, occlusion and editing cuts.

## Install

Python 3.12, Windows PowerShell, CPU only. Measured with torch 2.14.0+cpu, opencv 5.0.0, ultralytics 8.4.164.

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest
```

41 tests passed when last run (before the documentation and ground-truth commits; code unchanged since).

## Run

Videos are NOT in this repository (see "Data provenance"). Put your own clip in `data\input\`.

```
python run_perception.py VIDEO OUT_JSON [--sample-fps 3] [--device cpu]
python select_roi.py VIDEO OUT_JSON          # drag a box over the bed, press ENTER
python run_timeline.py CACHE ROI             # writes data\output\<name>_report.json
python make_gt.py LABELS_TXT OUT_JSON        # build ground truth from a label file
python run_eval.py GT_DIR REPORT_DIR [--tol 5 --step 0.5]
```

The cached observations, ROI configs, reports and ground truth for every run clip are committed, so `run_timeline.py` and `run_eval.py` can be re-run without the videos.

## Configuration

- Bed ROI: `config/<clip>_roi.json` with key `bed_roi` = [x1, y1, x2, y2] in pixels. One fixed ROI per clip, which assumes a fixed camera.
- Thresholds are constants in the source, all untuned placeholders:
  - classifier: walking speed 0.25 body-sizes/s, motion window 2.0 s.
  - temporal minimum durations (s): LYING 2.0, SITTING_ON_BED 2.0, SITTING_OUTSIDE_BED 2.0, STANDING 1.0, WALKING 1.0, OUT_OF_BED 2.0, UNKNOWN 3.0.
  - agent: UNKNOWN segments of at least 5.0 s are reviewed; an away state must last 3.0 s to confirm an exit.

## Input and output formats

- Perception cache (`data/cache/<clip>.json`): keys `video, fps, width, height, duration_sec, sample_fps, model, conf, samples`. Each sample has `t, n_people, track_id, box, box_conf, keypoints`.
- Report (`data/output/<clip>_report.json`): keys `observation_duration_sec, activity_duration_sec, bed_exit_count, bed_return_count, total_in_bed_sec, total_out_of_bed_sec, unknown_sec, longest_out_of_bed_period_sec, final_state, bed_events, rejected_candidates, timeline, alerts, overall_decision, agent_traces`. The timeline is a list of `{start, end, state}`.
- `total_out_of_bed_sec` is the sum of time in away states. `longest_out_of_bed_period_sec` is event-based (confirmed exit to confirmed return). They can differ.
- Ground truth (`data/gt/<clip>.json`): `duration_sec`, `segments` (start, end, state), `bed_events` (bed_exit / return_to_bed with time), `notes`. Label files are in `data/gt/*_labels.txt`.

## State definitions

LYING_IN_BED, SITTING_ON_BED, SITTING_OUTSIDE_BED, STANDING, WALKING, OUT_OF_BED, UNKNOWN.

- WALKING only when posture is STANDING and speed is at least 0.25 body-sizes/s. Speed is box-centre displacement divided by the longer box side, per second, for the same track id. WALKING was never observed in any run.
- OUT_OF_BED only when posture is unclear and the person is away from the bed.
- No person detected gives UNKNOWN with reason `no_person_detected`.
- With more than one person in view, confidence is reduced (reason suffix `multi_person`) but a state is still emitted.
- Per-sample reasons seen in runs: `upright_and_still`, `sitting_hips_inside_bed_roi`, `lying_posture_inside_bed_roi`, `lying_but_not_on_bed`, `posture_unclear`, `no_person_detected`.

## Bed exit and return

`detect_bed_events` in `src/bedmonitor/events.py` turns the smoothed timeline into bed exit and return events and out-of-bed periods. The exact confirmation rules were not re-read when this README was written; read the file for them. `agent.py` defines a `min_away_sec` setting of 3.0 s (an away state must persist this long to confirm an exit); how that interacts with `events.py` was not verified. Unit tests are in `tests/test_events.py`. **Not validated on real footage.**

## Agent

`agent.py` is deterministic. It reviews UNKNOWN segments of at least 5 s and records a trace. An optional VLM client interface exists; its output is validated (known state, confidence 0..1) and stored in the trace only. It never changes the timeline or events. The default `NullVLM` returns nothing, so the system is offline and reproducible. No VLM backend was implemented or tested.

## Alert rules

| Rule | Level |
|---|---|
| prolonged_out_of_bed, 300 s | ALERT |
| prolonged_unknown, 60 s | MONITOR |
| prolonged_edge_sitting, 120 s | MONITOR |
| each confirmed bed exit | MONITOR |

There is no time-of-day input. The rules are covered by `tests/test_alerts.py` using synthetic timelines. On real clips every decision was NORMAL and ALERT never fired.

## Evaluation method

- Ground truth was hand-labelled from contact sheets (1 s spacing; 1.5 s for wakeup_sitting). Segment boundaries are uncertain by about 1 s.
- `run_eval.py` compares prediction and ground truth on a 0.5 s grid. UNKNOWN counts as wrong. Reported: state accuracy, confusion matrix, per-state precision/recall/F1, duration error, and for bed events precision, recall, false exits, missed events and time error (match tolerance 5 s).
- Thresholds were not tuned. The same clips were used to find the failures, so these numbers are not an estimate of generalisation.
- No pooled number is reported. Five clips, one of them trivial and two chosen because they fail, would give a meaningless average.

## Results (measured 2026-10-04)

| Clip | Footage | State accuracy | Notes |
|---|---|---|---|
| test2 | stock, lying, no movement | 1.00 (27/27) | Smoke test only |
| wakeup_sitting | fixed camera | 0.95 (39/41) | Sitting reported 1.2 s late; ground-truth boundary uncertain (16.5-18 s) |
| wakeup_sitting_b | fixed camera | 0.90 (27/30) | Final 1.6 s of sitting missed (sitting recall 0.00) |
| turning_bed | fixed camera, two people | 0.00 (0/25) | Failure case 2 |
| edge_sitting | edited footage | 0.00 (0/10) | Failure case 1 |

Perception speed on CPU: 0.07-0.14 s per sample after the first run (the first run was 0.75 s per sample; cause not isolated). Bed-exit and return metrics: **Not measured**.

Run but not scored (no ground truth or unsuitable footage): low_light (night vision with camera-angle changes), standby_sitback (close-up push-in), blanket_occlusion (AI-generated, watermark visible), full_exit and getin_getout (edited instructional videos with cuts), walking_room (a fitness video with no bed; excluded), test1 (perception cache only). Their timelines are not results. clip_chair_bed, clip_elder_chair and clip_leave_frame were never run.

## Failure cases

**1. edge_sitting: sitting called STANDING, then person lost (measured).**
Ground truth: sits on the bed the whole 5.2 s. Raw samples (16): 1 SITTING_ON_BED, 4 STANDING at confidence 0.8 (t=1.33-2.33, `upright_and_still`), 11 UNKNOWN (5 of them `no_person_detected` from t=3.67). Final report: 5.2 s STANDING and 0.0 s UNKNOWN. Evaluation: accuracy 0.00. The smoother hid the 11 UNKNOWN samples from the report.
Footage: the contact sheet shows camera-angle changes (front view, rear view, close front, close rear) and a subject wrapped in a duvet. Cause not isolated; candidates are the cuts, the blanket occlusion and close-up cropping (detected boxes were wider than tall). An earlier hypothesis that a front-facing close camera alone breaks the posture rule is not supported, because this clip is edited.

**2. turning_bed: the wrong person is monitored (measured; cause supported, not verified frame by frame).**
The patient lies on the bed the whole time while a physiotherapist stands beside it. Raw samples (38): 31 STANDING, 5 SITTING_ON_BED, 2 UNKNOWN, 0 LYING_IN_BED. Report: 12.7 s STANDING. Accuracy 0.00.
Evidence: the code keeps the largest box (`perception.py`); most kept boxes are roughly 450-550 px wide and 740-850 px tall, standing-person sized; 4 of the 5 SITTING samples have wide, bed-level boxes (720 x about 470 px) that are probably the patient, and the fifth is 691 x 671 px. `multi_person` only lowers confidence; it does not change which person is chosen.

**3. low_light: person mostly not classified (observed on the contact sheet only; no ground truth, not scored).**
The sheet shows a person lying from about 5 s. The report says UNKNOWN for all 15.05 s. 23 of 43 samples had a person: 6 LYING_IN_BED, 3 SITTING_ON_BED and 14 UNKNOWN, mostly `lying_but_not_on_bed`; the 3 SITTING samples disagree with the sheet, which shows her lying on her side. 20 samples had no person.
Likely causes: the single ROI does not match the changed camera framing, and correct LYING samples occur in runs shorter than the 2.0 s minimum, so the smoother never commits them. The camera angle changes during the clip and orb overlays appear. This is a hypothesis supported by the reason strings.

## Limitations

- Footage: mostly not elderly, mostly edited or third-party, several with cuts. Edited footage violates the fixed-camera, fixed-ROI assumption.
- Placeholder thresholds; no tuning; no held-out split.
- The smoother can hide UNKNOWN time from the report (failure case 1).
- A trailing segment shorter than its minimum duration is dropped at the end. STANDING commits faster (1.0 s) than SITTING (2.0 s). The 1.6 s sitting segment of wakeup_sitting_b is shorter than the sitting minimum, which is consistent with the miss; raw states were not inspected for that clip.
- No logic for choosing the monitored person (largest box wins).
- Speed uses box-centre displacement, so motion toward the camera may be missed (hypothesis, untested).
- Alerts cannot be demonstrated on clips shorter than the alert thresholds. No time-of-day logic.
- 3 fps sampling; CPU only.
- Untested: dim light with an elderly subject, leave-and-return, a second person entering, real bed exit/return.

## Not measured

Bed-exit precision, recall, false exits and exit/return time error; ALERT behaviour on real footage; ByteTrack ID switches; a VLM backend; threshold tuning; any held-out set; performance on elderly subjects.

## Requirement audit

| Requirement | Implemented? | Evidence / file | Notes |
|---|---|---|---|
| Person state classification | Yes | `classifier.py`, `temporal.py` | WALKING never observed; 2 of 5 scored clips fail |
| Bed exit / return | Implemented, not validated | `events.py`, `tests/test_events.py` | No real exit/return ground truth |
| Durations | Yes | report `activity_duration_sec` | Duration error measured on 5 clips |
| Timeline | Yes | report `timeline` | |
| NORMAL / MONITOR / ALERT | Yes | `alerts.py`, `tests/test_alerts.py` | All real runs NORMAL |
| Agentic component | Partial | `agent.py` | Deterministic; VLM interface only, no backend |
| Evaluation metrics | Yes | `evaluation.py`, `run_eval.py` | Exit/return metrics not measured |
| At least 3 real failure cases | Yes | section above | 2 scored, 1 contact-sheet only |
| Elderly footage | No | `data/sources.md` | Only edited instructional clips show older adults; not scored |
| Tests | Yes | `tests/` | 41 passed |

## Data provenance

See `data/sources.md`. test1 and test2 came from Pixabay and Pexels. The other 13 clips were downloaded through a third-party downloader site; their platform and licences are NOT verified, so they are not treated as licensed stock. Video files are not in this repository and are not redistributed. The repository contains derived data only (pose keypoints and boxes, reports, ROI configs, ground-truth labels); no frames or images are committed.

## Future work

Person selection (track the one on or nearest the bed), per-shot ROI or cut detection, tuning on a proper split, a VLM backend for UNKNOWN review, fixed-camera elderly footage with real exit/return ground truth, and a smoother that reports UNKNOWN time honestly.
