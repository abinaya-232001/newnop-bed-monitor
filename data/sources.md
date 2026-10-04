# Video sources and licences

Provenance below was recovered on 2026-10-04 from the download metadata that Windows stores on each file. No footage was recorded by the author.

- test1.mp4 and test2.mp4 came from stock sites (Pixabay, Pexels).
- The other 13 clips were downloaded through the third-party downloader savefrom.net. The IDs below are the media IDs embedded in those download links. The originating platform and each uploader's licence have NOT been verified, so these clips are NOT treated as licensed stock footage.
- The video files are not in this repository (see .gitignore) and are not redistributed. The repository does contain derived data only: pose keypoints and boxes (data/cache), run reports (data/output), ROI configs (config) and ground-truth labels (data/gt). No frames or images are committed.
- Subjects are mostly NOT elderly. See the last column.

| Filename | Origin | ID or URL | Licence | Date downloaded | What it shows | Subject elderly? |
|---|---|---|---|---|---|---|
| test1.mp4 | Pixabay | File URL: https://cdn.pixabay.com/video/2025/08/17/297986.mp4 (page URL not recorded) | Pixabay site licence; not re-checked on the page | Not recorded | Person on chair, torso cut off | Not recorded |
| test2.mp4 | Pexels | File URL: https://videos.pexels.com/video-files/6689161/6689161-hd_1080_1920_25fps.mp4 (page URL not recorded) | Pexels site licence; not re-checked on the page | Not recorded | Lying under blanket, side-on, 13.6s | Not recorded |
| clip_edge_sitting.mp4 | savefrom.net | id=ulAyuTA3EiU, title: "POV: sitting on the edge of the bed for 25 minutes because you're tired #POV #relatable #Shorts" | Not verified | 2026-10-04 | Person sitting on bed edge, front-facing, 5.2s | Not verified |
| clip_blanket_occlusion.mp4 | savefrom.net | id=_QDM4yHYaVc, title: "Tossing & Turning All Night? This Weighted Blanket Helps" | Not verified | 2026-10-04 | Sitting then lying under blanket | Not verified |
| clip_standby_sitback.mp4 | savefrom.net | id=J5k_I1oYL7E, title: "Person sitting on bed breathing" | Not verified | 2026-10-04 | Sitting on bed | Not verified |
| clip_turning_bed.mp4 | savefrom.net | id=Lny5RrOxFdE, title: "Bed Mobility Training (Rolling side-to-side) by Physiotherapist" | Not verified | 2026-10-04 | Rolling in bed (training video) | Not verified |
| clip_wakeup_sitting.mp4 | savefrom.net | id=9PvK9W-3DmY, title: "How to Help Older Person Get Out of Bed Easily" | Not verified | 2026-10-04 | Lying then sitting up (instructional) | Not verified |
| clip_wakeup_sitting_b.mp4 | savefrom.net | id=zDg3t7wFhb0, title: "How to get out of bed with #backpain! Make sure you don't twist!! #backpainrelief" | Not verified | 2026-10-04 | Lying in bed (instructional) | Not verified |
| clip_low_light.mp4 | savefrom.net | id=lqShybgV2d0, title: "When I fall asleep paranormal activity begins. #paranormal #scary #orbs #occult #ghost #soul" | Not verified | 2026-10-04 | Dark bedroom footage, entertainment video | Not verified |
| clip_full_exit.mp4 | savefrom.net | id=iALyaq2E1T8, title: "Getting out of Bed - Exercise for Older Adults" | Not verified | 2026-10-04 | Edited instructional video, older woman | Appears older adult; not verified. Run, not scored: edited footage |
| clip_getin_getout.mp4 | savefrom.net | id=ah9INsRxvQA, title: "How to Safely Get In and Out of Bed - Step-by-Step Guide for Seniors and Mobility Aid Users" | Not verified | 2026-10-04 | Edited instructional video, older man with walking frame | Appears older adult; not verified. Run, not scored: edited footage |
| clip_walking_room.mp4 | savefrom.net | id=TsQqjoe0QkE, title: "Beginner Room Walk Indoor" | Not verified | 2026-10-04 | Fitness video, living room, no bed | No. Run, excluded from scoring |

## Downloaded but not run

These three are in data/input but were never processed (no cache, ROI or report) and are not used in any result.

- clip_chair_bed.mp4: id=6BPEotZIUOQ, "Lifting from Wheelchair to bed"
- clip_elder_chair.mp4: id=CfiqVEFckUE, "Power Lift Recliners for Seniors with Limited mobility"
- clip_leave_frame.mp4: id=6m3VgIGaxO4, "Parents Leave 5 Year Old Home Alone" (not relevant to the task)

## Clip types searched but not found (documented as untested)

- Dim-light footage of an elderly person, a second person entering the scene, and leave-and-return sequences were not tested.

## Notes

- Ground truth exists only for the clips listed in data/gt/.
- Results are reported per clip. Any pooled number covers only fixed-camera clips with confirmed ground truth and the README names which clips are in it.
