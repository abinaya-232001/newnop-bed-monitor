import glob
import os
import cv2

folder = r"C:\dev\newnop-bed-monitor\data\input"
files = []
for ext in ("*.mp4", "*.mov", "*.webm", "*.avi", "*.mkv"):
    files += glob.glob(os.path.join(folder, ext))
files = sorted(files)

if not files:
    print("No video files found in", folder)

for f in files:
    cap = cv2.VideoCapture(f)
    if not cap.isOpened():
        print(os.path.basename(f), "-> COULD NOT OPEN")
        continue
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = (n / fps) if fps else 0.0
    print("%-40s %6.1fs  %5.1f fps  %dx%d" % (os.path.basename(f), dur, fps, w, h))
    cap.release()
