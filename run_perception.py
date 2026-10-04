"""Usage: python run_perception.py VIDEO OUT_JSON [--sample-fps 3]"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from bedmonitor.perception import run_perception  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

p = argparse.ArgumentParser()
p.add_argument("video")
p.add_argument("out_json")
p.add_argument("--sample-fps", type=float, default=3.0)
p.add_argument("--device", default="cpu")
args = p.parse_args()

cache = run_perception(args.video, args.out_json, args.sample_fps, device=args.device)
seen = sum(1 for s in cache["samples"] if s["n_people"] > 0)
print(f"{len(cache['samples'])} samples, person seen in {seen}; wrote {args.out_json}")