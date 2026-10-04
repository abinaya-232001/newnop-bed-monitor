import json
import sys

cache_path = sys.argv[1]
with open(cache_path) as f:
    cache = json.load(f)

samples = cache["samples"]
print("top-level keys:", list(cache.keys()))
print("sample[0] keys:", list(samples[0].keys()))
print()

time_key = None
for k in ("t", "time", "t_sec", "timestamp", "time_sec"):
    if k in samples[0]:
        time_key = k
        break

for i, s in enumerate(samples):
    t = s.get(time_key) if time_key else None
    tstr = "%.2f" % t if isinstance(t, (int, float)) else str(t)
    print("%2d  t=%-6s  n_people=%s" % (i, tstr, s.get("n_people")))
