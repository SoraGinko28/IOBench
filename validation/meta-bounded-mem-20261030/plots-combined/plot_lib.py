import json, sys, os, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
BIN = 0.05
def load(rep_dir, phase, ops):
    res = json.load(open(os.path.join(rep_dir, f"{phase}-result.json")))
    evs = []
    for r in res:
        base = r["export"]["first_event_absolute_ns"]
        for l in open(r["export"]["output_path"]):
            e = json.loads(l); e["a"] = base + e["start_ns"]; e["b"] = e["a"] + e["duration_ns"]; evs.append(e)
    t0 = min(e["a"] for e in evs); t1 = max(e["b"] for e in evs)
    wall = (t1 - t0) / 1e9
    v = np.zeros(int(np.ceil(wall / BIN)) + 1)
    for e in evs:
        if e["op"] not in ops or not e.get("completed"): continue
        a, b = (e["a"] - t0) / 1e9, (e["b"] - t0) / 1e9
        ia, ib = int(a / BIN), int(b / BIN)
        if ia == ib: v[ia] += e["completed"]; continue
        rate = e["completed"] / (b - a)  # spread bytes evenly over the call
        for i in range(ia, ib + 1):
            lo, hi = max(a, i * BIN), min(b, (i + 1) * BIN)
            if hi > lo: v[i] += rate * (hi - lo)
    return wall, v / 1e6
