"""Plot Meta vs CPU aggregate pwrite/pread volume per 50 ms on real wall-clock time (fsync span included, not marked).

Usage: plot_realtime_io.py MEASUREMENTS_DIR {save|cold-load|warm-load} OUT.png TITLE
MEASUREMENTS_DIR holds r{N}-meta / r{N}-cpu repeat dirs; the repeat whose paired wall gap is closest to the median is drawn.
"""
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
mdir, phase, out, title = sys.argv[1:5]
ops = {"pwrite", "write"} if phase == "save" else {"pread", "read", "readinto"}
reps = [r for r in range(1, 6) if os.path.exists(f"{mdir}/r{r}-cpu/{phase}-result.json") and os.path.exists(f"{mdir}/r{r}-meta/{phase}-result.json")]
data = {r: {c: load(f"{mdir}/r{r}-{c}", phase, ops) for c in ("meta", "cpu")} for r in reps}
# representative repeat: paired Meta-CPU wall difference closest to the median difference
d = {r: data[r]["meta"][0] - data[r]["cpu"][0] for r in reps}
med = statistics.median(d.values()); rep = min(reps, key=lambda r: abs(d[r] - med))
fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
fig.subplots_adjust(left=.07, right=.98, top=.86, bottom=.13)
for c, label, color, ls in (("cpu", "CPU", "#1baf7a", "-"), ("meta", "Meta", "#2a78d6", "--")):
    wall, v = data[rep][c]
    ax.plot((np.arange(len(v)) + .5) * BIN, v, color=color, ls=ls, lw=1.3, label=f"{label}  {wall:.2f} s")
ax.set_xlim(0, max(data[rep][c][0] for c in data[rep]) * 1.02); ax.set_ylim(bottom=0)
ax.set_xlabel("Time (s)", fontsize=11)
ax.set_ylabel(f"{'Write' if phase == 'save' else 'Read'} volume (MB / 50 ms)", fontsize=11)
ax.set_title(title, loc="left", fontsize=15, pad=12)
ax.grid(color="#e5e5e5", lw=.5); ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=10.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
fig.savefig(out)
print(out, "repeat", rep, {c: round(data[rep][c][0], 2) for c in data[rep]})
