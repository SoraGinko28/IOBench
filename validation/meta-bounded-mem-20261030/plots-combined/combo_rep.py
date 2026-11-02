import sys, os, statistics, importlib.util
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
spec = importlib.util.spec_from_file_location("p", "/tmp/plot-shared-save/plot_lib.py"); p = importlib.util.module_from_spec(spec); spec.loader.exec_module(p)
mdir, out, title = sys.argv[1:4]
panels = (("save", {"pwrite", "write"}, "Save", "Write"), ("cold-load", {"pread", "read", "readinto"}, "Cold load", "Read"))
fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), dpi=150)
fig.subplots_adjust(left=.075, right=.98, top=.88, bottom=.08, hspace=.42)
fig.suptitle(title, x=.075, ha="left", fontsize=16)
for ax, (phase, ops, name, kind) in zip(axes, panels):
    reps = [r for r in range(1, 6) if os.path.exists(f"{mdir}/r{r}-cpu/{phase}-result.json")]
    data = {r: {c: p.load(f"{mdir}/r{r}-{c}", phase, ops) for c in ("meta", "cpu")} for r in reps}
    d = {r: data[r]["meta"][0] - data[r]["cpu"][0] for r in reps}
    med = statistics.median(d.values()); rep = int(os.environ["REP"]) if os.environ.get("REP") else min(reps, key=lambda r: abs(d[r] - med))
    for c, label, color, ls in (("cpu", "CPU", "#1baf7a", "-"), ("meta", "Meta", "#2a78d6", "--")):
        wall, v = data[rep][c]
        ax.plot((np.arange(len(v)) + .5) * p.BIN, v, color=color, ls=ls, lw=1.2, label=f"{label}  {wall:.2f} s")
    ax.set_xlim(0, max(data[rep][c][0] for c in data[rep]) * 1.01); ax.set_ylim(bottom=0)
    ax.set_title(name, loc="left", fontsize=13)
    ax.set_xlabel("Time (s)", fontsize=10.5); ax.set_ylabel(f"{kind} (MB / 50 ms)", fontsize=10.5)
    ax.grid(color="#e5e5e5", lw=.5); ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
    print(out, phase, "repeat", rep, {c: round(data[rep][c][0], 2) for c in data[rep]})
fig.savefig(out)
