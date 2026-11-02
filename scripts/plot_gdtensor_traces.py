"""Plot analyzer output with one clean request-volume chart per phase and rank.

Example: python scripts/plot_gdtensor_traces.py RUN/analysis --output RUN/plots
Save plots exclude fsync; cold/warm loads retain the full checkpoint-call axis.
"""

import argparse
import json
import os
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_gdtensor_traces import group_key  # noqa: E402

LABELS = {"meta": "Meta", "meta-sim": "Meta (simulated transfers)", "cpu": "CPU", "gpu": "GPU"}
COLORS = {"meta": "#8067ad", "meta-sim": "#2a78d6", "cpu": "#eb6834", "gpu": "#1baf7a"}


def representative(cases):
    target = statistics.median(c["bins"]["duration_s"] for c in cases)
    return min(cases, key=lambda c: (abs(c["bins"]["duration_s"] - target), c["row"]["repeat"]))


def plot_group(cases, output):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/metasl-trace-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = defaultdict(list)
    for case in cases:
        groups[(case["row"].get("run", ""), case["row"]["config"])].append(case)
    multiple_runs = len({key[0] for key in groups}) > 1
    row = cases[0]["row"]
    fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
    fig.subplots_adjust(left=.07, right=.98, top=.80, bottom=.13)
    representatives = {}
    for (run, config), selected in sorted(groups.items()):
        case = representative(selected)
        bins = case["bins"]
        values = [v / 1e6 for v in bins["binned_bytes_per_50ms"]]
        xs = [(i + .5) * .05 for i in range(len(values))]
        label = LABELS.get(config, config)
        if multiple_runs:
            label = f"{Path(run).name} {label}"
        ax.plot(xs, values, color=COLORS.get(config), ls="--" if config.startswith("meta") else "-", lw=1.3, label=f"{label}  {bins['duration_s']:.2f} s")
        representatives[f"{run}:{config}"] = dict(output=case["row"]["output"], repeat=case["row"]["repeat"], duration_s=bins["duration_s"], **case["provenance"])
    ax.set_xlim(0, max(c["bins"]["duration_s"] for c in cases) * 1.02)
    ax.set_ylim(bottom=0)
    save = row["phase"] == "save"
    ax.set_xlabel("Time (s, fsync excluded)" if save else "Time (s)", fontsize=11)
    ax.set_ylabel(f"{'Write' if save else 'Read'} volume (MB / 50 ms)", fontsize=11)
    format_label = "DCP" if row["checkpoint_format"] == "dcp" else "safetensors"
    title = f"{row['model'].upper()} {format_label} {row['phase'].replace('-', ' ')}"
    if row["world_size"] > 1:
        title += f", rank {row['rank']}"
    ax.set_title(title, loc="left", fontsize=15, pad=39)
    ax.grid(color="#e5e5e5", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=min(4, len(groups)))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)
    return dict(path=str(output), representatives=representatives, fsync_excluded=save, bin_ms=50)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path, help="analysis.json or its parent directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--phases", nargs="+", choices=("save", "cold-load", "warm-load", "load"))
    parser.add_argument("--ranks", nargs="+", type=int)
    args = parser.parse_args(argv)
    source = args.analysis / "analysis.json" if args.analysis.is_dir() else args.analysis
    groups = defaultdict(list)
    for case in json.loads(source.read_text())["cases"]:
        row = case["row"]
        if args.models and row["model"] not in args.models or args.phases and row["phase"] not in args.phases or args.ranks and row["rank"] not in args.ranks:
            continue
        groups[group_key(row) + (row["rank"],)].append(case)
    plots = []
    for index, (_, cases) in enumerate(sorted(groups.items(), key=lambda item: str(item[0]))):
        row = cases[0]["row"]
        name = f"{index:02d}-{row['model']}-{row['checkpoint_format']}-{row['phase']}-r{row['rank']:03d}.png"
        plots.append(plot_group(cases, args.output / name))
    if not plots:
        raise ValueError("No matching cases to plot")
    (args.output / "plots.json").write_text(json.dumps(dict(plots=plots), indent=2) + "\n")
    print(json.dumps(dict(plots=len(plots), output=str(args.output))))


if __name__ == "__main__":
    main()
