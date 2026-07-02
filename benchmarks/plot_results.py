"""Plot benchmark results: runtime and throughput vs. sequence length, and a
roofline chart of achieved TFLOP/s vs. arithmetic intensity.

    python benchmarks/plot_results.py results/results.csv --device L4-bf16-tensor
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (peak FLOP/s, peak HBM bytes/s) from published specs.
DEVICE_PEAKS = {
    "T4-fp32": (8.1e12, 300e9),
    "T4-fp16-tensor": (65e12, 300e9),
    "L4-fp32": (30.3e12, 300e9),
    "L4-bf16-tensor": (121e12, 300e9),
    "TPUv5e-bf16": (197e12, 819e9),
}

STYLE = {"naive": ("tab:red", "o"), "xla": ("tab:orange", "s"),
         "pallas": ("tab:blue", "^")}


def load(path: Path):
    rows = []
    with path.open() as f:
        header_comment = f.readline().strip("# \n")
        for row in csv.DictReader(f):
            if row["error"]:
                continue
            rows.append(dict(impl=row["impl"], seq_len=int(row["seq_len"]),
                             time_ms=float(row["time_ms"]),
                             tflops=float(row["tflops_per_s"]),
                             intensity=float(row["arithmetic_intensity"])))
    return rows, header_comment


def by_impl(rows):
    d = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["seq_len"]):
        d[r["impl"]].append(r)
    return d


def line_plot(groups, ykey, ylabel, title, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    for impl, rows in groups.items():
        color, marker = STYLE.get(impl, ("gray", "x"))
        ax.plot([r["seq_len"] for r in rows], [r[ykey] for r in rows],
                color=color, marker=marker, label=impl)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def roofline_plot(groups, device, out):
    peak_flops, peak_bw = DEVICE_PEAKS[device]
    ridge = peak_flops / peak_bw
    fig, ax = plt.subplots(figsize=(7, 5))
    xs = [2 ** (i / 4) for i in range(-8, 64)]
    ax.plot(xs, [min(peak_flops, peak_bw * x) / 1e12 for x in xs],
            "k-", lw=2, label=f"{device} roofline")
    ax.axvline(ridge, color="gray", ls=":", lw=1)
    ax.annotate(f"ridge point\nAI={ridge:.0f}", (ridge, peak_flops / 1e12 * 0.05),
                fontsize=8, color="gray")
    for impl, rows in groups.items():
        color, marker = STYLE.get(impl, ("gray", "x"))
        ax.scatter([r["intensity"] for r in rows], [r["tflops"] for r in rows],
                   color=color, marker=marker, label=impl, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("arithmetic intensity (FLOPs / HBM byte)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title(f"Roofline — {device}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("csv_path", type=Path)
    p.add_argument("--device", choices=DEVICE_PEAKS, default="L4-bf16-tensor")
    p.add_argument("--out-dir", type=Path, default=None,
                   help="defaults to the CSV's directory")
    args = p.parse_args()

    rows, meta = load(args.csv_path)
    if not rows:
        raise SystemExit("no successful rows in CSV")
    out_dir = args.out_dir or args.csv_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = by_impl(rows)

    line_plot(groups, "time_ms", "median time (ms)",
              f"Attention runtime — {meta}", out_dir / "runtime_vs_seqlen.png")
    line_plot(groups, "tflops", "achieved TFLOP/s",
              f"Attention throughput — {meta}", out_dir / "throughput_vs_seqlen.png")
    roofline_plot(groups, args.device, out_dir / "roofline.png")


if __name__ == "__main__":
    main()
