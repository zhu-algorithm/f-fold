"""Figures for the F-Fold project page."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def fig_teacher_vs_student(npz, metrics, out):
    d = np.load(npz)
    dt, dp = d["d_true"], d["d_pred"]
    pt, pp = d["plddt_true"], d["plddt_pred"]
    L = len(pt)

    fig = plt.figure(figsize=(7.2, 4.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.0, 1.0], hspace=0.55,
                          wspace=0.28)
    ax0, ax1 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    vmax = float(np.percentile(dt, 99))
    for ax, m, t in ((ax0, dt, "AlphaFold teacher"), (ax1, dp, "F-Fold student")):
        im = ax.imshow(m, cmap="viridis", vmin=0, vmax=vmax, origin="lower",
                       interpolation="nearest")
        ax.set_title(t)
        ax.set_xlabel("residue")
        ax.set_xticks([0, 50, 100, L - 1])
        ax.set_yticks([0, 50, 100, L - 1])
    ax0.set_ylabel("residue")
    ax1.set_yticklabels([])
    cb = fig.colorbar(im, ax=[ax0, ax1], fraction=0.035, pad=0.03)
    cb.set_label("Cβ–Cβ distance (Å)")

    ax2 = fig.add_subplot(gs[1, :])
    x = np.arange(1, L + 1)
    ax2.plot(x, pt, color="#111111", lw=1.4, label="AlphaFold teacher")
    ax2.plot(x, pp, color="#c2410c", lw=1.4, label="F-Fold student")
    ax2.set_xlabel("residue")
    ax2.set_ylabel("pLDDT")
    ax2.set_xlim(1, L)
    ax2.set_ylim(0, 100)
    ax2.set_title(f"Per-residue confidence tracks the teacher "
                  f"(Pearson r = {metrics['plddt_pearson']:.3f})")
    ax2.legend(frameon=False, loc="lower right", ncols=2)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def fig_params(csv_path, out):
    rows = list(csv.DictReader(open(csv_path)))
    rows = [r for r in rows if r["module"] != "TOTAL"]
    rows = [r for r in rows if int(r["ffold_base"]) > 0]
    rows.sort(key=lambda r: int(r["ffold_base"]))
    names = [r["module"].replace("_", " ") for r in rows]
    base = np.array([int(r["ffold_base"]) for r in rows], float)
    mini = np.array([int(r["ffold_mini"]) for r in rows], float)
    y = np.arange(len(rows))

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    present = mini > 0
    ax.hlines(y[present], mini[present], base[present], color="#cccccc", lw=1.2,
              zorder=1)
    ax.scatter(base, y, s=34, color="#111111", zorder=3, label="ffold-base (368.0M)")
    ax.scatter(mini[present], y[present], s=34, color="#c2410c", zorder=3,
               label="ffold-mini (8.8M)")
    # modules ffold-mini does not instantiate: mark the slot, don't imply a value
    for yi, bi in zip(y[~present], base[~present]):
        ax.annotate("not in mini", (bi, yi), xytext=(-8, 0),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=7, color="#c2410c", style="italic")
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xscale("log")
    ax.set_xlabel("parameters")
    ax.set_xlim(5e2, 5e8)
    ax.set_title("Where the parameters are, at AF3 dimensions")
    ax.legend(frameon=False, loc="lower right")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color="#eeeeee", lw=0.6)
    ax.set_axisbelow(True)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    return fig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="../results")
    ap.add_argument("--acc", default="P61626")
    ap.add_argument("--outdir", default="../repo/docs/assets")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)
    res = Path(args.results)
    metrics = json.loads((res / f"metrics_{args.acc}.json").read_text())

    try:
        from kernel import apply_figure_style  # type: ignore
        apply_figure_style(sizes=(9, 8, 7))
    except Exception:
        mpl.rcParams.update({"font.size": 9, "axes.titlesize": 9,
                             "axes.labelsize": 9, "xtick.labelsize": 7,
                             "ytick.labelsize": 7, "legend.fontsize": 8,
                             "figure.facecolor": "white",
                             "savefig.facecolor": "white"})

    f1 = fig_teacher_vs_student(res / f"ffold_{args.acc}_pred.npz", metrics,
                                out / "teacher_vs_student.png")
    f2 = fig_params("param_audit.csv", out / "param_audit.png")
    print("wrote", out / "teacher_vs_student.png", out / "param_audit.png")

    for name, fig in (("teacher_vs_student", f1), ("param_audit", f2)):
        r = fig.canvas.get_renderer()
        texts = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
                 if t.get_text().strip() and t.get_visible()]
        tick = {ax: set(ax.get_xticklabels() + ax.get_yticklabels())
                for ax in fig.axes}
        spines = [(s, s.get_window_extent(r)) for ax in fig.axes
                  for s in ax.spines.values() if s.get_visible()]
        ov = [(a.get_text(), b.get_text())
              for i, (a, ba) in enumerate(texts) for b, bb in texts[i + 1:]
              if ba.overlaps(bb)]
        ov += [(t.get_text(), "spine") for t, bt in texts for s, bs in spines
               if bt.overlaps(bs) and t not in tick.get(s.axes, set())]
        print(f"{name}: {len(ov)} overlaps", ov[:6])


if __name__ == "__main__":
    main()
