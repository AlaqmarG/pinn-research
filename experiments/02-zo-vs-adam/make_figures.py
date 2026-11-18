"""Summarise experiment 02: LaTeX table, summary.json and the convergence figure.

Usage (from the repository root):
    python experiments/02-zo-vs-adam/make_figures.py experiments/02-zo-vs-adam/results/final \
        experiments/02-zo-vs-adam/figures
"""
import json
import statistics as st
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

ORDER = [("adam", "autograd", "base", "Adam, 6,000 steps"),
         ("adam", "autograd", "long", "Adam, 30,000 steps"),
         ("zoadam", "exact", "final", "ZO-Adam, exact $du/d\\tau$"),
         ("zoadam", "fd", "final", "ZO-Adam, finite differences"),
         ("cmaes", "exact", "final", "CMA-ES, exact $du/d\\tau$"),
         ("cmaes", "fd", "final", "CMA-ES, finite differences")]
PLAIN = {k[:3]: k[3].replace("$du/d\\tau$", "du/dτ") for k in ORDER}
COLORS = {"adam": "#2a78d6", "zoadam": "#eb6834", "cmaes": "#1baf7a"}
STYLE = {"autograd": "-", "exact": "-", "fd": "--"}

runs = {}
for d in sorted(src.iterdir()):
    m = d / "metrics.json"
    if m.exists():
        metrics = json.loads(m.read_text())
        metrics["history"] = json.loads((d / "history.json").read_text())
        runs.setdefault((metrics["method"], metrics["deriv"], metrics["tag"]), []).append(metrics)


def fmt(x):
    return f"{x:.1e}".replace("e-0", "e-").replace("e+0", "e")


rows, summary = [], {}
for method, deriv, tag, label in ORDER:
    rs = sorted(runs.get((method, deriv, tag), []), key=lambda r: r["seed"])
    if not rs:
        continue
    rmse = [r["rmse_c"] for r in rs]
    mx = [r["max_abs_error_c"] for r in rs]
    val = [r["validation_score"] for r in rs]
    secs = [r["seconds"] for r in rs]
    evals = rs[0]["budget"]
    unit = "grad." if method == "adam" else "loss"
    summary[f"{method}-{deriv}-{tag}"] = {"seeds": [r["seed"] for r in rs], "rmse_c": rmse,
                                          "max_abs_error_c": mx, "validation_score": val,
                                          "seconds": secs, "budget": evals, "hp": rs[0]["hp"]}
    rows.append(f"{label} & {evals:,} {unit} & {st.median(rmse):.4f} [{min(rmse):.4f}, {max(rmse):.4f}] & "
                f"{st.median(mx):.4f} & {fmt(st.median(val))} & {st.median(secs):.0f} \\\\")

table = [r"\begin{tabular}{lrrrrr}", r"\hline",
         r"Method & Budget & RMSE ($^\circ$C), median [min, max] & Max.\ error ($^\circ$C) & Validation loss & Time (s) \\",
         r"\hline", *rows, r"\hline", r"\end{tabular}"]
(dst / "results_table.tex").write_text("\n".join(table) + "\n", encoding="utf-8")
(src / "summary.json").write_text(json.dumps(summary, indent=2))

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 9, "axes.edgecolor": "#5f5e58", "xtick.color": "#5f5e58",
                     "ytick.color": "#5f5e58", "axes.linewidth": 0.6})
fig, (a, b) = plt.subplots(1, 2, figsize=(6.3, 2.7))
for method, deriv, tag, _ in ORDER:
    rs = runs.get((method, deriv, tag), [])
    if not rs or (method == "adam" and tag == "base"):
        continue
    key = (method, deriv, tag)
    for i, r in enumerate(rs):
        h = r["history"]
        y = []
        best = float("inf")
        for row in h:
            best = min(best, row["validation_score"])
            y.append(best)
        kw = dict(color=COLORS[method], linestyle=STYLE[deriv], linewidth=1.1, alpha=0.9,
                  label=PLAIN[key] if i == 0 else None)
        a.semilogy([row["seconds"] for row in h], y, **kw)
        if method != "adam":
            b.semilogy([row["evals"] for row in h], y, **kw)
a.set(xlabel="Wall-clock time (s)", ylabel="Best validation loss so far")
a.set_title("(a) Against time (all methods)", fontsize=9, loc="left")
b.set(xlabel="Loss evaluations")
b.set_title("(b) Against loss evaluations (ZO only)", fontsize=9, loc="left")
for ax in (a, b):
    ax.grid(color="#e4e3df", linewidth=0.5)
    ax.spines[["top", "right"]].set_visible(False)
handles, labels = a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.13, 1, 1))
fig.savefig(dst / "convergence.pdf")
fig.savefig(dst / "convergence.png", dpi=200)
print("\n".join(table))
