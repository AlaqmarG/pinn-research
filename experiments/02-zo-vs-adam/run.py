"""Experiment 02: Adam vs zero-order training of the cooling PINN.

Run from the repository root, for example:
    python experiments/02-zo-vs-adam/run.py --config experiments/02-zo-vs-adam/configs/final.json \
        --output experiments/02-zo-vs-adam/results/final
Finished runs are skipped, so an interrupted sweep can be restarted.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cma
import numpy as np
import torch

from pinnlab.zo import FD_H, METHODS, N_PARAMS, final_metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="JSON list of runs")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    runs = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(json.dumps({
        "python": sys.version, "platform": platform.platform(), "torch": torch.__version__,
        "numpy": np.__version__, "cma": cma.__version__, "threads": 1, "dtype": "float64",
        "fd_h": FD_H, "n_params": N_PARAMS}, indent=2))
    for r in runs:
        name = f"{r['method']}-{r['deriv']}-{r['tag']}-seed{r['seed']}"
        out = args.output / name
        if (out / "metrics.json").exists():
            print("skip", name, flush=True)
            continue
        out.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        tr = METHODS[r["method"]](r["seed"], r["budget"], r["hp"], r["deriv"])
        seconds = time.perf_counter() - start
        metrics, predictions = final_metrics(tr.best_theta)
        metrics.update(method=r["method"], deriv=r["deriv"], tag=r["tag"], seed=r["seed"],
                       budget=r["budget"], hp=r["hp"], selected_evals=tr.best_evals,
                       validation_score=tr.best, seconds=seconds)
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "history.json").write_text(json.dumps(tr.history))
        np.savetxt(out / "predictions.csv", predictions, delimiter=",", comments="",
                   header="time_min,predicted_c,exact_c,absolute_error_c,residual_c_per_min")
        print(json.dumps({k: metrics[k] for k in ("method", "deriv", "tag", "seed", "rmse_c",
                                                   "validation_score", "selected_evals", "seconds")}), flush=True)


if __name__ == "__main__":
    main()
