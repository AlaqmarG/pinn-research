"""Experiment 02: Adam vs zero-order training of the cooling PINN.

Run from the repository root, for example:
    python experiments/02-zo-vs-adam/run.py --config experiments/02-zo-vs-adam/configs/final.json \
        --output experiments/02-zo-vs-adam/results/final
Finished runs are skipped, so an interrupted sweep can be restarted.
"""

from __future__ import annotations

import argparse
import itertools
import json
import platform
import sys
import time
from pathlib import Path

import cma
import numpy as np
import torch

from pinnlab.zo import FD_H, METHODS, N_PARAMS, final_metrics


def expand(config: dict) -> list[dict]:
    """Turn a compact config into one entry per run.

    A config has "seeds" and "runs". In each run, "deriv" and any value inside "hp" may be
    a list, and every combination is run for every seed. "tag" can use hp values, like
    "tune-lr{lr}". Runs are ordered by seed, then by their order in the file.
    """
    out = []
    for seed in config["seeds"]:
        for run in config["runs"]:
            derivs = run["deriv"] if isinstance(run["deriv"], list) else [run["deriv"]]
            grid = {k: v if isinstance(v, list) else [v] for k, v in run["hp"].items()}
            for deriv in derivs:
                for values in itertools.product(*grid.values()):
                    hp = dict(zip(grid, values, strict=False))
                    out.append(
                        {
                            "method": run["method"],
                            "deriv": deriv,
                            "tag": run["tag"].format(**hp),
                            "seed": seed,
                            "budget": run["budget"],
                            "hp": hp,
                        }
                    )
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True, help="config file with seeds and runs (see expand)")
    ap.add_argument("--output", type=Path, help="folder for the results (not needed with --list)")
    ap.add_argument("--list", action="store_true", help="print the expanded runs and exit")
    args = ap.parse_args()
    torch.set_num_threads(1)
    runs = expand(json.loads(args.config.read_text()))
    if args.list:
        for r in runs:
            print(f"{r['method']}-{r['deriv']}-{r['tag']}-seed{r['seed']}", json.dumps(r["hp"]))
        return
    if args.output is None:
        ap.error("--output is required unless --list is given")
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "environment.json").write_text(
        json.dumps(
            {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "cma": cma.__version__,
                "torch_threads": torch.get_num_threads(),
                "dtype": "float64",
                "fd_h": FD_H,
                "n_params": N_PARAMS,
            },
            indent=2,
        )
    )
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
        metrics.update(
            method=r["method"],
            deriv=r["deriv"],
            tag=r["tag"],
            seed=r["seed"],
            budget=r["budget"],
            hp=r["hp"],
            selected_evals=tr.best_evals,
            validation_score=tr.best,
            seconds=seconds,
        )
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "history.json").write_text(json.dumps(tr.history))
        np.savetxt(
            out / "predictions.csv",
            predictions,
            delimiter=",",
            comments="",
            header="time_min,predicted_c,exact_c,absolute_error_c,residual_c_per_min",
        )
        print(
            json.dumps(
                {
                    k: metrics[k]
                    for k in (
                        "method",
                        "deriv",
                        "tag",
                        "seed",
                        "rmse_c",
                        "validation_score",
                        "selected_evals",
                        "seconds",
                    )
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
