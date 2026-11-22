"""Experiment 01: physics-only forward cooling PINN trained with Adam.

Exact temperatures are used only for final evaluation, after checkpoint selection.
Run from the repository root:  python experiments/01-cooling-adam/run.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from pinnlab.cooling import Config, CoolingNet, evaluate, load_checkpoint, losses, write_csv, write_json


def plot_run(folder: Path, history: list[dict], predictions: np.ndarray, seed: int) -> None:
    t, pred, exact, error, _ = predictions.T
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, exact, label="Exact solution", color="black", linewidth=2)
    ax.plot(t, pred, label="PINN", color="#0072B2", linestyle="--")
    ax.set(xlabel="Time (min)", ylabel="Temperature (°C)", title=f"Cooling: seed {seed}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "temperature_overlay.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, error, color="#D55E00")
    ax.set(xlabel="Time (min)", ylabel="Absolute temperature error (°C)", title=f"Final evaluation: seed {seed}")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "absolute_error.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [
        ("train_physics_mse", "Training residual"),
        ("train_ic_mse", "Initial condition"),
        ("validation_score", "Validation residual + IC"),
    ]:
        ax.semilogy([row["step"] for row in history], [max(row[key], 1e-20) for row in history], label=label)
    ax.set(xlabel="Adam step", ylabel="Dimensionless squared loss", title=f"Physics-only training: seed {seed}")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(folder / "losses.png", dpi=180)
    plt.close(fig)


def run_seed(seed: int, c: Config, output: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    folder = output / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    # Training and validation times come from separate generators and stay fixed for the whole run.
    train = torch.rand((c.n_train, 1), generator=torch.Generator().manual_seed(seed + 10000), dtype=torch.float64)
    validation = torch.rand(
        (c.n_validation, 1), generator=torch.Generator().manual_seed(seed + 20000), dtype=torch.float64
    )
    test = np.linspace(0, 1, c.n_test)
    assert not np.intersect1d(train.numpy(), validation.numpy()).size
    assert not np.intersect1d(train.numpy(), test).size
    assert not np.intersect1d(validation.numpy(), test).size
    np.savetxt(folder / "train_times_min.csv", train.numpy() * c.end_min, delimiter=",", header="time_min", comments="")
    np.savetxt(
        folder / "validation_times_min.csv",
        validation.numpy() * c.end_min,
        delimiter=",",
        header="time_min",
        comments="",
    )
    model = CoolingNet(c)
    optimizer = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    best_score, best_step, best_state = float("inf"), 0, None
    history = []
    start = time.perf_counter()
    for step in range(c.steps + 1):
        if step % c.validation_every == 0 or step == c.steps:
            p, ic = losses(model, train, c, create_graph=False)
            vp, vic = losses(model, validation, c, create_graph=False)
            score = float(vp.detach() + c.ic_weight * vic.detach())
            history.append(
                {
                    "step": step,
                    "train_physics_mse": p.item(),
                    "train_ic_mse": ic.item(),
                    "validation_physics_mse": vp.item(),
                    "validation_ic_mse": vic.item(),
                    "validation_score": score,
                }
            )
            if score < best_score:
                best_score, best_step = score, step
                best_state = copy.deepcopy(model.state_dict())
        if step == c.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        p, ic = losses(model, train, c)
        (p + c.ic_weight * ic).backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    checkpoint_path = folder / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": best_state,
            "config": asdict(c),
            "seed": seed,
            "selected_step": best_step,
            "validation_score": best_score,
            "selection_rule": "minimum validation residual MSE + ic_weight * IC MSE; no exact labels",
        },
        checkpoint_path,
    )
    reloaded, loaded_config, _ = load_checkpoint(checkpoint_path)
    probe = torch.linspace(0, 1, 101, dtype=torch.float64)[:, None]
    with torch.no_grad():
        reload_difference = float((model(probe) - reloaded(probe)).abs().max())
    assert reload_difference == 0.0 and loaded_config == c
    metrics, predictions = evaluate(reloaded, c)
    metrics.update(
        seed=seed,
        selected_step=best_step,
        validation_score=best_score,
        training_seconds=elapsed,
        reload_max_abs_difference_dimensionless=reload_difference,
        checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),
    )
    write_json(folder / "config.json", asdict(c))
    write_json(folder / "metrics.json", metrics)
    write_csv(folder / "history.csv", history)
    np.savetxt(
        folder / "predictions.csv",
        predictions,
        delimiter=",",
        comments="",
        header="time_min,predicted_c,exact_c,absolute_error_c,residual_c_per_min",
    )
    plot_run(folder, history, predictions, seed)
    print(json.dumps(metrics), flush=True)
    return metrics


def summarize(results: list[dict], output: Path, c: Config) -> None:
    keys = ["mae_c", "rmse_c", "max_abs_error_c", "initial_abs_error_c", "test_residual_rms_c_per_min"]
    aggregate = {
        key: {
            "mean": float(np.mean([r[key] for r in results])),
            "sample_std": float(np.std([r[key] for r in results], ddof=1)) if len(results) > 1 else None,
            "min": min(r[key] for r in results),
            "max": max(r[key] for r in results),
        }
        for key in keys
    }
    write_json(output / "summary.json", {"runs": results, "aggregate": aggregate})
    lines = [
        "# Cooling PINN results",
        "",
        "These runs solve one forward cooling problem on 0–10 minutes.",
        "",
        f"Each seed used {c.steps} Adam steps. Checkpoints were selected using validation residual loss "
        "plus the soft initial-condition loss. Exact temperatures were used only after selection.",
        "",
        "| Seed | Selected step | MAE (°C) | RMSE (°C) | Max error (°C) | IC error (°C) | Residual RMS (°C/min) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        cells = [str(r["seed"]), str(r["selected_step"])]
        cells += [f"{r[k]:.6g}" for k in ("mae_c", "rmse_c", "max_abs_error_c", "initial_abs_error_c")]
        cells.append(f"{r['test_residual_rms_c_per_min']:.6g}")
        lines.append("| " + " | ".join(cells) + " |")
    lines.extend(["", "Across seeds (sample standard deviation uses n−1):", ""])
    for key in keys:
        a = aggregate[key]
        spread = f"{a['sample_std']:.6g}" if a["sample_std"] is not None else "undefined for one seed"
        lines.append(f"- {key}: mean {a['mean']:.6g}; sample SD {spread}; range [{a['min']:.6g}, {a['max']:.6g}].")
    lines.extend(
        [
            "",
            "Every checkpoint gave identical predictions at 101 times after being loaded back from disk.",
            "",
            f"Errors and residuals are measured at {c.n_test:,} evenly spaced times inside the training interval. "
            "Other coefficients, inverse problems, noisy data and extrapolation were not tested.",
            "",
        ]
    )
    (output / "RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 21, 42])
    parser.add_argument("--steps", type=int, default=6000)
    args = parser.parse_args()
    if args.steps < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("steps must be positive and seeds must be distinct")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    c = Config(steps=args.steps)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output / "environment.json",
        {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
            "device": "cpu",
            "dtype": "float64",
            "torch_threads": torch.get_num_threads(),
            "deterministic_algorithms": True,
            "command": ["run.py", "--seeds", *map(str, args.seeds), "--steps", str(args.steps)],
            "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "library_sha256": hashlib.sha256(
                Path(sys.modules[CoolingNet.__module__].__file__).read_bytes()
            ).hexdigest(),
        },
    )
    results = [run_seed(seed, c, args.output) for seed in args.seeds]
    summarize(results, args.output, c)


if __name__ == "__main__":
    main()
