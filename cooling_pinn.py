"""Physics-only forward PINN for Newton cooling; exact values are evaluation only."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class Config:
    ambient_c: float = 20.0
    initial_c: float = 80.0
    k_per_min: float = 0.1
    end_min: float = 10.0
    width: int = 24
    hidden_layers: int = 2
    steps: int = 6000
    learning_rate: float = 0.001
    ic_weight: float = 1.0
    n_train: int = 128
    n_validation: int = 257
    n_test: int = 1001
    validation_every: int = 50


class CoolingNet(nn.Module):
    """Predict u=(T-T_ambient)/(T_initial-T_ambient) from tau=t/end_min."""
    def __init__(self, config: Config):
        super().__init__()
        layers: list[nn.Module] = []
        last = 1
        for _ in range(config.hidden_layers):
            layers.extend([nn.Linear(last, config.width), nn.Tanh()])
            last = config.width
        layers.append(nn.Linear(last, 1))
        self.layers = nn.Sequential(*layers)
        self.double()

    def forward(self, tau: torch.Tensor) -> torch.Tensor:
        return self.layers(tau)


def residual(model: nn.Module, tau: torch.Tensor, c: Config,
             create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    x = tau.detach().clone().requires_grad_(True)
    u = model(x)
    du_dtau = torch.autograd.grad(u, x, torch.ones_like(u),
                                  create_graph=create_graph)[0]
    # dT/dt = (initial-ambient)/end_min * du/dtau.
    # Dimensionless residual r = du/dtau + k*end_min*u.
    return du_dtau + c.k_per_min * c.end_min * u, u


def losses(model: nn.Module, tau: torch.Tensor, c: Config,
           create_graph: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    r, _ = residual(model, tau, c, create_graph)
    physics = r.square().mean()
    ic = (model(torch.zeros((1, 1), dtype=torch.float64)) - 1).square().mean()
    return physics, ic


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoint(path: Path) -> tuple[CoolingNet, Config, dict]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    c = Config(**checkpoint["config"])
    model = CoolingNet(c)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, c, checkpoint


def evaluate(model: nn.Module, c: Config) -> tuple[dict, np.ndarray]:
    """Called only after training and physics-only checkpoint selection finish."""
    t = np.linspace(0, c.end_min, c.n_test)
    tau = torch.tensor(t[:, None] / c.end_min, dtype=torch.float64)
    r, u = residual(model, tau, c, create_graph=False)
    predicted = c.ambient_c + (c.initial_c - c.ambient_c) * u.detach().numpy().ravel()
    exact = c.ambient_c + (c.initial_c - c.ambient_c) * np.exp(-c.k_per_min * t)
    error = predicted - exact
    physical_residual = (c.initial_c-c.ambient_c)/c.end_min * r.detach().numpy().ravel()
    metrics = {
        "mae_c": float(np.abs(error).mean()),
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_c": float(np.abs(error).max()),
        "relative_l2_temperature": float(np.linalg.norm(error)/np.linalg.norm(exact)),
        "relative_l2_excess_temperature": float(np.linalg.norm(error)/np.linalg.norm(exact-c.ambient_c)),
        "initial_abs_error_c": float(abs(error[0])),
        "test_residual_rms_c_per_min": float(np.sqrt(np.mean(physical_residual**2))),
        "test_residual_max_abs_c_per_min": float(np.abs(physical_residual).max()),
    }
    return metrics, np.column_stack([t, predicted, exact, np.abs(error), physical_residual])


def plot_run(folder: Path, history: list[dict], predictions: np.ndarray, seed: int) -> None:
    t, pred, exact, error, _ = predictions.T
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, exact, label="Exact solution", color="black", linewidth=2)
    ax.plot(t, pred, label="PINN", color="#0072B2", linestyle="--")
    ax.set(xlabel="Time (min)", ylabel="Temperature (°C)", title=f"Cooling: seed {seed}")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(folder / "temperature_overlay.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t, error, color="#D55E00")
    ax.set(xlabel="Time (min)", ylabel="Absolute temperature error (°C)", title=f"Final evaluation: seed {seed}")
    ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(folder / "absolute_error.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in [("train_physics_mse", "Training residual"), ("train_ic_mse", "Initial condition"),
                       ("validation_score", "Validation residual + IC")]:
        ax.semilogy([row["step"] for row in history], [max(row[key], 1e-20) for row in history], label=label)
    ax.set(xlabel="Adam step", ylabel="Dimensionless squared loss", title=f"Physics-only training: seed {seed}")
    ax.legend(); ax.grid(alpha=.2); fig.tight_layout()
    fig.savefig(folder / "losses.png", dpi=180); plt.close(fig)


def run_seed(seed: int, c: Config, output: Path) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    folder = output / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    # Separate random streams; fixed points throughout each run. Neither uses labels.
    train = torch.rand((c.n_train, 1), generator=torch.Generator().manual_seed(seed+10000), dtype=torch.float64)
    validation = torch.rand((c.n_validation, 1), generator=torch.Generator().manual_seed(seed+20000), dtype=torch.float64)
    test = np.linspace(0, 1, c.n_test)
    assert not np.intersect1d(train.numpy(), validation.numpy()).size
    assert not np.intersect1d(train.numpy(), test).size
    assert not np.intersect1d(validation.numpy(), test).size
    np.savetxt(folder / "train_times_min.csv", train.numpy()*c.end_min, delimiter=",", header="time_min", comments="")
    np.savetxt(folder / "validation_times_min.csv", validation.numpy()*c.end_min, delimiter=",", header="time_min", comments="")
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
            history.append({"step": step, "train_physics_mse": p.item(), "train_ic_mse": ic.item(),
                            "validation_physics_mse": vp.item(), "validation_ic_mse": vic.item(),
                            "validation_score": score})
            if score < best_score:
                best_score, best_step = score, step
                best_state = copy.deepcopy(model.state_dict())
        if step == c.steps:
            break
        optimizer.zero_grad(set_to_none=True)
        p, ic = losses(model, train, c)
        (p + c.ic_weight*ic).backward()
        optimizer.step()
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    checkpoint_path = folder / "checkpoint.pt"
    torch.save({"model_state_dict": best_state, "config": asdict(c), "seed": seed,
                "selected_step": best_step, "validation_score": best_score,
                "selection_rule": "minimum validation residual MSE + ic_weight * IC MSE; no exact labels"}, checkpoint_path)
    reloaded, loaded_config, _ = load_checkpoint(checkpoint_path)
    probe = torch.linspace(0, 1, 101, dtype=torch.float64)[:, None]
    with torch.no_grad():
        reload_difference = float((model(probe)-reloaded(probe)).abs().max())
    assert reload_difference == 0.0 and loaded_config == c
    metrics, predictions = evaluate(reloaded, c)
    metrics.update(seed=seed, selected_step=best_step, validation_score=best_score,
                   training_seconds=elapsed, reload_max_abs_difference_dimensionless=reload_difference,
                   checkpoint_sha256=hashlib.sha256(checkpoint_path.read_bytes()).hexdigest())
    write_json(folder / "config.json", asdict(c))
    write_json(folder / "metrics.json", metrics)
    write_csv(folder / "history.csv", history)
    np.savetxt(folder / "predictions.csv", predictions, delimiter=",", comments="",
               header="time_min,predicted_c,exact_c,absolute_error_c,residual_c_per_min")
    plot_run(folder, history, predictions, seed)
    print(json.dumps(metrics), flush=True)
    return metrics


def summarize(results: list[dict], output: Path, c: Config) -> None:
    keys = ["mae_c", "rmse_c", "max_abs_error_c", "initial_abs_error_c", "test_residual_rms_c_per_min"]
    aggregate = {key: {"mean": float(np.mean([r[key] for r in results])),
                       "sample_std": float(np.std([r[key] for r in results], ddof=1)) if len(results)>1 else None,
                       "min": min(r[key] for r in results), "max": max(r[key] for r in results)} for key in keys}
    write_json(output / "summary.json", {"runs": results, "aggregate": aggregate})
    lines = ["# Cooling PINN results", "", "These runs solve one forward cooling problem on 0–10 minutes.", "",
             f"Each seed used {c.steps} Adam steps. Checkpoints were selected using validation residual loss plus the soft initial-condition loss. Exact temperatures were used only after selection.", "",
             "| Seed | Selected step | MAE (°C) | RMSE (°C) | Max error (°C) | IC error (°C) | Residual RMS (°C/min) |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['seed']} | {r['selected_step']} | {r['mae_c']:.6g} | {r['rmse_c']:.6g} | {r['max_abs_error_c']:.6g} | {r['initial_abs_error_c']:.6g} | {r['test_residual_rms_c_per_min']:.6g} |")
    lines.extend(["", "Across seeds (sample standard deviation uses n−1):", ""])
    for key in keys:
        a = aggregate[key]
        spread = f"{a['sample_std']:.6g}" if a['sample_std'] is not None else "undefined for one seed"
        lines.append(f"- {key}: mean {a['mean']:.6g}; sample SD {spread}; range [{a['min']:.6g}, {a['max']:.6g}].")
    lines.extend(["", "All saved checkpoints reproduced their in-memory predictions exactly on 101 probe times after reload.", "",
                  f"The {c.n_test:,}-point final grid checks solution error and residuals within the training interval. Its interior points are held out; the initial-condition endpoint is a known constraint. It does not establish an error bound between points. These {len(results)} seeds describe this small experiment; they do not establish broad reliability. No changed coefficients, inverse estimation, noisy observations, or extrapolation were tested.", ""])
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
    write_json(args.output / "environment.json", {"python": sys.version,
               "platform": platform.platform(), "torch": torch.__version__, "numpy": np.__version__,
               "matplotlib": matplotlib.__version__, "device": "cpu", "dtype": "float64",
               "torch_threads": torch.get_num_threads(), "deterministic_algorithms": True,
               "command": ["cooling_pinn.py", "--seeds", *map(str, args.seeds), "--steps", str(args.steps)],
               "privacy_note": "Executable and output paths are omitted; command records portable training arguments.",
               "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    results = [run_seed(seed, c, args.output) for seed in args.seeds]
    summarize(results, args.output, c)


if __name__ == "__main__":
    main()
