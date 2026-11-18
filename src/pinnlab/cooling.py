"""Newton cooling problem: configuration, network, residual, losses and evaluation.

Exact temperatures are used only in `evaluate`, after training and checkpoint selection.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

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


def residual(
    model: nn.Module, tau: torch.Tensor, c: Config, create_graph: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
    x = tau.detach().clone().requires_grad_(True)
    u = model(x)
    du_dtau = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=create_graph)[0]
    # dT/dt = (initial-ambient)/end_min * du/dtau.
    # Dimensionless residual r = du/dtau + k*end_min*u.
    return du_dtau + c.k_per_min * c.end_min * u, u


def losses(
    model: nn.Module, tau: torch.Tensor, c: Config, create_graph: bool = True
) -> tuple[torch.Tensor, torch.Tensor]:
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
    physical_residual = (c.initial_c - c.ambient_c) / c.end_min * r.detach().numpy().ravel()
    metrics = {
        "mae_c": float(np.abs(error).mean()),
        "rmse_c": float(np.sqrt(np.mean(error**2))),
        "max_abs_error_c": float(np.abs(error).max()),
        "relative_l2_temperature": float(np.linalg.norm(error) / np.linalg.norm(exact)),
        "relative_l2_excess_temperature": float(np.linalg.norm(error) / np.linalg.norm(exact - c.ambient_c)),
        "initial_abs_error_c": float(abs(error[0])),
        "test_residual_rms_c_per_min": float(np.sqrt(np.mean(physical_residual**2))),
        "test_residual_max_abs_c_per_min": float(np.abs(physical_residual).max()),
    }
    return metrics, np.column_stack([t, predicted, exact, np.abs(error), physical_residual])
