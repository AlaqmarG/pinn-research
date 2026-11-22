"""Pilot: Adam versus zero-order (ZO) training of the physics-only cooling PINN.

Same problem, network, initialization, collocation/validation points and
selection rule as GitHub/cooling-pinn/cooling_pinn.py. Exact temperatures are
used only for final evaluation after checkpoint selection.

Methods
  adam      reverse-mode gradients (autograd), as in the original baseline
  zoadam    two-point random-direction gradient estimate fed to Adam
  cmaes     CMA-ES (population-based, function values only)

Input-derivative variants for ZO methods (the residual needs du/dtau):
  exact     du/dtau propagated analytically through the tanh layers
            (forward mode, no backpropagation, no autograd)
  fd        central finite difference in tau, h = 1e-3 (training and
            checkpoint selection need network outputs only)

Budgets are counted in loss evaluations. One evaluation is the residual on all
128 collocation points plus the initial condition, for one parameter vector.
Adam is counted in gradient evaluations. A reverse-mode gradient costs a small
constant multiple of one loss evaluation, so wall-clock time is reported too.
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

REPO = Path(__file__).resolve().parents[3] / "GitHub" / "cooling-pinn"
sys.path.insert(0, str(REPO))
import cooling_pinn as cp  # noqa: E402

C = cp.Config()
KT = C.k_per_min * C.end_min  # dimensionless decay rate (1.0)
FD_H = 1e-3
SHAPES = [(24, 1), (24,), (24, 24), (24,), (1, 24), (1,)]
N_PARAMS = sum(int(np.prod(s)) for s in SHAPES)  # 673


def unflatten(theta: torch.Tensor):
    """theta (P, 673) -> list of batched weight tensors in CoolingNet order."""
    out, i = [], 0
    for s in SHAPES:
        n = int(np.prod(s))
        out.append(theta[:, i:i + n].reshape(theta.shape[0], *s))
        i += n
    return out


def flatten_model(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_flat(model: torch.nn.Module, theta: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(theta[i:i + n].reshape(p.shape))
            i += n


def forward(theta: torch.Tensor, x: torch.Tensor, with_derivative: bool):
    """Batched MLP. x (N,1). Returns u (P,N) and optionally du/dx (P,N)."""
    W1, b1, W2, b2, W3, b3 = unflatten(theta)
    z1 = torch.einsum("nk,pjk->pnj", x, W1) + b1[:, None, :]
    h1 = torch.tanh(z1)
    z2 = torch.einsum("pnk,pjk->pnj", h1, W2) + b2[:, None, :]
    h2 = torch.tanh(z2)
    u = (torch.einsum("pnk,pjk->pnj", h2, W3) + b3[:, None, :])[..., 0]
    if not with_derivative:
        return u, None
    dz1 = W1[:, None, :, 0].expand_as(z1)            # dz1/dx
    dh1 = (1 - h1 ** 2) * dz1
    dz2 = torch.einsum("pnk,pjk->pnj", dh1, W2)
    dh2 = (1 - h2 ** 2) * dz2
    du = torch.einsum("pnk,pjk->pnj", dh2, W3)[..., 0]
    return u, du


def batched_loss(theta: torch.Tensor, x: torch.Tensor, deriv: str):
    """Physics MSE + IC MSE for each row of theta. Returns (total, physics, ic)."""
    if deriv == "exact":
        u, du = forward(theta, x, True)
    elif deriv == "fd":
        u, _ = forward(theta, x, False)
        up, _ = forward(theta, x + FD_H, False)
        um, _ = forward(theta, x - FD_H, False)
        du = (up - um) / (2 * FD_H)
    else:
        raise ValueError(deriv)
    physics = (du + KT * u).square().mean(dim=1)
    u0, _ = forward(theta, torch.zeros((1, 1), dtype=torch.float64), False)
    ic = (u0[:, 0] - 1).square()
    return physics + C.ic_weight * ic, physics, ic


def points(seed: int):
    train = torch.rand((C.n_train, 1), generator=torch.Generator().manual_seed(seed + 10000), dtype=torch.float64)
    val = torch.rand((C.n_validation, 1), generator=torch.Generator().manual_seed(seed + 20000), dtype=torch.float64)
    return train, val


def init_theta(seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    return flatten_model(cp.CoolingNet(C))


def validation_score(theta: torch.Tensor, val: torch.Tensor, deriv: str = "exact") -> float:
    # Residual MSE + IC MSE on the validation times.
    with torch.no_grad():
        return float(batched_loss(theta[None], val, deriv)[0][0])


class Tracker:
    """Records validation score at regular eval counts and keeps the best."""

    def __init__(self, val, every, deriv="exact"):
        # deriv is how du/dtau is computed for the validation score. fd runs pass "fd"
        # so that selection, like training, needs network outputs only.
        self.val, self.every, self.next, self.deriv = val, every, 0, deriv
        self.best, self.best_theta, self.best_evals = float("inf"), None, 0
        self.history = []
        self.t0 = time.perf_counter()

    def maybe(self, evals, theta, train_loss):
        if evals < self.next:
            return
        self.next = evals + self.every
        s = validation_score(theta, self.val, self.deriv)
        self.history.append({"evals": evals, "seconds": time.perf_counter() - self.t0,
                             "train_loss": float(torch.as_tensor(train_loss).detach()), "validation_score": s})
        if s < self.best:
            self.best, self.best_theta, self.best_evals = s, theta.clone(), evals


def run_adam(seed, budget, hp, deriv="autograd"):
    train, val = points(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = cp.CoolingNet(C)
    opt = torch.optim.Adam(model.parameters(), lr=hp["lr"])
    tr = Tracker(val, hp["every"])
    for step in range(budget + 1):
        p, ic = cp.losses(model, train, C, create_graph=False)
        tr.maybe(step, flatten_model(model), p + ic)
        if step == budget:
            break
        opt.zero_grad(set_to_none=True)
        p, ic = cp.losses(model, train, C)
        (p + C.ic_weight * ic).backward()
        opt.step()
    return tr


def run_zoadam(seed, budget, hp, deriv):
    train, val = points(seed)
    theta = init_theta(seed)
    gen = torch.Generator().manual_seed(seed + 30000)
    q, mu, lr = hp["q"], hp["mu"], hp["lr"]
    m = torch.zeros_like(theta)
    v = torch.zeros_like(theta)
    b1, b2, eps = 0.9, 0.999, 1e-8
    tr = Tracker(val, hp["every"], deriv)
    evals, t = 0, 0
    with torch.no_grad():
        loss_now = batched_loss(theta[None], train, deriv)[0][0]
        evals += 1
        while True:
            tr.maybe(evals, theta, loss_now)
            if evals + 2 * q + 1 > budget:
                break
            U = torch.randn((q, N_PARAMS), generator=gen, dtype=torch.float64)
            batch = torch.cat([theta + mu * U, theta - mu * U])
            L = batched_loss(batch, train, deriv)[0]
            evals += 2 * q
            g = ((L[:q] - L[q:]) / (2 * mu)) @ U / q   # two-point estimate
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            theta = theta - lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)
            loss_now = batched_loss(theta[None], train, deriv)[0][0]
            evals += 1
    tr.next = 0
    tr.maybe(evals, theta, loss_now)
    return tr


def run_cmaes(seed, budget, hp, deriv):
    train, val = points(seed)
    theta0 = init_theta(seed)
    opts = {"seed": seed + 40000, "verbose": -9, "maxfevals": budget, "tolfun": 0, "tolx": 0,
            "tolfunhist": 0, "tolstagnation": 10 ** 9, "tolflatfitness": 10 ** 9}
    if hp.get("popsize"):
        opts["popsize"] = hp["popsize"]
    es = cma.CMAEvolutionStrategy(theta0.numpy(), hp["sigma0"], opts)
    tr = Tracker(val, hp["every"], deriv)
    evals = 0
    tr.maybe(0, theta0, float(batched_loss(theta0[None], train, deriv)[0][0]))
    with torch.no_grad():
        while evals + es.popsize <= budget:
            X = es.ask()
            L = batched_loss(torch.tensor(np.array(X)), train, deriv)[0].numpy()
            es.tell(X, L.tolist())
            evals += len(X)
            mean = torch.tensor(es.mean.copy())
            if evals >= tr.next:
                tr.maybe(evals, mean, float(L.min()))
    mean = torch.tensor(es.mean.copy())
    tr.next = 0
    tr.maybe(evals, mean, float("nan"))
    return tr


METHODS = {"adam": run_adam, "zoadam": run_zoadam, "cmaes": run_cmaes}


def final_metrics(theta):
    model = cp.CoolingNet(C)
    load_flat(model, theta)
    metrics, predictions = cp.evaluate(model, C)
    return metrics, predictions


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
