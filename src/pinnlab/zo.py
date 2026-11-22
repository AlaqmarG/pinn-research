"""Zero-order (ZO) training of the cooling PINN.

A batched copy of the cooling network that scores many parameter vectors at once.
du/dtau comes from either a hand-written chain rule or central finite differences.
There are three trainers: Adam as the reference, ZO-Adam and CMA-ES.
ZO budgets count loss evaluations and Adam's counts gradient steps.
Exact temperatures are used only in `final_metrics`.
"""

from __future__ import annotations

import time

import cma
import numpy as np
import torch

from pinnlab import cooling as cp

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
        out.append(theta[:, i : i + n].reshape(theta.shape[0], *s))
        i += n
    return out


def flatten_model(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def load_flat(model: torch.nn.Module, theta: torch.Tensor) -> None:
    i = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(theta[i : i + n].reshape(p.shape))
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
    dz1 = W1[:, None, :, 0].expand_as(z1)  # dz1/dx
    dh1 = (1 - h1**2) * dz1
    dz2 = torch.einsum("pnk,pjk->pnj", dh1, W2)
    dh2 = (1 - h2**2) * dz2
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
    """Fixed collocation and validation times for a seed, as in experiment 01."""
    train = torch.rand((C.n_train, 1), generator=torch.Generator().manual_seed(seed + 10000), dtype=torch.float64)
    val = torch.rand((C.n_validation, 1), generator=torch.Generator().manual_seed(seed + 20000), dtype=torch.float64)
    return train, val


def init_theta(seed: int) -> torch.Tensor:
    """Starting parameters for a seed, identical to CoolingNet's initialization."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    return flatten_model(cp.CoolingNet(C))


def validation_score(theta: torch.Tensor, val: torch.Tensor, deriv: str = "exact") -> float:
    """Residual MSE + IC MSE on the validation times."""
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
        self.history.append(
            {
                "evals": evals,
                "seconds": time.perf_counter() - self.t0,
                "train_loss": float(torch.as_tensor(train_loss).detach()),
                "validation_score": s,
            }
        )
        if s < self.best:
            self.best, self.best_theta, self.best_evals = s, theta.clone(), evals


def run_adam(seed, budget, hp, deriv="autograd"):
    """Train with Adam on backpropagated gradients. The budget is the number of gradient steps."""
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
    """Train with ZO-Adam: two-point random gradient estimates fed to the Adam update."""
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
            g = ((L[:q] - L[q:]) / (2 * mu)) @ U / q  # two-point estimate, averaged over q directions
            t += 1
            m = b1 * m + (1 - b1) * g
            v = b2 * v + (1 - b2) * g * g
            theta = theta - lr * (m / (1 - b1**t)) / ((v / (1 - b2**t)).sqrt() + eps)
            loss_now = batched_loss(theta[None], train, deriv)[0][0]
            evals += 1
    tr.next = 0
    tr.maybe(evals, theta, loss_now)
    return tr


def run_cmaes(seed, budget, hp, deriv):
    """Train with CMA-ES, scoring each generation's candidates in one batch."""
    train, val = points(seed)
    theta0 = init_theta(seed)
    opts = {
        "seed": seed + 40000,
        "verbose": -9,
        "maxfevals": budget,
        "tolfun": 0,
        "tolx": 0,
        "tolfunhist": 0,
        "tolstagnation": 10**9,
        "tolflatfitness": 10**9,
    }
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
    """Evaluate a parameter vector against the exact solution."""
    model = cp.CoolingNet(C)
    load_flat(model, theta)
    metrics, predictions = cp.evaluate(model, C)
    return metrics, predictions
