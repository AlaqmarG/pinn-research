# 02 · Adam vs zero-order training

How well can the cooling PINN from [experiment 01](../01-cooling-adam/) be trained using only loss values, with no gradients? This experiment compares Adam with two zero-order (ZO) methods on the same problem, network, starting weights and seeds.

## Setup

- **Problem:** `dT/dt = -0.1 (T - 20)`, `T(0) = 80 °C`, `0 ≤ t ≤ 10` minutes, physics-only (no temperature data in training).
- **Network:** 2 hidden layers of 24 tanh units, 673 parameters, float64, one CPU thread.
- **Seeds:** 7, 21 and 42, the same as experiment 01. Settings were tuned on seed 0 only.
- **Checkpoint rule:** lowest validation residual + starting-value loss on 257 held-out times. The exact solution is only used afterwards, on 1,001 test times.

| Method | What it uses | Budget |
|---|---|---|
| Adam | backpropagated gradients, learning rate 0.001 | 6,000 and 30,000 steps |
| ZO-Adam | two-point random gradient estimate (10 Gaussian directions, μ = 1e-3) fed to the Adam update, learning rate 0.003 | 300,000 loss evaluations |
| CMA-ES | population of 23, initial spread 0.05 ([pycma](https://github.com/CMA-ES/pycma)) | 300,000 loss evaluations |

Each ZO method runs twice: once with the exact `du/dτ` (chain rule through the tanh layers, no backpropagation) and once with a central finite difference, `h = 1e-3`. The finite-difference runs also use it for checkpoint selection, so they never compute a derivative of the network.

Tuning grid (seed 0, chosen by validation loss): ZO-Adam learning rate ∈ {0.001, 0.003, 0.01} × directions ∈ {10, 40}; CMA-ES initial spread ∈ {0.02, 0.05, 0.2}. Adam was not tuned.

## Results

Median over the three seeds, with min and max in brackets. Errors are against the exact solution.

| Method | Budget | RMSE (°C) | Max error (°C) | Time (s) |
|---|---:|---|---:|---:|
| Adam, 6,000 steps | 6,000 gradients | 0.0035 [0.0026, 0.0046] | 0.0073 | 9 |
| Adam, 30,000 steps | 30,000 gradients | **0.0013** [0.0003, 0.0014] | 0.0023 | 45 |
| ZO-Adam, exact du/dτ | 300,000 losses | 0.0120 [0.0048, 0.0189] | 0.0237 | 35 |
| ZO-Adam, finite differences | 300,000 losses | 0.0105 [0.0078, 0.0124] | 0.0180 | 65 |
| CMA-ES, exact du/dτ | 300,000 losses | 0.0017 [0.0008, 0.0022] | 0.0030 | 237 |
| CMA-ES, finite differences | 300,000 losses | 0.0017 [0.0008, 0.0021] | 0.0032 | 253 |

![Best validation loss against time and against loss evaluations](https://github.com/AlaqmarG/pinn-research/releases/download/raw-results-v1/02-zo-vs-adam-convergence.png)

- ZO-Adam was the least accurate and stopped improving early.
- CMA-ES halved the error of the 6,000-step Adam run, but took about 25 times longer.
- Adam with 30,000 steps was the most accurate and still about 5 times faster than CMA-ES.
- CMA-ES did just as well with finite differences, so the network trained without any derivative at all.

Adam was stable on this smooth problem, so this measures the cost of ZO training, not whether it helps with instability.

## Reproduce

From the repository root, after the shared setup in the [main README](../../README.md):

```sh
pip install -r experiments/02-zo-vs-adam/requirements.txt
bash experiments/02-zo-vs-adam/reproduce.sh out
```

This runs the tuning grid, the 18 final runs and the figures, about 75 minutes on one CPU thread, and writes everything to `out/`. Finished runs are skipped, so it can be stopped and restarted. Run it on an otherwise idle machine if you want the times to be comparable.

The outputs behind the table above are in [02-zo-vs-adam-raw-results.zip](https://github.com/AlaqmarG/pinn-research/releases/download/raw-results-v1/02-zo-vs-adam-raw-results.zip) on the [raw-results-v1 release](https://github.com/AlaqmarG/pinn-research/releases/tag/raw-results-v1): one folder per run (`metrics.json`, `history.json`, `predictions.csv`), `summary.json`, `environment.json` and the run log.

## Files

```
02-zo-vs-adam/
├── reproduce.sh        runs everything below in order
├── run.py              runs every entry in a config file
├── make_figures.py     summary.json, LaTeX table and convergence figure
├── configs/            tuning.json (seed 0 grid), final.json (18 runs)
└── provenance/         the exact script that produced the results
```

## Notes

- Results were produced with Python 3.12.14, PyTorch 2.15.0, NumPy 2.5.5 and cma 4.6.0 (`environment.json` in the release zip). The Adam rows differ from experiment 01 by less than 0.0001 °C because of the newer PyTorch.
- All 18 runs ran one after another in a single process. The times also depend on the implementation: the ZO methods score candidates in batches, and CMA-ES spends most of its time updating its 673 × 673 covariance matrix.
- `provenance/executed_zo_pilot.py` is the script as run. The code in [src/pinnlab/zo.py](../../src/pinnlab/zo.py) and [run.py](run.py) is the same code split into a module and a script; short runs give byte-identical predictions.
