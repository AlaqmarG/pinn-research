# 01: Cooling PINN trained with Adam

A small PyTorch PINN that solves Newton's law of cooling from the equation and the starting temperature alone.

```text
dT/dt = -0.1 (T - 20)
T(0) = 80 °C
0 <= t <= 10 minutes
```

No measured or exact temperatures are used in training or in picking the checkpoint. The exact solution, `T(t) = 20 + 60 exp(-0.1t)`, is only used afterwards to measure the error.

## Results

Three seeds, 6,000 Adam steps each. The saved checkpoint is the one with the lowest validation residual loss plus initial-condition loss.

| Seed | Selected step | MAE (°C) | RMSE (°C) | Maximum error (°C) |
|---|---:|---:|---:|---:|
| 7 | 6000 | 0.00252663 | 0.00351887 | 0.00727131 |
| 21 | 5900 | 0.00226556 | 0.00269999 | 0.00527257 |
| 42 | 6000 | 0.00376351 | 0.00459602 | 0.00863944 |

Mean RMSE over the seeds is 0.00360496 °C, with a sample standard deviation of 0.000950944 °C. Errors are measured at 1,001 evenly spaced times in the ten minutes.

![Predicted and exact cooling curves for seed 21](https://github.com/AlaqmarG/pinn-research/releases/download/raw-results-v1/01-cooling-adam-temperature-overlay.png)

Everything these runs wrote is in [01-cooling-adam-raw-results.zip](https://github.com/AlaqmarG/pinn-research/releases/download/raw-results-v1/01-cooling-adam-raw-results.zip) on the [raw-results-v1 release](https://github.com/AlaqmarG/pinn-research/releases/tag/raw-results-v1): `RESULTS.md`, `summary.json`, `environment.json`, the test transcript, and for each seed the checkpoint, config, sample times, loss history, predictions and plots. The local Python path was removed from `environment.json` before upload.

## Reproduce

Use Python 3.12. The results above came from Python 3.12.14, PyTorch 2.11.0+cu128, NumPy 2.5.3 and Matplotlib 3.11.2. Training runs on the CPU even if PyTorch has CUDA.

Windows PowerShell, from the repository root:

```powershell
py -3.12 -m venv .venv
& .venv/Scripts/python.exe -m pip install -r experiments/01-cooling-adam/requirements.txt
& .venv/Scripts/python.exe -m pip install -e .
& .venv/Scripts/python.exe -m unittest discover -s tests -v
& .venv/Scripts/python.exe experiments/01-cooling-adam/run.py --output results-reproduction
```

macOS or Linux:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r experiments/01-cooling-adam/requirements.txt
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python experiments/01-cooling-adam/run.py --output results-reproduction
```

- `--seeds` takes a list of distinct integers (default 7, 21 and 42) and `--steps` sets the number of Adam steps.
- Without `--output`, results go to `results/` in this folder, which git ignores.
- `bash experiments/01-cooling-adam/reproduce.sh out` runs the defaults and writes to `out/`.
- Numbers change slightly with other hardware or library versions. With PyTorch 2.15.0 the RMSEs differ by up to 0.00006 °C and seed 21 selects step 6000 instead of 5900.

## Method

The network takes scaled time `tau = t/10` and predicts scaled temperature `u = (T-20)/60`. Since `dT/dt = 6 du/dtau`, the residual of the equation is `r = du/dtau + u`. Training minimizes `mean(r²) + (u(0)-1)²`.

- The network has two hidden layers of 24 tanh units and a linear output, 673 parameters in float64.
- It is trained with Adam at learning rate 0.001 on one CPU thread.
- Each seed has 128 fixed random collocation times and 257 separate validation times. The 1,001 test times are evenly spaced.
- Validation is checked every 50 steps. Training always runs the full budget, so the saved checkpoint can come from an earlier step.
- Each checkpoint is loaded back from disk and must give identical predictions at 101 times.

The four unit tests check the residual and its units on a known solution, the derivative against finite differences, backpropagation through the derivative, and saving and loading a checkpoint.

## Provenance

[provenance/executed_cooling_pinn.py](provenance/executed_cooling_pinn.py) is the script that produced the results above, unchanged. Its hash is recorded in `environment.json` in the release zip. The current code is the same calculation split into [run.py](run.py) and [src/pinnlab/cooling.py](../../src/pinnlab/cooling.py). A 300-step run of both on the same library versions gives byte-identical predictions. The outputs differ in two files. `environment.json` leaves out the Python executable path, records the seeds and steps in place of the raw command line, and adds a hash of `cooling.py` next to the hash of `run.py`. The notes at the end of `RESULTS.md` are shorter.

Checkpoints hold the selected weights and the config, but not the optimizer state, so training cannot be resumed from them.

## Limits

This is one forward problem with known constants and three seeds. Other coefficients, inverse problems, noisy data and prediction past ten minutes were not tested. A small residual does not by itself mean a small error, which is why the runs are also compared with the exact solution.
