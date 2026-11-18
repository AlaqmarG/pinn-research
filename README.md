# pinn-research

Physics-informed neural network (PINN) experiments, starting with Newton's law of cooling.

| # | Experiment | What it does |
|---|---|---|
| 01 | [Cooling PINN trained with Adam](experiments/01-cooling-adam/) | Physics-only forward PINN, three seeds, checked against the exact solution |

Shared code lives in `src/pinnlab/` and tests in `tests/`. Install with `pip install -e .`, then run `python -m unittest discover -s tests`.
