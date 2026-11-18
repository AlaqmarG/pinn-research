# pinn-research

Physics-informed neural network (PINN) experiments, starting with Newton's law of cooling.

| # | Experiment | What it does |
|---|---|---|
| 01 | [Cooling PINN trained with Adam](experiments/01-cooling-adam/) | Physics-only forward PINN, three seeds, checked against the exact solution |
| 02 | [Adam vs zero-order training](experiments/02-zo-vs-adam/) | Trains the same PINN with ZO-Adam and CMA-ES using only loss values, and compares accuracy and cost with Adam |

Shared code lives in `src/pinnlab/` and tests in `tests/`. Install with `pip install -e .`, then run `python -m unittest discover -s tests`.
