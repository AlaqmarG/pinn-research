# pinn-research

[![tests](https://github.com/AlaqmarG/pinn-research/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/AlaqmarG/pinn-research/actions/workflows/tests.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Experiments with physics-informed neural networks (PINNs), small enough to run on a laptop CPU.

> Undergraduate research project. New experiments are added as numbered folders.

## Experiments

| # | Experiment | Question |
|---|---|---|
| 01 | [Cooling PINN trained with Adam](experiments/01-cooling-adam/) | Can a physics-only PINN match the exact cooling curve? |
| 02 | [Adam vs zero-order training](experiments/02-zo-vs-adam/) | How well does the same PINN train from loss values alone? |

Each experiment is self-contained. Its README covers the setup, the commands to reproduce it, the results and any library versions it needs.

## How the repo is organized

- `experiments/` has one numbered folder per study.
- `src/pinnlab/` holds code shared between experiments. Experiments import from it, never from each other.
- `tests/` checks the shared code, and runs on every push.
- Results and figures are not committed. Experiments are run locally, and their outputs are attached to a [release](https://github.com/AlaqmarG/pinn-research/releases) so the numbers match what was reported.

## Shared setup

Python 3.12 or newer. From the repository root:

```sh
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -e .
python -m unittest discover -s tests -v
```

Then follow the README of the experiment you want to run.

## Citation

If this code is useful to you, GitHub's "Cite this repository" button uses [CITATION.cff](CITATION.cff).

## License

[MIT](LICENSE)
