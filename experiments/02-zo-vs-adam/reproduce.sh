#!/usr/bin/env bash
# Reproduce experiment 02 (about 75 minutes on one CPU thread).
# Usage: bash experiments/02-zo-vs-adam/reproduce.sh <out-dir>
set -euo pipefail
out="${1:?usage: reproduce.sh <out-dir>}"
here="$(cd "$(dirname "$0")" && pwd)"
python "$here/run.py" --config "$here/configs/tuning.json" --output "$out/results/tuning"
python "$here/run.py" --config "$here/configs/final.json" --output "$out/results/final"
python "$here/make_figures.py" "$out/results/final" "$out/figures"
