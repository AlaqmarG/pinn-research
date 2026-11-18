#!/usr/bin/env bash
# Reproduce experiment 01. Usage: bash experiments/01-cooling-adam/reproduce.sh <out-dir>
set -euo pipefail
out="${1:?usage: reproduce.sh <out-dir>}"
here="$(cd "$(dirname "$0")" && pwd)"
python "$here/run.py" --output "$out/results"
