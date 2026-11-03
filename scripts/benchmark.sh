#!/usr/bin/env sh
set -eu
python -m minigpt.training.benchmark --config configs/tinystories.yaml --steps 100 --output reports/hardware_benchmark.json "$@"
