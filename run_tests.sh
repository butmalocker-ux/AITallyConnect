#!/usr/bin/env bash
# Runs the full suite (unit + fuzz + adversarial + regression), then the mutation test.
set -e
cd "$(dirname "$0")"
python3 -m pytest -q -p no:cacheprovider tests
python3 mutation_test.py
