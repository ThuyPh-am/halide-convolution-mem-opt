#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PEOJECT_ROOT="$(dirname "$SCRIPR_DIR")"

cd "$PROJECT_ROOT/scripts"

python3 parse_measurements.py
python3 normalize_data.py
python3 visualize.py