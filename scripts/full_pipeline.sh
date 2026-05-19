#!/bin/bash

SCRIPT_DIR = "$("$(dirname "$0")" && pwd)"
PROJECT_ROOT = "$(dirname "$SCRIPT_DIR")"

./run_experiment > "PROJECT_ROOT/build/loop_nest.txt" 2>&1
python3 parse_measurements.py