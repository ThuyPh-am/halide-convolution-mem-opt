#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT/build"
make -j4

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi

./run_experiment > "$PROJECT_ROOT/build/loop_nest.txt" 2>&1