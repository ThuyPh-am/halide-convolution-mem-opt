#!/bin/bash

cd build
make -j4

if [ $? -ne 0 ]; then
    echo "Build failed"
    exit 1
fi
./run_experiment