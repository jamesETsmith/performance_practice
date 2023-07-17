#!/usr/bin/env bash

set -xeu -o pipefail

for HINT in 0 1 2 4 8
do
    perf stat -- ./build/concurrent_map ${HINT} 50000000
done