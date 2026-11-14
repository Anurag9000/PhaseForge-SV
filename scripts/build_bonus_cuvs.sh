#!/usr/bin/env bash
set -euo pipefail
# Requires the actual cuVS C++ SDK, CUDA toolkit and CMake.
# cuvs-cu12/cu13 Python wheels alone do not supply a complete C++ SDK.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cmake -S "$ROOT/benchmarks/cuvs" -B "$ROOT/build/bonus-cuvs" \
  -DCMAKE_BUILD_TYPE=Release "$@"
cmake --build "$ROOT/build/bonus-cuvs" --config Release --parallel 4
echo "Binary: $ROOT/build/bonus-cuvs/bonus_cuvs_gmm"
