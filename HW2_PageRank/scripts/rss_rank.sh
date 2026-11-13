#!/bin/bash
set -euo pipefail

OUTDIR="$1"
RANK="${OMPI_COMM_WORLD_RANK:?MPI rank required}"

exec /usr/bin/time -v \
  -o "$OUTDIR/rank_${RANK}.time" \
  /shared/hw2/pagerank \
  file /data/hw2/uk-2007-05.bin \
  0.85 1e-10 500 10 20
