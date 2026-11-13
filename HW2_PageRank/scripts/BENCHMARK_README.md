# HW2 scaling benchmark
# (Generated/assisted by ChatGPT GPT-6, 2026-11-13)

## AWS node1 use

Copy `scripts/run_scaling_benchmark.sh` to the EFS root as
`/shared/hw2/run_scaling_benchmark.sh`. Execute via:

```bash
bash /shared/hw2/run_scaling_benchmark.sh
```

- Uses `hosts1`, `hosts2`, `hosts4` (one node = 4 ranks).
- Runs 20 PageRank iterations at damping 0.85 in **benchmark** mode.
- Repeats every node count exactly 3 times; sequential runs avoid contention.
- Saves raw logs and `measurements.csv`, `summary.csv` in a new
  `/shared/hw2/results/scaling_YYYYMMDDTHHMMSSZ/` folder.
- Summary reports per-node-count median/min/max, speedup and efficiency.
- It doesn't replace independent **4-node 16-rank** vector correctness checks,
  which have already passed; no 808MB vector dumps during repeated timing.

**Important:** The host IPs refer to the particular EC2 instances used by this
2026-11-13 test. Change host files if those instances/IPs change. All source
`/data/hw2/uk-2007-05.bin` files must be on each server's **local EBS**,
not EFS. AWS charges accrue until EC2/EFS/network resources are cleaned up.

For the report, obtain rank maximum RSS high water mark separately; the
master-process `/usr/bin/time -v mpirun` is NOT valid for remote-rank RSS.
