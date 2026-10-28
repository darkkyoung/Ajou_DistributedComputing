#!/usr/bin/env python3
"""Repeatable HW1 performance measurements on one local machine.

AI-generated experiment helper — ChatGPT (OpenAI), GPT-6, 2026-10-28.
Run: python3 scripts/run_performance.py
Optional: --nx 2048 --ny 2048 --steps 300 --alpha 0.20 --repeats 3
Do NOT present CI or mock-MPI timings as laptop strong-scaling evidence.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import re
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = [(1, 1, 1), (2, 1, 2), (4, 2, 2), (4, 1, 4), (4, 4, 1)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nx", type=int, default=2048)
    parser.add_argument("--ny", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="results/hw1_performance.csv")
    args = parser.parse_args()

    if args.repeats < 3:
        parser.error("The assignment requires at least 3 runs per configuration")
    if args.nx <= 0 or args.ny <= 0 or args.steps < 0 or not (0 < args.alpha <= 0.25):
        parser.error("Invalid problem size, steps, or alpha")
    if any(args.nx % px or args.ny % py for _, px, py in CONFIGS):
        parser.error("NX and NY must be divisible by every requested process grid")

    with tempfile.TemporaryDirectory(prefix="hw1-performance-") as temp:
        binary = Path(temp) / "heat_diffusion"
        subprocess.run(
            ["mpicc", "-O2", "-std=c11", "-Wall", "-Wextra",
             "-o", str(binary), str(ROOT / "HW1_heat_diffusion.c")],
            check=True,
        )

        all_runs = []
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        print(f"Machine experiment: NX={args.nx} NY={args.ny} "
              f"steps={args.steps} alpha={args.alpha} repeats={args.repeats}")
        print("Only the maximum rank loop time is used; initialization/stat gathering excluded.")

        for n, px, py in CONFIGS:
            records = []
            for repeat in range(1, args.repeats + 1):
                env = os.environ.copy()
                env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
                proc = subprocess.run(
                    ["mpirun", "--oversubscribe", "-np", str(n), str(binary),
                     str(args.nx), str(args.ny), str(px), str(py),
                     str(args.steps), str(args.alpha)],
                    capture_output=True, text=True, env=env, timeout=600, check=True,
                )
                line = next(line for line in proc.stdout.splitlines()
                            if line.startswith("[Result] "))
                fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
                records.append((float(fields["Time"]),
                                float(fields["Sum"]), float(fields["Max"])))
                print(f"{n:>2} ranks {px}x{py} repeat {repeat}: "
                      f"{records[-1][0]:.9f} s")
            med = statistics.median(r[0] for r in records)
            # Use Sum/Max from one real run; they should agree across repeats.
            sum_value, max_value = records[0][1], records[0][2]
            for record in records:
                if abs(record[1]-sum_value) > 1e-7 or abs(record[2]-max_value) > 1e-7:
                    raise ValueError("Numerical results vary across repeated runs")
            all_runs.append({
                "Procs": n, "Grid": f"{px}x{py}",
                "Local size": f"{args.nx//px}x{args.ny//py}",
                "Halo elems/step": 2*((px-1)*args.ny+(py-1)*args.nx),
                "Median time (s)": med, "Sum": sum_value, "Max": max_value,
            })

        base = all_runs[0]["Median time (s)"]
        for row in all_runs:
            p = row["Procs"]
            time = row["Median time (s)"]
            row["Speedup"] = base / time if time > 0 else float("nan")
            row["Efficiency"] = row["Speedup"] / p

        with output.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_runs[0]))
            writer.writeheader()
            writer.writerows(all_runs)

        print("\nProcs Grid Local size  Halo/step Median(s) Speedup Efficiency")
        for row in all_runs:
            print(f"{row['Procs']:>5} {row['Grid']:>4} {row['Local size']:>10} "
                  f"{row['Halo elems/step']:>10} {row['Median time (s)']:>9.6f} "
                  f"{row['Speedup']:>7.3f} {row['Efficiency']:>10.3f}")
        print(f"\nSaved raw summary: {output}")
        print("Report machine, MPI version, core limits, placement and actual run screenshots.")


if __name__ == "__main__":
    main()
