#!/usr/bin/env python3
"""HW1 MPI correctness regression tests (real Open MPI required).

AI-generated test harness — ChatGPT (OpenAI), GPT-6, 2026-10-28.
Run in WSL2/Ubuntu: python3 tests/check_correctness.py
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = [
    [28.32, 17.60, 6.72, 1.60],
    [17.60, 10.56, 3.52, 0.64],
    [6.72, 3.52, 0.96, 0.00],
    [1.60, 0.64, 0.00, 0.00],
]


def assert_close(actual: float, expected: float, atol: float = 1e-8) -> None:
    assert math.isfinite(actual) and abs(actual - expected) <= atol, (
        f"Expected {expected:.14g}, got {actual:.14g}"
    )


def parse_output(stdout: str) -> tuple[dict[str, str], list[list[float]]]:
    result_line = next(
        (line for line in stdout.splitlines() if line.startswith("[Result] ")), None
    )
    assert result_line is not None, f"No [Result] line:\n{stdout}"
    fields = dict(re.findall(r"(\w+)=([^\s]+)", result_line))
    grid: list[list[float]] = []
    for line in stdout.splitlines():
        if line.startswith("[Grid] y="):
            row_text = line.split(":", 1)[1]
            grid.append([float(x) for x in row_text.split()])
    return fields, grid


def run(binary: Path, n: int, nx: int, ny: int, px: int, py: int,
        steps: int, alpha: float) -> tuple[dict[str, str], list[list[float]]]:
    env = os.environ.copy()
    # Open MPI on CI may run under a root-owned container.
    env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
    cmd = ["mpirun", "--oversubscribe", "-np", str(n), str(binary),
           str(nx), str(ny), str(px), str(py), str(steps), str(alpha)]
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=60)
    assert r.returncode == 0, (
        f"Command failed or deadlocked: {cmd}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}"
    )
    return parse_output(r.stdout)


def same_results(a: dict[str, str], b: dict[str, str], atol: float = 1e-8) -> None:
    for key in ("Sum", "Max"):
        assert_close(float(a[key]), float(b[key]), atol)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="hw1-mpi-check-") as tempdir:
        binary = Path(tempdir) / "heat_diffusion"
        subprocess.run(
            ["mpicc", "-O2", "-std=c11", "-Wall", "-Wextra", "-Werror",
             "-o", str(binary), str(ROOT / "HW1_heat_diffusion.c")],
            check=True,
        )
        print("PASS: mpicc -O2 -std=c11 -Wall -Wextra -Werror")

        # Required 4x4 verification plus 1D process grids.
        small_configs = [(1, 1, 1), (2, 1, 2), (2, 2, 1),
                         (4, 2, 2), (4, 1, 4), (4, 4, 1)]
        baseline = None
        for n, px, py in small_configs:
            fields, rows = run(binary, n, 4, 4, px, py, 4, 0.20)
            assert_close(float(fields["Sum"]), 100.0)
            assert_close(float(fields["Max"]), 28.32)
            assert len(rows) == 4 and all(len(row) == 4 for row in rows)
            for y in range(4):
                for x in range(4):
                    assert_close(rows[y][x], REFERENCE[y][x])
            if baseline is None:
                baseline = fields
            else:
                same_results(fields, baseline)
            print(f"PASS: 4x4, steps=4, grid={px}x{py}, all 16 cells")

        # Required large-domain verification; it does not replace the 4x4 check.
        large_one, _ = run(binary, 1, 512, 512, 1, 1, 200, 0.20)
        large_four, _ = run(binary, 4, 512, 512, 2, 2, 200, 0.20)
        for fields in (large_one, large_four):
            assert_close(float(fields["Sum"]), 100.0)
        same_results(large_one, large_four)
        print("PASS: 512x512, steps=200, 1x1 vs 2x2")

        # Optional additional checks (not substitutes for assignment evidence).
        zero, _ = run(binary, 1, 4, 4, 1, 1, 0, 0.25)
        assert_close(float(zero["Sum"]), 100.0)
        assert_close(float(zero["Max"]), 100.0)
        print("PASS: steps=0 initialization")

        uneven_base, _ = run(binary, 1, 8, 12, 1, 1, 5, 0.20)
        uneven_split, _ = run(binary, 4, 8, 12, 2, 2, 5, 0.20)
        same_results(uneven_base, uneven_split)
        print("PASS: rectangular global grid 8x12")

        # Each 2-rank case transfers >= 32 KiB per boundary message.
        # This helps expose implementations that rely on eager MPI buffering.
        wide_one, _ = run(binary, 1, 8192, 8, 1, 1, 2, 0.20)
        wide_two, _ = run(binary, 2, 8192, 8, 1, 2, 2, 0.20)
        same_results(wide_one, wide_two)
        tall_one, _ = run(binary, 1, 8, 8192, 1, 1, 2, 0.20)
        tall_two, _ = run(binary, 2, 8, 8192, 2, 1, 2, 0.20)
        same_results(tall_one, tall_two)
        print("PASS: large row/column messages (no hang)")

        env = os.environ.copy()
        env.update(OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1")
        invalid = subprocess.run(
            ["mpirun", "--oversubscribe", "-np", "2", str(binary),
             "4", "4", "1", "1", "4", "0.20"],
            capture_output=True, text=True, env=env, timeout=20
        )
        assert invalid.returncode != 0, "Mismatched process grid was accepted"
        print("PASS: reject PX*PY != process count")
        print("\nALL MPI CORRECTNESS CHECKS PASSED")


if __name__ == "__main__":
    main()
