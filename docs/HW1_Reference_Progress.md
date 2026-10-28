# HW1 reference implementation — staged learning trail

**Branch:** reference/hw1-step-by-step

**Status:** AI-generated reference for comparison. These are pedagogical milestones,
**not authentic student trial-and-error.** Do not claim this commit trail represents
your own independent debugging history.

**AI declaration:** ChatGPT / OpenAI GPT-6 / 2026-10-28. Entire reference C
implementation, progression examples, and scripts were generated with AI support.
The actual final report must disclose AI assistance, relevant conversations, and
code scopes, plus independently interpreted experimental results.

## Why a separate branch?

The main branch contains the pre-implementation Design Note at
docs/HW1_Design_Note.md, committed BEFORE code generation.
The main branch has not been changed by this reference implementation.

## Eight cumulative C source versions

Each commit updates the SAME root file HW1_heat_diffusion.c.
Open the GitHub commit page (Files changed) to compare a milestone.

| Version | SHA | New functionality | Supported check |
|---|---|---|---|
| V0 | 00a162f | MPI init, input validation, rank map | 4 4 1 1 0 0.20 with 1 rank |
| V1 | f6e4877 | Old/new grid allocation, initial hot cell | 4 4 1 1 0 0.20 with 1 rank |
| V2 | 8d1063a | Single-rank stencil, Neumann, double buffering | 4 4 1 1 4 0.20 with 1 rank |
| V3 | 8018e92 | Neighbors and 2D domain decomposition | 4 4 2 2 0 0.20 with 4 ranks |
| V4 | e85f085 | Up/Down row halo exchange | 4 4 1 2 4 0.20 with 2 ranks |
| V5 | 62b1737 | Left/Right column pack/unpack | 4 4 2 1 4 0.20 with 2 ranks |
| V6 | d857ced | Complete 2D four-direction halo exchange | 4 4 2 2 4 0.20 with 4 ranks |
| V7 | 97d56d9 | Global sum/max/time gather, 4x4 output | 4 4 2 2 4 0.20 with 4 ranks |

- V1 accepts steps=0 only.
- V2 accepts one MPI rank only.
- V3 permits multiple ranks with steps=0; computation with multiple
  ranks is not yet supported.
- V4 supports only PX=1 (vertical partition).
- V5 supports only process grids with PX=1 or PY=1 (stripes).
- V6 supports 2D grids but prints per-rank LOCAL statistics, not global.
- V7 is the complete reference and prints the required global Result line,
  plus Grid lines for the 4x4 diagnostic (outside the measured loop).

## Compare versions

Fetch the branch and inspect a version without changing your local files:

    git fetch origin
    git show 00a162f:HW1_heat_diffusion.c
    git show 8d1063a:HW1_heat_diffusion.c
    git show 97d56d9:HW1_heat_diffusion.c

For a specific change:

    git diff 8d1063a 8018e92 -- HW1_heat_diffusion.c

To work independently while preserving this reference:

    git switch main
    git pull --ff-only origin main
    git switch -c work/hw1-my-implementation

## Build and real-MPI correctness verification

Use WSL2/Ubuntu with Open MPI and Python 3:

    sudo apt update
    sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3
    mpicc -O2 -std=c11 -Wall -Wextra -o heat_diffusion HW1_heat_diffusion.c
    mpirun -np 4 ./heat_diffusion 4 4 2 2 4 0.20
    python3 tests/check_correctness.py

4x4 expected global interior temperature distribution after 4 steps:

    28.32  17.60   6.72   1.60
    17.60  10.56   3.52   0.64
     6.72   3.52   0.96   0.00
     1.60   0.64   0.00   0.00

Expected Sum=100 and Max=28.32.

The test suite also compares 512x512 (steps=200), 1D process-grid
shapes, rectangular domains, steps=0, large row/column messages,
and one invalid argument combination.

## Laptop performance study

Run on YOUR own hardware, with identical compiler flags and conditions:

    python3 scripts/run_performance.py --nx 2048 --ny 2048 --steps 300 --alpha 0.20 --repeats 3

This runs all five required configurations, takes median of 3 loop times,
computes Speedup and Efficiency relative to one rank, and produces
results/hw1_performance.csv. The timer measures the simulation loop
(including Neumann boundary, halo exchange, and stencil) and the reported
time is the MAXIMUM elapsed time across ranks.

Theoretical halo count formula, counting only sent elements:

    H = 2 * ((PX - 1) * NY + (PY - 1) * NX)

| Procs | Grid | Local size | Halo elems per step |
|---|---|---|---|
| 1 | 1x1 | 2048x2048 | 0 |
| 2 | 1x2 | 2048x1024 | 4096 |
| 4 | 2x2 | 1024x1024 | 8192 |
| 4 | 1x4 | 2048x512 | 12288 |
| 4 | 4x1 | 512x2048 | 12288 |

## Validation distinction

- The algorithm was syntax checked with local GCC using an MPI API shim
  because real Open MPI is unavailable in that particular development container.
- Synchronous multi-process mock-MPI tests passed for 4x4 in six grid layouts,
  512x512 at 1x1 and 2x2, and steps=0.
- Mock-MPI tests do NOT certify actual Open MPI behavior.
- A GitHub Actions workflow was added to run REAL Open MPI tests on Ubuntu.
  Check whether that action has completed successfully. Otherwise execute
  the correctness script in WSL2.
- Mock timings are not suitable for scaling results. No laptop performance
  data are included.

## Submission requirements

The reference filename deliberately omits the unknown student number. Rename
it to STUDENT_ID_HW1_heat_diffusion.c for submission with your actual ID.
The assignment also requires a PDF report: Design Note (<=half page),
MPI explanation, real correctness and performance evidence, answers to A–F,
screenshots, sources, and full AI usage declaration. The final source also
contains an AI scope comment.

This branch is intended as learning material and code comparison, not
a substitute for your own reasoning and authentic experimental results.
