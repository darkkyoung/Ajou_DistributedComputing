# Ajou Distributed Computing — HW1 reference branch

This is the **AI-generated, step-by-step reference branch** for Homework 1
(2D Heat Diffusion using blocking MPI_Send/MPI_Recv).

**Start here:** [HW1 staged commit trail](docs/HW1_Reference_Progress.md)

- [Pre-implementation Design Note](docs/HW1_Design_Note.md)
- [C source — final reference state](HW1_heat_diffusion.c)
- [Real-MPI correctness tests](tests/check_correctness.py)
- [Laptop performance experiment runner](scripts/run_performance.py)
- [GitHub Actions real-MPI test workflow](.github/workflows/hw1-reference-test.yml)

The original **main** branch is unchanged, so develop your own solution
from main and compare with the reference branch only after attempting
each stage. The staged commit sequence is not a genuine record of
student debugging or experiments. Do not submit performance figures
from a simulation or from another machine.

## Quickstart (Ubuntu/WSL2)

    sudo apt install -y build-essential openmpi-bin libopenmpi-dev python3
    mpicc -O2 -std=c11 -Wall -Wextra -o heat_diffusion HW1_heat_diffusion.c
    mpirun -np 4 ./heat_diffusion 4 4 2 2 4 0.20
    python3 tests/check_correctness.py

Expected: Sum=100, Max=28.32 and all 16 values matching the assignment.

After correctness is verified on your machine, execute:

    python3 scripts/run_performance.py --repeats 3

**AI disclosure:** GPT-6 / ChatGPT (OpenAI) / 2026-10-28.
The C source includes a top-of-file AI attribution and per-function notes.
The complete final PDF report and actual measurements remain the student's
responsibility. For submission, rename C source to match your own student ID.
