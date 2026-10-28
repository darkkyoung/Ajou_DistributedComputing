
#!/usr/bin/env python3
"""
HW1 V8 - MPI Performance Experiments

AI-generated: ChatGPT (OpenAI), GPT-6
Date: 2026-10-29

Runs five process-grid configurations.
Uses three measured repetitions and one warm-up per grid.
Saves raw results and performance summaries as CSV.
"""

import csv
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SOURCE = ROOT / "20216851_HW1_heat_diffusion.c"
BINARY = ROOT / "heat_diffusion"
RESULTS = ROOT / "results"

NX = 2048
NY = 2048
STEPS = 300
ALPHA = 0.20

REPEATS = 3
WARMUPS = 1

CONFIGS = [
    (1, 1, 1),
    (2, 1, 2),
    (4, 2, 2),
    (4, 1, 4),
    (4, 4, 1),
]


def run_case(procs, px, py):
    command = [
        "mpirun", "-np", str(procs),
        str(BINARY),
        str(NX), str(NY),
        str(px), str(py),
        str(STEPS), str(ALPHA),
    ]

    process = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600
    )

    if process.returncode != 0:
        raise RuntimeError(
            f"MPI execution failed:\n"
            f"{' '.join(command)}\n"
            f"{process.stderr}"
        )

    match = re.search(
        r"^\[Result\].*$",
        process.stdout,
        re.MULTILINE
    )

    if match is None:
        raise RuntimeError(
            f"No result found:\n{process.stdout}"
        )

    fields = dict(
        re.findall(r"(\w+)=([^\s]+)", match.group())
    )

    return {
        "Time": float(fields["Time"]),
        "Sum": float(fields["Sum"]),
        "Max": float(fields["Max"]),
    }


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    RESULTS.mkdir(exist_ok=True)

    # Compile once with identical options for every experiment.
    subprocess.run([
        "mpicc", "-O2", "-std=c11",
        "-Wall", "-Wextra",
        "-o", str(BINARY), str(SOURCE)
    ], check=True)

    print("V8 Performance Experiments")
    print(f"Grid={NX}x{NY}, Steps={STEPS}, Alpha={ALPHA}")

    # Warm-up runs are excluded from the three measurements.
    for procs, px, py in CONFIGS:
        for _ in range(WARMUPS):
            run_case(procs, px, py)
        print(f"Warm-up complete: {px}x{py}")

    raw_rows = []

    # Round-robin measurement order.
    for repeat in range(1, REPEATS + 1):
        for procs, px, py in CONFIGS:
            result = run_case(procs, px, py)

            raw_rows.append({
                "Repeat": repeat,
                "Procs": procs,
                "PX": px,
                "PY": py,
                "Time_s": result["Time"],
                "Sum": result["Sum"],
                "Max": result["Max"],
            })

            print(
                f"Repeat {repeat}: {px}x{py} "
                f"Time={result['Time']:.9f}s "
                f"Sum={result['Sum']:.12f} "
                f"Max={result['Max']:.12f}"
            )

    # Verify that every configuration produces the same result.
    reference_sum = raw_rows[0]["Sum"]
    reference_max = raw_rows[0]["Max"]

    for row in raw_rows:
        if abs(row["Sum"] - 100.0) > 1e-8:
            raise RuntimeError("Global sum conservation failed")

        if (
            abs(row["Sum"] - reference_sum) > 1e-8 or
            abs(row["Max"] - reference_max) > 1e-8
        ):
            raise RuntimeError("Results differ between process grids")

    summary = []

    for procs, px, py in CONFIGS:
        selected = [
            row for row in raw_rows
            if row["PX"] == px and row["PY"] == py
        ]

        median_time = statistics.median(
            row["Time_s"] for row in selected
        )

        halo_elements = 2 * (
            (px - 1) * NY + (py - 1) * NX
        )

        messages = 2 * (
            (px - 1) * py + (py - 1) * px
        )

        summary.append({
            "Procs": procs,
            "Grid": f"{px}x{py}",
            "LocalSize": f"{NX//px}x{NY//py}",
            "HaloElemsPerStep": halo_elements,
            "MessagesPerStep": messages,
            "MedianTime_s": median_time,
            "Speedup": 0.0,
            "Efficiency": 0.0,
            "Sum": selected[0]["Sum"],
            "Max": selected[0]["Max"],
        })

    baseline = summary[0]["MedianTime_s"]

    for row in summary:
        row["Speedup"] = baseline / row["MedianTime_s"]
        row["Efficiency"] = row["Speedup"] / row["Procs"]

    save_csv(RESULTS / "v8_raw.csv", raw_rows)
    save_csv(RESULTS / "v8_summary.csv", summary)

    print("\n=== Performance Summary ===")

    for row in summary:
        print(
            f"{row['Procs']} ranks | "
            f"{row['Grid']} | "
            f"Median={row['MedianTime_s']:.9f}s | "
            f"Speedup={row['Speedup']:.4f} | "
            f"Efficiency={row['Efficiency']:.4f}"
        )

    print("\nSaved results/v8_raw.csv")
    print("Saved results/v8_summary.csv")


if __name__ == "__main__":
    main()
