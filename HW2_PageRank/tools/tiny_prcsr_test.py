#!/usr/bin/env python3
"""Generate a tiny valid PRCSR1 graph, or check V2 binary PageRank dumps.

AI-assisted test tool (ChatGPT GPT-6, 2026-11-13).
Five-node graph: 0->{1,2}, 1->{2}, 2->{0}, 3->{2}, 4->{}.
This includes a dangling node, uneven destination in-degree, and N % 4 != 0.
No NumPy and no real AWS dataset are required.
"""
import argparse
import math
import pathlib
import struct
import sys

OUTGOING = ((1, 2), (2,), (0,), (2,), ())
N = len(OUTGOING)
M = sum(map(len, OUTGOING))
D = sum(not x for x in OUTGOING)


def graph_parts():
    incoming = [[] for _ in range(N)]
    for source, dests in enumerate(OUTGOING):
        for dest in dests:
            incoming[dest].append(source)
    row_ptr = [0]
    col = []
    for sources in incoming:
        col.extend(sorted(sources))
        row_ptr.append(len(col))
    outdeg = [len(x) for x in OUTGOING]
    return row_ptr, outdeg, col


def generate(path):
    row_ptr, outdeg, col = graph_parts()
    header = struct.pack("<8sqqqi", b"PRCSR1\0\0", N, M, D, 1)
    header = header.ljust(64, b"\0")
    data = header
    data += struct.pack(f"<{N+1}q", *row_ptr)
    data += struct.pack(f"<{N}i", *outdeg)
    data += struct.pack(f"<{M}i", *col)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    print(f"CREATED {path}: N={N} M={M} D={D} bytes={len(data)}")


def expected(iterations=20, damping=0.85):
    row_ptr, outdeg, col = graph_parts()
    old = [1.0 / N] * N
    for _ in range(iterations):
        dangling = sum(old[u] for u in range(N) if outdeg[u] == 0)
        new = []
        for v in range(N):
            incoming = sum(old[col[j]] / outdeg[col[j]]
                           for j in range(row_ptr[v], row_ptr[v + 1]))
            new.append((1.0 - damping) / N
                       + damping * (incoming + dangling / N))
        old = new
    return old


def check(paths):
    target = expected()
    previous = None
    for path in paths:
        data = path.read_bytes()
        if len(data) != N * 8:
            raise ValueError(f"{path}: expected {N * 8} bytes, got {len(data)}")
        observed = struct.unpack(f"<{N}d", data)
        l1 = math.fsum(abs(x - y) for x, y in zip(observed, target))
        maxabs = max(abs(x - y) for x, y in zip(observed, target))
        mass = math.fsum(observed)
        if l1 > 1e-11 or maxabs > 1e-11 or abs(mass - 1) > 1e-11:
            raise AssertionError(f"{path}: FAILED L1={l1:.3e} "
                                 f"maxabs={maxabs:.3e} sum={mass:.15f}")
        print(f"PASS {path}: L1={l1:.3e} maxabs={maxabs:.3e} "
              f"sum={mass:.15f}")
        if previous is not None:
            diff = math.fsum(abs(x-y) for x,y in zip(observed, previous))
            if diff > 1e-11:
                raise AssertionError(f"Cross-rank vector mismatch: {diff}")
            print(f"PASS cross-rank L1={diff:.3e}")
        previous = observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("create")
    a.add_argument("path", type=pathlib.Path)
    b = sub.add_parser("check")
    b.add_argument("dumps", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    if args.command == "create":
        generate(args.path)
    else:
        check(args.dumps)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, AssertionError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
