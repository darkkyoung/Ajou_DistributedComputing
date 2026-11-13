#!/usr/bin/env bash
# HW2 AWS 1/2/4-node scaling benchmark; 3 repeats each.
# AI-assisted (ChatGPT GPT-6, 2026-11-13). Execute from AWS node1 only.
# Requirements: EFS at /shared/hw2 on all nodes; local graph at /data/hw2.
set -euo pipefail

ROOT="/shared/hw2"
GRAPH="/data/hw2/uk-2007-05.bin"
EXE="$ROOT/pagerank"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="$ROOT/results/scaling_$STAMP"
mkdir -p "$OUT"

cat > "$ROOT/hosts1" <<'HOSTS1'
172.31.13.23 slots=4
HOSTS1
cat > "$ROOT/hosts2" <<'HOSTS2'
172.31.13.23 slots=4
172.31.3.104 slots=4
HOSTS2
cat > "$ROOT/hosts4" <<'HOSTS4'
172.31.13.23 slots=4
172.31.3.104 slots=4
172.31.7.154 slots=4
172.31.8.59 slots=4
HOSTS4

[[ -x "$EXE" && -r "$GRAPH" ]] || {
  echo "ERROR: PageRank binary or local official graph missing on node1" >&2
  exit 1
}

echo "BENCHMARK_DIR=$OUT"
echo "Settings: N=105896555 M=3738733648 damp=0.85 benchmark_iters=20"
echo "1 node=4 ranks; 2 nodes=8 ranks; 4 nodes=16 ranks"
for nodes in 1 2 4; do
  ranks=$((nodes * 4))
  # Confirm source files exist in every participating node's local EBS.
  while IFS=' ' read -r host _; do
    ssh -o BatchMode=yes -o ConnectTimeout=8 -i "$HOME/.ssh/id_ed25519" \
      "ubuntu@$host" "test -r '$GRAPH' && test -x '$EXE'" || {
      echo "ERROR: Missing graph or binary on $host" >&2
      exit 1
    }
  done < "$ROOT/hosts$nodes"

  for rep in 1 2 3; do
    log="$OUT/nodes${nodes}_ranks${ranks}_rep${rep}.txt"
    echo "=== Starting nodes=$nodes ranks=$ranks rep=$rep ==="
    # No dump path for repeat timings; writing ~808MiB would distort workflow.
    mpirun --hostfile "$ROOT/hosts$nodes" -np "$ranks" \
      --map-by ppr:4:node --bind-to core \
      "$EXE" file "$GRAPH" 0.85 1e-10 500 10 20 \
      2>&1 | tee "$log"

    # Reject partial output, wrong rank count, and anomalous mass.
    python3 - "$log" "$nodes" "$ranks" <<'PY'
import re,sys
path,nodes,ranks=sys.argv[1],int(sys.argv[2]),int(sys.argv[3])
content=open(path,encoding="utf8").read()
result=re.search(r"^\[Result\] (.+)$",content,re.M)
stats=re.search(r"^\[Stats\] (.+)$",content,re.M)
if not result or not stats:
    sys.exit("ERROR: benchmark log lacks Result/Stats")
fields=dict(re.findall(r"(\w+)=([^\s]+)",result.group(1)))
sfields=dict(re.findall(r"(\w+)=([^\s]+)",stats.group(1)))
if (fields.get("Graph")!="file" or int(fields["Nodes"])!=nodes
        or int(fields["Procs"])!=ranks or int(fields["Iters"])!=20
        or fields.get("Mode")!="benchmark"):
    sys.exit("ERROR: unexpected benchmark parameters")
if abs(float(sfields["Sum"])-1.0)>1e-8:
    sys.exit("ERROR: PageRank mass check failed")
print("PASS: verified 20 iterations and rank/node counts")
PY
  done
done

python3 - "$OUT" <<'PY'
import csv, pathlib, re, statistics, sys
out=pathlib.Path(sys.argv[1])
rows=[]
for nodes in (1,2,4):
    for rep in (1,2,3):
        path=out/f"nodes{nodes}_ranks{nodes*4}_rep{rep}.txt"
        content=path.read_text(encoding="utf8")
        match=re.search(r"^\[Result\] (.+)$",content,re.M)
        balance=re.search(r"^\[Balance\] (.+)$",content,re.M)
        if not match or not balance:
            raise RuntimeError(f"Missing fields: {path}")
        vals=dict(re.findall(r"(\w+)=([^\s]+)",match.group(1)))
        vals.update(dict(re.findall(r"(\w+)=([^\s]+)",balance.group(1))))
        rows.append(dict(Nodes=nodes,Ranks=nodes*4,Repeat=rep,
            **{k:vals[k] for k in ("Total","Compute","Comm","Allgatherv",
            "Allreduce","Load","InEdgesMin","InEdgesMax","ComputeMin","ComputeMax")}))
csvpath=out/"measurements.csv"
with csvpath.open("w",newline="",encoding="utf8") as fp:
    writer=csv.DictWriter(fp,fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)
print(f"Saved {csvpath}")
base=statistics.median(float(r["Total"]) for r in rows if r["Nodes"]==1)
summary=[]
for n in (1,2,4):
    subset=[r for r in rows if r["Nodes"]==n]
    times=[float(r["Total"]) for r in subset]
    median=statistics.median(times)
    speedup=base/median
    summary.append((n,median,min(times),max(times),speedup,speedup/n))
print("Nodes | median(s) | min(s) | max(s) | speedup | efficiency")
for row in summary:
    print("%5d | %9.3f | %6.3f | %6.3f | %7.3f | %.3f" % row)
with (out/"summary.csv").open("w",newline="",encoding="utf8") as fp:
    writer=csv.writer(fp)
    writer.writerow(["Nodes","MedianSeconds","MinSeconds","MaxSeconds","Speedup","Efficiency"])
    writer.writerows(summary)
PY
echo "DONE: $OUT"
