#!/usr/bin/env python3
"""fak_node_compare.py -- fold the per-node benchmark results into one cross-hardware
table. Each benchmark node writes its results
under fak/experiments/fleet-nodes/<host>/; this reads them all and compares the
headline kernel numbers across ISA/OS. Read-only; stdlib only.

  python tools/fak_node_compare.py            # table to stdout
  python tools/fak_node_compare.py --json     # machine-readable
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NODES = os.path.join(ROOT, "fak", "experiments", "fleet-nodes")


def parse_q8kernel(path):
    """-> {gomaxprocs, rows:{name:{ms,gbs,xf32}}}"""
    out = {"gomaxprocs": None, "rows": {}}
    if not os.path.exists(path):
        return out
    txt = open(path, encoding="utf-8", errors="replace").read()
    m = re.search(r"GOMAXPROCS=(\d+)", txt)
    if m:
        out["gomaxprocs"] = int(m.group(1))
    # rows like: "  int8xf32(WO)    1.00 ms     28.5 weight-GB/s   0.39x vs f32"
    for name, ms, gbs, x in re.findall(
        r"^\s*(\S+)\s+([\d.]+)\s*ms\s+([\d.]+)\s*weight-GB/s\s+([\d.]+)x", txt, re.M
    ):
        out["rows"][name] = {"ms": float(ms), "gbs": float(gbs), "xf32": float(x)}
    return out


def parse_batchbench(path):
    """-> {b1_tok_s, bmax, bmax_tok_s}"""
    out = {"b1_tok_s": None, "bmax": None, "bmax_tok_s": None}
    json_path = os.path.join(os.path.dirname(path), "batchbench-q8.json")
    if os.path.exists(json_path):
        try:
            rep = json.load(open(json_path, encoding="utf-8"))
            peak = rep.get("peak") or {}
            out["b1_tok_s"] = rep.get("baseline_b1_tok_per_sec")
            out["bmax"] = peak.get("batch")
            out["bmax_tok_s"] = peak.get("agg_tok_per_sec")
            return out
        except (OSError, json.JSONDecodeError):
            pass
    if not os.path.exists(path):
        return out
    txt = open(path, encoding="utf-8", errors="replace").read()
    rows = re.findall(r"B=(\d+)\s+step=.*?agg=\s*([\d.]+)\s*tok/s", txt)
    if rows:
        out["b1_tok_s"] = next((float(a) for b, a in rows if b == "1"), None)
        b, a = max(rows, key=lambda r: int(r[0]))
        out["bmax"], out["bmax_tok_s"] = int(b), float(a)
    return out


def load_nodes():
    nodes = []
    for info_path in sorted(glob.glob(os.path.join(NODES, "*", "node-info.json"))):
        d = os.path.dirname(info_path)
        try:
            info = json.load(open(info_path, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            info = {"host": os.path.basename(d)}
        info["q8"] = parse_q8kernel(os.path.join(d, "q8kernel.txt"))
        info["batch"] = parse_batchbench(os.path.join(d, "batchbench.txt"))
        nodes.append(info)
    return nodes


def main():
    nodes = load_nodes()
    if not nodes:
        print(f"no node results under {NODES}\n"
              f"run `bash tools/fak_node_bench.sh` on each node first.", file=sys.stderr)
        return 1
    if "--json" in sys.argv:
        print(json.dumps(nodes, indent=2))
        return 0

    print(f"==== fak cross-node kernel comparison ({len(nodes)} node(s)) ====")
    hdr = ("HOST", "OS/ARCH", "CORES", "q8 f32 ms", "q8 i8xf32 ms", "i8xf32 vs f32", "batch B1 t/s", "batch Bmax t/s")
    print("{:<16} {:<14} {:>5} {:>10} {:>12} {:>13} {:>12} {:>15}".format(*hdr))
    for n in nodes:
        q8 = n.get("q8", {}).get("rows", {})
        f32 = q8.get("f32", {})
        i8 = q8.get("int8xf32(WO)", {})
        b = n.get("batch", {})
        bmax = f"{b.get('bmax_tok_s') or '-'} (B{b.get('bmax') or '-'})" if b.get("bmax_tok_s") else "-"
        print("{:<16} {:<14} {:>5} {:>10} {:>12} {:>13} {:>12} {:>15}".format(
            str(n.get("host", "?"))[:16],
            f"{n.get('os','?')}/{n.get('arch','?')}"[:14],
            str(n.get("cores", "?")),
            f"{f32.get('ms','-')}" if f32 else "-",
            f"{i8.get('ms','-')}" if i8 else "-",
            f"{i8.get('xf32','-')}x" if i8 else "-",
            f"{b.get('b1_tok_s') or '-'}",
            bmax,
        ))
    print("\nlower ms = faster kernel; i8xf32 vs f32 <1.0 = quantized GEMV beats f32;")
    print("batch t/s = aggregate decode throughput (continuous batching).")
    print("cpu per node:")
    for n in nodes:
        print(f"  {n.get('host','?'):<16} {n.get('cpu','?')}  ({n.get('go','?')}, git {n.get('git','?')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
