#!/usr/bin/env python3
"""Summarize a tools/_registry/soak overnight run.

Usage:
  python tools/summarize_overnight_soak.py tools/_registry/soak/overnight-...
  python tools/summarize_overnight_soak.py tools/_registry/soak/overnight-... --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def fmt_num(value, digits=1):
    if value is None:
        return "-"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_bool(value):
    if value is None:
        return "-"
    return "yes" if bool(value) else "no"


def tick_rows(summary):
    rows = []
    for tick in summary.get("ticks") or []:
        m = tick.get("metrics") or {}
        bench = m.get("fakbench") or {}
        turntax = m.get("turntax") or {}
        agent = m.get("agent_offline") or {}
        model = m.get("modelbench_q8") or {}
        batch = m.get("batchbench_q8") or {}
        nodes = m.get("node_compare") or {}
        rows.append(
            {
                "tick": tick.get("tick"),
                "failures": tick.get("failures"),
                "duration_sec": tick.get("duration_sec"),
                "boundary": tick.get("boundary_heavy"),
                "fak_gate": bench.get("gate"),
                "fak_p50_ns": bench.get("p50_ns"),
                "speedup_x": bench.get("speedup_x"),
                "turns_saved": turntax.get("turns_saved"),
                "tokens_saved": turntax.get("tokens_saved"),
                "agent_turns_saved": agent.get("turns_saved"),
                "baseline_injection": agent.get("baseline_injection_in_context"),
                "fak_injection": agent.get("fak_injection_in_context"),
                "model_tok_s": model.get("decode_tok_per_sec"),
                "model_cap": model.get("workload_prefill_cap"),
                "batch_tok_s": batch.get("agg_tok_per_sec"),
                "batch": batch.get("batch"),
                "nodes": nodes.get("nodes"),
                "hosts": nodes.get("hosts") or [],
            }
        )
    return rows


def endpoint_state(run_root: Path):
    out = {}
    for tick_dir in sorted(run_root.glob("h[0-9][0-9]")):
        endpoints = load_json(tick_dir / "endpoints.json")
        if not isinstance(endpoints, list):
            continue
        out[tick_dir.name] = [
            {
                "name": e.get("name"),
                "state": e.get("state"),
                "serve": e.get("serve"),
                "ssh": e.get("ssh"),
                "rtt": e.get("rtt"),
            }
            for e in endpoints
        ]
    return out


def aux_gates(run_root: Path):
    gates = []
    for d in sorted(run_root.glob("post-*-source-gate*")):
        rep = load_json(d / "summary.json")
        if not isinstance(rep, dict):
            continue
        gates.append(
            {
                "name": d.name,
                "started_utc": rep.get("started_utc"),
                "ended_utc": rep.get("ended_utc"),
                "go_vet_exit": rep.get("go_vet_exit"),
                "go_test_exit": rep.get("go_test_exit"),
                "note": rep.get("note"),
            }
        )
    return gates


def latest_run(root: Path) -> Path:
    runs = [p for p in root.glob("overnight-*") if p.is_dir()]
    if not runs:
        raise FileNotFoundError(f"no overnight-* run under {root}")
    return max(runs, key=lambda p: p.stat().st_mtime)


def summarize(run_root: Path):
    summary = load_json(run_root / "summary.json") or {
        "run_root": str(run_root),
        "ticks": [],
        "completed_samples": 0,
        "requested_samples": None,
        "failures": None,
    }
    rows = tick_rows(summary)
    return {
        "run_root": str(run_root),
        "updated_utc": summary.get("updated_utc"),
        "completed_samples": summary.get("completed_samples"),
        "requested_samples": summary.get("requested_samples"),
        "failures": summary.get("failures"),
        "ticks": rows,
        "endpoints": endpoint_state(run_root),
        "aux_gates": aux_gates(run_root),
    }


def print_table(report):
    print(f"run: {report['run_root']}")
    print(
        "samples: "
        f"{report.get('completed_samples')}/{report.get('requested_samples')}  "
        f"failures={report.get('failures')}  updated={report.get('updated_utc') or '-'}"
    )
    print()
    header = (
        "tick", "fail", "sec", "gate", "p50ns", "speedup", "turns",
        "model tok/s", "batch tok/s", "B", "nodes", "inject now->fak"
    )
    print("{:<5} {:>4} {:>7} {:<5} {:>8} {:>8} {:>5} {:>11} {:>11} {:>4} {:>5} {:>14}".format(*header))
    for r in report["ticks"]:
        inj = f"{fmt_bool(r.get('baseline_injection'))}->{fmt_bool(r.get('fak_injection'))}"
        print(
            "{:<5} {:>4} {:>7} {:<5} {:>8} {:>8} {:>5} {:>11} {:>11} {:>4} {:>5} {:>14}".format(
                r.get("tick") or "-",
                r.get("failures"),
                fmt_num(r.get("duration_sec"), 0),
                r.get("fak_gate") or "-",
                r.get("fak_p50_ns") or "-",
                fmt_num(r.get("speedup_x"), 1),
                r.get("turns_saved") if r.get("turns_saved") is not None else "-",
                fmt_num(r.get("model_tok_s"), 1),
                fmt_num(r.get("batch_tok_s"), 1),
                r.get("batch") or "-",
                r.get("nodes") if r.get("nodes") is not None else "-",
                inj,
            )
        )
    if report["endpoints"]:
        print("\nendpoints:")
        for tick, endpoints in report["endpoints"].items():
            parts = []
            for e in endpoints:
                serve = "serve" if e.get("serve") else "no-serve"
                ssh = "ssh" if e.get("ssh") else "no-ssh"
                parts.append(f"{e.get('name')}={e.get('state')}:{serve}:{ssh}:rtt{e.get('rtt')}")
            print(f"  {tick}: " + "; ".join(parts))
    if report["aux_gates"]:
        print("\naux source gates:")
        for g in report["aux_gates"]:
            print(
                "  "
                f"{g.get('name')}: vet={g.get('go_vet_exit')} "
                f"test={g.get('go_test_exit')} ended={g.get('ended_utc') or '-'}"
            )


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", nargs="?", help="overnight run root; defaults to latest under tools/_registry/soak")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    if args.run_root:
        run_root = Path(args.run_root).resolve()
    else:
        run_root = latest_run(Path("tools") / "_registry" / "soak").resolve()
    report = summarize(run_root)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_table(report)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
