#!/usr/bin/env python3
"""Run realistic fak-vs-llama turn-agent sweeps.

The smoke grid proves wiring. These profiles are the benchmark grids worth citing:

  interactive: 10/20 turns, 8-20 agents, fat tool-result context, 8k-fit.
  long:        40/80 turns, 8-20 agents, compressed/rolled context, 8k-fit.
  bounded-long: endpoint baseline for the long grid when the full grid is too slow.
  mac-longer:  80/120 turns, 8-20 agents, longer 8k-fit Mac/native node probe.
  mac-stress:  beyond-8k capacity stress profile, opt-in only.
  probe:       one realistic-ish cell for local validation before a long run.

The reported per-agent context is the semantic sequence length to keep under the
model's train context. The llama peer uses a single unified KV arena, so its
context allocation also scales with agent count.
"""
import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import fleet_version


PROFILES = {
    "probe": {
        "description": "single realistic validation cell; not canonical",
        "prefix": 1024,
        "turns": "10",
        "agents": "8",
        "decode": 32,
        "result": 64,
        "reps": 1,
    },
    "interactive": {
        "description": "10-20 turn active agents with uncompressed/fat result context",
        "prefix": 2048,
        "turns": "10,20",
        "agents": "8,12,16,20",
        "decode": 64,
        "result": 256,
        "reps": 3,
    },
    "long": {
        "description": "40-80 turn agents with compacted/rolled per-turn context",
        "prefix": 1024,
        "turns": "40,80",
        "agents": "8,12,16,20",
        "decode": 32,
        "result": 48,
        "reps": 3,
    },
    "bounded-long": {
        "description": "endpoint baseline for long profile: 40/80 turns at 8/20 agents",
        "prefix": 1024,
        "turns": "40,80",
        "agents": "8,20",
        "decode": 32,
        "result": 48,
        "reps": 1,
        "ablation": False,
    },
    "mac-longer": {
        "description": "80-120 turn native-node probe; longer than local long while staying 8k-fit",
        "prefix": 2048,
        "turns": "80,120",
        "agents": "8,12,16,20",
        "decode": 32,
        "result": 16,
        "reps": 1,
        "ablation": False,
    },
    "mac-stress": {
        "description": "beyond-8k native-node capacity stress; do not cite as model-context-valid",
        "prefix": 4096,
        "turns": "120,160",
        "agents": "8,12,16",
        "decode": 32,
        "result": 16,
        "reps": 1,
        "ablation": False,
    },
}


def parse_ints(s):
    return [int(x) for x in s.split(",") if x.strip()]


def repo_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def profile_estimates(p):
    app_ver = fleet_version.app_version()
    turns = parse_ints(p["turns"])
    agents = parse_ints(p["agents"])
    rows = []
    for t in turns:
        per_agent_tail = t * p["decode"] + max(0, t - 1) * p["result"]
        per_agent_ctx = p["prefix"] + per_agent_tail
        for a in agents:
            rows.append(
                {
                    "version": app_ver,
                    "turns": t,
                    "agents": a,
                    "per_agent_context_tokens": per_agent_ctx,
                    "unified_kv_cells": p["prefix"] + a * per_agent_tail,
                    "llama_n_ctx_with_slack": p["prefix"] + a * per_agent_tail + 64,
                    "agent_turns": t * a,
                }
            )
    return rows


def run(cmd, cwd, dry_run):
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=cwd, check=True)


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--profile",
        choices=["probe", "interactive", "long", "bounded-long", "mac-longer", "mac-stress", "all"],
        default="probe",
        help="which sweep profile to run",
    )
    ap.add_argument("--engine", choices=["both", "fak", "llama"], default="both")
    ap.add_argument("--reps", type=int, default=0, help="override profile reps")
    ap.add_argument("--out-dir", default="fak/experiments/model-baseline/turn-agent-realistic")
    ap.add_argument("--gguf", default="experiments/model-baseline/gguf/SmolLM2-135M-Instruct-Q8_0.gguf")
    ap.add_argument("--baseline-label", default="tuned-llama.cpp",
                    help="label for the tuned reference in compare artifacts")
    ap.add_argument("--candidate-tuning-mult", type=float, default=1.0,
                    help="explicit projection multiplier for future tuned FAK throughput in compare artifacts")
    ap.add_argument("--candidate-tuning-label", default="measured-current",
                    help="label for the candidate tuning projection")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    if args.candidate_tuning_mult <= 0:
        ap.error("--candidate-tuning-mult must be > 0")

    root = repo_root()
    fak_dir = os.path.join(root, "fak")
    out_dir = args.out_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(root, out_dir)
    os.makedirs(out_dir, exist_ok=True)

    names = ["interactive", "long"] if args.profile == "all" else [args.profile]
    manifest = {
        "schema": "fak.turn-agent-realistic-sweep.v1",
        "app_version": fleet_version.app_version(Path(root)),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "engine": args.engine,
        "baseline_policy": {
            "headline_reference": args.baseline_label,
            "baseline_role": "tuned_external_reference",
            "candidate_tuning_label": args.candidate_tuning_label,
            "candidate_tuning_multiplier": args.candidate_tuning_mult,
            "naive_or_noreuse_role": "internal_ablation_only",
        },
        "profiles": [],
    }

    for name in names:
        p = dict(PROFILES[name])
        if args.reps > 0:
            p["reps"] = args.reps
        p["version"] = fleet_version.app_version(Path(root))
        p["name"] = name
        p["estimates"] = profile_estimates(p)
        manifest["profiles"].append(p)

        fak_out = os.path.join(out_dir, f"{name}-fak-q8.json")
        llama_out = os.path.join(out_dir, f"{name}-llamacpp-q8.json")
        compare_out = os.path.join(out_dir, f"{name}-compare.json")

        if args.engine in ("both", "fak"):
            run(
                [
                    "go",
                    "run",
                    "./cmd/fleetserve",
                    "-quant",
                    "-prefix",
                    str(p["prefix"]),
                    "-turns",
                    p["turns"],
                    "-decode",
                    str(p["decode"]),
                    "-result",
                    str(p["result"]),
                    "-concurrency",
                    p["agents"],
                    "-reps",
                    str(p["reps"]),
                    f"-ablation={str(p.get('ablation', True)).lower()}",
                    "-out",
                    fak_out,
                ],
                cwd=fak_dir,
                dry_run=args.dry_run,
            )

        if args.engine in ("both", "llama"):
            run(
                [
                    sys.executable,
                    "internal/model/bench_llamacpp_turn_agents.py",
                    "--gguf",
                    args.gguf,
                    "--prefix",
                    str(p["prefix"]),
                    "--turns",
                    p["turns"],
                    "--agents",
                    p["agents"],
                    "--decode",
                    str(p["decode"]),
                    "--result",
                    str(p["result"]),
                    "--reps",
                    str(p["reps"]),
                    "--out",
                    llama_out,
                ],
                cwd=fak_dir,
                dry_run=args.dry_run,
            )

        if args.engine == "both":
            run(
                [
                    sys.executable,
                    "tools/fak_llama_turn_agent_compare.py",
                    "--fak",
                    fak_out,
                    "--llama",
                    llama_out,
                    "--baseline-label",
                    args.baseline_label,
                    "--candidate-tuning-mult",
                    str(args.candidate_tuning_mult),
                    "--candidate-tuning-label",
                    args.candidate_tuning_label,
                    "--out",
                    compare_out,
                ],
                cwd=root,
                dry_run=args.dry_run,
            )

    manifest_path = os.path.join(out_dir, "manifest.json")
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        print(f"dry-run: would write {manifest_path}")
    else:
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        print(f"wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
