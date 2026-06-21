#!/usr/bin/env python3
"""Compare fak fleetserve against a tuned turn-agent reference artifact.

The reference artifact is normally llama.cpp with its tuned shared-prefix settings
(`kv_unified` + `seq_cp`) enabled. The old fak-vs-fak no-reuse speedup remains in
the output as an ablation, but the headline comparator is the tuned reference.
"""
import argparse
import json
import os
import sys

import fleet_version


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def key_fak(point):
    return int(point["turns"]), int(point["concurrency"])


def key_llama(point):
    return int(point["turns"]), int(point["agents"])


def ratio(a, b):
    if b in (0, None):
        return None
    return a / b


def scaled(value, multiplier):
    if value is None:
        return None
    return value * multiplier


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fak", required=True, help="fleetserve JSON artifact")
    ap.add_argument("--llama", required=True, help="tuned reference JSON artifact, usually bench_llamacpp_turn_agents.py output")
    ap.add_argument("--baseline-label", default="tuned-llama.cpp",
                    help="label for the tuned reference in the JSON summary")
    ap.add_argument("--candidate-tuning-mult", type=float, default=1.0,
                    help="explicit projection multiplier for future tuned FAK throughput; 1.0 means measured current FAK only")
    ap.add_argument("--candidate-tuning-label", default="measured-current",
                    help="label for the candidate tuning projection")
    ap.add_argument("--out", default="", help="optional JSON summary path")
    args = ap.parse_args(argv)
    if args.candidate_tuning_mult <= 0:
        ap.error("--candidate-tuning-mult must be > 0")

    fak = load(args.fak)
    llama = load(args.llama)
    app_ver = fleet_version.app_version()
    fak_points = {key_fak(p): p for p in fak.get("points", [])}
    llama_points = {key_llama(p): p for p in llama.get("points", [])}

    rows = []
    for key in sorted(set(fak_points) & set(llama_points)):
        fp = fak_points[key]
        lp = llama_points[key]
        f_turns = fp.get("reuse_agent_turns_per_sec")
        l_turns = lp.get("agent_turns_per_sec")
        f_agents = fp.get("reuse_agents_per_sec")
        l_agents = lp.get("agents_per_sec")
        projected_turns = scaled(f_turns, args.candidate_tuning_mult)
        projected_agents = scaled(f_agents, args.candidate_tuning_mult)
        rows.append(
            {
                "version": app_ver,
                "turns": key[0],
                "agents": key[1],
                "fak_reuse_total_ms": fp.get("reuse_total_ms"),
                "llama_total_ms": lp.get("total_ms"),
                "fak_reuse_agent_turns_per_sec": f_turns,
                "llama_agent_turns_per_sec": l_turns,
                "fak_vs_llama_agent_turns": ratio(f_turns, l_turns),
                "tuned_reference_agent_turns_per_sec": l_turns,
                "fak_current_vs_tuned_reference": ratio(f_turns, l_turns),
                "fak_current_gap_to_tuned_reference": ratio(l_turns, f_turns),
                "fak_projected_agent_turns_per_sec": projected_turns,
                "fak_projected_vs_tuned_reference": ratio(projected_turns, l_turns),
                "fak_reuse_agents_per_sec": f_agents,
                "llama_agents_per_sec": l_agents,
                "fak_vs_llama_agents": ratio(f_agents, l_agents),
                "tuned_reference_agents_per_sec": l_agents,
                "fak_current_agents_vs_tuned_reference": ratio(f_agents, l_agents),
                "fak_projected_agents_per_sec": projected_agents,
                "fak_projected_agents_vs_tuned_reference": ratio(projected_agents, l_agents),
                "fak_reuse_speedup_vs_noreuse": fp.get("reuse_speedup_vs_noreuse"),
            }
        )

    summary = {
        "schema": "fak.llama-turn-agent-compare.v1",
        "app_version": app_ver,
        "fak": args.fak,
        "llama": args.llama,
        "fak_engine": fak.get("engine"),
        "llama_engine": llama.get("engine"),
        "baseline_policy": {
            "headline_reference": args.baseline_label,
            "baseline_role": "tuned_external_reference",
            "candidate_tuning_label": args.candidate_tuning_label,
            "candidate_tuning_multiplier": args.candidate_tuning_mult,
            "naive_or_noreuse_role": "internal_ablation_only",
        },
        "prefix_len": fak.get("prefix_len"),
        "decode_steps_per_turn": fak.get("decode_steps_per_turn"),
        "result_tokens_between_turns": fak.get("result_tokens_between_turns"),
        "cells_compared": len(rows),
        "cells": rows,
    }

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")

    if not rows:
        print("no overlapping (turns, agents) cells", file=sys.stderr)
        return 1

    if args.candidate_tuning_mult == 1.0:
        print("turns agents fak_turns/s tuned_ref_turns/s fak_current/tuned_ref")
    else:
        print("turns agents fak_turns/s tuned_ref_turns/s current/tuned_ref projected/tuned_ref")
    for row in rows:
        current = row["fak_current_vs_tuned_reference"]
        if args.candidate_tuning_mult == 1.0:
            print(
                f'{row["turns"]:>5} {row["agents"]:>6} '
                f'{row["fak_reuse_agent_turns_per_sec"]:>11.3f} '
                f'{row["tuned_reference_agent_turns_per_sec"]:>18.3f} '
                f'{current:>21.3f}x'
            )
        else:
            projected = row["fak_projected_vs_tuned_reference"]
            print(
                f'{row["turns"]:>5} {row["agents"]:>6} '
                f'{row["fak_reuse_agent_turns_per_sec"]:>11.3f} '
                f'{row["tuned_reference_agent_turns_per_sec"]:>18.3f} '
                f'{current:>17.3f}x {projected:>19.3f}x'
            )
    if args.out:
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
