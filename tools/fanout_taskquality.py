#!/usr/bin/env python3
"""Task-quality litmus for the fanbench one-goal -> N-subagent grid (issue #429).

`fanbench` (FANOUT-BENCH-RESULTS.md) prices the KERNEL COST GEOMETRY of fanning one
master goal out to N sub-agents: the measured cross-agent tool-result dedup and the
exact shared-prefix KV-reuse `(N-1)*P` the kernel never redoes. It is deliberately
silent on whether wider fan-out improves TASK SUCCESS, coverage, or answer quality --
the research brief (`experiments/fanout/RESEARCH-BRIEF-fanout-2026-06-17.md` Section 8)
lists "real task success / coverage@N / realized@N" as an explicit out-of-scope seam.

This tool closes the honest gap GitHub issue #429 names. It runs a CONTROLLED LITMUS
task suite -- one master goal whose full solution is a known ground-truth set of G
sub-result "atoms" -- through the same flat fan-out of N sub-agents fanbench sweeps,
with a fold/verifier and an adversarial-injection fraction, and reports the TASK
metrics the cost grid cannot:

  coverage@N        distinct ground-truth atoms any sub-agent produced / G
  realized@N        atoms the imperfect verifier/fold actually accepted / G
  verifier_success  precision of the accepted set (accepted-correct / accepted-total)
  duplicate_work    fraction of competent sub-agent outputs that re-derive a covered atom
  failed_rate       fraction of sub-agents that produced nothing useful (decoy/irrelevant)
  injection_*       sub-agents fed an adversarial tool result, and how many were CONTAINED
                    under the fak arm (quarantined) vs the naive arm (poisoned -> atom lost)

Each row is JOINED to the real fanbench cost cell for the same N
(`experiments/fanout/fanbench-research.csv`: calls, cross_uplift, tax_clawed_back,
parallel_speedup) so the artifact literally connects a litmus task run to the
fanbench-like N grid -- the issue's first acceptance bullet.

It also carries a MATCHED-BUDGET SINGLE-AGENT CONTROL: one agent given the fan-out's
total call budget as a single sequential trajectory (no cross-agent redundancy, no
parallelism). Holding total compute constant is the research brief's design law
(Section 6) -- it is what lets the doc say honestly "fan-out saves cost/latency but at
matched budget does NOT prove better quality."

HONESTY (the [SIMULATED] tag this carries in CLAIMS.md): the task OUTCOMES are a
TRANSPARENT, knobbed model, NOT a real-model run. The numbers are grounded in published
anchors -- homogeneous pools saturate ~4 agents (Agent Forest, arXiv:2402.05120);
imperfect-verifier realized accuracy peaks then declines, compute-optimal K<=5
(arXiv:2411.17501); step-repetition is the single most frequent MAST failure mode at
15.7% (arXiv:2503.13657); naive MAS sits ~33-59% correct (MAST). The injection arms are
grounded in the REAL quarantine evidence in `docs/benchmarks/LIVE-RESULTS.md` (Gemini /
local Qwen: poisoned tool output reaches the naive baseline, fak quarantines it). What is
modeled is labeled modeled; what is measured elsewhere is cited, never re-derived here.

Determinism IS the gate (same discipline as `fanout_longctx_probe.py`): a fixed
(profile, N-grid, trials, seed) yields a byte-identical artifact. `--check` regenerates
in memory and diffs against the checked-in JSON, exiting non-zero on any drift.

Usage:
  python tools/fanout_taskquality.py                 # regenerate the checked-in artifact
  python tools/fanout_taskquality.py --check          # gate: re-run must reproduce it
  python tools/fanout_taskquality.py --trials 256     # tighter medians (changes artifact)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import random
import statistics
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_CSV = os.path.join(REPO_ROOT, "experiments", "fanout", "fanbench-research.csv")
OUT_JSON = os.path.join(REPO_ROOT, "experiments", "fanout", "taskquality-litmus.json")
OUT_CSV = os.path.join(REPO_ROOT, "experiments", "fanout", "taskquality-litmus.csv")

# --- The controlled litmus task -------------------------------------------------------
# One master goal whose complete solution is G ground-truth atoms. Atom i is discovered
# by a competent sub-agent with weight w_i ~ 1/(i+1) (Zipf-ish): a few atoms are easy,
# a long tail is hard. A skewed pool is what makes coverage grow log-linearly and
# SATURATE (the brief's homogeneous-pool ~4-agent knee) instead of growing forever.
PROFILE = {
    "name": "litmus-research-goal",
    "atoms": 12,            # G: ground-truth sub-results that fully solve the goal
    "p_fail": 0.40,         # per-sub-agent: produces a decoy/irrelevant output (MAST ~33-59% correct)
    "p_inject": 0.20,       # per-sub-agent: fed an adversarial tool result
    "verifier_recall": 0.85,  # fold accepts a produced correct atom w.p. recall
    "verifier_decoy_fa": 0.30,  # fold false-accepts a decoy w.p. fa (imperfect verifier)
    "fold_capacity": 8,     # the fold can surface at most this many candidates (selection bottleneck)
    "subturns": 4,          # attempts per sub-agent (mirrors fanbench sub_turns=4)
}
N_GRID = [1, 2, 4, 8, 16, 64, 256]
DEFAULT_TRIALS = 64
DEFAULT_SEED = 429102931  # fixed -> byte-identical artifact (the gate)


def _atom_weights(g):
    w = [1.0 / (i + 1) for i in range(g)]
    s = sum(w)
    return [x / s for x in w]


def _draw_atom(rng, weights):
    r = rng.random()
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1


def _run_fanout_trial(rng, n, prof, weights, arm):
    """One seeded fan-out trial. arm in {'fak','naive'} -> injection containment policy.

    Returns the per-trial task metrics for a width-N fan-out over the litmus goal.
    """
    produced = set()          # distinct correct atoms surfaced (coverage numerator)
    competent_outputs = 0     # non-failed sub-agent atom productions
    duplicate_outputs = 0     # of those, ones re-deriving an already-covered atom
    failed = 0                # sub-agents that produced no useful atom
    injected = 0              # sub-agents fed an adversarial tool result
    contained = 0             # of those, quarantined (fak) vs poisoned (naive)
    decoys = 0                # wrong candidates entering the fold

    for _ in range(n):
        inj = rng.random() < prof["p_inject"]
        if inj:
            injected += 1
        poisoned = inj and arm == "naive"  # fak quarantines -> sub-agent still produces
        if inj and arm == "fak":
            contained += 1

        # subturns attempts; the best of them is the sub-agent's output.
        got_atom = None
        for _ in range(prof["subturns"]):
            if rng.random() < prof["p_fail"]:
                continue
            a = _draw_atom(rng, weights)
            if got_atom is None:
                got_atom = a

        if poisoned:
            # adversarial tool output derails the naive sub-agent: useful work lost.
            got_atom = None

        if got_atom is None:
            failed += 1
            decoys += 1  # an irrelevant/decoy output still reaches the fold
            continue

        competent_outputs += 1
        if got_atom in produced:
            duplicate_outputs += 1
        produced.add(got_atom)

    g = prof["atoms"]
    coverage = len(produced) / g

    # Fold / verifier: an imperfect selector over (correct candidates + decoys) with a
    # finite capacity. As N grows the decoy pile grows, crowding correct atoms out of the
    # surfaced set -> realized accuracy PEAKS then DECLINES (compute-optimal K<=5).
    candidates = []  # (is_correct, atom_or_None)
    for a in produced:
        candidates.append((True, a))
    for _ in range(decoys):
        candidates.append((False, None))
    rng.shuffle(candidates)

    accepted_correct = set()
    accepted_total = 0
    cap = prof["fold_capacity"]
    for is_correct, a in candidates:
        if accepted_total >= cap:
            break
        if is_correct:
            if rng.random() < prof["verifier_recall"]:
                accepted_correct.add(a)
                accepted_total += 1
        else:
            if rng.random() < prof["verifier_decoy_fa"]:
                accepted_total += 1  # false accept consumes a fold slot

    realized = len(accepted_correct) / g
    verifier_success = (len(accepted_correct) / accepted_total) if accepted_total else 1.0
    dup_rate = (duplicate_outputs / competent_outputs) if competent_outputs else 0.0
    failed_rate = failed / n
    inj_contain_rate = (contained / injected) if injected else 1.0

    return {
        "coverage": coverage,
        "realized": realized,
        "verifier_success": verifier_success,
        "duplicate_work": dup_rate,
        "failed_rate": failed_rate,
        "injected": injected,
        "injection_contained_rate": inj_contain_rate,
    }


def _matched_budget_single_agent(rng, total_calls, prof, weights):
    """One sequential agent given the fan-out's TOTAL call budget (matched compute).

    No cross-agent redundancy (it sees its own context, so it never re-attempts an atom
    it already holds) and no parallelism. This is the budget-controlled control the
    research brief demands: at equal compute a single agent matches/beats sequential
    fan-out, which is exactly why a fan-out 'win' must be read as cost/latency, not
    quality.
    """
    produced = set()
    failed = 0
    for _ in range(total_calls):
        if rng.random() < prof["p_fail"]:
            failed += 1
            continue
        a = _draw_atom(rng, weights)
        produced.add(a)  # self-context: a re-derived atom is simply already held
    g = prof["atoms"]
    return {
        "coverage": len(produced) / g,
        "failed_rate": failed / total_calls if total_calls else 0.0,
        "calls": total_calls,
    }


def _load_cost_grid():
    """Join key cost columns from the real fanbench artifact, keyed by N (agents)."""
    grid = {}
    if not os.path.exists(COST_CSV):
        return grid
    with open(COST_CSV, newline="") as f:
        for row in csv.DictReader(f):
            grid[int(row["agents"])] = {
                "calls": int(row["calls"]),
                "cross_uplift": int(row["cross_uplift_p50"]),
                "tax_clawed_back": float(row["tax_clawed_back"]),
                "parallel_speedup": float(row["parallel_speedup"]),
            }
    return grid


def _med(xs):
    m = statistics.median(xs)
    return round(m, 4)


def build(trials, seed):
    prof = PROFILE
    weights = _atom_weights(prof["atoms"])
    cost = _load_cost_grid()
    cells = []
    for n in N_GRID:
        fak, naive, single = [], [], []
        for t in range(trials):
            rng = random.Random(seed * 1000003 + n * 101 + t)
            fak.append(_run_fanout_trial(rng, n, prof, weights, "fak"))
            naive.append(_run_fanout_trial(rng, n, prof, weights, "naive"))
            c = cost.get(n, {}).get("calls", prof["subturns"] * n + prof["subturns"])
            single.append(_matched_budget_single_agent(rng, c, prof, weights))

        def col(rows, k):
            return [r[k] for r in rows]

        cell = {
            "n": n,
            "trials": trials,
            "coverage_at_n": _med(col(fak, "coverage")),
            "realized_at_n": _med(col(fak, "realized")),
            "verifier_success": _med(col(fak, "verifier_success")),
            "duplicate_work_rate": _med(col(fak, "duplicate_work")),
            "failed_subagent_rate": _med(col(fak, "failed_rate")),
            "coverage_naive_arm": _med(col(naive, "coverage")),
            "injection_contained_fak": _med(col(fak, "injection_contained_rate")),
            "injection_contained_naive": _med(col(naive, "injection_contained_rate")),
            "matched_budget_single_coverage": _med(col(single, "coverage")),
            "matched_budget_single_calls": int(statistics.median(col(single, "calls"))),
            # joined cost cell from the real fanbench grid:
            "cost_calls": cost.get(n, {}).get("calls"),
            "cost_cross_uplift": cost.get(n, {}).get("cross_uplift"),
            "cost_tax_clawed_back": cost.get(n, {}).get("tax_clawed_back"),
            "cost_parallel_speedup": cost.get(n, {}).get("parallel_speedup"),
        }
        cells.append(cell)

    artifact = {
        "schema": "fanout-taskquality-litmus/v1",
        "issue": 429,
        "kind": "SIMULATED",
        "banner": (
            "CONTROLLED LITMUS, NOT a real-model run. Task outcomes are a transparent "
            "knobbed model grounded in published anchors (Agent Forest saturation ~4; "
            "imperfect-verifier K<=5 inversion; MAST step-repetition 15.7%; naive MAS "
            "~33-59% correct) and the real injection quarantine evidence in "
            "docs/benchmarks/LIVE-RESULTS.md. Cost columns are joined verbatim from the "
            "MEASURED fanbench artifact experiments/fanout/fanbench-research.csv. This "
            "lane proves the cost/quality SEPARATION; it does not prove real-model quality."
        ),
        "profile": prof,
        "n_grid": N_GRID,
        "trials": trials,
        "seed": seed,
        "cost_source": "experiments/fanout/fanbench-research.csv",
        "cells": cells,
    }
    return artifact


def to_csv(artifact):
    buf = io.StringIO()
    cols = [
        "n", "coverage_at_n", "realized_at_n", "verifier_success",
        "duplicate_work_rate", "failed_subagent_rate", "coverage_naive_arm",
        "injection_contained_fak", "injection_contained_naive",
        "matched_budget_single_coverage", "matched_budget_single_calls",
        "cost_calls", "cost_cross_uplift", "cost_tax_clawed_back", "cost_parallel_speedup",
    ]
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(cols)
    for c in artifact["cells"]:
        w.writerow([c.get(k) for k in cols])
    return buf.getvalue()


def _dumps(artifact):
    return json.dumps(artifact, indent=2, sort_keys=False) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--check", action="store_true",
                    help="regenerate in memory and diff against the checked-in JSON; "
                         "exit non-zero on drift (the determinism gate)")
    args = ap.parse_args(argv)

    artifact = build(args.trials, args.seed)
    new_json = _dumps(artifact)
    new_csv = to_csv(artifact)

    if args.check:
        ok = True
        if not os.path.exists(OUT_JSON):
            print(f"CHECK FAIL: missing {OUT_JSON}", file=sys.stderr)
            return 1
        with open(OUT_JSON, encoding="utf-8") as f:
            cur = f.read()
        if cur != new_json:
            print("CHECK FAIL: taskquality-litmus.json drifted from a fresh "
                  f"--trials {args.trials} --seed {args.seed} run", file=sys.stderr)
            ok = False
        if os.path.exists(OUT_CSV):
            with open(OUT_CSV, encoding="utf-8") as f:
                if f.read() != new_csv:
                    print("CHECK FAIL: taskquality-litmus.csv drifted", file=sys.stderr)
                    ok = False
        if ok:
            print(f"CHECK OK: artifact reproduces byte-for-byte "
                  f"({len(artifact['cells'])} N cells, {args.trials} trials)")
            return 0
        return 1

    with open(OUT_JSON, "w", encoding="utf-8", newline="") as f:
        f.write(new_json)
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        f.write(new_csv)
    print(f"wrote {os.path.relpath(OUT_JSON, REPO_ROOT)} and "
          f"{os.path.relpath(OUT_CSV, REPO_ROOT)} "
          f"({len(artifact['cells'])} N cells, {args.trials} trials, seed {args.seed})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
