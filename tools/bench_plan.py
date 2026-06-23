#!/usr/bin/env python3
r"""bench_plan.py -- the recurring hardware-benchmark PLANNER.

The benchmark catalog (experiments/benchmark/catalog.json, maintained by
tools/bench_catalog.py) is a passive REGISTRY: it records what every bench-node
(macbook / datacenter-A100 / cloud-L4 / RTX-laptop) HAS run. It never decides what to run
NEXT. This tool is the missing brain: a deterministic, read-only DRIVER that folds
the catalog at an injected ``--now`` stamp into a ranked plan of the single
highest-value next test per bench-node, spanning the four operator intents:

  * benchmark      -- (re)measure throughput on hardware
  * learn-collect  -- first-ever data on a new node / model
  * regression     -- re-measure a recorded baseline before it drifts (the
                      tok/s analogue of tools/bench_witness.py's keep-floor)
  * coverage       -- fill an empty (machine x workload-kind) cell

It can ONLY plan. There is no execute path, no ``--live`` arm; it never writes the
catalog and never spawns a run. Rendering the plan IS its only effect -- the
strongest possible honesty guarantee, and correct because THIS box is the
agent-host (role != bench-node), so a real run is a later human/remote action on
the bench-node. It mirrors tools/dispatch_status.py's pure read-only ``--md`` fold.

SCORING is coverage-first BY DOMINANCE but four-intent BY CONSTRUCTION: an empty
feasible cell lexicographically outranks any non-empty cell before the weighted
sum is consulted (so a100 at 0 runs and the thin L4 always lead), while WITHIN a
tier a weighted sum blends marginal-information-gain (machine novelty, model
diversity) and regression urgency (per-kind re-check interval, recorded-baseline
drift). Every number is a literal catalog fact; "today" is the explicit ``--now``
stamp (no wall-clock is read in this tool), so a fixed ``--now`` yields
byte-identical output and a unit test can pin it.

    python tools/bench_plan.py --now 20260622T140000Z
    python tools/bench_plan.py --now 20260622T140000Z --json
    python tools/bench_plan.py --now 20260622T140000Z --md docs/bench-plan.md
    python tools/bench_plan.py --now 20260622T140000Z --machine a100 --intent coverage
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "bench-plan/1"

# --- the canonical workload-kinds the planner schedules (machine x kind cells) ---
# A run OCCUPIES cell (m, k) when k is in its tags. Headline kinds carry a concrete
# subject model; the model-agnostic workloads do not. The suggested_command names a
# REAL ./cmd binary (verified present) so the operator hint is runnable, not invented.
KIND_META: dict[str, dict[str, Any]] = {
    "model-benchmark":   {"cmd": "go run ./cmd/modelbench -quant",                 "model": ("SmolLM2-135M-Instruct", "q8_0"), "headline": True},
    "gpu-benchmark":     {"cmd": "go run -tags cuda ./cmd/gpucheck",               "model": ("qwen2.5-3b", "Q8_0"),            "headline": True},
    "qwen36":            {"cmd": "fak serve + fak agent (qwen3.6-27b via gateway)", "model": ("qwen3.6-27b", "q8"),             "headline": True},
    "radix-benchmark":   {"cmd": "go run ./cmd/radixbench",                         "model": ("SmolLM2-135M", "q8"),            "headline": False},
    "session-benchmark": {"cmd": "go run ./cmd/sessionbench",                       "model": (None, "n/a"),                     "headline": False},
    "fan-benchmark":     {"cmd": "go run ./cmd/fanbench",                           "model": (None, "n/a"),                     "headline": False},
    "agent-live":        {"cmd": "go run ./cmd/fak agent --task <task>",            "model": (None, "n/a"),                     "headline": False},
    "turn-tax":          {"cmd": "go run ./cmd/fak turntax --suite turntax-airline", "model": (None, "n/a"),                    "headline": False},
    "parity":            {"cmd": "go run ./cmd/paritybench",                        "model": (None, "n/a"),                     "headline": False},
}
KINDS: list[str] = list(KIND_META)
MODEL_KINDS = {k for k, v in KIND_META.items() if v["model"][0] is not None}

# CUDA kinds need an NVIDIA GPU; struck out BEFORE scoring on a non-NVIDIA node (so the
# planner never recommends a gpu-benchmark on the Apple mac). METAL is reserved for a
# future Metal-only kind; empty today, so the load-bearing rule is "no CUDA on the mac".
CUDA_KINDS = {"gpu-benchmark"}
METAL_KINDS: set[str] = set()

# Per-kind re-check interval (days): how long a recorded measurement stays "fresh"
# before a re-measure is overdue -- the time analogue of bench_witness.py's
# --tolerance-pct. Headline throughput drifts fastest; one-off workloads slowest.
RECHECK_DAYS = {
    "model-benchmark": 7, "gpu-benchmark": 7,
    "qwen36": 14, "radix-benchmark": 14, "session-benchmark": 14, "fan-benchmark": 14,
    "agent-live": 30, "turn-tax": 30, "parity": 30,
}
DEFAULT_RECHECK_DAYS = 14
DEFAULT_STALE_HORIZON_DAYS = 14

# Dimension weights (sum to 1.0). coverage_gap dominates AND is the lexicographic
# primary key, so an empty cell always outranks a non-empty one; the rest reorder
# within a tier.
WEIGHTS = {
    "coverage_gap": 0.34,
    "machine_novelty": 0.20,
    "staleness_overdue": 0.16,
    "baseline_drift": 0.16,
    "model_diversity": 0.14,
}
# The four operator intents. A cell's label is a SEMANTIC role (cell_intent), not an
# argmax over the scoring dimensions: argmax would let coverage_gap (0.34) swallow every
# empty cell and never surface learn-collect or benchmark, defeating the "show all four
# intents" goal. The dimensions still drive the SCORE; the intent classifies the cell so
# the per-intent sections are populated and an operator can filter (--intent benchmark).
INTENTS = ["benchmark", "learn-collect", "regression", "coverage"]


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


# ----------------------------- time (deterministic) -----------------------------

def parse_stamp(s: str | None) -> datetime | None:
    """Parse a catalog/``--now`` stamp to an aware UTC datetime, or None.

    Accepts the compact catalog form ``20260620T164719Z`` and ISO-8601
    (``2026-06-22T14:00:00Z`` / ``2026-06-22``). No wall-clock is read here.
    """
    if not s:
        return None
    s = s.strip()
    m = re.fullmatch(r"(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})(\d{2})Z?", s)
    if m:
        y, mo, d, h, mi, se = (int(g) for g in m.groups())
        try:
            return datetime(y, mo, d, h, mi, se, tzinfo=timezone.utc)
        except ValueError:
            return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def age_days(now: datetime, stamp: str | None) -> float | None:
    """Whole-and-fractional days from ``stamp`` to ``now``, clamped >= 0 (a ``--now``
    earlier than the stamp -- clock skew -- never yields a negative age). None if the
    stamp is missing/unparseable."""
    ts = parse_stamp(stamp)
    if ts is None:
        return None
    return max(0.0, (now - ts).total_seconds() / 86400.0)


# ------------------------------- catalog folding --------------------------------

def load_catalog(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else None
    except (OSError, ValueError):
        return None


def run_kinds(run: dict[str, Any]) -> set[str]:
    """Canonical kinds a run occupies. A run with a recorded throughput number
    (peak_tok_per_sec / speedup) IS a model-benchmark result however it was tagged --
    this is what hangs the lone phase0 baseline (31.02 tok/s, tagged kernel/batch/phase0)
    onto the model-benchmark cell so the regression leg has a watch-point. A run whose
    tags map to no canonical kind (e.g. the L4's gcp/gpu/ada tags) occupies no cell: a
    deliberate canonical-coverage choice, not a claim the node is unused."""
    tags = set(run.get("tags") or [])
    kinds = {k for k in KINDS if k in tags}
    if run.get("peak_tok_per_sec") is not None or run.get("speedup") is not None:
        kinds.add("model-benchmark")
    return kinds


def feasible(machine: dict[str, Any], kind: str) -> tuple[bool, str]:
    """Hardware feasibility, evaluated BEFORE scoring. Infeasible cells are never
    scored-to-zero; they render '-' in the matrix and never enter the ranking."""
    gpu = (machine.get("gpu") or "").lower()
    os_ = (machine.get("os") or "").lower()
    arch = (machine.get("arch") or "").lower()
    is_nvidia = "nvidia" in gpu
    if kind in CUDA_KINDS:
        if not is_nvidia:
            return False, f"needs an NVIDIA GPU (has '{machine.get('gpu')}')"
        return True, f"NVIDIA GPU present ({machine.get('gpu')})"
    if kind in METAL_KINDS:
        if not (("macos" in os_ or "darwin" in os_) and "arm64" in arch):
            return False, f"needs Apple Metal (macOS/arm64; is {machine.get('os')}/{machine.get('arch')})"
        return True, "Apple Metal node"
    return True, "hardware-agnostic"


def most_common(pairs: list[tuple]) -> tuple | None:
    """Deterministic mode of a list of hashables (ties -> smallest value)."""
    if not pairs:
        return None
    counts = Counter(pairs)
    return min(counts, key=lambda v: (-counts[v], v))


# --------------------------------- scoring --------------------------------------

def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def score_cell(*, machine: dict[str, Any], kind: str, cell_runs: list[dict[str, Any]],
               now: datetime, recheck_days: dict[str, int], stale_horizon: int,
               model_counts: Counter, pair_seen: set[tuple], total_runs: int,
               cand_model: str | None, cand_prec: str | None) -> dict[str, Any]:
    runs_in_cell = len(cell_runs)
    is_empty = runs_in_cell == 0
    interval = recheck_days.get(kind, stale_horizon)

    # newest run in the cell, and the age of the latest measurement.
    last_ts = max((r.get("timestamp") or "" for r in cell_runs), default="") or None
    cell_age = age_days(now, last_ts) if last_ts else None
    has_baseline = any(r.get("peak_tok_per_sec") is not None or r.get("speedup") is not None
                       for r in cell_runs)
    baseline_val = next((r.get("peak_tok_per_sec") for r in cell_runs
                         if r.get("peak_tok_per_sec") is not None), None)

    # coverage_gap: an empty cell is the coverage hole (1.0); a covered cell decays.
    coverage_gap = 1.0 if is_empty else _clamp(1.0 / (1 + runs_in_cell))

    # machine_novelty: a never-measured / thin node first-touch is most informative.
    m_runs = int(machine.get("runs") or 0)
    machine_novelty = 1.0 if (machine.get("last_run") in (None, "") or m_runs == 0) \
        else _clamp(1.0 / (1 + m_runs))

    # staleness_overdue: empty == maximally stale; else age vs the per-kind interval,
    # saturating at 2x the interval.
    if is_empty:
        staleness_overdue = 1.0
    elif cell_age is None:
        staleness_overdue = 0.5  # data-quality: a run with no parseable timestamp
    else:
        staleness_overdue = _clamp(cell_age / interval, 0.0, 2.0) / 2.0

    # baseline_drift: only a cell with a RECORDED number is a regression watch-point;
    # the drift risk grows as that number ages past its re-check interval.
    if has_baseline and cell_age is not None:
        baseline_drift = _clamp(cell_age / interval, 0.0, 1.0)
    else:
        baseline_drift = 0.0

    # model_diversity: rewards a coverage cell that proposes a rare/new model; a
    # re-measure of an already-covered model adds little model information.
    if not is_empty:
        model_diversity = 0.1
    elif kind not in MODEL_KINDS:
        model_diversity = 0.5  # model-agnostic workload -> neutral
    else:
        share = (model_counts.get(cand_model, 0) / total_runs) if total_runs else 0.0
        model_diversity = 1.0 - share
        if (cand_model, cand_prec) not in pair_seen:
            model_diversity = min(1.0, model_diversity + 0.25)
    model_diversity = round(model_diversity, 4)

    dims = {
        "coverage_gap": round(coverage_gap, 4),
        "machine_novelty": round(machine_novelty, 4),
        "staleness_overdue": round(staleness_overdue, 4),
        "baseline_drift": round(baseline_drift, 4),
        "model_diversity": model_diversity,
    }
    cell_score = round(sum(WEIGHTS[d] * dims[d] for d in WEIGHTS), 4)

    return {
        "runs_in_cell": runs_in_cell, "is_empty": is_empty,
        "last_run_in_cell": last_ts, "age_days": (round(cell_age, 2) if cell_age is not None else None),
        "recheck_days": interval, "has_baseline": has_baseline,
        "baseline_tok_per_sec": baseline_val,
        "dimension_scores": dims, "cell_score": cell_score,
    }


def cell_intent(*, is_empty: bool, machine_is_new: bool, has_baseline: bool) -> str:
    """The cell's semantic goal-intent (one of INTENTS). Guarantees a meaningful split:
    a never-measured node's empty cell is LEARN-COLLECT (first data on new hardware), any
    other empty cell is COVERAGE (fill a gap), a cell with a recorded number is REGRESSION
    (defend it before it drifts), and a cell that ran but captured no number is BENCHMARK
    (re-run to measure perf)."""
    if is_empty:
        return "learn-collect" if machine_is_new else "coverage"
    return "regression" if has_baseline else "benchmark"


def build_plan(catalog: dict[str, Any], *, now: datetime, machine_filter: str | None,
               intent_filter: str, top: int, recheck_days: dict[str, int],
               stale_horizon: int) -> dict[str, Any]:
    machines: dict[str, Any] = catalog.get("machines") or {}
    runs: list[dict[str, Any]] = catalog.get("runs") or []
    total_runs = len(runs)

    # global model multiset (for the anti-monoculture model_diversity dimension).
    model_counts: Counter = Counter(r.get("model") for r in runs if r.get("model"))
    pair_seen = {(r.get("model"), r.get("precision")) for r in runs}

    # pre-index runs per (machine, kind) so a duplicate run_id can't double-count.
    cells_by_mk: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in runs:
        mid = r.get("machine_id")
        for k in run_kinds(r):
            cells_by_mk.setdefault((mid, k), []).append(r)

    bench_nodes = {mid: m for mid, m in machines.items() if (m.get("role") == "bench-node")}
    excluded = [{"machine_id": mid, "why": f"role={m.get('role')}"}
                for mid, m in machines.items() if m.get("role") != "bench-node"]

    entries: list[dict[str, Any]] = []
    matrix: dict[str, dict[str, Any]] = {}
    n_infeasible = 0
    notes: list[str] = []

    for mid, m in sorted(bench_nodes.items()):
        if machine_filter and mid != machine_filter:
            continue
        matrix[mid] = {}
        for kind in KINDS:
            feas, fnote = feasible(m, kind)
            if not feas:
                matrix[mid][kind] = {"feasible": False, "note": fnote}
                n_infeasible += 1
                continue
            cell_runs = cells_by_mk.get((mid, kind), [])
            if cell_runs:
                cand_model, cand_prec = most_common(
                    [(r.get("model"), r.get("precision")) for r in cell_runs]) or (None, None)
            else:
                cand_model, cand_prec = KIND_META[kind]["model"]
            sc = score_cell(machine={"runs": m.get("runs"), "last_run": m.get("last_run")},
                            kind=kind, cell_runs=cell_runs, now=now, recheck_days=recheck_days,
                            stale_horizon=stale_horizon, model_counts=model_counts,
                            pair_seen=pair_seen, total_runs=total_runs,
                            cand_model=cand_model, cand_prec=cand_prec)
            matrix[mid][kind] = {"feasible": True, "runs": sc["runs_in_cell"], "score": sc["cell_score"]}

            machine_is_new = int(m.get("runs") or 0) == 0 or not m.get("last_run")
            intent = cell_intent(is_empty=sc["is_empty"], machine_is_new=machine_is_new,
                                 has_baseline=sc["has_baseline"])
            feas_note = fnote
            if kind == "qwen36" and sc["is_empty"]:
                feas_note += "; no prior qwen3.6-27B run here -- verify it fits (hybrid offload) first"
            entries.append({
                "machine_id": mid,
                "machine_summary": f"{m.get('gpu')} · {m.get('arch')}/{m.get('os')} · "
                                   f"{m.get('runs')} runs · last {m.get('last_run') or 'never'}",
                "workload_kind": kind,
                "model": cand_model, "precision": cand_prec,
                "model_is_new": (cand_model, cand_prec) not in pair_seen,
                "intent": intent, "feasible": True, "feasibility_note": feas_note,
                "suggested_command": f"on {mid} ({m.get('gpu')}): {KIND_META[kind]['cmd']}  "
                                     f"# HINT for the remote bench-node -- not run by this tool",
                **{k: sc[k] for k in ("runs_in_cell", "is_empty", "last_run_in_cell",
                                      "age_days", "recheck_days", "has_baseline",
                                      "baseline_tok_per_sec", "dimension_scores", "cell_score")},
            })

    # now-precedes-a-run sanity note (clock skew / bad stamp).
    newest = max((parse_stamp(r.get("timestamp")) or datetime.min.replace(tzinfo=timezone.utc)
                  for r in runs), default=None)
    if newest is not None and newest > now:
        notes.append("--now precedes the newest catalog run; ages were clamped to 0.")

    # deterministic two-level sort: empty tier first (dominance), then weighted score,
    # then staler, then thinner machine, then lexical -- byte-stable for a fixed --now.
    def sort_key(e: dict[str, Any]) -> tuple:
        age = e["age_days"] if e["age_days"] is not None else 10 ** 9  # empty == oldest
        m_runs = int(machines.get(e["machine_id"], {}).get("runs") or 0)
        return (0 if e["is_empty"] else 1, -e["cell_score"], -age, m_runs,
                e["machine_id"], e["workload_kind"])
    entries.sort(key=sort_key)

    for i, e in enumerate(entries, 1):
        e["rank"] = i
        e["reason"] = _reason(e, machines.get(e["machine_id"], {}))

    # the single NEXT test per bench-node (first entry per machine in ranked order).
    per_machine: dict[str, Any] = {}
    for e in entries:
        per_machine.setdefault(e["machine_id"], e)

    # per-intent grouping (guarantees all four intents are surfaced in the doc).
    by_intent = {it: [e for e in entries if e["intent"] == it] for it in INTENTS}

    ranked = entries if intent_filter == "all" else [e for e in entries if e["intent"] == intent_filter]

    n_empty = sum(1 for e in entries if e["is_empty"])
    n_baseline = sum(1 for e in entries if e["has_baseline"])
    return {
        "schema": SCHEMA, "ok": True,
        "now": now.strftime("%Y%m%dT%H%M%SZ"),
        "honesty": "PLAN ONLY -- no benchmark was run; every figure is a literal catalog "
                   "fact and all ages are computed against --now.",
        "catalog_last_updated": catalog.get("last_updated"),
        "totals": {"bench_nodes": len(bench_nodes), "feasible_cells": len(entries),
                   "infeasible_cells": n_infeasible, "empty_cells": n_empty,
                   "cells_with_baseline": n_baseline, "total_runs": total_runs},
        "per_machine_next": {mid: per_machine.get(mid) for mid in sorted(per_machine)},
        "ranked": ranked[:top],
        "by_intent": {it: by_intent[it][:top] for it in INTENTS},
        "matrix": matrix,
        "excluded": excluded,
        "notes": notes,
    }


def _reason(e: dict[str, Any], machine: dict[str, Any]) -> str:
    mid, kind = e["machine_id"], e["workload_kind"]
    if e["is_empty"]:
        if (machine.get("runs") or 0) == 0:
            return (f"{mid} has 0 runs (last_run={machine.get('last_run')}); the {kind} cell "
                    f"has never run -- a first-ever measurement on this node.")
        return (f"{mid} has run other workloads but the {kind} cell is empty -- a coverage "
                f"gap to fill ({e['model'] or 'agent workload'}).")
    if e["has_baseline"]:
        base = round(e["baseline_tok_per_sec"], 2) if e["baseline_tok_per_sec"] is not None else "?"
        return (f"{mid}/{kind} has a recorded {base} tok/s baseline aged "
                f"{e['age_days']}d (re-check every {e['recheck_days']}d) -- re-measure to catch drift.")
    return (f"{mid}/{kind} ran {e['runs_in_cell']}x, last {e['age_days']}d ago "
            f"(re-check {e['recheck_days']}d) but recorded no number -- re-run to capture one.")


# --------------------------------- rendering ------------------------------------

def _cell_glyph(c: dict[str, Any]) -> str:
    if not c.get("feasible"):
        return "-"
    return "." if c.get("runs", 0) == 0 else str(c["runs"])


def render_md(p: dict[str, Any]) -> str:
    t = p["totals"]
    out = [
        "---",
        'title: "fak hardware bench plan: what to benchmark next per machine"',
        'description: "Auto-generated hardware benchmark plan: the next highest-value test '
        'per bench-node (macbook / datacenter-A100 / cloud-L4 / RTX-laptop) across coverage, '
        'regression, and new-data intents, ranked from the benchmark catalog."',
        "---",
        "",
        f"# Hardware bench plan — {p['now']}",
        "",
        "_Auto-generated by `tools/bench_plan.py --md`. Do not hand-edit; re-run the tool "
        "(or the `FleetBenchPlanDoc` task) to refresh._",
        "",
        f"**{p['honesty']}**",
        "",
        f"Generated from `catalog.json` (last_updated {p.get('catalog_last_updated')}): "
        f"{t['bench_nodes']} bench-nodes, {t['feasible_cells']} feasible cells "
        f"({t['empty_cells']} empty coverage holes, {t['cells_with_baseline']} with a recorded "
        f"baseline), {t['infeasible_cells']} infeasible cell"
        f"{'' if t['infeasible_cells'] == 1 else 's'}, {t['total_runs']} runs on record.",
        "",
        "## Coverage matrix (bench-node × workload-kind)",
        "",
        "`.` = empty feasible cell (a coverage hole) · `-` = infeasible on this hardware "
        "(e.g. a CUDA kind on the Apple mac) · a number = runs on record.",
        "",
    ]
    short = {k: k.replace("-benchmark", "-bench").replace("model-bench", "model")
             .replace("session-bench", "session").replace("radix-bench", "radix")
             .replace("fan-bench", "fan").replace("gpu-bench", "gpu") for k in KINDS}
    out.append("| machine | " + " | ".join(short[k] for k in KINDS) + " |")
    out.append("|---|" + "|".join("---" for _ in KINDS) + "|")
    for mid in sorted(p["matrix"]):
        row = p["matrix"][mid]
        out.append(f"| {mid} | " + " | ".join(_cell_glyph(row[k]) for k in KINDS) + " |")

    out += ["", "## Next test per machine", "",
            "| machine | next workload | model / precision | intent | score | why |",
            "|---|---|---|---|---|---|"]
    for mid in sorted(p["per_machine_next"]):
        e = p["per_machine_next"][mid]
        if not e:
            continue
        mp = f"{e['model']} / {e['precision']}" + (" **NEW**" if e.get("model_is_new") and e["model"] else "")
        out.append(f"| {mid} | {e['workload_kind']} | {mp} | {e['intent']} | {e['cell_score']} | {e['reason']} |")

    out += ["", "## Do next — global ranked plan", "",
            "| # | machine | workload | intent | score | suggested command (hint) |",
            "|---|---|---|---|---|---|"]
    for e in p["ranked"]:
        out.append(f"| {e['rank']} | {e['machine_id']} | {e['workload_kind']} | {e['intent']} "
                   f"| {e['cell_score']} | `{KIND_META[e['workload_kind']]['cmd']}` |")

    out += ["", "## By intent", "",
            "_The planner guarantees all four operator intents are surfaced._", ""]
    labels = {"benchmark": "Benchmark perf", "learn-collect": "Learn / collect new data",
              "regression": "Prevent regression", "coverage": "Fill coverage gaps"}
    for it in INTENTS:
        out.append(f"### {labels[it]}")
        rows = p["by_intent"][it]
        if not rows:
            if it == "regression":
                out.append("_Latent today — the catalog holds only one recorded tok/s baseline, "
                           "so there is little to re-measure yet._")
            else:
                out.append("_None ranked this round._")
        else:
            for e in rows[:5]:
                out.append(f"- **{e['machine_id']} → {e['workload_kind']}** "
                           f"(score {e['cell_score']}): {e['reason']}")
        out.append("")

    out += ["## Frontier & honesty call-outs", ""]
    for n in _frontier_callouts(p):
        out.append(f"- {n}")
    for n in p.get("notes") or []:
        out.append(f"- {n}")

    out += ["", "## Excluded (not a bench target)", ""]
    if p["excluded"]:
        for x in p["excluded"]:
            out.append(f"- `{x['machine_id']}` — {x['why']} (run-on-bench-nodes-by-default).")
    else:
        out.append("- None.")

    out += ["", "## Methodology & honesty", "",
            "Each feasible `(machine × workload-kind)` cell is scored on five dimensions, "
            "each in `[0,1]`:", "",
            "| dimension | weight | intent | reads |",
            "|---|---|---|---|",
            "| coverage_gap | 0.34 | coverage | run count in the cell (empty = 1.0) |",
            "| machine_novelty | 0.20 | learn-collect | machine total runs / last_run |",
            "| staleness_overdue | 0.16 | regression | newest cell timestamp vs --now, per-kind interval |",
            "| baseline_drift | 0.16 | regression | a recorded peak_tok_per_sec aging past its interval |",
            "| model_diversity | 0.14 | coverage | anti-monoculture: how rare the proposed model is |",
            "",
            "Cells sort empty-tier-first (coverage dominance), then by weighted score. "
            "Feasibility (no CUDA on the mac) is applied before scoring. `--now` is injected, "
            "never read from the wall clock, so a fixed stamp yields identical output. "
            "marginal-information-gain (novelty / diversity) is a heuristic proxy, not a "
            "measured quantity — the ranking is a guide, not a verdict. This tool only PLANS; "
            "a run is a later action on the remote bench-node.",
            ""]
    return "\n".join(out) + "\n"


def _frontier_callouts(p: dict[str, Any]) -> list[str]:
    out: list[str] = []
    mx = p["matrix"]
    for mid in sorted(mx):
        feas = [c for c in mx[mid].values() if c.get("feasible")]
        empties = [c for c in feas if c.get("runs", 0) == 0]
        if feas and len(empties) == len(feas):
            out.append(f"`{mid}` is a total coverage frontier — every feasible workload-kind "
                       f"({len(feas)}) has never run here.")
    if p["totals"]["cells_with_baseline"] <= 1:
        out.append(f"Only {p['totals']['cells_with_baseline']} cell has a recorded tok/s baseline, "
                   "so regression-prevention is honestly latent today — most cells have no number "
                   "to defend yet.")
    return out


def render(p: dict[str, Any]) -> str:
    lines = [f"╔═ BENCH PLAN @ {p['now']}  ({p['totals']['empty_cells']} empty cells, "
             f"{p['totals']['cells_with_baseline']} baselines, {p['totals']['total_runs']} runs)",
             f"║ {p['honesty']}", "║ next per bench-node:"]
    for mid in sorted(p["per_machine_next"]):
        e = p["per_machine_next"][mid]
        if e:
            lines.append(f"║   {mid:<14} → {e['workload_kind']:<16} [{e['intent']}] "
                         f"score={e['cell_score']}  {e['model'] or '-'}")
    lines.append("╚═ top: " + "; ".join(
        f"{e['machine_id']}/{e['workload_kind']}({e['cell_score']})" for e in p["ranked"][:5]))
    return "\n".join(lines)


def _parse_recheck_override(s: str) -> dict[str, int]:
    out = dict(RECHECK_DAYS)
    for pair in (s or "").split(","):
        pair = pair.strip()
        if not pair:
            continue
        k, _, v = pair.partition("=")
        try:
            out[k.strip()] = int(v)
        except ValueError:
            pass
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Recurring hardware-benchmark planner (plans only).")
    ap.add_argument("--now", required=True,
                    help="deterministic 'today' stamp, e.g. 20260622T140000Z (drives all age math)")
    ap.add_argument("--catalog", default="", help="catalog.json path (default: experiments/benchmark/catalog.json)")
    ap.add_argument("--workspace", default="", help="repo root override (default: tool's repo root)")
    ap.add_argument("--md", default="", help="write the committed markdown plan doc to this path")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable payload")
    ap.add_argument("--machine", default="", help="restrict to one bench-node id")
    ap.add_argument("--intent", default="all",
                    choices=["benchmark", "learn-collect", "regression", "coverage", "all"],
                    help="filter the ranked list to one goal-intent (default: all)")
    ap.add_argument("--top", type=int, default=10, help="length of the global ranked list (default: 10)")
    ap.add_argument("--recheck-days", default="", help="override per-kind intervals, e.g. model-benchmark=3,qwen36=7")
    ap.add_argument("--stale-horizon-days", type=int, default=DEFAULT_STALE_HORIZON_DAYS,
                    help=f"fallback re-check interval for kinds without one (default: {DEFAULT_STALE_HORIZON_DAYS})")
    args = ap.parse_args(argv)

    now = parse_stamp(args.now)
    if now is None:
        print(json.dumps({"schema": SCHEMA, "ok": False,
                          "reason": f"unparseable --now stamp: {args.now!r}"}))
        return 2

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    catalog_path = Path(args.catalog) if args.catalog else root / "experiments" / "benchmark" / "catalog.json"
    catalog = load_catalog(catalog_path)
    if catalog is None:
        payload = {"schema": SCHEMA, "ok": False, "now": now.strftime("%Y%m%dT%H%M%SZ"),
                   "reason": f"catalog missing or unreadable: {catalog_path}",
                   "honesty": "PLAN ONLY -- no benchmark was run.",
                   "per_machine_next": {}, "ranked": [], "by_intent": {it: [] for it in INTENTS},
                   "matrix": {}, "excluded": [], "notes": [],
                   "totals": {"bench_nodes": 0, "feasible_cells": 0, "infeasible_cells": 0,
                              "empty_cells": 0, "cells_with_baseline": 0, "total_runs": 0}}
    else:
        payload = build_plan(catalog, now=now, machine_filter=(args.machine or None),
                             intent_filter=args.intent, top=args.top,
                             recheck_days=_parse_recheck_override(args.recheck_days),
                             stale_horizon=args.stale_horizon_days)

    if args.md:
        md_path = Path(args.md)
        if not md_path.is_absolute():
            md_path = root / md_path
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(render_md(payload), encoding="utf-8")
        if not args.json:
            print(f"wrote {md_path} ({payload['totals']['empty_cells']} empty cells, "
                  f"{payload['totals']['cells_with_baseline']} baselines)")

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.md:
        print(render(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
