#!/usr/bin/env python3
"""
cross_agent_ablate.py — the CROSS-AGENT (Regime B) ablation controller (epic #607, rung 3).

Where `fak ablate` (internal/ablate, rung 1) sweeps a fak feature on/off over ONE frozen
tool-call trace and binds every arm to a single WorkloadHash — Regime A: deterministic, $0,
no model — this controller measures the OTHER half: `claude_code` (bare `claude -p`) vs
`claude_code+fak` (`fak guard -- claude -p`) on the SAME task. That is Regime B:
an EXTERNAL model emits different tool calls every run, so the WorkloadHash guard does NOT
apply. The validity model is therefore DISTRIBUTIONAL, not identical-workload:

  1. Success-gate    — never report a "saved" number unless BOTH arms completed AND succeeded.
  2. N-run variance  — every metric is mean ± CI95 over K>=5 reps (external models vary).
  3. Model-named     — each arm names its model; the headline states which model fak ran.
                       The kernel-efficiency number is REFUSED unless the model is held
                       constant across the two arms (you cannot attribute a token delta to
                       the kernel if the model also changed).
  4. Decompose       — kernel-efficiency (model held constant) and agent-capability (model
                       varies) are reported as TWO numbers, never collapsed into one.

ARCHITECTURE. The pure core (the `session_audit` adapter, the journal counter, the CI95
stats, the success-gate, the report assembly) is dependency-free and hermetically tested
(tools/cross_agent_ablate_test.py runs with NO network and NO external binary). The thin
`run` subcommand is the only part that shells out to `claude` / `fak guard`; it records each
rep as raw data embedded in the artifact so the report can be regenerated offline with
`report --reps FILE` — a committed cross-agent artifact is self-contained and reproducible.

Token accounting is ALWAYS decomposed (input / output / provider_cache_read / cache_create)
and NEVER summed into a single scalar — the four move independently (the kernel hop changes
the cache split even when output is unchanged), so a single "tokens" number would hide the
very effect this rung measures.

Usage:
  python cross_agent_ablate.py run --task pong --k 5 --out experiments/ablate/cross-agent-pong.json
  python cross_agent_ablate.py report --reps reps.json [--out report.json]   # offline re-assembly
  python cross_agent_ablate.py tasks                                          # list built-in tasks
"""
from __future__ import annotations

import argparse
import glob
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

SCHEMA = "fak.cross-agent-ablation.v1"

# The three arms this rung knows. The acceptance requires the two Claude arms; `pure_fak`
# (the in-kernel planner over a frozen trace, metrics from kernel.Counters) is a different
# regime (a trace replayer, not a live file-writing agent) and is OUT of the same-task
# comparison by design — see the module docstring. Listed for vocabulary completeness.
ARM_PURE_FAK = "pure_fak"
ARM_CLAUDE = "claude_code"
ARM_CLAUDE_FAK = "claude_code+fak"


# --------------------------------------------------------------------------------------
# session_audit adapter — emit the unified, DECOMPOSED token vector from a transcript.
# --------------------------------------------------------------------------------------

def _load_session_audit():
    """Import the sibling tools/session_audit.py as a module (the transcript oracle)."""
    here = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("session_audit", here / "session_audit.py")
    if not (spec and spec.loader):  # pragma: no cover - defensive
        raise RuntimeError("cross_agent_ablate: cannot load tools/session_audit.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def audit_transcript(path):
    """The `session_audit` ADAPTER: run session_audit.analyze on a Claude transcript JSONL
    and project it onto the unified shape — tokens DECOMPOSED (never summed), turns, tools,
    the primary model named. This is the canonical token source for a measured rep; it folds
    each billed turn exactly once (session_audit keys on message.id), so the numbers match
    the provider's own accounting.

    The TOOL histogram is counted SEPARATELY (`_tools_from_transcript`), NOT taken from
    session_audit: Claude Code splits one billed turn across multiple transcript lines that
    share a `message.id` (the streamed text on one line, the `tool_use` on the next), and
    session_audit's id de-dup — correct for not double-counting USAGE — skips the later lines
    and so loses their tool_use blocks. Counting DISTINCT `tool_use` ids across every line
    recovers the true tool mix without re-inflating the token totals.
    """
    sa = _load_session_audit()
    a = sa.analyze(path)
    if "error" in a:
        raise RuntimeError(f"cross_agent_ablate: session_audit failed on {path}: {a['error']}")
    tok = a["tokens"]
    return {
        "tokens": _token_vector(tok["input"], tok["output"], tok["cache_read"], tok["cache_create"]),
        "turns": int(a["assistant_turns"]),
        "tools": _tools_from_transcript(path),
        "model": _primary_model(a.get("models", {})),
        "transcript": os.path.basename(path),
    }


def _tools_from_transcript(path):
    """Count DISTINCT tool calls from a transcript by their unique `tool_use` id (`toolu_...`),
    scanning EVERY assistant line. Distinct-by-id is robust to both failure modes of a naive
    count: the same-message-id line split (a tool_use stranded on a de-duped line) is still
    seen, and a streamed duplicate of the same block (same id repeated) is counted once."""
    seen = set()
    tools = {}
    try:
        lines = open(path, encoding="utf-8", errors="replace").read().splitlines()
    except OSError:
        return tools
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if r.get("type") != "assistant":
            continue
        for b in (r.get("message", {}) or {}).get("content", []) or []:
            if not (isinstance(b, dict) and b.get("type") == "tool_use"):
                continue
            tid = b.get("id") or f"{b.get('name')}:{json.dumps(b.get('input', {}), sort_keys=True)}"
            if tid in seen:
                continue
            seen.add(tid)
            name = b.get("name", "?")
            tools[name] = tools.get(name, 0) + 1
    return tools


def _token_vector(inp, out, cache_read, cache_create):
    """The decomposed token vector — the four fields kept SEPARATE, on purpose. There is
    deliberately no 'total' field that sums input+output: the contract is never-summed."""
    return {
        "input": int(inp),
        "output": int(out),
        "provider_cache_read": int(cache_read),
        "cache_create": int(cache_create),
    }


def total_input_tokens(tv):
    """Total INGESTED context = input + cache_read + cache_create. This is an input-side
    aggregate (the I in I:O); it never folds in output, so the input/output split stays
    visible. Used only for the kernel-efficiency lens, never as 'the' token number."""
    return tv["input"] + tv["provider_cache_read"] + tv["cache_create"]


def _primary_model(models):
    """The model that did the work this turn-set: the most-used model id in the transcript's
    model histogram (a sub-agent haiku call does not unseat the opus that ran the turns)."""
    if not models:
        return "unknown"
    return max(models.items(), key=lambda kv: kv[1])[0]


# --------------------------------------------------------------------------------------
# claude -p result-JSON adapter (the robust live token source) + model naming.
# --------------------------------------------------------------------------------------

def rep_from_result_json(result, *, arm, success, completed, wall_seconds,
                         transcript_audit=None, adjudication=None):
    """Build ONE unified rep record from a `claude -p --output-format json` result object.

    Tokens come from the transcript audit when available (the issue's contract: tokens from
    session_audit on the transcript); otherwise they fall back to the result JSON's own
    `usage` block (same decomposition). Tools come from the transcript audit (the result JSON
    does not carry a tool histogram). `adjudication` is the +fak journal tally, or None.
    """
    u = (result or {}).get("usage", {}) or {}
    if transcript_audit is not None:
        tokens = transcript_audit["tokens"]
        turns = transcript_audit["turns"]
        tools = transcript_audit["tools"]
        model = transcript_audit["model"]
    else:
        tokens = _token_vector(
            u.get("input_tokens", 0) or 0,
            u.get("output_tokens", 0) or 0,
            u.get("cache_read_input_tokens", 0) or 0,
            u.get("cache_creation_input_tokens", 0) or 0,
        )
        turns = int((result or {}).get("num_turns", 0) or 0)
        tools = {}
        model = result_primary_model(result)
    rep = {
        "arm": arm,
        "model": model,
        "session_id": (result or {}).get("session_id", ""),
        "completed": bool(completed),
        "success": bool(success),
        "turns": turns,
        "tokens": tokens,
        "tools": tools,
        "wall_seconds": round(float(wall_seconds), 4),
    }
    if adjudication is not None:
        rep["adjudication"] = adjudication
    return rep


def result_primary_model(result):
    """Name the model from a claude -p result's `modelUsage` map: the model with the most
    ingested tokens (input + cache_creation) is the one that ran the turns."""
    mu = (result or {}).get("modelUsage", {}) or {}
    if not mu:
        return "unknown"
    def ingest(v):
        return (v.get("inputTokens", 0) or 0) + (v.get("cacheCreationInputTokens", 0) or 0)
    return max(mu.items(), key=lambda kv: ingest(kv[1]))[0]


# --------------------------------------------------------------------------------------
# FAK_AUDIT_JOURNAL counter — the +fak arm's adjudication tally (denies/repairs/quarantines).
# --------------------------------------------------------------------------------------

def count_adjudications(journal_path):
    """Count the kernel's verdicts from a `fak guard --audit` JSONL journal (hash-chained,
    one Row per line). Buckets by the Row `verdict` field, from the closed vocabulary
    (internal/journal): ALLOW / DENY / TRANSFORM(=repair) / QUARANTINE / WITNESS / DEFER.
    VDSO_HIT rows are cache hits, not adjudications, so they are tallied SEPARATELY and never
    counted as decisions. Returns zeros (not an error) when the journal is absent/empty — a
    tool-free task legitimately produces no decisions."""
    counts = {"allowed": 0, "denied": 0, "repaired": 0, "quarantined": 0,
              "deferred": 0, "vdso_hits": 0, "journal_rows": 0}
    if not journal_path or not os.path.exists(journal_path):
        counts["journal"] = os.path.basename(journal_path) if journal_path else ""
        return counts
    with open(journal_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # tolerate a torn final line (crash mid-write); Verify flags it
            kind = row.get("kind", "")
            verdict = row.get("verdict", "")
            if kind == "VDSO_HIT":
                counts["vdso_hits"] += 1
                continue
            counts["journal_rows"] += 1
            if verdict == "ALLOW":
                counts["allowed"] += 1
            elif verdict == "DENY":
                counts["denied"] += 1
            elif verdict == "TRANSFORM":
                counts["repaired"] += 1
            elif verdict == "QUARANTINE":
                counts["quarantined"] += 1
            elif verdict == "DEFER":
                counts["deferred"] += 1
    # Store only the basename: the live journal lives in an ephemeral per-rep temp dir, so the
    # absolute path is machine noise in a committed artifact (the COUNTS are the witness).
    counts["journal"] = os.path.basename(journal_path)
    return counts


# --------------------------------------------------------------------------------------
# N-run variance — mean ± CI95 (Student's t, no third-party deps).
# --------------------------------------------------------------------------------------

# t_{0.975, df} critical values for df 1..30 (two-sided 95%). For df>30 the normal 1.96 is
# within ~5% and we use it; this keeps the CI honest for small K without scipy.
_T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
          8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
          15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 21: 2.080,
          22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048,
          29: 2.045, 30: 2.042}


def _t_crit(df):
    if df <= 0:
        return None
    if df in _T_975:
        return _T_975[df]
    return 1.96


def mean_ci95(xs):
    """mean ± CI95 over a sample. n<1 -> no estimate; n==1 -> a mean with no interval
    (a single rep cannot bound variance — honestly None, never a fake 0). n>=2 -> the
    Student-t half-width t_{0.975,n-1} * stdev/sqrt(n)."""
    xs = [float(x) for x in xs if x is not None]
    n = len(xs)
    if n == 0:
        return {"mean": None, "ci95": None, "n": 0}
    mean = statistics.mean(xs)
    if n == 1:
        return {"mean": round(mean, 4), "ci95": None, "n": 1}
    stdev = statistics.stdev(xs)
    half = _t_crit(n - 1) * stdev / math.sqrt(n)
    return {"mean": round(mean, 4), "ci95": round(half, 4), "n": n}


# --------------------------------------------------------------------------------------
# Arm aggregation + the success-gated, decomposed report.
# --------------------------------------------------------------------------------------

def aggregate_arm(arm_id, reps):
    """Roll K reps of one arm into an ArmReport: success accounting + mean±CI95 of every
    metric over the SUCCESSFUL reps (a failed rep's token count is not comparable, so it is
    excluded from the resource means but still counts against the success rate)."""
    completed = [r for r in reps if r.get("completed")]
    # The contract is both.completed && both.success: a rep flagged success WITHOUT
    # completing is incoherent data (the live runner makes success imply completed; only
    # a hand-supplied offline reps file can desync them), so it is NOT counted as a
    # success and never enters the resource means — the gate must not open on it.
    successful = [r for r in reps if r.get("success") and r.get("completed")]
    models = sorted({r.get("model", "unknown") for r in reps})
    def metric(key):
        return mean_ci95([r["tokens"][key] for r in successful]) if successful else mean_ci95([])
    agg = {
        "arm": arm_id,
        "model": _modal([r.get("model", "unknown") for r in reps]),
        "models_seen": models,
        "reps": len(reps),
        "completed_reps": len(completed),
        "success_reps": len(successful),
        "success_rate": round(len(successful) / len(reps), 4) if reps else None,
        "metrics": {
            "output_tokens": metric("output"),
            "input_tokens": metric("input"),
            "provider_cache_read_tokens": metric("provider_cache_read"),
            "cache_creation_tokens": metric("cache_create"),
            "total_input_tokens": mean_ci95([total_input_tokens(r["tokens"]) for r in successful]),
            "turns": mean_ci95([r["turns"] for r in successful]),
            "wall_seconds": mean_ci95([r["wall_seconds"] for r in successful]),
        },
        "raw_reps": reps,
    }
    adj = [r["adjudication"] for r in reps if r.get("adjudication") is not None]
    if adj:
        agg["adjudication_totals"] = {
            k: sum(a.get(k, 0) for a in adj)
            for k in ("allowed", "denied", "repaired", "quarantined", "deferred",
                      "vdso_hits", "journal_rows")
        }
    return agg


def _modal(xs):
    if not xs:
        return "unknown"
    return statistics.mode(xs) if len(set(xs)) > 1 else xs[0]


def _ratio(treat, base):
    """treatment/baseline of two mean_ci95 blocks; None when either mean is missing/zero."""
    bm, tm = base.get("mean"), treat.get("mean")
    if bm in (None, 0) or tm is None:
        return None
    return round(tm / bm, 4)


def _delta(base, treat):
    bm, tm = base.get("mean"), treat.get("mean")
    if bm is None or tm is None:
        return None
    return round(bm - tm, 4)


def decompose(baseline, treatment):
    """The TWO numbers, never one.

    kernel-efficiency — MODEL HELD CONSTANT. The treatment(+fak) vs baseline(bare) ratio on
    resource metrics. Because both arms run the SAME model, a token/turn delta is attributable
    to the KERNEL (the gateway hop, cache reshaping, adjudication overhead). Positive `saved_*`
    means the kernel was cheaper; NEGATIVE means it cost more — reported with its real sign,
    never dressed up. REFUSED (None) unless the two arms ran the same primary model.

    agent-capability — MODEL VARIES. The per-arm success rate: did the agent complete the task
    correctly. This is the model's capability and is what swings across nondeterministic reps.
    It is kept STRICTLY separate from the resource ratio so 'fak saved tokens' can never be
    misread as 'fak is more capable'.
    """
    bm = baseline.get("metrics", {})
    tm = treatment.get("metrics", {})
    # "Held constant" must hold BOTH across the arms AND within each arm. The cross-arm
    # check is the modal model match; the within-arm check is models_seen, because a single
    # arm that drifted models (e.g. 3 opus + 2 sonnet reps) has a modal model that would
    # falsely match the baseline while its token means pool across two models — exactly the
    # unattributable delta the contract says to REFUSE.
    base_models = baseline.get("models_seen", [])
    treat_models = treatment.get("models_seen", [])
    within_arm_drift = len(base_models) > 1 or len(treat_models) > 1
    model_held_constant = (baseline.get("model") == treatment.get("model")
                           and baseline.get("model") not in (None, "unknown")
                           and not within_arm_drift)
    kernel = None
    if model_held_constant:
        kernel = {
            "model": baseline.get("model"),
            "output_tokens_ratio": _ratio(tm["output_tokens"], bm["output_tokens"]),
            "total_input_tokens_ratio": _ratio(tm["total_input_tokens"], bm["total_input_tokens"]),
            "turns_ratio": _ratio(tm["turns"], bm["turns"]),
            "saved_output_tokens": _delta(bm["output_tokens"], tm["output_tokens"]),
            "saved_total_input_tokens": _delta(bm["total_input_tokens"], tm["total_input_tokens"]),
            "note": ("model held constant -> the delta is the kernel's, not the model's; "
                     "+saved = kernel cheaper, -saved = kernel overhead"),
        }
    elif within_arm_drift:
        kernel = {
            "refused": True,
            "reason": ("model NOT held constant WITHIN an arm "
                       f"(baseline models_seen={base_models}, treatment models_seen={treat_models}) "
                       "-- the per-arm token means pool across models, so a delta cannot be "
                       "attributed to the kernel"),
        }
    else:
        kernel = {
            "refused": True,
            "reason": ("model NOT held constant across arms "
                       f"({baseline.get('model')} vs {treatment.get('model')}) "
                       "-- a token delta cannot be attributed to the kernel"),
        }
    capability = {
        "baseline": {"arm": baseline["arm"], "success_rate": baseline.get("success_rate")},
        "treatment": {"arm": treatment["arm"], "success_rate": treatment.get("success_rate")},
        "note": ("success rate is the MODEL's capability and varies across reps; with both "
                 "arms on one model it is ~held here, so kernel-efficiency is the live number"),
    }
    return {"kernel_efficiency": kernel, "agent_capability": capability}


def build_report(task, arms, *, k, generated_by, command, wall_clock_utc=None,
                 baseline=ARM_CLAUDE, treatment=ARM_CLAUDE_FAK, app_version="unknown"):
    """Assemble the success-GATED AblationReport from a list of ArmReports.

    The comparison block (the 'saved' numbers) is emitted ONLY when BOTH the baseline and the
    treatment arm have >=1 successful rep — the success-gate: never a saved number off a failed
    arm. `variance_ok` flags whether K>=5 successful reps backed the interval (the N-run
    contract). The headline names the model fak ran.
    """
    by_id = {a["arm"]: a for a in arms}
    base_arm = by_id.get(baseline)
    treat_arm = by_id.get(treatment)

    comparison = {"baseline_arm": baseline, "treatment_arm": treatment}
    if base_arm and treat_arm and base_arm["success_reps"] >= 1 and treat_arm["success_reps"] >= 1:
        comparison["gated"] = True
        comparison["variance_ok"] = base_arm["success_reps"] >= 5 and treat_arm["success_reps"] >= 5
        comparison.update(decompose(base_arm, treat_arm))
    else:
        comparison["gated"] = False
        comparison["reason"] = (
            "success-gate CLOSED: a 'saved' number is only reported when BOTH arms have a "
            "successful rep (both.completed && both.success). "
            f"baseline success_reps={base_arm['success_reps'] if base_arm else 'n/a'}, "
            f"treatment success_reps={treat_arm['success_reps'] if treat_arm else 'n/a'}")

    headline_model = (treat_arm or base_arm or {}).get("model", "unknown") if (treat_arm or base_arm) else "unknown"
    return {
        "schema": SCHEMA,
        "regime": "B (nondeterministic; external model varies; the WorkloadHash guard does NOT apply)",
        "provenance": {
            "app_version": app_version,
            "command": command,
            "os": sys.platform,
            "generated_by": generated_by,
            "wall_clock_utc": wall_clock_utc,
        },
        "task": task,
        "k_reps": k,
        "headline_model": headline_model,
        "validity": {
            "success_gate": "no 'saved' number unless both.completed && both.success",
            "variance": "mean +/- CI95 (Student-t) over K>=5 reps; external models are nondeterministic",
            "model_named": "each arm names its model; kernel-efficiency refused unless model held constant",
            "decompose": "kernel-efficiency (model constant) and agent-capability (model varies) as TWO numbers",
        },
        "arms": arms,
        "comparison": comparison,
    }


# --------------------------------------------------------------------------------------
# Built-in tasks — a prompt + a DETERMINISTIC success predicate.
# --------------------------------------------------------------------------------------

def _check_pong(workdir, result_text):
    p = os.path.join(workdir, "RESULT.txt")
    try:
        return open(p, encoding="utf-8", errors="replace").read().strip() == "PONG"
    except OSError:
        return False


TASKS = {
    "pong": {
        "id": "pong",
        "prompt": ("Create a file named RESULT.txt in the current directory containing exactly "
                   "the single word PONG and nothing else. Use your file tools. Then reply done."),
        "success_check": "RESULT.txt strip()=='PONG'",
        "_check": _check_pong,
        "_induces_tool_call": True,
    },
}


# --------------------------------------------------------------------------------------
# Live runner — the only part that shells out (NOT exercised by the hermetic test).
# --------------------------------------------------------------------------------------

def _claude_config_roots():
    """Where claude -p writes its transcripts: <CLAUDE_CONFIG_DIR or ~/.claude>/projects."""
    roots = []
    cfg = os.environ.get("CLAUDE_CONFIG_DIR", "")
    if cfg:
        cfg = cfg.split(os.pathsep)[0]
        roots.append(os.path.join(cfg, "projects"))
    roots.append(os.path.join(os.path.expanduser("~/.claude"), "projects"))
    return [r for r in roots if os.path.isdir(r)]


def _find_transcript(session_id):
    """Locate the transcript JSONL for a session id under the claude projects roots."""
    if not session_id:
        return None
    for root in _claude_config_roots():
        hits = glob.glob(os.path.join(root, "**", f"{session_id}.jsonl"), recursive=True)
        if hits:
            return max(hits, key=os.path.getmtime)
    return None


def _claude_argv(prompt):
    return ["claude", "-p", prompt, "--output-format", "json",
            "--permission-mode", "acceptEdits", "--allowedTools", "Write", "Edit", "Bash"]


def run_one_rep(arm, task, fak_bin, timeout):
    """Run a single rep of one arm in a fresh temp workdir; return a unified rep record."""
    workdir = tempfile.mkdtemp(prefix="xagent-")
    journal = os.path.join(workdir, "guard-audit.jsonl") if arm == ARM_CLAUDE_FAK else None
    if arm == ARM_CLAUDE:
        argv = _claude_argv(task["prompt"])
    elif arm == ARM_CLAUDE_FAK:
        argv = [fak_bin, "guard", "--audit", journal, "--"] + _claude_argv(task["prompt"])
    else:
        raise ValueError(f"cross_agent_ablate: arm {arm!r} is not a live cross-agent arm")

    t0 = time.time()
    completed, result = True, {}
    try:
        proc = subprocess.run(argv, cwd=workdir, capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.stdout.strip() else {}
        except (json.JSONDecodeError, IndexError):
            result = {}
        completed = proc.returncode == 0 and bool(result) and not result.get("is_error", True)
    except subprocess.TimeoutExpired:
        completed = False
    wall = time.time() - t0

    success = completed and task["_check"](workdir, (result or {}).get("result", ""))

    transcript_audit = None
    tpath = _find_transcript((result or {}).get("session_id"))
    if tpath:
        try:
            transcript_audit = audit_transcript(tpath)
        except Exception:  # noqa: BLE001 - a missing/locked transcript falls back to result JSON
            transcript_audit = None
    adjudication = count_adjudications(journal) if arm == ARM_CLAUDE_FAK else None
    return rep_from_result_json(result, arm=arm, success=success, completed=completed,
                               wall_seconds=wall, transcript_audit=transcript_audit,
                               adjudication=adjudication)


def cmd_run(a):
    task = TASKS.get(a.task)
    if not task:
        sys.exit(f"cross_agent_ablate: unknown task {a.task!r} (have: {', '.join(TASKS)})")
    arms = a.arms.split(",") if a.arms else [ARM_CLAUDE, ARM_CLAUDE_FAK]
    public_task = {k: v for k, v in task.items() if not k.startswith("_")}
    app_version = _app_version(a.fak)

    arm_reports = []
    for arm in arms:
        reps = []
        for i in range(a.k):
            print(f"[{arm}] rep {i + 1}/{a.k} ...", file=sys.stderr, flush=True)
            rep = run_one_rep(arm, task, a.fak, a.timeout)
            print(f"    completed={rep['completed']} success={rep['success']} "
                  f"out_tok={rep['tokens']['output']} turns={rep['turns']}", file=sys.stderr, flush=True)
            reps.append(rep)
        arm_reports.append(aggregate_arm(arm, reps))

    report = build_report(
        public_task, arm_reports, k=a.k, generated_by="tools/cross_agent_ablate.py",
        command=f"python tools/cross_agent_ablate.py run --task {a.task} --k {a.k}",
        wall_clock_utc=a.wall_clock_utc, app_version=app_version)
    _emit(report, a.out)


def reps_from_doc(doc):
    """Extract (task, k, reps_by_arm, wall_clock_utc, app_version) from EITHER input shape:
    a saved reps file ({"task":..,"k":..,"reps":{arm:[...]}}) OR a full report ARTIFACT
    (which embeds each arm's raw_reps). The latter lets a COMMITTED cross-agent artifact
    regenerate every aggregate from itself — the self-contained-reproducible contract."""
    if "reps" in doc:  # the reps-file shape
        reps_by_arm = doc["reps"]
        return (doc["task"], doc.get("k", max((len(v) for v in reps_by_arm.values()), default=0)),
                reps_by_arm, doc.get("wall_clock_utc"), doc.get("app_version", "unknown"))
    # a full report artifact: re-derive the reps from arms[].raw_reps
    reps_by_arm = {arm["arm"]: arm["raw_reps"] for arm in doc["arms"]}
    prov = doc.get("provenance", {})
    return (doc["task"], doc.get("k_reps", max((len(v) for v in reps_by_arm.values()), default=0)),
            reps_by_arm, prov.get("wall_clock_utc"), prov.get("app_version", "unknown"))


def cmd_report(a):
    """Re-assemble a report OFFLINE from a saved reps file OR a committed report artifact."""
    doc = json.load(open(a.reps, encoding="utf-8"))
    task, k, reps_by_arm, wall, app_version = reps_from_doc(doc)
    arm_reports = [aggregate_arm(arm, reps) for arm, reps in reps_by_arm.items()]
    report = build_report(
        task, arm_reports, k=k,
        generated_by="tools/cross_agent_ablate.py report",
        command="python tools/cross_agent_ablate.py report --reps " + os.path.basename(a.reps),
        wall_clock_utc=wall, app_version=app_version)
    _emit(report, a.out)


def cmd_tasks(_a):
    for tid, t in TASKS.items():
        print(f"{tid}: {t['prompt']}")
        print(f"     success: {t['success_check']}")


def _app_version(fak_bin):
    try:
        r = subprocess.run([fak_bin, "version"], capture_output=True, text=True, timeout=20)
        return (r.stdout or r.stderr or "unknown").strip().split()[-1]
    except Exception:  # noqa: BLE001
        return "unknown"


def _emit(report, out):
    blob = json.dumps(report, indent=2) + "\n"
    if out:
        Path(out).write_text(blob, encoding="utf-8")
        print(f"wrote {out}", file=sys.stderr)
    else:
        sys.stdout.write(blob)


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:  # noqa: BLE001
        pass
    p = argparse.ArgumentParser(description="Cross-agent (Regime B) ablation controller — claude vs fak-guarded claude.")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run K reps of each arm live (shells out to claude / fak guard)")
    r.add_argument("--task", default="pong")
    r.add_argument("--k", type=int, default=5, help="reps per arm (the contract wants K>=5)")
    r.add_argument("--arms", default="", help="comma list (default: claude_code,claude_code+fak)")
    r.add_argument("--fak", default="./fak.exe" if sys.platform == "win32" else "./fak",
                   help="path to the fak binary for the +fak arm")
    r.add_argument("--timeout", type=int, default=240, help="per-rep timeout seconds")
    r.add_argument("--wall-clock-utc", default=None, help="ISO timestamp of the run (provenance)")
    r.add_argument("--out", default=None)
    r.set_defaults(func=cmd_run)

    rp = sub.add_parser("report", help="re-assemble a report offline from a saved reps file")
    rp.add_argument("--reps", required=True)
    rp.add_argument("--out", default=None)
    rp.set_defaults(func=cmd_report)

    tp = sub.add_parser("tasks", help="list the built-in tasks")
    tp.set_defaults(func=cmd_tasks)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
