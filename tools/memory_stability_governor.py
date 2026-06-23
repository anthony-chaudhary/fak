#!/usr/bin/env python3
"""Memory-stability governor — the TRAJECTORY layer over fak's evolving recall store.

fak's memory governance is strong on POINT-IN-TIME hazards but blind to CUMULATIVE
drift of an evolving store. Every existing recall check fires at a SINGLE reload:
a sealed page stays sealed across the process boundary, a witness-revoked page is
re-sealed when the trust epoch advances (``trustGate``, recall.go:378-387), a benign
page is RE-SCREENED under today's tightened detector chain on page-in (``reScreen``,
recall.go:404-433). All three are SLOPE-BLIND: when the recall/RSI path repeatedly
reads, folds, and re-writes memory across many cycles, slow drift — compounding
summarization loss, low-grade poison that each cycle slips BELOW the per-reload
threshold, or accreted contradictions — is invisible. ``stale_reuse`` /
``poison_leakage`` (#515) are measured at a POINT in a replay, not as a SLOPE across
iterations.

This governor adds the missing trajectory layer (#542). It mirrors the rsiloop
keep-or-revert/baseline SHAPE (a frozen ``BaselineMetric`` + a non-forgeable keep
bit) but governs an evolving MEMORY store rather than CODE candidates:

  1. FREEZE A BASELINE  — cycle 0's snapshot (a ``recall.Manifest`` id) is the stable
     reference every later cycle is replayed against.
  2. DRIFT TRAJECTORY   — after each evolve/fold cycle, drift is a WITNESSED,
     replay-derived scalar (NEVER a model self-assessment): the deterministic-replay
     divergence vs the baseline PLUS the real #515 ``stale_reuse`` / ``poison_leakage``
     counters, normalized by the fixed probe count. We report a drift SLOPE (a
     least-squares trend across iterations), not a point value — that is the signal a
     per-reload check structurally cannot see.
  3. STABILITY SCORE    — a per-component score (reliability / recency / consistency)
     that ranks each cycle's trustworthiness. It is fail-closed by the SAME shape as
     the recall quarantine gate: the OVERALL score is the WEAKEST component (a min),
     so one bad axis seals the view.
  4. STABILITY BUDGET   — cumulative drift is governed against a budget. The instant
     the running sum breaches the budget the governor FREEZES the store fail-closed
     (refuses to admit the drifted view) and offers ROLLBACK to the last in-budget
     snapshot. Default-deny kernel authority: the agent cannot self-certify its own
     memory as stable — the verdict is a deterministic rule over witnessed counters.
  5. VALIDATION + FORGET — each new cycle is validated against the baseline floor
     BEFORE it is allowed to evolve the store; cycles below the floor are surfaced
     for selective forgetting (an ACTIVE governance action, not just sealing the
     obviously-poisoned).

The load-bearing honesty property (the kernel twist): a single cycle's drift can sit
BELOW the per-reload threshold ``tau`` — so every point-in-time recall check passes —
yet the CUMULATIVE sum still breaches the stability budget. That is the gap a
point-in-time gate cannot close and this trajectory governor does. The
``proofs_witness_test`` proves non-vacuity: an injected-drift replay (progressive
summarization loss + sub-threshold poison) moves the slope off zero and trips the
freeze/rollback, while a benign replay stays in budget.

This tool is the PURE governance kernel and its near-free, engine-agnostic oracle.
The witnessed counters (``divergence`` / ``stale_reuse`` / ``poison_leakage``) are
fed in as data here; in production they arrive from the #502-505 replay-as-fitness
substrate and real ``k.Syscall`` counters. Read-only: it decides; it never edits a
memory store.

Run: ``python tools/memory_stability_governor.py --trajectory <file.json>``
Test: ``python tools/memory_stability_governor_test.py``  (exit 0 = all pass).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SCHEMA = "fleet-memory-stability-governor/1"

# Verdicts (closed set).
STABLE = "STABLE_IN_BUDGET"      # whole trajectory stayed in budget — safe to evolve.
FROZEN = "DRIFT_FROZEN"          # cumulative drift breached the budget — fail-closed.

# Governance defaults. tau is the PER-RELOAD point threshold a single cycle's drift
# must clear to trip an existing recall check; budget is the CUMULATIVE ceiling this
# governor adds on top. floor is the per-cycle stability-score admission floor.
DEFAULT_TAU = 0.20
DEFAULT_BUDGET = 0.50
DEFAULT_FLOOR = 0.60


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def cycle_drift(rec: dict[str, Any]) -> float:
    """The witnessed, replay-derived drift scalar for one evolve/fold cycle.

    Two witnessed inputs, both from the replay substrate / real counters — never a
    model's opinion of its own memory:

      * ``divergence`` in [0,1] — the deterministic-replay divergence of the fixed
        probe trajectory through THIS cycle's store vs the frozen baseline store
        (progressive summarization loss, accreted contradictions).
      * ``stale_reuse`` + ``poison_leakage`` (the #515 counters), normalized by the
        fixed ``probes`` count so the scalar is dimensionless and engine-agnostic.

    The sum is the per-cycle increment the stability budget integrates.
    """
    probes = max(1, int(rec.get("probes", 1)))
    divergence = _clamp01(float(rec.get("divergence", 0.0)))
    leak = int(rec.get("stale_reuse", 0)) + int(rec.get("poison_leakage", 0))
    return _clamp01(divergence + leak / probes)


def stability_score(rec: dict[str, Any]) -> dict[str, float]:
    """Per-component trustworthiness for one cycle, fail-closed to the weakest axis.

    Same shape as the recall quarantine gate (most-restrictive-wins): ``overall`` is
    the MIN of the three components, so one bad axis seals the view — a high
    consistency score cannot launder a poisoned reliability score.
    """
    probes = max(1, int(rec.get("probes", 1)))
    reliability = _clamp01(1.0 - int(rec.get("poison_leakage", 0)) / probes)
    recency = _clamp01(1.0 - int(rec.get("stale_reuse", 0)) / probes)
    consistency = _clamp01(1.0 - float(rec.get("divergence", 0.0)))
    overall = min(reliability, recency, consistency)
    return {
        "reliability": round(reliability, 6),
        "recency": round(recency, 6),
        "consistency": round(consistency, 6),
        "overall": round(overall, 6),
    }


def drift_slope(drifts: list[float]) -> float:
    """Least-squares slope of per-cycle drift vs cycle index — the TRAJECTORY signal.

    A benign store holds drift flat (slope ~ 0); progressive summarization loss or
    accreting sub-threshold poison bends it UP (slope > 0). This is the measurement a
    per-reload check structurally cannot make — it sees one point, never the trend.
    """
    n = len(drifts)
    if n < 2:
        return 0.0
    mean_x = (n - 1) / 2.0
    mean_y = sum(drifts) / n
    num = sum((i - mean_x) * (d - mean_y) for i, d in enumerate(drifts))
    den = sum((i - mean_x) ** 2 for i in range(n))
    return num / den if den else 0.0


def build_payload(
    *,
    workspace: str,
    trajectory: list[dict[str, Any]],
    baseline: str | None = None,
    tau: float = DEFAULT_TAU,
    budget: float = DEFAULT_BUDGET,
    floor: float = DEFAULT_FLOOR,
    slope_tol: float = 0.05,
    error: str | None = None,
) -> dict[str, Any]:
    """Pure grader: a replay trajectory in, a governance verdict out.

    ``trajectory`` is the ordered list of per-cycle replay records. Cycle 0 is the
    frozen baseline snapshot (its ``manifest`` is the rollback floor). Each later
    record carries the witnessed ``divergence`` / ``stale_reuse`` / ``poison_leakage``
    / ``probes`` for that evolve/fold cycle, plus a ``manifest`` snapshot id.
    """
    if error:
        return {
            "schema": SCHEMA, "workspace": workspace, "verdict": FROZEN,
            "ok": False, "finding": error, "next_action": "fix the trajectory source",
            "error": error, "cycles": [],
        }
    if not trajectory:
        return {
            "schema": SCHEMA, "workspace": workspace, "verdict": STABLE,
            "ok": True, "finding": "empty trajectory — nothing to govern",
            "next_action": "freeze a baseline snapshot and replay a cycle",
            "metrics": {"drift_slope": 0.0, "cumulative_drift": 0.0, "budget": budget,
                        "frozen": False, "freeze_cycle": None, "rollback_to": None},
            "cycles": [], "forget": [],
        }

    base = trajectory[0]
    baseline_id = baseline or str(base.get("manifest") or "cycle-0")

    drifts = [cycle_drift(r) for r in trajectory]
    cumulative = 0.0
    breach_cumulative: float | None = None
    frozen = False
    freeze_cycle: int | None = None
    rollback_to: str | None = None
    last_in_budget_id = baseline_id
    cycles: list[dict[str, Any]] = []
    forget: list[dict[str, Any]] = []

    for i, rec in enumerate(trajectory):
        d = drifts[i]
        score = stability_score(rec)
        mid = str(rec.get("manifest") or f"cycle-{i}")
        # A new cycle is admissible only if its stability score clears the floor AND
        # admitting its drift keeps the running sum within budget. Fail-closed.
        post = cumulative + d
        score_ok = score["overall"] >= floor
        budget_ok = post <= budget + 1e-9
        admit = i == 0 or (score_ok and budget_ok and not frozen)

        row = {
            "cycle": i,
            "manifest": mid,
            "drift": round(d, 6),
            "cumulative": round(post, 6),
            "below_tau": d < tau,           # would the per-reload point gate stay silent?
            "stability": score,
            "admitted": admit,
        }
        cycles.append(row)

        if i == 0:
            cumulative = post
            continue

        if not score_ok:
            forget.append({"cycle": i, "manifest": mid, "overall": score["overall"],
                           "reason": "stability score below floor"})

        if not frozen and (not budget_ok):
            # First budget breach — freeze fail-closed, offer rollback to the last
            # snapshot whose cumulative drift was still in budget.
            frozen = True
            freeze_cycle = i
            rollback_to = last_in_budget_id
            breach_cumulative = post  # the running sum that tripped the budget

        if admit:
            cumulative = post
            last_in_budget_id = mid

    slope = drift_slope(drifts)
    drifting = slope > slope_tol
    ok = not frozen and not drifting
    # Report the running sum that TRIPPED the budget (over the ceiling) when frozen;
    # otherwise the largest in-budget cumulative reached.
    reported_cumulative = breach_cumulative if breach_cumulative is not None else cumulative

    if frozen:
        finding = (f"cumulative drift {round(reported_cumulative, 4)} breached budget {budget} "
                   f"at cycle {freeze_cycle}; store FROZEN fail-closed")
        next_action = f"rollback to in-budget snapshot {rollback_to}; forget low-stability cycles"
        verdict = FROZEN
    elif drifting:
        finding = (f"drift slope {round(slope, 4)} > tol {slope_tol} (still in budget) "
                   f"— store is trending toward breach")
        next_action = "tighten fold loss / forget contradictory items before the next cycle"
        verdict = STABLE
    else:
        finding = f"drift slope {round(slope, 4)} flat, cumulative in budget — store stable"
        next_action = "continue evolving; re-validate each new cycle against the baseline"
        verdict = STABLE

    # The honesty witness: cycles whose per-cycle drift is BELOW tau (a per-reload
    # point gate would stay silent) yet which still pushed cumulative over budget.
    sub_threshold_breach = frozen and all(
        c["below_tau"] for c in cycles[1:freeze_cycle + 1]
    ) if freeze_cycle else False

    return {
        "schema": SCHEMA,
        "workspace": workspace,
        "verdict": verdict,
        "ok": ok,
        "finding": finding,
        "next_action": next_action,
        "baseline": baseline_id,
        "metrics": {
            "drift_slope": round(slope, 6),
            "cumulative_drift": round(reported_cumulative, 6),
            "in_budget_cumulative": round(cumulative, 6),
            "budget": budget,
            "tau": tau,
            "floor": floor,
            "frozen": frozen,
            "freeze_cycle": freeze_cycle,
            "rollback_to": rollback_to,
            "sub_threshold_breach": sub_threshold_breach,
        },
        "cycles": cycles,
        "forget": forget,
        "totals": {"cycles": len(trajectory), "forget": len(forget)},
    }


# ---------------------------------------------------------------------------
# Wiring + CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def collect(workspace: Path, *, trajectory_path: str | None = None,
            tau: float = DEFAULT_TAU, budget: float = DEFAULT_BUDGET,
            floor: float = DEFAULT_FLOOR) -> dict[str, Any]:
    root = workspace.resolve()
    if not trajectory_path:
        return build_payload(workspace=str(root), trajectory=[],
                             tau=tau, budget=budget, floor=floor)
    p = Path(trajectory_path)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return build_payload(workspace=str(root), trajectory=[],
                             error=f"cannot read trajectory {trajectory_path}: {e}")
    traj = raw.get("trajectory") if isinstance(raw, dict) else raw
    if not isinstance(traj, list):
        return build_payload(workspace=str(root), trajectory=[],
                             error="trajectory must be a JSON list (or {trajectory:[...]})")
    baseline = raw.get("baseline") if isinstance(raw, dict) else None
    return build_payload(workspace=str(root), trajectory=traj, baseline=baseline,
                         tau=tau, budget=budget, floor=floor)


def render(payload: dict[str, Any]) -> str:
    m = payload.get("metrics") or {}
    lines = [
        f"memory-stability governor: {payload.get('verdict')} ({payload.get('finding')})",
        (f"slope={m.get('drift_slope')}  cumulative={m.get('cumulative_drift')}"
         f"/budget={m.get('budget')}  frozen={m.get('frozen')}"),
        f"next: {payload.get('next_action')}",
    ]
    if m.get("frozen"):
        lines.append(f"  ROLLBACK -> {m.get('rollback_to')} (freeze at cycle {m.get('freeze_cycle')})")
        if m.get("sub_threshold_breach"):
            lines.append("  NOTE: every breaching cycle was BELOW tau — a per-reload gate "
                         "would have stayed silent (trajectory-only catch).")
    forget = payload.get("forget") or []
    if forget:
        lines.append("  FORGET (stability below floor):")
        for f in forget[:20]:
            lines.append(f"    cycle {f['cycle']:>3} {f.get('manifest')}  overall={f.get('overall')}")
    return "\n".join(lines)


def exit_code(payload: dict[str, Any]) -> int:
    """Process exit: non-zero ONLY when the governor fail-closed — it FROZE the store
    (a real cumulative-drift breach) or could not source the trajectory.

    A stable store exits 0, INCLUDING one that is merely drifting-but-in-budget: a
    rising drift slope that has not yet breached the budget is a SOFT warning (verdict
    stays ``STABLE_IN_BUDGET``, ``next_action`` says tighten the fold), not a refusal.
    Only a freeze (or an unreadable trajectory) is the hard default-deny. ``ok`` in the
    payload is the stricter "stable AND not trending" signal; the exit code tracks the
    fail-closed contract, so a drift warning does not masquerade as a breach.
    """
    m = payload.get("metrics") or {}
    return 1 if (m.get("frozen") or payload.get("error")) else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Memory-stability governor: replay-derived drift trajectory + "
                    "stability budget + rollback-to-stable-snapshot (read-only)."
    )
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--trajectory", default="", help="path to a replay-trajectory JSON file")
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU, help="per-reload point threshold")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET, help="cumulative drift budget")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR, help="per-cycle stability floor")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, trajectory_path=args.trajectory or None,
                      tau=args.tau, budget=args.budget, floor=args.floor)

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))

    return exit_code(payload)


if __name__ == "__main__":
    raise SystemExit(main())
