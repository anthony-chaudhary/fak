#!/usr/bin/env python3
"""Tests for the memory-stability governor (#542).

Drives the PURE grader (build_payload) so the tests need no replay engine — the
witnessed counters are fed as data, exactly as the production replay substrate would
hand them in. The load-bearing case is ``test_proofs_witness``: it proves the
governor is NON-VACUOUS — an injected-drift replay whose every cycle sits BELOW the
per-reload threshold (so each point-in-time recall check stays silent) STILL trips
the cumulative freeze/rollback, while a benign replay stays in budget.

Run: ``python tools/memory_stability_governor_test.py``  (exit 0 = all pass).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import memory_stability_governor as g  # noqa: E402


def _cycle(i, *, divergence=0.0, stale=0, poison=0, probes=100):
    return {"manifest": f"m{i}", "divergence": divergence,
            "stale_reuse": stale, "poison_leakage": poison, "probes": probes}


def _benign(n=8):
    # Flat near-zero drift: tiny noise that does not trend.
    noise = [0.00, 0.01, 0.00, 0.01, 0.00, 0.01, 0.00, 0.01]
    return [_cycle(i, divergence=noise[i % len(noise)]) for i in range(n)]


def _injected_drift(n=8):
    # Progressive summarization loss (divergence creeps up each cycle) PLUS one unit
    # of sub-threshold poison every cycle. Each cycle's drift stays below tau, but the
    # running sum accretes past the budget — the exact blindness #542 names.
    traj = [_cycle(0)]
    for i in range(1, n):
        traj.append(_cycle(i, divergence=0.03 * i, poison=1, probes=100))
    return traj


def main() -> int:
    failures: list[str] = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        print(f"  [{'ok' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if not cond and detail else ""))
        if not cond:
            failures.append(name)

    ws = "/repo"

    # 1) Benign replay: flat slope, in budget, ok, not frozen.
    p = g.build_payload(workspace=ws, trajectory=_benign())
    m = p["metrics"]
    benign_slope = m["drift_slope"]
    check("benign: not frozen", m["frozen"] is False)
    check("benign: ok", p["ok"] is True)
    check("benign: verdict STABLE", p["verdict"] == g.STABLE)
    check("benign: slope ~0", abs(m["drift_slope"]) <= 0.01, f"slope={m['drift_slope']}")
    check("benign: cumulative in budget", m["cumulative_drift"] <= m["budget"])
    check("benign: no rollback", m["rollback_to"] is None)

    # 2) proofs_witness — NON-VACUITY. Injected drift trips freeze/rollback even
    #    though every breaching cycle is below the per-reload threshold tau.
    inj = _injected_drift()
    p = g.build_payload(workspace=ws, trajectory=inj)
    m = p["metrics"]
    check("injected: frozen", m["frozen"] is True)
    check("injected: not ok (fail-closed)", p["ok"] is False)
    check("injected: verdict FROZEN", p["verdict"] == g.FROZEN)
    check("injected: slope off zero (positive trend)", m["drift_slope"] > 0.01, f"slope={m['drift_slope']}")
    check("injected: slope clearly exceeds benign", m["drift_slope"] > benign_slope + 0.01,
          f"injected={m['drift_slope']} benign={benign_slope}")
    check("injected: cumulative breached budget", m["cumulative_drift"] > m["budget"],
          f"cum={m['cumulative_drift']} budget={m['budget']}")
    check("injected: rollback offered to a snapshot", bool(m["rollback_to"]))
    check("injected: freeze_cycle set", m["freeze_cycle"] is not None)

    # The honesty witness: every cycle up to (and incl.) the freeze was below tau, so
    # a POINT-IN-TIME recall gate would have stayed silent — only the trajectory
    # governor catches it. This is what makes the check non-vacuous.
    breach_cycles = p["cycles"][1:m["freeze_cycle"] + 1]
    check("injected: every breaching cycle below tau", all(c["below_tau"] for c in breach_cycles),
          str([(c["cycle"], c["drift"]) for c in breach_cycles]))
    check("injected: sub_threshold_breach flagged", m["sub_threshold_breach"] is True)

    # 3) Rollback target is the LAST in-budget snapshot, not the baseline blindly and
    #    not the breaching cycle.
    fc = m["freeze_cycle"]
    expected_rollback = inj[fc - 1]["manifest"]
    check("rollback = last in-budget snapshot", m["rollback_to"] == expected_rollback,
          f"got {m['rollback_to']} expected {expected_rollback}")
    # The breaching cycle itself must NOT be admitted (fail-closed refusal to evolve).
    check("breaching cycle not admitted", p["cycles"][fc]["admitted"] is False)

    # 4) Stability score is fail-closed to the weakest axis (min), like the quarantine gate.
    s = g.stability_score(_cycle(1, divergence=0.0, stale=0, poison=80, probes=100))
    check("stability: reliability tanks on poison", s["reliability"] <= 0.2, str(s))
    check("stability: overall is the min axis", s["overall"] == min(
        s["reliability"], s["recency"], s["consistency"]), str(s))

    # 5) Selective forgetting: a cycle below the stability floor lands on the forget list.
    traj = [_cycle(0), _cycle(1, divergence=0.0, poison=70, probes=100)]
    p = g.build_payload(workspace=ws, trajectory=traj, budget=10.0)  # budget high so freeze is not the cause
    check("forget: low-stability cycle surfaced", any(f["cycle"] == 1 for f in p["forget"]),
          str(p["forget"]))

    # 6) Validation gate: a below-floor new cycle is refused admission even in budget.
    admitted_flags = [c["admitted"] for c in p["cycles"]]
    check("validation: below-floor cycle refused admission", admitted_flags[1] is False)

    # 7) Drift slope is a least-squares TREND, not a point value.
    check("slope: rising series has positive slope", g.drift_slope([0.0, 0.1, 0.2, 0.3]) > 0)
    check("slope: flat series ~ zero", abs(g.drift_slope([0.1, 0.1, 0.1, 0.1])) < 1e-9)
    check("slope: single point is zero", g.drift_slope([0.3]) == 0.0)

    # 8) Empty trajectory is a clean zero-state pass, not an error.
    p = g.build_payload(workspace=ws, trajectory=[])
    check("empty: ok", p["ok"] is True and p["verdict"] == g.STABLE)

    # 9) Exit-code contract: the process exit is the FAIL-CLOSED signal (freeze or a
    #    source error), NOT the stricter `ok` field. A merely-drifting-but-in-budget
    #    store is a soft slope warning (verdict STABLE, ok False) and must exit 0 — a
    #    drift trend must not masquerade as a hard breach.
    steep = [_cycle(0), _cycle(1, divergence=0.10), _cycle(2, divergence=0.20),
             _cycle(3, divergence=0.30)]
    drifting = g.build_payload(workspace=ws, trajectory=steep, budget=10.0)
    check("exit: drifting verdict STABLE", drifting["verdict"] == g.STABLE)
    check("exit: drifting not frozen / in budget", drifting["metrics"]["frozen"] is False)
    check("exit: drifting is the ok=False case", drifting["ok"] is False,
          f"slope={drifting['metrics']['drift_slope']}")
    check("exit: drifting-but-in-budget exits 0 (soft warning)", g.exit_code(drifting) == 0)
    check("exit: stable flat exits 0", g.exit_code(g.build_payload(workspace=ws, trajectory=_benign())) == 0)
    check("exit: FROZEN store exits 1 (fail-closed)",
          g.exit_code(g.build_payload(workspace=ws, trajectory=_injected_drift())) == 1)
    check("exit: unreadable trajectory exits 1",
          g.exit_code(g.build_payload(workspace=ws, trajectory=[], error="boom")) == 1)
    check("exit: empty trajectory exits 0", g.exit_code(g.build_payload(workspace=ws, trajectory=[])) == 0)

    print()
    if failures:
        print(f"FAILED: {len(failures)} check(s): {failures}")
        return 1
    print("all memory-stability-governor checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
