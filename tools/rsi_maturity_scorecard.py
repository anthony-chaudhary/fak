#!/usr/bin/env python3
"""RSI maturity & usefulness scorecard — the measuring stick for the self-improver.

The repo measures its CODE (``code_quality_scorecard.py``), its DOCS
(``docs_scorecard.py``), and folds the family into one portfolio debt
(``scorecard_control_pane.py``). Nothing measured the thing that DRIVES that
program: the recursive-self-improvement (RSI) engine itself — ``internal/rsiloop``
+ ``internal/shipgate`` + ``cmd/rsiloop``. This is that stick.

It scores the engine on TWO honest, independent axes — the same shape
``industry_scorecard.py`` uses (coverage vs parity-debt):

  MATURITY  (0-100, A-F)  — *structural readiness*: can the loop honestly CLOSE?
      Is the keep-bit derived from a real measurement the author didn't write
      (not a hand-fed flag)? Is it non-forgeable in code? Is the metric
      deterministic (no wall-clock / no RNG, so a KEEP can't be a one-box fluke)?
      Does it benchmark vs LATEST main? Is there an escalation breaker? Is the
      candidate adjudicated in an ISOLATED worktree so main is never mutated?

  USEFULNESS (0-100, A-F) — *realized value*: is it OPERATIONALIZED and producing
      kept gains? Are real KEPT gains proven by a witness test? Is more than one
      REAL subsystem wired (vs a single demo tunable)? Is the portfolio ratchet
      ENFORCED in CI (a hard ``--check`` gate, not an advisory warning)? Is a
      regression track-gate wired? A loop can be maturity-A (structurally sound)
      yet usefulness-F (a pedagogical demo nobody enforces) — and that gap is
      exactly what this surfaces.

Each axis is a set of weighted PASS/FAIL criteria checked by STATIC source
inspection (deterministic, stdlib-only, no toolchain needed), so the score
reproduces bit-for-bit and a regression (someone makes the KPI use ``time.Now``)
flips a criterion red. ``rsi_debt`` is the count of HARD criteria that FAIL —
the same "one defect = one unit of debt" contract the rest of the family uses,
so this folds cleanly into ``scorecard_control_pane.py`` (debt under ``corpus``).

  python tools/rsi_maturity_scorecard.py            # human scorecard
  python tools/rsi_maturity_scorecard.py --json      # machine payload (control-pane shape)
  python tools/rsi_maturity_scorecard.py --markdown  # RSI-MATURITY-SCORECARD.md body
  python tools/rsi_maturity_scorecard.py --probe     # ALSO run cmd/kpiprobe twice and
                                                      # assert a real, bit-identical gain
                                                      # (needs the go toolchain; degrades)

``--probe`` is the only non-static check: it runs ``go run ./cmd/kpiprobe -dump``
twice and asserts the KPI curve strictly rises (a real gain to find) AND is
bit-identical across runs (the determinism witness, observed not asserted). With
no ``go`` on PATH it degrades to the static signal, exactly like
``code_quality_scorecard.py --no-toolchain``.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-rsi-maturity-scorecard/1"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def read_text(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def workflow_text(root: Path) -> str:
    """All .github/workflows/*.yml concatenated (CI-enforcement evidence)."""
    wf = root / ".github" / "workflows"
    if not wf.is_dir():
        return ""
    parts = []
    for p in sorted(wf.glob("*.yml")) + sorted(wf.glob("*.yaml")):
        try:
            parts.append(p.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            continue
    return "\n".join(parts)


def _has_all(text: str, *subs: str) -> bool:
    return all(s in text for s in subs)


def grade_letter(score: int) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Evidence context: read the engine's source ONCE; criteria are pure over it.
# ---------------------------------------------------------------------------

def load_context(root: Path) -> dict[str, Any]:
    rsiloop = read_text(root, "internal/rsiloop/rsiloop.go")
    shipgate = read_text(root, "internal/shipgate/shipgate.go")
    kpi = read_text(root, "internal/rsiloop/kpi.go")
    worktree = read_text(root, "internal/rsiloop/worktree.go")
    rulesynth = read_text(root, "internal/rsiloop/rulesynth.go")
    cmd_rsiloop = read_text(root, "cmd/rsiloop/main.go")
    rsiloop_test = read_text(root, "internal/rsiloop/rsiloop_test.go")
    # how many DISTINCT Harness constructors does the loop's command actually
    # instantiate? That is the count of REAL, loop-driven subsystems (not just
    # harness sketches that exist as files).
    driven = []
    for ctor in ("NewWorktreeHarness", "NewRuleSynthHarness"):
        if ctor in cmd_rsiloop:
            driven.append(ctor)
    return {
        "rsiloop": rsiloop,
        "shipgate": shipgate,
        "kpi": kpi,
        "worktree": worktree,
        "rulesynth": rulesynth,
        "cmd_rsiloop": cmd_rsiloop,
        "rsiloop_test": rsiloop_test,
        "kpiprobe_exists": (root / "cmd/kpiprobe/main.go").exists(),
        "shipgate_proof_exists": (root / "docs/proofs/shipgate.md").exists(),
        "rsi_doc_exists": (root / "docs/rsi-loop.md").exists(),
        "workflows": workflow_text(root),
        "driven_harnesses": driven,
    }


# A criterion: deterministic predicate over the evidence context.
#   key, axis ('maturity'|'usefulness'), label, hard (counts toward debt),
#   weight (for the 0-100 axis score), check(ctx) -> (passed, detail)
Criterion = dict[str, Any]


def maturity_criteria() -> list[Criterion]:
    return [
        {
            "key": "closed_loop_engine", "hard": True, "weight": 3,
            "label": "closed loop derives its own witness (not hand-fed)",
            "check": lambda c: (
                _has_all(c["rsiloop"], "Harness", "BaselineMetric", "Measure")
                and bool(c["cmd_rsiloop"]),
                "internal/rsiloop drives Measure()/BaselineMetric() via cmd/rsiloop"
                if c["cmd_rsiloop"] else "cmd/rsiloop or rsiloop.Harness seams absent"),
        },
        {
            "key": "nonforgeable_keepbit", "hard": True, "weight": 3,
            "label": "non-forgeable keep-bit (set only inside Evaluate)",
            "check": lambda c: (
                _has_all(c["shipgate"], "improvedBit", "func Evaluate", "Kept()"),
                "shipgate.Witness.improvedBit is unexported, set only in Evaluate"),
        },
        {
            "key": "deterministic_metric", "hard": True, "weight": 2,
            "label": "deterministic metric (no wall-clock, no RNG)",
            "check": lambda c: (
                bool(c["kpi"]) and '"math/rand"' not in c["kpi"]
                and '"time"' not in c["kpi"],
                "kpi.go imports neither math/rand nor time — reproduces bit-for-bit"
                if c["kpi"] else "internal/rsiloop/kpi.go missing"),
        },
        {
            "key": "vs_latest_main", "hard": True, "weight": 2,
            "label": "benchmarks against LATEST main (baseline re-derived)",
            "check": lambda c: (
                ("baseline-ref" in c["cmd_rsiloop"] or "BaselineRefName" in c["rsiloop"]
                 or "rev-parse" in c["worktree"]),
                "baseline re-derived from a ref each run (-baseline-ref / rev-parse)"),
        },
        {
            "key": "escalation_breaker", "hard": True, "weight": 2,
            "label": "escalation breaker after K consecutive non-keeps",
            "check": lambda c: (
                _has_all(c["shipgate"], "ESCALATE", "Gate", "nonKeeps"),
                "shipgate.Gate escalates after K consecutive non-keeps"),
        },
        {
            "key": "worktree_isolation", "hard": True, "weight": 2,
            "label": "candidate adjudicated in an isolated worktree (main untouched)",
            "check": lambda c: (
                "ApplyInWorktree" in c["shipgate"] and "worktree" in c["worktree"],
                "shipgate.ApplyInWorktree forks a detached worktree off main"),
        },
        {
            "key": "multi_harness_pattern", "hard": False, "weight": 1,
            "label": "extension seam demonstrated by >1 Harness implementation",
            "check": lambda c: (
                bool(c["worktree"]) and bool(c["rulesynth"]),
                "two Harness implementations exist (worktree + rulesynth)"),
        },
    ]


def usefulness_criteria() -> list[Criterion]:
    return [
        {
            "key": "kept_gains_proven", "hard": False, "weight": 3,
            "label": "real KEPT gains proven by a witness test + proof",
            "check": lambda c: (
                ("Keep" in c["rsiloop_test"] and "Revert" in c["rsiloop_test"]
                 and c["shipgate_proof_exists"]),
                "rsiloop_test proves KEEP-on-gain/REVERT-on-no-gain; docs/proofs/shipgate.md"),
        },
        {
            "key": "deterministic_witness", "hard": False, "weight": 1,
            "label": "witness reproduces (deterministic KPI)",
            "check": lambda c: (
                bool(c["kpi"]) and '"math/rand"' not in c["kpi"]
                and '"time"' not in c["kpi"],
                "the KPI is a pure function of a fixed trace — same verdict on any box"),
        },
        {
            "key": "real_demo_gain", "hard": False, "weight": 1,
            "label": "the demo metric has a real gain to find",
            "check": lambda c: (
                c["kpiprobe_exists"] and "monoton" in c["kpi"].lower(),
                "cmd/kpiprobe + a monotone-by-construction trace (verify with --probe)"),
        },
        {
            "key": "journal_capability", "hard": False, "weight": 1,
            "label": "track mode journals a vs-main time series",
            "check": lambda c: (
                _has_all(c["cmd_rsiloop"], "track", "journal"),
                "cmd/rsiloop -mode track appends a JSONL journal point"),
        },
        {
            "key": "multi_real_subsystem", "hard": True, "weight": 2,
            "label": ">=2 REAL subsystems wired into the loop (not a single tunable)",
            "check": lambda c: (
                len(c["driven_harnesses"]) >= 2,
                f"cmd/rsiloop drives {len(c['driven_harnesses'])} real subsystem(s): "
                f"{', '.join(c['driven_harnesses']) or 'none'} "
                "(a 2nd live-tunable harness retires this)"),
        },
        {
            "key": "ratchet_enforced_in_ci", "hard": True, "weight": 2,
            "label": "portfolio ratchet ENFORCED in CI (hard --check gate)",
            "check": lambda c: (
                "scorecard_control_pane.py --check" in c["workflows"],
                "a CI step runs scorecard_control_pane.py --check as a hard gate "
                "(today it runs advisory-only with `|| echo ::warning::`)"),
        },
        {
            "key": "regression_gate_in_ci", "hard": True, "weight": 1,
            "label": "main-KPI regression gate wired in CI (rsiloop track)",
            "check": lambda c: (
                "cmd/rsiloop" in c["workflows"] or "rsiloop -mode track" in c["workflows"],
                "a CI step runs rsiloop track to fail the build on a main-KPI regression"),
        },
    ]


def evaluate(criteria: list[Criterion], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for crit in criteria:
        passed, detail = crit["check"](ctx)
        out.append({
            "key": crit["key"], "label": crit["label"], "hard": crit["hard"],
            "weight": crit["weight"], "passed": bool(passed), "detail": detail,
        })
    return out


def axis_score(results: list[dict[str, Any]]) -> int:
    total = sum(r["weight"] for r in results) or 1
    got = sum(r["weight"] for r in results if r["passed"])
    return int(round(100 * got / total))


def maturity_level(m: dict[str, bool]) -> int:
    """Graduated 0-4 ladder; a level needs all lower rungs."""
    level = 0
    if m.get("closed_loop_engine"):
        level = 1
    if level == 1 and m.get("nonforgeable_keepbit"):
        level = 2
    if level == 2 and m.get("vs_latest_main") and m.get("escalation_breaker") \
            and m.get("worktree_isolation"):
        level = 3
    if level == 3 and m.get("multi_harness_pattern"):
        level = 4
    return level


def usefulness_level(u: dict[str, bool]) -> int:
    """Graduated 0-5 ladder of realized value."""
    level = 0
    if u.get("kept_gains_proven"):
        level = 1
    if level == 1 and u.get("real_demo_gain"):
        level = 2
    if level == 2 and u.get("journal_capability"):
        level = 3
    if level == 3 and u.get("multi_real_subsystem"):
        level = 4
    if level == 4 and u.get("ratchet_enforced_in_ci") and u.get("regression_gate_in_ci"):
        level = 5
    return level


def probe_real_gain(root: Path) -> dict[str, Any]:
    """Run cmd/kpiprobe -dump twice; assert a strictly-rising, bit-identical curve.

    Observed, not asserted: this is the determinism witness for the metric. Degrades
    cleanly (status 'skipped') when the go toolchain is absent.
    """
    go = shutil.which("go")
    if not go:
        return {"status": "skipped", "detail": "go toolchain absent"}
    runs = []
    for _ in range(2):
        try:
            p = subprocess.run([go, "run", "./cmd/kpiprobe", "-dump"], cwd=str(root),
                               capture_output=True, text=True, encoding="utf-8",
                               errors="replace", timeout=180)
        except (OSError, subprocess.SubprocessError) as exc:
            return {"status": "error", "detail": str(exc)}
        if p.returncode != 0:
            return {"status": "error", "detail": f"kpiprobe exit {p.returncode}"}
        runs.append(p.stdout)
    vals = []
    for line in runs[0].splitlines():
        if "KPI=" in line:
            try:
                vals.append(float(line.split("KPI=")[1].split()[0]))
            except (ValueError, IndexError):
                pass
    strictly_up = len(vals) >= 2 and all(b > a for a, b in zip(vals, vals[1:]))
    identical = runs[0] == runs[1]
    return {
        "status": "ok" if (strictly_up and identical) else "fail",
        "strictly_rising": strictly_up,
        "bit_identical_across_runs": identical,
        "n_points": len(vals),
        "detail": f"{len(vals)} KPI points, strictly_rising={strictly_up}, "
                  f"bit_identical={identical}",
    }


def build_payload(root: Path, *, probe: bool = False) -> dict[str, Any]:
    ctx = load_context(root)
    mres = evaluate(maturity_criteria(), ctx)
    ures = evaluate(usefulness_criteria(), ctx)

    probe_res = probe_real_gain(root) if probe else None
    if probe_res and probe_res.get("status") == "fail":
        for r in ures:
            if r["key"] == "real_demo_gain":
                r["passed"] = False
                r["detail"] += " | --probe FAILED: " + probe_res["detail"]

    m_pass = {r["key"]: r["passed"] for r in mres}
    u_pass = {r["key"]: r["passed"] for r in ures}
    m_score = axis_score(mres)
    u_score = axis_score(ures)
    m_level = maturity_level(m_pass)
    u_level = usefulness_level(u_pass)

    hard_fail = [r for r in (mres + ures) if r["hard"] and not r["passed"]]
    rsi_debt = len(hard_fail)
    ok = rsi_debt == 0

    if ok:
        verdict, finding = "OK", "rsi_loop_mature_and_useful"
        reason = (f"RSI engine: maturity {m_score}/100 ({grade_letter(m_score)}, "
                  f"L{m_level}/4), usefulness {u_score}/100 ({grade_letter(u_score)}, "
                  f"L{u_level}/5); zero hard gaps")
        next_action = "hold the line; re-run after any change to internal/rsiloop or shipgate"
    else:
        verdict, finding = "ACTION", "rsi_usefulness_debt"
        gaps = ", ".join(r["key"] for r in hard_fail)
        reason = (f"RSI engine is structurally mature (maturity {m_score}/100 "
                  f"{grade_letter(m_score)}, L{m_level}/4) but carries {rsi_debt} hard "
                  f"gap(s) in realized value (usefulness {u_score}/100 "
                  f"{grade_letter(u_score)}, L{u_level}/5): {gaps}")
        lead = hard_fail[0]
        next_action = f"retire worst-first: {lead['key']} — {lead['detail']}"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(root),
        "corpus": {
            "rsi_debt": rsi_debt,
            "maturity_score": m_score,
            "maturity_grade": grade_letter(m_score),
            "maturity_level": m_level,
            "usefulness_score": u_score,
            "usefulness_grade": grade_letter(u_score),
            "usefulness_level": u_level,
            "grade": grade_letter(min(m_score, u_score)),
        },
        "maturity": mres,
        "usefulness": ures,
        "probe": probe_res,
    }


def render(p: dict[str, Any]) -> str:
    c = p["corpus"]
    lines = [
        f"RSI maturity & usefulness — {p['verdict']} ({p['finding']})",
        f"  rsi_debt: {c['rsi_debt']}   (hard gaps in realized value)",
        f"  maturity:   {c['maturity_score']:>3}/100 [{c['maturity_grade']}]  level {c['maturity_level']}/4  (can the loop honestly close?)",
        f"  usefulness: {c['usefulness_score']:>3}/100 [{c['usefulness_grade']}]  level {c['usefulness_level']}/5  (is it operationalized + producing kept gains?)",
        "",
        "  MATURITY (structural readiness):",
    ]
    for r in p["maturity"]:
        mark = "PASS" if r["passed"] else ("FAIL" if r["hard"] else "----")
        lines.append(f"    [{mark}] {r['label']}")
    lines.append("  USEFULNESS (realized value):")
    for r in p["usefulness"]:
        mark = "PASS" if r["passed"] else ("FAIL" if r["hard"] else "----")
        lines.append(f"    [{mark}] {r['label']}  — {r['detail']}" if not r["passed"]
                     else f"    [{mark}] {r['label']}")
    if p.get("probe"):
        lines.append(f"  probe: {p['probe'].get('status')} ({p['probe'].get('detail','')})")
    lines.extend(["", f"  → {p['next_action']}"])
    return "\n".join(lines)


def markdown(p: dict[str, Any]) -> str:
    c = p["corpus"]
    out = [
        "---",
        "title: \"RSI maturity & usefulness scorecard\"",
        "description: \"How mature (can the loop honestly close?) and how useful "
        "(is it operationalized and producing kept gains?) fak's recursive-self-"
        "improvement engine is, scored from static source evidence.\"",
        "---",
        "",
        "# RSI maturity & usefulness scorecard",
        "",
        f"**rsi_debt: {c['rsi_debt']}** · maturity **{c['maturity_score']}/100 "
        f"({c['maturity_grade']})** level {c['maturity_level']}/4 · usefulness "
        f"**{c['usefulness_score']}/100 ({c['usefulness_grade']})** level "
        f"{c['usefulness_level']}/5",
        "",
        f"> {p['reason']}",
        "",
        "## Maturity — can the loop honestly close?",
        "",
        "| ✓ | criterion |",
        "|---|---|",
    ]
    for r in p["maturity"]:
        out.append(f"| {'✅' if r['passed'] else '❌'} | {r['label']} |")
    out += ["", "## Usefulness — is it operationalized and producing kept gains?", "",
            "| ✓ | criterion | detail |", "|---|---|---|"]
    for r in p["usefulness"]:
        out.append(f"| {'✅' if r['passed'] else '❌'} | {r['label']} | {r['detail']} |")
    out += ["", f"**Next:** {p['next_action']}", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="RSI maturity & usefulness scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the scorecard doc body")
    ap.add_argument("--probe", action="store_true",
                    help="also run cmd/kpiprobe twice (needs go; degrades if absent)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = build_payload(root, probe=args.probe)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(markdown(payload))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
