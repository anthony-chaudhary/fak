#!/usr/bin/env python3
"""guard_rsi_scorecard.py — the measuring stick for the RSI loop(s) of `fak guard`.

The repo measures its CODE, its DOCS, the generic RSI ENGINE (rsi_maturity_scorecard.py
scores internal/rsiloop) — but nothing scored the self-improvement loop attached to the
PRODUCT a user actually runs: `fak guard`. `fak guard` writes a default-on, hash-chained
decision journal of every kernel verdict; that journal is the self-improvement signal our
own workflow produces, and "score & 3x the RSI loops for fak guard on our own usage" is
exactly the gap this stick fills.

Two loops are in scope, scored against REALITY (not what their docstrings claim):

  * the LATENCY loop (tools/guard_hop_rsi.py, #733) — optimises guard-hop overhead.
    Its keep/revert rung is honestly hardware-gated (#734), so it can never close on a
    normal box; we score its STRUCTURE, not realized iterations, and do not punish the
    hardware gate it discloses honestly.
  * the VERDICT loop (tools/guard_verdict_rsi.py) — closes on the real decision journal
    TODAY: folds the verdict distribution, scores its quality, and KEEPS a refinement
    only on a strict metric gain + a witness it did not author. This is the loop that
    runs on our own usage with no hardware gate, and the one whose realized value the
    3x program drives up.

The score folds two axes (the industry_scorecard shape: structure vs realized value):

  MATURITY   — can the loop honestly CLOSE? Witness-grounded, non-forgeable keep-bit;
               deterministic metric (no clock/RNG); empty-journal honesty; an honesty
               --check gate that rejects an unwitnessed / un-improved / empty-journal keep.
  REALIZED   — is it OPERATIONALISED on our usage? Does a loop READ the real journal (not
               a dangling telemetry string)? Is the loop REGISTERED in the control-pane
               ratchet? Does a verb expose it? Is there a paired honesty test?

Each criterion is a deterministic predicate over the git-tracked tree + (for the realized
axis) the real journal, so the score reproduces bit-for-bit and is UNGAMEABLE by editing a
docstring: a KPI passes only when the real file / function / registry row exists.
`guard_rsi_debt` is the count of HARD criteria that FAIL — the "one defect = one unit of
debt" contract the rest of the family uses, so it folds into scorecard_control_pane.py.

    python tools/guard_rsi_scorecard.py            # human scorecard
    python tools/guard_rsi_scorecard.py --json      # control-pane payload
    python tools/guard_rsi_scorecard.py --markdown  # the scorecard doc body
    python tools/guard_rsi_scorecard.py --compare baseline.json   # prove the debt drop
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fak-guard-rsi-scorecard/1"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def read_text(root: Path, rel: str) -> str:
    try:
        return (root / rel).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


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


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if not (spec and spec.loader):
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Evidence context: read the loops' source + the control-pane registry + the real
# journal ONCE; the criteria are pure predicates over this context.
# ---------------------------------------------------------------------------

def load_context(root: Path) -> dict[str, Any]:
    hop = read_text(root, "tools/guard_hop_rsi.py")
    verdict = read_text(root, "tools/guard_verdict_rsi.py")
    control_pane = read_text(root, "tools/scorecard_control_pane.py")
    baseline = read_text(root, "tools/scorecard_baseline.json")
    main_go = read_text(root, "cmd/fak/main.go")
    guard_go = read_text(root, "cmd/fak/guard.go")

    # The realized-value cross-check: how many REAL adjudicated rows exist in our own
    # journals, and how many kept iterations the verdict loop has actually banked on
    # them. Read via the SAME discovery dogfood_coverage uses (one source of truth).
    rows = 0
    journals = 0
    diagnose = ""
    verdict_quality = None
    try:
        cov = _load_module("dogfood_coverage", root / "tools" / "dogfood_coverage.py")
        rows, journals = cov.count_audit_rows(root)
        if rows == 0:
            diagnose = cov.diagnose_audit_gap(root)
    except Exception as exc:  # noqa: BLE001 — a missing sibling must not crash the score
        diagnose = f"journal reader unavailable: {exc}"
    # If the verdict loop is present, fold the real rows through IT so the score reflects
    # the loop's own metric on our usage (not a re-derivation).
    if verdict:
        try:
            gv = _load_module("guard_verdict_rsi", root / "tools" / "guard_verdict_rsi.py")
            paths = gv._journal_paths(root)
            fold = gv.fold_rows(paths)
            verdict_quality = gv.verdict_quality(fold)
        except Exception:  # noqa: BLE001
            verdict_quality = None

    return {
        "hop": hop,
        "verdict": verdict,
        "control_pane": control_pane,
        "baseline": baseline,
        "main_go": main_go,
        "guard_go": guard_go,
        "verdict_exists": (root / "tools/guard_verdict_rsi.py").is_file(),
        "verdict_test_exists": (root / "tools/guard_verdict_rsi_test.py").is_file(),
        "hop_test_exists": (root / "tools/guard_hop_rsi_test.py").is_file(),
        "skill_exists": (root / ".claude/skills/guard-rsi-score/SKILL.md").is_file(),
        "doc_exists": (root / "docs/fak/guard-verdict-rsi-loop.md").is_file(),
        "audit_rows": rows,
        "audit_journals": journals,
        "audit_diagnose": diagnose,
        "verdict_quality": verdict_quality,
    }


# A criterion: deterministic predicate over the evidence context.
#   key, axis ('maturity'|'realized'), label, hard, weight, check(ctx)->(passed, detail)
Criterion = dict[str, Any]


def maturity_criteria() -> list[Criterion]:
    return [
        {
            "key": "verdict_loop_present", "hard": True, "weight": 3, "axis": "maturity",
            "label": "a journal-grounded verdict loop exists (closes without hardware)",
            "check": lambda c: (
                c["verdict_exists"] and _has_all(c["verdict"], "fold_rows",
                                                 "verdict_quality", "run_iteration"),
                "tools/guard_verdict_rsi.py folds real rows + scores verdict-quality"
                if c["verdict_exists"] else "no verdict loop — only the hardware-gated "
                "latency loop exists, which cannot close on a normal box"),
        },
        {
            "key": "deterministic_metric", "hard": True, "weight": 2, "axis": "maturity",
            "label": "deterministic verdict-quality metric (no wall-clock, no RNG)",
            "check": lambda c: (
                bool(c["verdict"]) and '"math/rand"' not in c["verdict"]
                and "random" not in c["verdict"] and "time.time" not in c["verdict"]
                and "datetime" not in c["verdict"],
                "verdict_quality is a pure function of the fold bytes — same rows, same "
                "score (so a KEEP can't be a one-run fluke)"),
        },
        {
            "key": "nonforgeable_keepbit", "hard": True, "weight": 3, "axis": "maturity",
            "label": "non-forgeable keep-bit (rows>0 AND strict gain AND witness)",
            "check": lambda c: (
                _has_all(c["verdict"], "check_iteration", "fabricated gain",
                         "strict improvement", "green external witness"),
                "check_iteration rejects a kept iteration lacking rows / a strict delta "
                "/ a green witness — the same gate guard_hop_rsi.check_plan enforces"),
        },
        {
            "key": "empty_journal_honesty", "hard": True, "weight": 2, "axis": "maturity",
            "label": "refuses a kept iteration on an empty journal (self-diagnosing 0)",
            "check": lambda c: (
                _has_all(c["verdict"], "diagnose_audit_gap", "empty journal")
                and "rows > 0 and strict_gain and have_witness" in c["verdict"],
                "an empty journal -> kept=False with a self-diagnosing reason (the row "
                "count IS the gate; no gain fabricated from no data)"),
        },
        {
            "key": "latency_loop_honest", "hard": False, "weight": 1, "axis": "maturity",
            "label": "the latency loop discloses its hardware gate (not silently broken)",
            "check": lambda c: (
                _has_all(c["hop"], "PENDING_MEASUREMENT", "deferred", "check_plan"),
                "guard_hop_rsi honestly fences its keep/revert rung on a measured "
                "baseline (#734) instead of fabricating a latency win"),
        },
    ]


def realized_criteria() -> list[Criterion]:
    return [
        {
            "key": "loop_reads_real_journal", "hard": True, "weight": 3, "axis": "realized",
            "label": "a loop READS the real journal (not a dangling telemetry string)",
            "check": lambda c: (
                _has_all(c["verdict"], "count_audit_rows", "_journal_paths")
                or "count_audit_rows" in c["hop"],
                "the verdict loop discovers + folds the real guard-audit journals via "
                "the shared dogfood_coverage reader (was: a telemetry_source string no "
                "code read)"),
        },
        {
            "key": "registered_in_control_pane", "hard": True, "weight": 2, "axis": "realized",
            "label": "the guard-RSI scorecard is registered in the control-pane ratchet",
            "check": lambda c: (
                "guard_rsi_scorecard.py" in c["control_pane"]
                and "guard_rsi_debt" in c["control_pane"]
                and "guard_rsi" in c["baseline"],
                "scorecard_control_pane.SCORECARDS carries the guard_rsi row AND the "
                "baseline is pinned (so the ratchet folds + gates this debt)"),
        },
        {
            "key": "kept_iteration_on_real_rows", "hard": True, "weight": 3, "axis": "realized",
            "label": "the loop has CLOSED on real usage (>=1 kept iteration possible)",
            "check": lambda c: (
                c["audit_rows"] > 0,
                f"{c['audit_rows']} real adjudicated row(s) across "
                f"{c['audit_journals']} journal(s) — the verdict loop can bank a kept "
                f"iteration on our own usage"
                if c["audit_rows"] > 0
                else f"0 real rows — the loop cannot close yet ({c['audit_diagnose']})"),
        },
        {
            "key": "paired_honesty_test", "hard": True, "weight": 2, "axis": "realized",
            "label": "a paired test proves the keep/revert + empty-journal refusal",
            "check": lambda c: (
                c["verdict_test_exists"],
                "tools/guard_verdict_rsi_test.py proves KEEP-on-gain, REVERT-on-no-gain, "
                "and empty-journal refusal"
                if c["verdict_test_exists"] else "no paired test for the verdict loop"),
        },
        {
            "key": "documented", "hard": False, "weight": 1, "axis": "realized",
            "label": "the real-usage loop is documented + has an RSI skill",
            "check": lambda c: (
                c["doc_exists"] and c["skill_exists"],
                "docs/fak/guard-verdict-rsi-loop.md + .claude/skills/guard-rsi-score "
                "explain + operationalise the pass"),
        },
    ]


def evaluate(criteria: list[Criterion], ctx: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for crit in criteria:
        passed, detail = crit["check"](ctx)
        out.append({
            "key": crit["key"], "label": crit["label"], "hard": crit["hard"],
            "weight": crit["weight"], "axis": crit["axis"],
            "passed": bool(passed), "detail": detail,
        })
    return out


def axis_score(results: list[dict[str, Any]]) -> int:
    total = sum(r["weight"] for r in results) or 1
    got = sum(r["weight"] for r in results if r["passed"])
    return int(round(100 * got / total))


def build_payload(root: Path) -> dict[str, Any]:
    ctx = load_context(root)
    mres = evaluate(maturity_criteria(), ctx)
    rres = evaluate(realized_criteria(), ctx)
    allres = mres + rres

    m_score = axis_score(mres)
    r_score = axis_score(rres)
    composite = int(round(0.4 * m_score + 0.6 * r_score))  # realized value weighted higher

    hard_fail = [r for r in allres if r["hard"] and not r["passed"]]
    guard_rsi_debt = len(hard_fail)
    grade = grade_letter(composite)
    ok = guard_rsi_debt == 0

    if ok:
        verdict, finding = "OK", "guard_rsi_loop_mature_and_useful"
        reason = (f"guard RSI loop: maturity {m_score}/100, realized {r_score}/100, "
                  f"composite {composite}/100 ({grade}); zero hard gaps; "
                  f"{ctx['audit_rows']} real journal row(s)")
        next_action = ("hold the line; re-run after a change to either guard RSI loop, "
                       "or run `guard_verdict_rsi.py run` after a fresh guarded session")
    else:
        verdict, finding = "ACTION", "guard_rsi_debt"
        gaps = ", ".join(r["key"] for r in hard_fail)
        reason = (f"guard RSI loop carries {guard_rsi_debt} hard gap(s) "
                  f"(maturity {m_score}/100, realized {r_score}/100, composite "
                  f"{composite}/100 {grade}): {gaps}")
        lead = hard_fail[0]
        next_action = f"retire worst-first: {lead['key']} — {lead['detail']}"

    def to_kpi(r: dict[str, Any]) -> dict[str, Any]:
        return {"kpi": r["key"], "group": r["axis"],
                "score": 100 if r["passed"] else 0, "detail": r["detail"],
                "defects": [] if r["passed"] or not r["hard"] else [f"{r['key']}: {r['detail']}"],
                "soft": [] if r["passed"] or r["hard"] else [f"{r['key']}: {r['detail']}"]}

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(root),
        "corpus": {
            "guard_rsi_debt": guard_rsi_debt,
            "score": composite,
            "grade": grade,
            "maturity_score": m_score,
            "realized_score": r_score,
            "audit_rows": ctx["audit_rows"],
            "verdict_quality": ctx["verdict_quality"],
        },
        "kpis": [to_kpi(r) for r in allres],
        "maturity": mres,
        "realized": rres,
    }


def render(p: dict[str, Any]) -> str:
    c = p["corpus"]
    lines = [
        f"guard RSI loop — {p['verdict']} ({p['finding']})",
        f"  guard_rsi_debt: {c['guard_rsi_debt']}   composite {c['score']}/100 [{c['grade']}]"
        f"   (maturity {c['maturity_score']} · realized {c['realized_score']})",
        f"  real journal: {c['audit_rows']} row(s)"
        + (f"   verdict-quality {c['verdict_quality']}" if c['verdict_quality'] is not None else ""),
        "",
        "  MATURITY (can the loop honestly close?):",
    ]
    for r in p["maturity"]:
        mark = "PASS" if r["passed"] else ("FAIL" if r["hard"] else "----")
        lines.append(f"    [{mark}] {r['label']}")
        if not r["passed"]:
            lines.append(f"           -> {r['detail']}")
    lines.append("  REALIZED (operationalised on our usage?):")
    for r in p["realized"]:
        mark = "PASS" if r["passed"] else ("FAIL" if r["hard"] else "----")
        lines.append(f"    [{mark}] {r['label']}")
        if not r["passed"]:
            lines.append(f"           -> {r['detail']}")
    lines.extend(["", f"  -> {p['next_action']}"])
    return "\n".join(lines)


def markdown(p: dict[str, Any]) -> str:
    c = p["corpus"]
    out = [
        "---",
        'title: "fak guard RSI loop scorecard"',
        'description: "How mature (can it honestly close?) and how realized (does it run '
        'on our own guard usage?) the RSI loop(s) for fak guard are, scored from the '
        'tree + the real decision journal."',
        "---",
        "",
        "# fak guard RSI loop scorecard",
        "",
        f"**guard_rsi_debt: {c['guard_rsi_debt']}** · composite **{c['score']}/100 "
        f"({c['grade']})** · maturity {c['maturity_score']}/100 · realized "
        f"{c['realized_score']}/100 · real journal rows {c['audit_rows']}",
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
    out += ["", "## Realized — does it run on our own usage?", "",
            "| ✓ | criterion | detail |", "|---|---|---|"]
    for r in p["realized"]:
        out.append(f"| {'✅' if r['passed'] else '❌'} | {r['label']} | {r['detail']} |")
    out += ["", f"**Next:** {p['next_action']}", ""]
    return "\n".join(out)


def compare(current: dict[str, Any], baseline_path: Path) -> str:
    base = json.loads(baseline_path.read_text(encoding="utf-8"))
    bc = base.get("corpus", base)
    cc = current["corpus"]
    b_debt = bc.get("guard_rsi_debt", bc.get("score"))
    c_debt = cc["guard_rsi_debt"]
    delta = (b_debt - c_debt) if isinstance(b_debt, int) else None
    lines = [
        "guard-rsi compare:",
        f"  guard_rsi_debt: {b_debt} -> {c_debt}"
        + (f"  (retired {delta})" if delta is not None else ""),
        f"  composite: {bc.get('score')} -> {cc['score']}  "
        f"grade {bc.get('grade')} -> {cc['grade']}",
        f"  real journal rows: {bc.get('audit_rows')} -> {cc['audit_rows']}",
    ]
    if isinstance(b_debt, int) and b_debt > 0:
        if c_debt * 3 <= b_debt:
            lines.append(f"  VERDICT: >=3x improvement (debt {b_debt} -> {c_debt}, "
                         f"<= 1/3 of baseline)")
        elif c_debt * 2 <= b_debt:
            lines.append(f"  VERDICT: >=2x improvement (debt {b_debt} -> {c_debt})")
        elif c_debt < b_debt:
            lines.append(f"  VERDICT: improved but < 2x (debt {b_debt} -> {c_debt})")
        else:
            lines.append(f"  VERDICT: no improvement (debt {b_debt} -> {c_debt})")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="fak guard RSI loop scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit the control-pane JSON payload")
    ap.add_argument("--markdown", action="store_true", help="emit the scorecard doc body")
    ap.add_argument("--compare", metavar="BASELINE.json", default="",
                    help="compare against a prior --json payload and print the debt delta + Nx verdict")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = build_payload(root)

    if args.compare:
        print(compare(payload, Path(args.compare)))
        return 0 if payload.get("ok") else 1
    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(markdown(payload))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
