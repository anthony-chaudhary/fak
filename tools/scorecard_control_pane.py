#!/usr/bin/env python3
"""Unified scorecard debt control-pane — fold every *-debt into one tracked trend.

The repo has deterministic scorecards, each emitting a debt integer plus a
control-pane payload (``schema/ok/verdict/finding/reason/next_action``): docs,
README freshness, code, doc-appeal, seo, demo-quality, demo-robustness, repo-hygiene, the one
OUTWARD-facing stick — industry-parity (fak vs SOTA) — agent-readiness (can an
agent discover, adopt, and build on fak), product (can a PERSON pick up each fak
concept and use it today — durable / real / useful-today), persona (are the
top-10 personas who land on fak — free-tier dev through researcher — each served),
and stability (can we tell when a regression / tail-wag / confusion landed, and
revert to a stable version). They run independently and advisory. Nothing folds
{doc_debt, readme_debt, code_debt, appeal_debt, seo_debt, demo_debt, robustness_debt,
hygiene_debt, parity_debt, friction_debt, product_debt, persona_debt,
stability_debt} into one number, pins a per-metric baseline, and shows the trend
commit-over-commit.

This is that fold — the RSI checking layer for the whole scorecard family. It
runs each scorecard, extracts the debt integer + grade, sums one portfolio
``total_debt``, and compares against a pinned per-metric baseline so the answer
to "is the repo getting better or worse" is one query.

  python tools/scorecard_control_pane.py            # human snapshot + trend
  python tools/scorecard_control_pane.py --json      # machine payload
  python tools/scorecard_control_pane.py --pin       # pin today's debt as the baseline
  python tools/scorecard_control_pane.py --check     # CI ratchet gate (fail only on regression)

The baseline lives in a tracked file (``tools/scorecard_baseline.json``) so the
trend is commit-over-commit and shared: re-pin after a debt drop to ratchet it
down. Pure-stdlib Python, repo root like the other honesty gates.

``--check`` is the RSI ratchet the repo-3x epic (#506) names: it turns the one
folded number into an enforceable gate. Unlike the default exit code (green only
at ZERO debt), ``--check`` is GREEN while the portfolio holds at-or-below its
pinned baseline and RED only when debt *regresses* above it (or a scorecard
fails to report). That is the honest CI contract — debt may stay or fall, never
silently rise — without demanding the whole family be at zero first. Issue #509.
The README freshness scorecard is deliberately wired here, not as a bespoke
``--min-score`` CI line: its baseline pins ``readme_debt`` at zero, so a front-page
score-affordance regression reds through the existing green ratchet (#779/#893).

The portfolio ratchet has one blind spot: it folds every metric into one sum, so
a single metric's regression can hide under another metric's improvement (seo
rose 6->8 while the portfolio fell 44->40 — the ratchet stayed green, and a blind
``--pin`` would have blessed the seo rise as the new floor). The per-metric
EARLY-WARNING lens (#712) closes it: any metric whose debt rose vs its pinned
value is reported as an advisory WARN even when the portfolio total is green —
the trend carries an ``early_warning`` list, ``--check`` appends it to the
RATCHET OK line WITHOUT tripping the gate (the portfolio ratchet semantics are
unchanged), and the human snapshot prints it. So a hidden per-metric regression
surfaces BEFORE a re-pin locks it in.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-scorecard-control-pane/1"
BASELINE_SCHEMA = "fak-scorecard-control-pane.baseline/1"
BASELINE_REL = "tools/scorecard_baseline.json"

# The scorecard family, in the canonical order the issue lists them. Each entry
# binds the scorecard's script to the debt integer it emits; the runner folds
# every debt key into one portfolio number.
SCORECARDS: list[dict[str, str]] = [
    {"key": "doc", "debt": "doc_debt", "script": "docs_scorecard.py", "label": "docs"},
    {"key": "readme", "debt": "readme_debt", "script": "readme_freshness_audit.py", "label": "readme-freshness"},
    {"key": "code", "debt": "code_debt", "script": "code_quality_scorecard.py", "label": "code"},
    {"key": "appeal", "debt": "appeal_debt", "script": "doc_appeal_scorecard.py", "label": "doc-appeal"},
    {"key": "seo", "debt": "seo_debt", "script": "seo_aeo_scorecard.py", "label": "seo"},
    {"key": "demo", "debt": "demo_debt", "script": "demo_quality_scorecard.py", "label": "demo-quality"},
    {"key": "robustness", "debt": "robustness_debt", "script": "demo_robustness_scorecard.py", "label": "demo-robustness"},
    {"key": "hygiene", "debt": "hygiene_debt", "script": "repo_hygiene_scorecard.py", "label": "repo-hygiene"},
    {"key": "parity", "debt": "parity_debt", "script": "industry_scorecard.py", "label": "industry-parity"},
    {"key": "agent", "debt": "friction_debt", "script": "agent_readiness_scorecard.py", "label": "agent-readiness"},
    {"key": "product", "debt": "product_debt", "script": "product_scorecard.py", "label": "product"},
    {"key": "persona", "debt": "persona_debt", "script": "persona_readiness_scorecard.py", "label": "persona"},
    {"key": "stability", "debt": "stability_debt", "script": "stability_scorecard.py", "label": "stability"},
    {"key": "slop", "debt": "slop_debt", "script": "code_slop_scorecard.py", "label": "code-slop"},
    {"key": "steer", "debt": "steerability_debt", "script": "steerability_scorecard.py", "label": "steerability"},
    {"key": "conflation", "debt": "conflation_debt", "script": "", "cmd": "go run ./cmd/fak conflation-scorecard --json", "label": "conflation"},
    {"key": "disambiguation", "debt": "disambiguation_debt", "script": "concept_disambiguation_scorecard.py", "label": "concept-disambiguation"},
    {"key": "intent_literal", "debt": "intent_literal_debt", "script": "intent_literal_scorecard.py", "label": "intent-literal"},
    {"key": "tokendefaults", "debt": "token_defaults_debt", "script": "", "cmd": "go run ./cmd/fak token-defaults-scorecard --json", "label": "token-defaults"},
    {"key": "guard_rsi", "debt": "guard_rsi_debt", "script": "", "cmd": "go run ./cmd/fak guard-rsi-scorecard --json", "label": "guard-rsi"},
    {"key": "dogfood", "debt": "dogfood_debt", "script": "", "cmd": "go run ./cmd/fak dogfood-score --json", "label": "dogfood-loop"},
    {"key": "growth", "debt": "growth_debt", "script": "growth_debt_scorecard.py", "label": "growth-debt"},
]


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _git_line(args: list[str], root: Path) -> str:
    try:
        p = subprocess.run(["git", *args], cwd=str(root), capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    if p.returncode != 0:
        return ""
    return p.stdout.strip()


def head_commit(root: Path) -> str:
    return _git_line(["rev-parse", "--short", "HEAD"], root) or "unknown"


# --- pure extraction / folding (the tested surface) ------------------------

def find_int(payload: Any, key: str) -> int | None:
    """First int value stored under ``key`` anywhere in the payload.

    The debt integer lives under ``corpus.<debt>`` for most scorecards and
    ``doc.<debt>`` for doc-appeal; a tolerant search keeps the fold from caring
    which nesting a given scorecard chose.
    """
    if isinstance(payload, dict):
        for nest in ("corpus", "doc"):
            sub = payload.get(nest)
            if isinstance(sub, dict) and isinstance(sub.get(key), bool) is False \
                    and isinstance(sub.get(key), int):
                return int(sub[key])
        val = payload.get(key)
        if isinstance(val, int) and not isinstance(val, bool):
            return int(val)
        for v in payload.values():
            got = find_int(v, key)
            if got is not None:
                return got
    elif isinstance(payload, list):
        for v in payload:
            got = find_int(v, key)
            if got is not None:
                return got
    return None


def find_grade(payload: Any) -> str | None:
    """The portfolio grade a scorecard reports at corpus/doc level, if any."""
    if isinstance(payload, dict):
        for nest in ("corpus", "doc"):
            sub = payload.get(nest)
            if isinstance(sub, dict) and isinstance(sub.get("grade"), str):
                return str(sub["grade"])
        if isinstance(payload.get("grade"), str):
            return str(payload["grade"])
    return None


def metric_from_payload(card: dict[str, str], payload: dict[str, Any] | None,
                        error: str = "") -> dict[str, Any]:
    debt_key = card["debt"]
    if error or not isinstance(payload, dict):
        return {
            "key": card["key"],
            "label": card["label"],
            "debt_key": debt_key,
            "debt": None,
            "grade": None,
            "ok": False,
            "verdict": "ERROR",
            "error": error or "no payload",
        }
    debt = find_int(payload, debt_key)
    return {
        "key": card["key"],
        "label": card["label"],
        "debt_key": debt_key,
        "debt": debt,
        "grade": find_grade(payload),
        "ok": bool(payload.get("ok")),
        "verdict": str(payload.get("verdict") or ""),
        "error": "" if debt is not None else f"missing {debt_key} in payload",
    }


def fold(metrics: list[dict[str, Any]], baseline: dict[str, Any] | None,
         *, workspace: str, commit: str) -> dict[str, Any]:
    """Fold per-scorecard metrics into one control-pane payload + trend."""
    measured = [m for m in metrics if isinstance(m.get("debt"), int)]
    errors = [m for m in metrics if not isinstance(m.get("debt"), int)]
    total_debt = sum(int(m["debt"]) for m in measured)

    trend = compute_trend(metrics, baseline, total_debt)

    by_debt = sorted(measured, key=lambda m: int(m["debt"]), reverse=True)
    breakdown = ", ".join(f"{m['label']} {m['debt']}" for m in by_debt) or "none"

    regressed = trend["direction"] == "regressed"
    early_warning = trend.get("early_warning") or []
    ew_note = ""
    if early_warning and not regressed:
        # The hidden case the early-warning lens exists for (#712): a metric rose
        # but the portfolio held, so the ratchet stays green. Surface it advisory —
        # don't flip the verdict (the portfolio ratchet semantics are unchanged).
        ew_note = ("; EARLY-WARNING (advisory): "
                   + ", ".join(f"{e['label']} {e['from']}->{e['to']} (+{e['delta']})"
                               for e in early_warning)
                   + " rose vs baseline under a green portfolio — a hidden per-metric "
                     "regression; review before --pin re-floors it")
    if errors:
        ok, verdict, finding = False, "ACTION", "scorecard_unmeasured"
        reason = (f"{len(errors)} scorecard(s) failed to report a debt integer "
                  f"({', '.join(m['label'] for m in errors)}); portfolio debt "
                  f"{total_debt} across {len(measured)} measured")
        next_action = ("repair the failing scorecard(s) so the fold is complete; "
                       "re-run python tools/scorecard_control_pane.py")
    elif regressed:
        ok, verdict, finding = False, "ACTION", "scorecard_regressed"
        reason = (f"portfolio debt rose {trend['total_delta']:+d} to {total_debt} "
                  f"vs baseline @{trend['baseline_commit']} ({breakdown}); "
                  f"worsened: {', '.join(trend['worsened']) or 'see deltas'}")
        next_action = ("retire the regressed metric(s) worst-first with the owning "
                       "scorecard's skill, then re-pin: "
                       "python tools/scorecard_control_pane.py --pin")
    elif total_debt > 0:
        ok, verdict, finding = False, "ACTION", "scorecard_debt"
        reason = (f"portfolio debt {total_debt} across {len(measured)} scorecards "
                  f"({breakdown}); trend {trend['summary']}")
        next_action = ("retire debt worst-first (heaviest: "
                       f"{by_debt[0]['label']} {by_debt[0]['debt']}) with that "
                       "scorecard's skill; re-run to prove the portfolio drop")
    else:
        ok, verdict, finding = True, "OK", "all_clear"
        reason = (f"zero portfolio debt across {len(measured)} scorecards; "
                  f"trend {trend['summary']}")
        next_action = "hold the line; re-pin the baseline to lock the clean state"

    reason += ew_note
    if ew_note:
        # Point the operator at the offending metric(s) regardless of the verdict
        # ladder branch — the early-warning is the actionable signal here.
        next_action = ("review the per-metric early-warning ("
                       + ", ".join(e["label"] for e in early_warning)
                       + ") with that scorecard's skill BEFORE `--pin`, so a hidden "
                       "regression isn't blessed as the new floor; then: " + next_action)

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "commit": commit,
        "total_debt": total_debt,
        "measured": len(measured),
        "errored": len(errors),
        "early_warning": early_warning,
        "metrics": metrics,
        "trend": trend,
    }


def compute_trend(metrics: list[dict[str, Any]], baseline: dict[str, Any] | None,
                  total_debt: int) -> dict[str, Any]:
    """Per-metric + portfolio delta vs a pinned baseline."""
    base_metrics = {}
    base_commit = ""
    base_total = None
    if isinstance(baseline, dict):
        base_metrics = baseline.get("metrics") or {}
        base_commit = str(baseline.get("commit") or "")
        bt = baseline.get("total_debt")
        base_total = int(bt) if isinstance(bt, int) and not isinstance(bt, bool) else None

    if not base_metrics or base_total is None:
        return {
            "direction": "unpinned",
            "summary": "unpinned (no baseline; run --pin)",
            "total_delta": 0,
            "baseline_commit": base_commit,
            "baseline_total": base_total,
            "deltas": {},
            "worsened": [],
            "improved": [],
            "early_warning": [],
        }

    deltas: dict[str, int] = {}
    worsened: list[str] = []
    improved: list[str] = []
    # The per-metric early-warning lens (#712): EVERY metric whose debt rose vs its
    # pinned value, independent of where the portfolio total landed. The portfolio
    # ratchet only trips when the SUM regresses, so a single metric's rise can hide
    # under another's improvement (seo 6->8 while the portfolio fell 44->40). This
    # list surfaces that first downward move WITHIN a healthy envelope — before a
    # blind --pin blesses it as the new floor.
    early_warning: list[dict[str, Any]] = []
    for m in metrics:
        if not isinstance(m.get("debt"), int):
            continue
        prior = base_metrics.get(m["key"])
        if not isinstance(prior, int) or isinstance(prior, bool):
            continue
        delta = int(m["debt"]) - int(prior)
        deltas[m["key"]] = delta
        if delta > 0:
            worsened.append(m["label"])
            early_warning.append({"key": m["key"], "label": m["label"],
                                  "delta": delta, "from": int(prior), "to": int(m["debt"])})
        elif delta < 0:
            improved.append(m["label"])

    total_delta = total_debt - base_total
    if total_delta > 0:
        direction = "regressed"
    elif total_delta < 0:
        direction = "improved"
    else:
        direction = "flat"
    summary = (f"{direction} {total_delta:+d} vs @{base_commit or 'baseline'} "
               f"(was {base_total}, now {total_debt})")
    return {
        "direction": direction,
        "summary": summary,
        "total_delta": total_delta,
        "baseline_commit": base_commit,
        "baseline_total": base_total,
        "deltas": deltas,
        "worsened": worsened,
        "improved": improved,
        "early_warning": early_warning,
    }


def baseline_doc(payload: dict[str, Any]) -> dict[str, Any]:
    """The baseline file body to pin from a folded control-pane payload."""
    metrics = {
        m["key"]: int(m["debt"])
        for m in payload.get("metrics", [])
        if isinstance(m.get("debt"), int)
    }
    return {
        "schema": BASELINE_SCHEMA,
        "commit": payload.get("commit", ""),
        "total_debt": payload.get("total_debt", 0),
        "metrics": metrics,
        "_doc": ("Pinned per-metric scorecard-debt baseline for the unified "
                 "control pane. Re-pin after a debt drop to ratchet the trend down: "
                 "python tools/scorecard_control_pane.py --pin"),
    }


# --- live runner -----------------------------------------------------------

def run_scorecard(root: Path, card: dict[str, str] | str, *, python: str, timeout: int) -> tuple[dict[str, Any] | None, str]:
    if isinstance(card, dict) and card.get("cmd"):
        argv = shlex.split(card["cmd"])
    else:
        script = card["script"] if isinstance(card, dict) else card
        script_path = root / "tools" / script
        if not script_path.exists():
            return None, f"missing scorecard: tools/{script}"
        argv = [python, str(script_path), "--json"]
    try:
        proc = subprocess.run(
            argv,
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, f"timed out after {timeout}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, str(exc)
    try:
        return json.loads(proc.stdout), ""
    except ValueError:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        return None, f"non-JSON output (exit {proc.returncode}): {tail[0][:160]}"


def collect(root: Path, *, python: str = "", timeout: int = 120) -> list[dict[str, Any]]:
    python = python or sys.executable
    metrics: list[dict[str, Any]] = []
    for card in SCORECARDS:
        payload, error = run_scorecard(root, card, python=python, timeout=timeout)
        metrics.append(metric_from_payload(card, payload, error))
    return metrics


def load_baseline(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"scorecard control pane — {payload['verdict']} ({payload['finding']})",
        f"  portfolio debt: {payload['total_debt']}  "
        f"({payload['measured']} measured, {payload['errored']} errored)  @{payload['commit']}",
        f"  trend: {payload['trend']['summary']}",
        "",
    ]
    for m in payload["metrics"]:
        debt = m["debt"] if m["debt"] is not None else f"ERR ({m['error']})"
        grade = f" [{m['grade']}]" if m.get("grade") else ""
        lines.append(f"  {m['label']:<16} {m['debt_key']:<16} {debt}{grade}")
    early_warning = payload.get("early_warning") or []
    if early_warning:
        lines.append("")
        for e in early_warning:
            lines.append(f"  WARN early-warning: {e['label']} rose {e['from']}->{e['to']} "
                         f"(+{e['delta']}) vs baseline — hidden under a green portfolio")
    lines.extend(["", f"  → {payload['next_action']}"])
    return "\n".join(lines)


def check_gate(payload: dict[str, Any]) -> tuple[int, str]:
    """The CI ratchet decision over a folded payload (pure: exit code + message).

    The default exit code is green only at ZERO portfolio debt — too strict to
    gate a repo that still carries real debt. This is the ratchet contract the
    repo-3x epic (#506) wants instead: debt may hold or fall, never rise.

      0  flat / improved   — the ratchet held (green even with nonzero debt)
      1  regressed         — debt rose above the pinned baseline (or unmeasured)
      2  unpinned          — no baseline to ratchet against; run --pin first
    """
    if int(payload.get("errored", 0)) > 0:
        return 1, (f"RATCHET FAIL: {payload['errored']} scorecard(s) unmeasured — "
                   f"{payload['reason']}")
    trend = payload.get("trend") or {}
    direction = trend.get("direction")
    if direction == "unpinned":
        return 2, ("RATCHET UNPINNED: no baseline to ratchet against; run "
                   "`python tools/scorecard_control_pane.py --pin` to set one")
    if direction == "regressed":
        return 1, f"RATCHET FAIL: {trend['summary']}; worsened: {', '.join(trend['worsened']) or 'see deltas'}"
    msg = f"RATCHET OK: {trend['summary']} (debt {payload['total_debt']} held at-or-below baseline)"
    # The early-warning lens (#712): the portfolio ratchet held (exit 0), but a
    # per-metric rise is hiding under it — surface it ADVISORY without tripping the
    # gate, so it's seen before a re-pin re-floors it as the new baseline.
    early_warning = (trend.get("early_warning") or []) if isinstance(trend, dict) else []
    if early_warning:
        msg += ("; EARLY-WARNING (advisory, gate still green): "
                + ", ".join(f"{e['label']} +{e['delta']}" for e in early_warning)
                + " rose vs baseline — a hidden per-metric regression; review before --pin")
    return 0, msg


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Unified scorecard debt control-pane (read-only unless --pin).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--pin", action="store_true",
                    help=f"pin the current debt as the baseline ({BASELINE_REL})")
    ap.add_argument("--check", action="store_true",
                    help="CI ratchet gate: exit non-zero only if debt regressed above baseline (#506)")
    ap.add_argument("--baseline", default="", help=f"baseline JSON path (default: {BASELINE_REL})")
    ap.add_argument("--timeout", type=int, default=120, help="per-scorecard timeout seconds")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    baseline_path = Path(args.baseline).resolve() if args.baseline else (root / BASELINE_REL)

    metrics = collect(root, timeout=args.timeout)
    baseline = load_baseline(baseline_path)
    payload = fold(metrics, baseline, workspace=str(root), commit=head_commit(root))

    if args.pin:
        doc = baseline_doc(payload)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
        if not args.json:
            print(f"pinned baseline @{doc['commit']} total_debt={doc['total_debt']} -> {baseline_path}")

    if args.check:
        code, message = check_gate(payload)
        if args.json:
            # Under --check the tool's contract IS the ratchet, not the raw fold:
            # ok/verdict reflect "did the portfolio hold at-or-below baseline?"
            # (green even with residual debt), not "is debt zero?". This is what a
            # loop runner reads to fold the pane — keep gate_exit/gate_message for
            # the literal exit code. #509.
            gated = {
                **payload,
                "ok": code == 0,
                "verdict": "OK" if code == 0 else "ACTION",
                "gate_exit": code,
                "gate_message": message,
            }
            print(json.dumps(gated, indent=2))
        else:
            print(message)
        return code

    if args.json:
        print(json.dumps(payload, indent=2))
    elif not args.pin:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
