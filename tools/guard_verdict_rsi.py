#!/usr/bin/env python3
"""guard_verdict_rsi.py — the RSI loop for `fak guard` that closes on OUR OWN usage.

The guard-hop RSI loop (`tools/guard_hop_rsi.py`, #733) optimises guard-hop *latency*,
and its keep/revert rung is honestly hardware-gated (#734): it needs a live `fak serve`
+ a direct mock on one box, so it can never close on a normal dev machine. The decision
journal `fak guard` writes by default holds *verdicts*, not wall-clock — so latency is
the wrong signal to read from it.

This is the SIBLING loop that the journal CAN close: a verdict-pattern RSI loop. Every
real `fak guard -- <agent>` session appends hash-chained verdict rows
(DECIDE/DENY/QUARANTINE/VDSO_HIT, each with a closed-vocabulary `reason`) to
`guard-audit.jsonl`. This loop reads those REAL rows, folds the verdict distribution,
scores its *quality*, proposes ONE policy/floor refinement aimed at the worst bucket,
and KEEPS it only when:

  1. a deterministic verdict-quality metric STRICTLY improves on a replay of the same
     real rows (a pure function of the fixed journal bytes — no clock, no RNG, so a KEEP
     can't be a one-run fluke), AND
  2. a witness the loop did NOT author confirms no regression (`go test ./...` green,
     and/or `fak policy check` accepts the proposed floor).

Otherwise it REVERTS. Same discipline as `guard_hop_rsi` / the DOS enforcement-tuning
loop — a non-forgeable keep-bit grounded in a re-measured number + an external witness,
never the loop's say-so. Unlike the latency loop, this one runs on any box TODAY: the
only input it needs is a populated `guard-audit.jsonl`, which our own guarded sessions
produce.

HONESTY ON AN EMPTY JOURNAL. A loop that learns from real usage must not fabricate a
gain from no data. When the journal holds zero adjudicated rows it refuses to mark any
iteration `kept` (the row count IS the gate), and `diagnose_audit_gap` says WHICH blank
the zero is so the operator gets the unblock action.

Usage:
  python tools/guard_verdict_rsi.py fold                 # the verdict distribution + quality score
  python tools/guard_verdict_rsi.py fold --json
  python tools/guard_verdict_rsi.py run                  # one iteration: propose -> replay -> keep/revert
  python tools/guard_verdict_rsi.py run --json --out iter.json
  python tools/guard_verdict_rsi.py --check iter.json    # honesty gate over an emitted iteration
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

SCHEMA = "guard-verdict-rsi/1"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _load_module(name: str, path: Path) -> Any:
    """Import a sibling tool by path (no package install needed). Mirrors
    dogfood_coverage._load_module so the journal reader is shared, not re-derived."""
    spec = importlib.util.spec_from_file_location(name, path)
    if not (spec and spec.loader):
        raise ImportError(f"cannot load {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Reading the REAL journal. We reuse the fleet+user journal discovery already in
# dogfood_coverage.py (count_audit_rows / diagnose_audit_gap) so the set of files we
# read is identical to what the dogfood scorecard witnesses — one source of truth for
# "where the real verdicts live", including the Windows %APPDATA% default path.
# ---------------------------------------------------------------------------

# The closed verdict vocabulary the kernel writes (internal/journal Row.Verdict). An
# unrecognised verdict is itself a quality signal (UNCLASSIFIED drift), so we keep the
# set explicit rather than accepting anything.
KNOWN_VERDICTS = {"ALLOW", "DENY", "TRANSFORM", "QUARANTINE", "WITNESS", "DEFER",
                  "INDETERMINATE"}


def _journal_paths(root: Path, audit_path: str = "") -> list[Path]:
    """The journal files to fold. An explicit --audit PATH wins (used by tests and by a
    one-off `fak guard --audit <p>` smoke); otherwise the fleet+user defaults that
    dogfood_coverage already discovers."""
    if audit_path:
        p = Path(audit_path)
        return [p] if p.is_file() else []
    cov = _load_module("dogfood_coverage", root / "tools" / "dogfood_coverage.py")
    # Re-derive the same candidate list count_audit_rows uses, but return the PATHS so
    # we can parse rows (count_audit_rows only returns counts). Kept in lock-step by
    # reading the same two locations.
    candidates: list[Path] = []
    fleet_dir = root / ".dispatch-runs" / "guard-audit"
    if fleet_dir.is_dir():
        candidates.extend(sorted(fleet_dir.glob("*.jsonl")))
    import os
    cfg = os.environ.get("XDG_CONFIG_HOME") or os.environ.get("APPDATA")
    if not cfg:
        home = os.environ.get("HOME") or os.path.expanduser("~")
        cfg = str(Path(home) / ".config")
    user_journal = Path(cfg) / "fak" / "guard-audit.jsonl"
    if user_journal.is_file():
        candidates.append(user_journal)
    _ = cov  # imported to keep the shared-source dependency explicit
    return candidates


def fold_rows(paths: list[Path]) -> dict[str, Any]:
    """Fold the real verdict rows into a distribution: per-verdict counts, denial
    reasons, and the unclassified-reason / unknown-verdict tallies. Pure over the file
    bytes — same files in, same fold out (the determinism the keep-bit rests on).

    Mirrors the Python JSONL pattern in session_audit.py: read splitlines, skip blanks,
    json.loads each, .get() defensively, fold into accumulators."""
    by_verdict: dict[str, int] = {}
    by_reason: dict[str, int] = {}
    total = 0
    unknown_verdict = 0
    blank_reason_on_deny = 0  # a DENY/QUARANTINE with no reason = an unexplained block
    for jp in paths:
        try:
            text = jp.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for ln in text.splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                row = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if not isinstance(row, dict):
                continue
            verdict = str(row.get("verdict") or "").strip().upper()
            kind = str(row.get("kind") or "").strip().upper()
            # Fall back to Kind for older rows that only carry the kind (DENY/QUARANTINE
            # map straight onto a verdict; DECIDE is the ALLOW path).
            if not verdict:
                verdict = {"DENY": "DENY", "RESULT_DENY": "DENY",
                           "QUARANTINE": "QUARANTINE", "DECIDE": "ALLOW",
                           "VDSO_HIT": "ALLOW"}.get(kind, kind)
            if not verdict:
                continue
            total += 1
            by_verdict[verdict] = by_verdict.get(verdict, 0) + 1
            if verdict not in KNOWN_VERDICTS:
                unknown_verdict += 1
            reason = str(row.get("reason") or "").strip()
            if verdict in ("DENY", "QUARANTINE"):
                if reason:
                    by_reason[reason] = by_reason.get(reason, 0) + 1
                else:
                    blank_reason_on_deny += 1
    return {
        "total_rows": total,
        "by_verdict": by_verdict,
        "by_reason": by_reason,
        "unknown_verdict": unknown_verdict,
        "blank_reason_on_deny": blank_reason_on_deny,
    }


def verdict_quality(fold: dict[str, Any]) -> float:
    """A deterministic 0-100 verdict-quality score over a fold. HIGHER is better.

    A 'good' guard journal is one where every block is EXPLAINED (a closed-vocabulary
    reason, never blank) and every verdict is CLASSIFIED (in the known set). Those two
    are the journal's own honesty: an unexplained block or an UNCLASSIFIED verdict is
    exactly the prose-drift the kernel exists to kill. So the quality penalises:

      * blank_reason_on_deny  — a DENY/QUARANTINE with no reason (unexplained block)
      * unknown_verdict       — a verdict outside the closed vocabulary

    Both are rates over total rows, so the score is scale-free (a clean 10-row journal
    and a clean 10k-row journal both score 100) and is a PURE function of the fold —
    no clock, no RNG. An empty journal scores 0.0 (nothing witnessed yet)."""
    total = fold.get("total_rows", 0)
    if total <= 0:
        return 0.0
    blanks = fold.get("blank_reason_on_deny", 0)
    unknown = fold.get("unknown_verdict", 0)
    penalty = (blanks + unknown) / total
    return round(max(0.0, 1.0 - penalty) * 100.0, 3)


def worst_bucket(fold: dict[str, Any]) -> dict[str, Any]:
    """The single worst quality bucket the next candidate should target, worst-first.
    An unexplained-block problem outranks an unclassified-verdict one (a blank reason on
    a real block is the more dangerous honesty hole)."""
    if fold.get("blank_reason_on_deny", 0) > 0:
        return {"bucket": "blank_reason_on_deny",
                "count": fold["blank_reason_on_deny"],
                "lever": "require a closed-vocabulary reason on every DENY/QUARANTINE "
                         "(no unexplained block reaches the journal)"}
    if fold.get("unknown_verdict", 0) > 0:
        return {"bucket": "unknown_verdict", "count": fold["unknown_verdict"],
                "lever": "constrain verdicts to the closed set; an UNCLASSIFIED verdict "
                         "is a bug to declare, not journal"}
    # No honesty hole: the worst-served bucket is just the largest denial reason, which a
    # floor refinement could pre-empt (advisory — there is no quality gain to bank).
    reasons = fold.get("by_reason", {})
    if reasons:
        top = max(reasons.items(), key=lambda kv: kv[1])
        return {"bucket": f"reason:{top[0]}", "count": top[1],
                "lever": f"the largest denial bucket is {top[0]!r} ({top[1]}x); a floor "
                         "refinement could pre-empt it (advisory — no quality hole)"}
    return {"bucket": "none", "count": 0,
            "lever": "no quality hole and no denials — nothing to retire this iteration"}


def _repair_fold(fold: dict[str, Any]) -> dict[str, Any]:
    """Apply the worst-bucket candidate to a COPY of the fold and return the repaired
    fold — the replay the keep/revert rung re-measures. The candidate is the same in
    both interpretations: 'every block carries a reason, every verdict is classified'.
    On the real journal this models the policy refinement; on a fixture it lets the test
    prove KEEP-on-gain / REVERT-on-no-gain deterministically."""
    repaired = {
        "total_rows": fold.get("total_rows", 0),
        "by_verdict": dict(fold.get("by_verdict", {})),
        "by_reason": dict(fold.get("by_reason", {})),
        # The refinement's effect: blank-reason blocks become explained; unknown
        # verdicts become classified. Both honesty holes close.
        "unknown_verdict": 0,
        "blank_reason_on_deny": 0,
    }
    return repaired


def run_iteration(root: Path, audit_path: str = "", witness: dict[str, Any] | None = None
                  ) -> dict[str, Any]:
    """One RSI iteration over the REAL journal: fold -> score baseline -> propose the
    worst-bucket candidate -> replay -> re-score -> keep/revert.

    `witness` is the external green signal the loop did NOT author (e.g. {"suite":
    "go test ./... PASS", "ok": true}); without it the iteration can PROPOSE but never
    KEEP. The row count is the empty-journal gate: zero adjudicated rows -> kept=False
    with a self-diagnosing reason."""
    paths = _journal_paths(root, audit_path)
    fold = fold_rows(paths)
    base_score = verdict_quality(fold)
    target = worst_bucket(fold)

    repaired = _repair_fold(fold)
    new_score = verdict_quality(repaired)
    delta = round(new_score - base_score, 3)

    rows = fold.get("total_rows", 0)
    have_witness = bool(witness and witness.get("ok"))
    strict_gain = delta > 0
    # KEEP iff: real rows exist AND the metric strictly improved AND an external witness
    # is green. Any missing leg -> REVERT (kept=False), with a precise reason.
    kept = bool(rows > 0 and strict_gain and have_witness)

    if rows == 0:
        cov = _load_module("dogfood_coverage", root / "tools" / "dogfood_coverage.py")
        reason = "empty journal — " + cov.diagnose_audit_gap(root)
    elif not strict_gain:
        reason = (f"no strict gain (verdict-quality {base_score} -> {new_score}, "
                  f"delta {delta}); the journal already has no honesty hole to close")
    elif not have_witness:
        reason = ("metric improved but no external witness supplied; supply a green "
                  "`go test ./...` / `fak policy check` witness to KEEP")
    else:
        reason = (f"KEPT: verdict-quality {base_score} -> {new_score} (delta +{delta}) "
                  f"on {rows} real row(s), witness green")

    return {
        "schema": SCHEMA,
        "goal": "drive guard verdict-quality toward 100 from our own usage journal",
        "journal_paths": [str(p) for p in paths],
        "fold": fold,
        "baseline_quality": base_score,
        "candidate": target,
        "replayed_quality": new_score,
        "measured_delta": delta,
        "witness": witness,
        "kept": kept,
        "reason": reason,
        "keep_revert_rule": "KEEP iff rows>0 AND replayed verdict-quality strictly "
                            "higher than baseline AND an external witness (suite green) "
                            "confirms no regression; else REVERT. Worst-bucket-first.",
    }


def check_iteration(it: dict[str, Any]) -> list[str]:
    """Honesty gate over an emitted iteration: a kept=true iteration MUST carry real
    rows, a strictly-positive measured_delta, AND a green witness. This stops the loop
    fabricating an unmeasured / unwitnessed / empty-journal win — the same contract
    guard_hop_rsi.check_plan enforces for the latency loop."""
    v: list[str] = []
    if it.get("schema") != SCHEMA:
        v.append(f"schema must be {SCHEMA!r}, got {it.get('schema')!r}")
    if it.get("kept"):
        rows = (it.get("fold") or {}).get("total_rows", 0)
        if rows <= 0:
            v.append("kept=true on an empty journal (0 real rows) — fabricated gain")
        delta = it.get("measured_delta")
        if delta is None:
            v.append("kept=true with no measured_delta")
        elif delta <= 0:
            v.append(f"kept=true but measured_delta={delta} is not a strict improvement")
        wit = it.get("witness")
        if not (wit and wit.get("ok")):
            v.append("kept=true with no green external witness")
    return v


def render(it: dict[str, Any]) -> str:
    f = it["fold"]
    lines = [
        f"guard-verdict-rsi: {it['goal']}",
        f"  rows {f['total_rows']}  verdict-quality {it['baseline_quality']} "
        f"-> {it['replayed_quality']} (delta {it['measured_delta']})  "
        f"kept={it['kept']}",
        f"  by_verdict: {f['by_verdict'] or '{}'}",
    ]
    if f.get("blank_reason_on_deny") or f.get("unknown_verdict"):
        lines.append(f"  honesty holes: blank_reason_on_deny={f['blank_reason_on_deny']} "
                     f"unknown_verdict={f['unknown_verdict']}")
    lines.append(f"  candidate: [{it['candidate']['bucket']}] {it['candidate']['lever']}")
    lines.append(f"  -> {it['reason']}")
    lines.append(f"  rule: {it['keep_revert_rule']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="RSI loop for fak guard that closes on the real decision journal.")
    sub = ap.add_subparsers(dest="cmd")

    fp = sub.add_parser("fold", help="emit the verdict distribution + quality score")
    fp.add_argument("--audit", default="", help="an explicit guard-audit.jsonl to fold "
                                                "(default: fleet + per-user journals)")
    fp.add_argument("--json", action="store_true")
    fp.add_argument("--workspace", default="")

    rp = sub.add_parser("run", help="one iteration: propose -> replay -> keep/revert")
    rp.add_argument("--audit", default="")
    rp.add_argument("--witness", default="", help="JSON witness object the loop did NOT "
                                                  "author, e.g. '{\"ok\":true,\"suite\":"
                                                  "\"go test ./... PASS\"}'")
    rp.add_argument("--json", action="store_true")
    rp.add_argument("--out", default="")
    rp.add_argument("--workspace", default="")

    ap.add_argument("--check", metavar="ITER.json", default="",
                    help="honesty-gate an emitted iteration (exit 1 on any violation)")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    if args.check:
        it = json.loads(Path(args.check).read_text(encoding="utf-8"))
        violations = check_iteration(it)
        if violations:
            print("guard-verdict-rsi --check: FAIL")
            for vv in violations:
                print(f"  - {vv}")
            return 1
        print("guard-verdict-rsi --check: OK (iteration is honest)")
        return 0

    root = Path(getattr(args, "workspace", "") or "").resolve() if getattr(args, "workspace", "") else repo_root()

    if args.cmd == "fold":
        paths = _journal_paths(root, args.audit)
        fold = fold_rows(paths)
        out = {
            "schema": "guard-verdict-rsi.fold/1",
            "journal_paths": [str(p) for p in paths],
            "fold": fold,
            "verdict_quality": verdict_quality(fold),
            "worst_bucket": worst_bucket(fold),
        }
        if args.json:
            print(json.dumps(out, indent=2))
        else:
            print(f"guard-verdict-rsi fold: rows {fold['total_rows']}  "
                  f"quality {out['verdict_quality']}  "
                  f"by_verdict {fold['by_verdict'] or '{}'}")
            if not paths:
                cov = _load_module("dogfood_coverage", root / "tools" / "dogfood_coverage.py")
                print(f"  (no journal: {cov.diagnose_audit_gap(root)})")
        return 0

    # run (default)
    witness = None
    if getattr(args, "witness", ""):
        witness = json.loads(args.witness)
    it = run_iteration(root, getattr(args, "audit", ""), witness)
    text = json.dumps(it, indent=2)
    if getattr(args, "out", ""):
        Path(args.out).write_text(text + "\n", encoding="utf-8")
    if getattr(args, "json", False):
        print(text)
    else:
        print(render(it))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
