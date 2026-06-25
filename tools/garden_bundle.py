#!/usr/bin/env python3
"""The garden bundle — one default-on fold over the repo's read-only gardening passes.

fak already measures itself with three orchestrators, each folding a different
slice of repo health, and each run by hand or in CI but never as ONE thing:

  * ``tools/scorecard_control_pane.py`` folds the 13-scorecard family into one
    portfolio ``total_debt`` with a pinned-baseline ratchet (``--check``).
  * ``tools/fresh_status.py`` folds the four top-level domains (git, benchmarks,
    work, industry) into one cross-domain snapshot.
  * ``tools/fleet_control_pane.py loop-audit`` runs the gardening loop catalog and
    buckets each loop healthy / action / broken.

This is the fold over those folds — the single read-only "is the garden tended?"
verdict, so "run the gardening" is one command instead of three. It is the Rung-0
member of the garden bundle (full design:
``docs/notes/GARDENING-BUNDLE-DEFAULT-ON-2026-06-25.md``). It is deliberately
READ-ONLY: it runs each member, reads its control-pane payload, and folds one
``schema/ok/verdict/finding/reason/next_action`` envelope. It mutates nothing and
fixes nothing — auto-fix is a later, witness-gated rung.

  python tools/garden_bundle.py            # human snapshot of every member
  python tools/garden_bundle.py --json      # machine payload
  python tools/garden_bundle.py --check     # CI gate: exit non-zero if a member gates RED
  python tools/garden_bundle.py --deep      # also run the slower fleet loop-audit member

``--check`` is the bundle's CI contract: it exits non-zero when any *gating* member
reports a hard regression (today only the scorecard ratchet gates; fresh-status and
loop-audit are advisory panes that surface conditions without failing the bundle —
a loop in the ``action`` bucket is the loop WORKING, not the garden broken). A
member that fails to RUN (errored) always trips the gate, so a silently-broken pass
can't masquerade as a clean garden.

The bundle is skipped entirely when ``FAK_GARDEN`` is set to an off value
(``0/off/false/no/disable/disabled``) — the env-side half of the governor brake, so
a scheduled tick can be silenced on a host without editing the loop policy file.

Pure-stdlib Python, run from the repo root like the other honesty folds. The member
list is an inline ``MEMBERS`` table (the same discipline ``SCORECARDS`` uses in the
control pane): adding a pass is one row, not a code change.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fak-garden-bundle/1"

# The off-switch vocabulary, shared with the fleet's other default-on guards
# (FLEET_DOGFOOD_GUARD). Any of these as FAK_GARDEN means "skip the bundle".
_OFF = {"0", "off", "false", "no", "disable", "disabled"}

# The loop-audit subset used by the opt-in --deep member. Restricted to the FAST,
# LOCAL, deterministic loops — no network, no external tool — so the deep pass stays
# safe on any clone. The catalog's other loops are excluded on purpose: the
# `scorecard-control-pane` / `docs-scorecard` / `seo-aeo-scorecard` loops re-run
# scorecards ALREADY covered by the scorecard member, and `issue-closure-audit` (the
# `gh` API) / `memory-recall-audit` (a `dos` store) need external state.
LOOP_AUDIT_NAMES = (
    "readme-freshness",   # tools/readme_freshness_audit.py — local tree
    "gofmt-debt-audit",   # tools/gofmt_debt_audit.py — gofmt over .go, local
    "public-leak-scan",   # tools/scrub_public_copy.py --audit-tree — local scan
)

# Each member binds a label to the argv that produces its control-pane payload and
# whether a RED verdict from that member GATES the bundle (--check exits non-zero).
# The scorecard ratchet is the only hard gate today; the others are advisory panes
# whose ACTION verdict is a surfaced condition, not a broken garden.
#
# The DEFAULT bundle is the two fast, canonical folds that already speak the same
# control-pane envelope: the scorecard control pane (the 13-scorecard portfolio
# ratchet) and fresh-status (the four-domain rollup). It is deliberately small so a
# scheduled tick is fast and host-safe. The fleet loop-audit is a third member added
# only under --deep: it goes through fleet_control_pane's per-loop orchestration,
# which carries enough overhead (minutes, not the seconds the underlying tools take)
# that it doesn't belong on the default tick.
MEMBERS: list[dict[str, Any]] = [
    {
        "key": "scorecard",
        "label": "scorecard control pane",
        "argv": ["tools/scorecard_control_pane.py", "--check", "--json"],
        "gates": True,
        "kind": "envelope",
    },
    {
        "key": "fresh_status",
        "label": "fresh status",
        "argv": ["tools/fresh_status.py", "--json"],
        "gates": False,
        "kind": "envelope",
    },
]

# The opt-in deep member (added by --deep). Non-gating advisory.
DEEP_MEMBER: dict[str, Any] = {
    "key": "loop_audit",
    "label": "fleet loop-audit",
    "argv": ["tools/fleet_control_pane.py", "loop-audit",
             "--names", ",".join(LOOP_AUDIT_NAMES), "--json"],
    "gates": False,
    "kind": "loop_audit",
}


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


def garden_off() -> bool:
    """True when FAK_GARDEN names an off value (the env-side governor brake)."""
    return os.environ.get("FAK_GARDEN", "").strip().lower() in _OFF


# --- pure interpretation of a member payload (the tested surface) ----------

def interpret(member: dict[str, Any], payload: dict[str, Any] | None,
              exit_code: int, error: str = "") -> dict[str, Any]:
    """Fold one member's raw payload into a uniform member-result row.

    ``state`` is the bundle's normalization of a member into one of:
      * ``ok``       — the member ran and reports nothing to do
      * ``action``   — the member ran and surfaces a real condition (advisory)
      * ``red``      — the member's gate tripped (a hard regression)
      * ``errored``  — the member could not run / produced no usable payload
    """
    base = {
        "key": member["key"],
        "label": member["label"],
        "gates": bool(member["gates"]),
        "exit_code": exit_code,
    }
    if error or not isinstance(payload, dict):
        return {**base, "state": "errored", "ok": False,
                "verdict": "ERROR", "detail": error or "no payload"}

    if member["kind"] == "loop_audit":
        counts = payload.get("counts") or {}
        broken = int(counts.get("broken") or 0)
        action = int(counts.get("action") or 0)
        # loop-audit is a NON-GATING advisory member: it ran (it produced a payload),
        # so a broken sub-loop is a condition to surface, NOT the bundle's own
        # inability to measure. A peripheral check that can't run on this host (e.g.
        # public-leak-scan without the private companion repo) must not be able to
        # gate the whole garden red — that's the tail wagging the dog. Only a member
        # that produces NO payload at all is `errored` (handled above). The broken
        # count rides in the detail so it's still visible.
        if broken > 0 or action > 0:
            state, ok, verdict = "action", True, "ACTION"
            bits = []
            if broken:
                bits.append(f"{broken} loop(s) broken")
            if action:
                bits.append(f"{action} surfacing a condition")
            detail = "; ".join(bits) + " (advisory; does not gate)"
        else:
            state, ok, verdict = "ok", True, "OK"
            detail = f"{int(counts.get('healthy') or 0)} loop(s) healthy"
        return {**base, "state": state, "ok": ok, "verdict": verdict,
                "detail": detail, "counts": counts}

    # Standard control-pane envelope (scorecard, fresh_status).
    ok = bool(payload.get("ok"))
    verdict = str(payload.get("verdict") or "")
    detail = str(payload.get("reason") or "")
    if member["gates"] and not ok:
        state = "red"
    elif not ok:
        state = "action"
    else:
        state = "ok"
    return {**base, "state": state, "ok": ok, "verdict": verdict or ("OK" if ok else "ACTION"),
            "detail": detail}


def fold(results: list[dict[str, Any]], *, workspace: str, commit: str) -> dict[str, Any]:
    """Fold member results into one garden-bundle control-pane payload."""
    errored = [r for r in results if r["state"] == "errored"]
    red = [r for r in results if r["state"] == "red"]
    action = [r for r in results if r["state"] == "action"]

    if errored:
        ok, verdict, finding = False, "ACTION", "garden_member_unmeasured"
        reason = ("garden bundle could not measure "
                  + ", ".join(r["label"] for r in errored)
                  + " — a gardening pass failed to run, so the garden is not proven tended")
        next_action = ("repair the failing member(s): "
                       + "; ".join(f"{r['label']}: {r['detail']}" for r in errored))
    elif red:
        ok, verdict, finding = False, "ACTION", "garden_gate_red"
        reason = ("garden gate RED — "
                  + ", ".join(f"{r['label']} ({r['detail']})" for r in red))
        next_action = ("retire the regression worst-first with the owning scorecard's "
                       "skill, then re-run python tools/garden_bundle.py --check")
    elif action:
        ok, verdict, finding = True, "OK", "garden_advisory"
        reason = ("garden tended; advisory conditions surfaced by "
                  + ", ".join(r["label"] for r in action)
                  + " (these don't gate — a pass surfacing a condition is the pass working)")
        next_action = ("optional: address the advisory condition(s) — "
                       + "; ".join(f"{r['label']}: {r['detail']}" for r in action))
    else:
        ok, verdict, finding = True, "OK", "garden_clear"
        reason = f"garden tended; all {len(results)} members clear"
        next_action = "hold the line; the scheduled garden tick keeps it tended"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "commit": commit,
        "members": results,
        "member_count": len(results),
        "gating": [r["key"] for r in results if r["gates"]],
    }


def skipped_payload(*, workspace: str, commit: str) -> dict[str, Any]:
    """The well-formed payload for an FAK_GARDEN=off skip (still ok=True)."""
    return {
        "schema": SCHEMA,
        "ok": True,
        "verdict": "OK",
        "finding": "garden_skipped",
        "reason": "FAK_GARDEN is set off; garden bundle skipped (env-side governor brake)",
        "next_action": "unset FAK_GARDEN (or set it on) to run the bundle",
        "workspace": workspace,
        "commit": commit,
        "members": [],
        "member_count": 0,
        "gating": [],
        "skipped": True,
    }


# --- live runner -----------------------------------------------------------

def run_member(root: Path, member: dict[str, Any], *, python: str,
               timeout: int) -> tuple[dict[str, Any] | None, int, str]:
    argv = list(member["argv"])
    script = root / argv[0]
    if not script.exists():
        return None, -1, f"missing member script: {argv[0]}"
    try:
        proc = subprocess.run(
            [python, str(script), *argv[1:]],
            cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return None, -1, f"timed out after {timeout}s"
    except (OSError, subprocess.SubprocessError) as exc:
        return None, -1, str(exc)
    try:
        return json.loads(proc.stdout), proc.returncode, ""
    except ValueError:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-1:] or [""]
        return None, proc.returncode, f"non-JSON output (exit {proc.returncode}): {tail[0][:160]}"


def collect(root: Path, *, python: str = "", timeout: int = 240,
            deep: bool = False) -> list[dict[str, Any]]:
    python = python or sys.executable
    members = list(MEMBERS) + ([DEEP_MEMBER] if deep else [])
    results: list[dict[str, Any]] = []
    for member in members:
        payload, code, error = run_member(root, member, python=python, timeout=timeout)
        results.append(interpret(member, payload, code, error))
    return results


def render(payload: dict[str, Any]) -> str:
    icon = {"ok": "✓", "action": "•", "red": "✗", "errored": "✗"}
    lines = [
        f"garden bundle — {payload['verdict']} ({payload['finding']})  @{payload['commit']}",
        "",
    ]
    if payload.get("skipped"):
        lines.append("  (skipped: FAK_GARDEN is off)")
    for m in payload["members"]:
        gate = " [gates]" if m["gates"] else ""
        lines.append(f"  {icon.get(m['state'], '?')} {m['label']:<24} {m['state']:<8}{gate}  {m['detail']}")
    lines.extend(["", f"  → {payload['next_action']}"])
    return "\n".join(lines)


def check_gate(payload: dict[str, Any]) -> tuple[int, str]:
    """The CI gate decision over a folded payload (pure: exit code + message).

      0  garden tended (clear or advisory-only)
      1  a gating member regressed, or a member failed to run
    """
    finding = payload.get("finding")
    if finding in ("garden_clear", "garden_advisory", "garden_skipped"):
        return 0, f"GARDEN OK: {payload['reason']}"
    return 1, f"GARDEN RED: {payload['reason']}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="The garden bundle — one read-only fold over the gardening passes.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--check", action="store_true",
                    help="CI gate: exit non-zero if a gating member regressed or failed to run")
    ap.add_argument("--deep", action="store_true",
                    help="also run the fleet loop-audit member (slower; non-gating advisory)")
    ap.add_argument("--timeout", type=int, default=240, help="per-member timeout seconds")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    root = Path(args.workspace).resolve() if args.workspace else repo_root()
    commit = head_commit(root)

    if garden_off():
        payload = skipped_payload(workspace=str(root), commit=commit)
    else:
        results = collect(root, timeout=args.timeout, deep=args.deep)
        payload = fold(results, workspace=str(root), commit=commit)

    if args.check:
        code, message = check_gate(payload)
        if args.json:
            gated = {**payload, "ok": code == 0,
                     "verdict": "OK" if code == 0 else "ACTION",
                     "gate_exit": code, "gate_message": message}
            print(json.dumps(gated, indent=2))
        else:
            print(message)
        return code

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
