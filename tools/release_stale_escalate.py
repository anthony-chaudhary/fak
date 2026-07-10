#!/usr/bin/env python3
r"""release-stale-escalate — decide whether a VERY_STALE `@latest` deserves more than
the informational `::warning::` the cadence tick already prints, and shape the escalation
so it rides the SHARED witnessable-signal contract instead of a second notification feed.

THE GAP (#4025). `release-cadence.yml` prints publish staleness as a non-gating
`::warning::` (the auto-cut steps below it are what actually close the lag). That warning
is the right default — WHILE auto-cut is armed. But the kill switch `FAK_AUTO_RELEASE=0`
turns auto-cut OFF, and then the one loop that would cut the tag is gone: a VERY_STALE
`@latest` can rot behind HEAD indefinitely while the only signal is a warning nobody is
paged on. The auto-cut attempt-path instrumentation cannot observe this — there IS no
attempt to instrument when the switch is off. This tool is the missing escalation for
exactly that blind spot.

THE DECISION (pure, unit-tested — `decide`):

  * verdict is not `very_stale`            -> informational  (the warning stands)
  * `very_stale` but auto-cut is LIVE      -> informational  (the cadence closes the lag)
  * `very_stale` AND auto-cut DISABLED
        - repo opted into a hard fail       -> fail-tick      (surface a red status)
        - default                           -> file-issue     (open/bump ONE deduped issue)

Only `very_stale` under a killed cadence escalates; `fresh`/`stale` and every armed tick
stay exactly as loud as today. The fail-vs-file choice is the operator's: default to a
single deduped tracking issue; set the repo variable `FAK_STALE_ESCALATION_FAIL=1` to
fail the tick outright instead.

THE HAND-OFF — we do NOT duplicate the issue-filing machinery. On `file-issue` this tool
shapes a gate-signal-compatible envelope (`--emit-envelope`) enriched with the kill-switch
context and a `.github/workflows/release-cadence.yml` owning path, then the workflow pipes
it into `tools/gate_signal.py --live` — the same deduped, lane-routed, marker-anchored
feeder score-signal/gate-signal already use. One OPEN issue per failure signature
(`release-cadence:publish_very_stale_under_killed_cadence`), re-fileable after a close,
routed to the `ci` lane. That is the "share the witnessable-signal contract, not the
notification feed" requirement made concrete.

    python tools/release_stale_escalate.py --from staleness.json \
        --auto-cut-disabled true --emit-envelope escalate-envelope.json --json
    python tools/gate_signal.py --from escalate-envelope.json --live   # only on file-issue

The tool itself NEVER files an issue and NEVER mutates: it decides and (on file-issue)
writes an envelope. Filing is the workflow's explicit `gate_signal --live` arm, and the
fail-tick gate is the workflow's explicit `exit 1` — the caller enforces, so the decision
stays hermetic and testable. Exit 0 for every decided outcome; exit 2 only on an
unreadable envelope (an infra error).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fak-release-stale-escalate/1"

# The gate-signal envelope identity. `source` is what gate_signal._envelope_source folds
# to the finding-key prefix; `finding` is the stable per-signature suffix; `where` is the
# owning path that makes the filed issue path-confirm the `ci` lane (the escalation's fix
# lives in the cadence workflow). These three are the load-bearing dedup + routing anchors.
ESCALATION_SOURCE = "release-cadence"
ESCALATION_FINDING = "publish_very_stale_under_killed_cadence"
ESCALATION_WHERE = ".github/workflows/release-cadence.yml"

# The truthy spellings a workflow might hand `--auto-cut-disabled` (a bare shell bool).
_TRUE = {"1", "true", "yes", "on"}

# Actions, in escalation order. Kept as module constants so the workflow and the tests
# reference the same closed vocabulary rather than bare strings.
INFORMATIONAL = "informational"
FILE_ISSUE = "file-issue"
FAIL_TICK = "fail-tick"


# ============================================================================
# Pure core (no I/O) — unit-tested directly.
# ============================================================================
def decide(verdict: str, auto_cut_disabled: bool, fail_opt_in: bool) -> tuple[str, str]:
    """The escalation rule. Returns (action, reason). `action` is one of INFORMATIONAL /
    FILE_ISSUE / FAIL_TICK. Only a `very_stale` verdict under a disabled auto-cut escalates
    — every other combination keeps today's informational warning."""
    v = (verdict or "").strip().lower()
    if v != "very_stale":
        return (INFORMATIONAL,
                f"verdict {v or '(none)'} is not very_stale — the informational "
                f"publish-staleness warning stands; no escalation.")
    if not auto_cut_disabled:
        return (INFORMATIONAL,
                "very_stale, but auto-cut is armed — the scheduled cadence closes the lag "
                "on the next substantive green window; no escalation.")
    if fail_opt_in:
        return (FAIL_TICK,
                "very_stale AND auto-cut disabled (FAK_AUTO_RELEASE=0); "
                "FAK_STALE_ESCALATION_FAIL=1 — failing the tick so the red status is loud.")
    return (FILE_ISSUE,
            "very_stale AND auto-cut disabled (FAK_AUTO_RELEASE=0) — the one loop that "
            "closes the lag is off; open/bump a single deduped tracking issue via "
            "gate-signal.")


def _fmt_num(n: Any) -> str:
    """Render a JSON number without a trailing `.0` so days/age read cleanly in prose."""
    if isinstance(n, bool) or n is None:
        return "?"
    if isinstance(n, float) and n.is_integer():
        return str(int(n))
    return str(n)


def escalation_envelope(payload: dict[str, Any]) -> dict[str, Any]:
    """Shape the gate-signal-compatible envelope for the FILE_ISSUE path from a
    release-staleness payload. The envelope is a single not-ok BLOCKING member whose
    `finding`/`source` anchor a stable dedup key and whose `where` routes the filed issue
    to the `ci` lane. The reason NAMES the kill switch as the crux — a filed issue must
    explain that the cadence will not self-heal until the switch is lifted."""
    commits = _fmt_num(payload.get("commits_behind"))
    days = _fmt_num(payload.get("days_behind"))
    age = payload.get("age_days")
    latest = payload.get("latest_tag") or "(none)"
    upstream_next = str(payload.get("next_action") or "").strip()

    age_clause = ""
    if isinstance(age, (int, float)) and not isinstance(age, bool) and age:
        age_clause = f", published {_fmt_num(age)}d ago"

    reason = (
        f"`@latest` = {latest} is VERY_STALE — {commits} commit(s) / {days}d behind HEAD"
        f"{age_clause}. Auto-cut is disabled (FAK_AUTO_RELEASE=0), so the scheduled "
        f"release cadence will NOT cut the tag: `@latest` keeps rotting behind HEAD until "
        f"a release is cut by hand or the kill switch is lifted."
    )

    next_action = (
        "Re-enable auto-cut (unset FAK_AUTO_RELEASE, or set it to any value other than "
        "'0'), or cut a release manually with `/release`; then `@latest` tracks HEAD "
        "again and this issue can close."
    )
    if upstream_next:
        next_action = f"{upstream_next} {next_action}"

    return {
        "schema": SCHEMA,
        "source": ESCALATION_SOURCE,
        "ok": False,
        "verdict": "BLOCKING",
        "finding": ESCALATION_FINDING,
        "reason": reason,
        "next_action": next_action,
        "where": ESCALATION_WHERE,
    }


# ============================================================================
# I/O boundary — read the staleness envelope, write the escalation envelope.
# ============================================================================
def load_payload(from_arg: str) -> dict[str, Any]:
    """Read the release-staleness envelope from --from (a file or '-' for stdin). Raises
    RuntimeError on any failure so the caller can exit 2 cleanly."""
    if not from_arg:
        raise RuntimeError("--from is required (a release-staleness --json envelope file "
                           "or '-' for stdin)")
    raw = sys.stdin.read() if from_arg == "-" else Path(from_arg).read_text(
        encoding="utf-8")
    try:
        payload = json.loads(raw)
    except ValueError as e:
        raise RuntimeError(f"envelope is not JSON: {e}") from e
    if not isinstance(payload, dict):
        raise RuntimeError("envelope is not a JSON object")
    return payload


def _parse_bool(text: str) -> bool:
    return str(text).strip().lower() in _TRUE


# ============================================================================
# Driver.
# ============================================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Decide whether a VERY_STALE @latest under a killed auto-cut should "
                    "escalate to a deduped tracking issue or a failed tick.")
    ap.add_argument("--from", dest="from_arg", default="",
                    help="the release-staleness --json envelope ('-' for stdin).")
    ap.add_argument("--auto-cut-disabled", default="false",
                    help="whether the auto-cut kill switch is on (FAK_AUTO_RELEASE=0): "
                         "true/false. Default false.")
    ap.add_argument("--fail-on-escalate", action="store_true",
                    help="opt in (FAK_STALE_ESCALATION_FAIL=1): fail the tick instead of "
                         "filing a tracking issue.")
    ap.add_argument("--emit-envelope", default="",
                    help="on file-issue, write the gate-signal envelope to this path for "
                         "`tools/gate_signal.py --from <path> --live`.")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    args = ap.parse_args(argv)

    try:
        payload = load_payload(args.from_arg)
    except (OSError, RuntimeError) as e:
        print(f"refuse: could not load the release-staleness envelope: {e}",
              file=sys.stderr)
        return 2

    verdict = str(payload.get("verdict") or "")
    auto_cut_disabled = _parse_bool(args.auto_cut_disabled)
    action, reason = decide(verdict, auto_cut_disabled, args.fail_on_escalate)

    envelope_path = ""
    if action == FILE_ISSUE and args.emit_envelope:
        env = escalation_envelope(payload)
        Path(args.emit_envelope).write_text(
            json.dumps(env, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        envelope_path = args.emit_envelope

    result = {
        "schema": SCHEMA,
        "action": action,
        "reason": reason,
        "verdict": verdict,
        "auto_cut_disabled": auto_cut_disabled,
        "fail_on_escalate": bool(args.fail_on_escalate),
        "envelope": envelope_path,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"release-stale-escalate: {action}")
        print(f"  {reason}")
        if envelope_path:
            print(f"  gate-signal envelope written to {envelope_path} — file it with:")
            print(f"    python tools/gate_signal.py --from {envelope_path} --live")
        elif action == FAIL_TICK:
            print("  the caller (workflow) fails the tick on this action.")

    # Exit 0 for every DECIDED outcome: this tool decides, the caller enforces (the
    # workflow arms `gate_signal --live` on file-issue and `exit 1` on fail-tick). Only an
    # unreadable envelope (handled above) is a nonzero infra error.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
