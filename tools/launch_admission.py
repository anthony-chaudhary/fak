#!/usr/bin/env python3
r"""launch_admission.py -- the single launch-admission gate for the fleet (#617).

WHY
---
On 2026-06-24 an out-of-tree resume launcher (``resafe_resume``) fired **15
``claude --resume`` launches onto ONE throttled account (``q-netra``) inside
~60s**. q-netra 429'd; a retry wave then scattered onto other accounts and also
hit limits -- a self-inflicted rate-limit storm. The capped, ledger-gated path
(``tools/fleet_resume_watchdog.py``, ``FAK_MAX_PER_TICK=2``) that would have
stopped it was **DISABLED** as a scheduled task, so an ad-hoc launcher bypassed
every safety rail. The safety logic existed, but it lived in tooling a launcher
could route around.

WHAT
----
This is the SINGLE admission point every launcher -- committed, ad-hoc, or
future -- should pass through before it fires a launch (resume **or** fresh). It
does NOT launch; it returns a **verdict** (``ADMIT`` or ``DEFER``) so the caller
self-gates. A DEFER carries a STRUCTURED reason from a closed set --
``LAUNCH_RATE_EXCEEDED`` / ``GLOBAL_LAUNCH_CAP`` / ``ACCOUNT_THROTTLED`` -- plus a
machine-checkable ``retry_after``, not free-text prose a launcher can shrug off.
Register each token in ``dos.toml [reasons.*]`` (with THIS file as its named
floor -- the same producer/verifier split as ``OFF_TRUNK`` ->
``tools/githooks/reference-transaction``) so ``dos_check_reason <token>`` returns
known=true and the DEFER is a verifiable member of the kernel's refusal
vocabulary. The exact, validated block ships with #617; the gate runs standalone
without it.

Three gates, refused conservatively (most-specific first):

  1. THROTTLE-GATE    -- the account's current verdict is throttled/blocked with a
     future reset -> DEFER ``ACCOUNT_THROTTLED``, ``retry_after`` = the reset.
  2. GLOBAL CAP       -- launches across ALL accounts in the window >= the global
     cap -> DEFER ``GLOBAL_LAUNCH_CAP``, ``retry_after`` = oldest-global + window.
  3. PER-ACCOUNT RATE -- launches onto THIS account in the window >= the per-account
     ceiling -> DEFER ``LAUNCH_RATE_EXCEEDED``, ``retry_after`` = oldest-acct + window.
  else ADMIT.

The per-account / global counters are read from the SAME durable launch ledger the
resume watchdog already appends to (``tools/_registry/resume_ledger.jsonl``), so the
gate sees EVERY recorded launch regardless of which launcher made it. The launch
TARGET is the ledger record's ``resume_account`` (falling back to ``account``).

This gate ENFORCES the launch rate; it does NOT re-derive account health -- the
caller supplies the throttle verdict (from ``fleet_accounts.py`` / the
account-health-verdict layer) via ``--throttled`` / ``--throttle-reset``. Keeping
the two concerns separate is deliberate: this stays a small, hermetic decision.

CEILINGS (env-overridable, fleet convention):

  FAK_LAUNCH_MAX_PER_ACCOUNT  (default 3)   N launches / account / window
  FAK_LAUNCH_WINDOW_MIN       (default 5)   the rolling window, in minutes
  FAK_LAUNCH_GLOBAL_CAP       (default 10)  launches across all accounts / window

CLI::

  python tools/launch_admission.py admit --account .claude-q-netra
      # JSON verdict on stdout; exit 0 = ADMIT, 3 = DEFER, 2 = usage.
  python tools/launch_admission.py admit --account X --throttle-reset "3pm"
  python tools/launch_admission.py admit --account X --record   # append on ADMIT
  python tools/launch_admission.py reasons   # the structured tokens it can emit
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LEDGER = os.path.join(
    os.environ.get("FLEET_REG_DIR", os.path.join(HERE, "_registry")),
    "resume_ledger.jsonl",
)

VERDICT_ADMIT = "ADMIT"
VERDICT_DEFER = "DEFER"

# The structured DEFER reasons this gate emits. Register each in dos.toml
# [reasons.*] (with THIS file as its named floor) so it becomes a real member of
# the closed refusal vocabulary -- `dos_check_reason <token>` returns known=true,
# not the UNCLASSIFIED prose-drift the kernel exists to kill (companion step, #617).
REASON_THROTTLED = "ACCOUNT_THROTTLED"
REASON_GLOBAL_CAP = "GLOBAL_LAUNCH_CAP"
REASON_RATE = "LAUNCH_RATE_EXCEEDED"
EMITTABLE_REASONS = (REASON_THROTTLED, REASON_GLOBAL_CAP, REASON_RATE)

# Ledger phases that are NOT a fired launch (a deferral/consideration is not
# launch pressure, so counting it would let our own DEFERs cascade into more).
_NON_LAUNCH_PHASES = {"deferred", "considered", "skipped"}

ISO = "%Y-%m-%dT%H:%M:%SZ"


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def default_ceilings() -> dict:
    """The launch ceilings, env-overridable (fleet convention)."""
    return {
        "max_per_account": _env_int("FAK_LAUNCH_MAX_PER_ACCOUNT", 3),
        "window_min": _env_int("FAK_LAUNCH_WINDOW_MIN", 5),
        "global_cap": _env_int("FAK_LAUNCH_GLOBAL_CAP", 10),
    }


def parse_ts(s) -> datetime | None:
    """Parse an ISO8601-Z ledger timestamp to an aware UTC datetime; None on failure."""
    if not s:
        return None
    try:
        return datetime.strptime(str(s), ISO).replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        pass
    try:
        d = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        return d if d.tzinfo else d.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


def _fmt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime(ISO)


def _in_window(timestamps, now: datetime, window: timedelta) -> list[datetime]:
    """The subset of `timestamps` strictly inside (now - window, now], sorted."""
    cutoff = now - window
    return sorted(t for t in timestamps if t is not None and cutoff < t <= now)


def admit(
    account: str,
    *,
    now: datetime,
    account_launches,
    global_launches,
    throttled: bool = False,
    throttle_reset=None,
    max_per_account: int = 3,
    window_min: int = 5,
    global_cap: int = 10,
) -> dict:
    """Pure admission decision -- the testable core of the gate.

    account_launches / global_launches: iterables of aware datetimes for prior
    launches onto THIS account / across ALL accounts. ``now`` is an aware datetime
    (the caller's clock -- passed in so the decision is deterministic). ``throttled``
    + ``throttle_reset`` are the caller-supplied account-health verdict.

    Returns a verdict dict: {verdict, account, reason, retry_after, detail, ...}.
    ``retry_after`` is an ISO8601-Z string (or the raw reset marker for a throttle)
    the launcher can wait on; None on ADMIT.
    """
    window = timedelta(minutes=window_min)

    def verdict(v, reason=None, retry_after=None, detail=""):
        return {
            "verdict": v,
            "account": account,
            "reason": reason,
            "retry_after": retry_after,
            "detail": detail,
            "now": _fmt(now),
            "window_min": window_min,
            "max_per_account": max_per_account,
            "global_cap": global_cap,
        }

    # 1. THROTTLE-GATE: never launch onto a throttled/blocked account. The reset is
    #    surfaced verbatim as retry_after -- "DEFER bound to the account's reset".
    if throttled:
        return verdict(
            VERDICT_DEFER,
            REASON_THROTTLED,
            retry_after=throttle_reset,
            detail=(
                f"account {account} verdict is throttled/blocked; "
                f"retry after reset {throttle_reset or '<unknown>'}"
            ),
        )

    # 2. GLOBAL CAP: bound total launch pressure across the whole fleet.
    g = _in_window(global_launches, now, window)
    if len(g) >= global_cap:
        return verdict(
            VERDICT_DEFER,
            REASON_GLOBAL_CAP,
            retry_after=_fmt(g[0] + window),
            detail=(
                f"{len(g)} launches across all accounts in the last {window_min}m "
                f">= global cap {global_cap}"
            ),
        )

    # 3. PER-ACCOUNT RATE: bound the burst onto any one account.
    a = _in_window(account_launches, now, window)
    if len(a) >= max_per_account:
        return verdict(
            VERDICT_DEFER,
            REASON_RATE,
            retry_after=_fmt(a[0] + window),
            detail=(
                f"{len(a)} launches onto {account} in the last {window_min}m "
                f">= per-account ceiling {max_per_account}"
            ),
        )

    return verdict(
        VERDICT_ADMIT,
        detail=(
            f"{len(a)}/{max_per_account} acct + {len(g)}/{global_cap} global "
            f"launches in the last {window_min}m -- admitted"
        ),
    )


def simulate_burst(
    account: str,
    n: int,
    *,
    start: datetime,
    interval_sec: int = 4,
    throttled: bool = False,
    throttle_reset=None,
    **ceilings,
) -> list[dict]:
    """Replay N back-to-back launch attempts onto `account`, feeding each ADMIT
    back as a prior launch (the gate is STATEFUL across a burst -- the i-th attempt
    must see the i-1 already admitted). Single-account, so global == account here.

    Models the 2026-06-24 15->1 q-netra storm (#617). Returns the N verdicts.
    """
    admitted: list[datetime] = []
    verdicts: list[dict] = []
    for i in range(n):
        now = start + timedelta(seconds=i * interval_sec)
        v = admit(
            account,
            now=now,
            account_launches=list(admitted),
            global_launches=list(admitted),
            throttled=throttled,
            throttle_reset=throttle_reset,
            **ceilings,
        )
        if v["verdict"] == VERDICT_ADMIT:
            admitted.append(now)
        verdicts.append(v)
    return verdicts


# ---------------------------------------------------------------------------
# Ledger I/O (the live CLI path; tests drive the pure functions above).
# ---------------------------------------------------------------------------

def _launch_target(rec: dict) -> str | None:
    """The account a ledger record launched ONTO (resume target, else origin)."""
    return rec.get("resume_account") or rec.get("account")


def _is_launch(rec: dict) -> bool:
    return str(rec.get("phase") or "").strip().lower() not in _NON_LAUNCH_PHASES


def load_launches(ledger_path: str, account: str) -> tuple[list[datetime], list[datetime]]:
    """Read the durable ledger -> (launches onto `account`, launches across all).

    Tolerant of a missing file (fresh node) and of malformed lines (skip, never
    crash the gate). Account match is on the bare tag so `.claude-q-netra`,
    `q-netra`, and the dir basename all resolve to the same account.
    """
    want = _account_tag(account)
    acct: list[datetime] = []
    glob: list[datetime] = []
    try:
        fh = open(ledger_path, encoding="utf-8", errors="replace")
    except OSError:
        return acct, glob
    with fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except (ValueError, TypeError):
                continue
            if not isinstance(rec, dict) or not _is_launch(rec):
                continue
            target = _launch_target(rec)
            ts = parse_ts(rec.get("ts"))
            if target is None or ts is None:
                continue
            glob.append(ts)
            if _account_tag(target) == want:
                acct.append(ts)
    return acct, glob


def _account_tag(account: str) -> str:
    """Normalize an account key to its bare tag for robust matching.

    `.claude-q-netra` -> `q-netra`; `.claude` -> `default`; `opencode-glm` -> `glm`;
    a bare `q-netra` stays `q-netra`. Lowercased.
    """
    s = str(account or "").strip().rstrip("/\\")
    s = os.path.basename(s).lower()
    if s in ("", ".claude", "claude"):
        return "default"
    for pre in (".claude-", "claude-", "opencode-", ".opencode-"):
        if s.startswith(pre):
            return s[len(pre) :] or "default"
    if s in ("opencode", ".opencode"):
        return "default"
    return s


def record_launch(ledger_path: str, account: str, now: datetime, **extra) -> dict:
    """Append one admitted-launch record to the ledger (the schema the watchdog uses)."""
    rec = {
        "ts": _fmt(now),
        "account": account,
        "resume_account": account,
        "phase": "launched",
        "via": "launch_admission",
    }
    rec.update(extra)
    os.makedirs(os.path.dirname(ledger_path), exist_ok=True)
    with open(ledger_path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")
    return rec


def _cmd_admit(args) -> int:
    ceilings = default_ceilings()
    if args.max_per_account is not None:
        ceilings["max_per_account"] = args.max_per_account
    if args.window_min is not None:
        ceilings["window_min"] = args.window_min
    if args.global_cap is not None:
        ceilings["global_cap"] = args.global_cap

    now = parse_ts(args.now) if args.now else datetime.now(timezone.utc)
    if now is None:
        print(json.dumps({"verdict": "ERROR", "detail": f"unparseable --now {args.now!r}"}))
        return 2

    acct, glob = load_launches(args.ledger, args.account)
    throttled = bool(args.throttled or args.throttle_reset)
    v = admit(
        args.account,
        now=now,
        account_launches=acct,
        global_launches=glob,
        throttled=throttled,
        throttle_reset=args.throttle_reset,
        **ceilings,
    )
    if v["verdict"] == VERDICT_ADMIT and args.record:
        record_launch(args.ledger, args.account, now, cause=args.cause or "admitted")
    print(json.dumps(v, indent=2 if args.pretty else None))
    return 0 if v["verdict"] == VERDICT_ADMIT else 3


def _cmd_reasons(args) -> int:
    print(json.dumps({"emittable_reasons": list(EMITTABLE_REASONS),
                      "note": "register each in dos.toml [reasons.*], then verify with dos_check_reason (companion step, #617)"},
                     indent=2))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Single launch-admission gate: ADMIT or a structured DEFER (#617).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("admit", help="decide whether a launch onto --account is admitted")
    a.add_argument("--account", required=True, help="target account (e.g. .claude-q-netra)")
    a.add_argument("--throttled", action="store_true", help="account verdict is throttled/blocked")
    a.add_argument("--throttle-reset", default=None, help="the reset marker (implies --throttled)")
    a.add_argument("--ledger", default=DEFAULT_LEDGER, help="launch ledger path")
    a.add_argument("--now", default=None, help="ISO8601-Z clock override (default: now)")
    a.add_argument("--max-per-account", type=int, default=None, dest="max_per_account")
    a.add_argument("--window-min", type=int, default=None, dest="window_min")
    a.add_argument("--global-cap", type=int, default=None, dest="global_cap")
    a.add_argument("--record", action="store_true", help="append the launch to the ledger on ADMIT")
    a.add_argument("--cause", default=None, help="cause stamped on a --record append")
    a.add_argument("--pretty", action="store_true", help="indent the JSON verdict")
    a.set_defaults(func=_cmd_admit)

    r = sub.add_parser("reasons", help="list the structured DEFER tokens this gate emits")
    r.set_defaults(func=_cmd_reasons)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
