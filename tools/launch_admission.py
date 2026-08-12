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
  python tools/launch_admission.py plan --account A --account B  # whole-wave dry run
      # JSON {granted, shortfall, retry_after, lanes}; exit 0 = all launchable, 3 = not.
  python tools/launch_admission.py reasons   # the structured tokens it can emit

``plan`` exists because a dry run that prices SEATS but never asks this gate advertises
a grant that is not launchable: on 2026-08-11 a detached wave plan reported
``granted=4, shortfall=0`` and the matching ``-Launch`` dispatched ``0/4``, every lane
DEFERRED ``LAUNCH_RATE_EXCEEDED`` (and, one report later, ``GLOBAL_LAUNCH_CAP`` for a
one-worker plan). ``plan`` walks the SAME per-spawn decision over the whole proposed
wave, statefully and WITHOUT recording, so planning and launching cannot disagree (#6492).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import fleet_regdir  # noqa: E402  -- the host's one registry dir (never a second one)

# The resume ledger lives in the RUNTIME registry: $FLEET_REG_DIR when the fleet names it,
# else the host ladder. Reading a second, forked registry here would grade admission
# against a ledger no launcher writes -- every launch looks like a first launch.
DEFAULT_LEDGER = os.path.join(fleet_regdir.reg_dir(), "resume_ledger.jsonl")

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

# The plan-side verdict envelope (#6492): a dry run that reports a grant it cannot
# launch is a lie the operator acts on, so planning answers with the SAME gate the
# launcher self-gates on, walked over the whole proposed wave.
PLAN_SCHEMA = "fak.launch-admission-plan.v1"

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


def plan_lanes(
    accounts,
    *,
    now: datetime,
    account_launches=None,
    global_launches=None,
    throttled=None,
    throttle_resets=None,
    max_per_account: int = 3,
    window_min: int = 5,
    global_cap: int = 10,
) -> dict:
    """Run the per-spawn gate over a WHOLE proposed wave, statefully (#6492).

    A launcher's dry run used to plan capacity with a gate the launch did not consult:
    the plan said ``granted=4, shortfall=0`` and the matching ``-Launch`` dispatched
    ``0/4``, every lane DEFERRED ``LAUNCH_RATE_EXCEEDED`` (and, in a second report, the
    global cap). The plan was not wrong about SEATS -- it simply never asked the
    admission gate, so it advertised a grant that was not launchable.

    This is that question, asked once for the ordered list of lanes a plan proposes.
    It is the same :func:`admit` decision the launcher self-gates on per spawn, walked
    in launch order and STATEFUL: each ADMIT is fed back as a prior launch (onto that
    account and globally), exactly as ``--record`` does live, so lane i sees lanes
    0..i-1. It records NOTHING -- planning must never consume launch budget.

    ``account_launches`` maps account tag -> prior launch datetimes onto that account;
    ``global_launches`` is the fleet-wide list; ``throttled`` / ``throttle_resets`` are
    the caller-supplied per-account health verdicts (same split as :func:`admit`).

    Returns ``{requested, granted, shortfall, retry_after, reasons, lanes}`` where
    ``retry_after`` is the EARLIEST time any deferred lane could be retried (the honest
    "come back at" for the wave) and ``lanes`` is the per-lane verdict in order.
    """
    accounts = [str(a) for a in accounts]
    acct_hist: dict[str, list[datetime]] = {}
    for tag, times in (account_launches or {}).items():
        acct_hist[_account_tag(tag)] = list(times or [])
    glob_hist: list[datetime] = list(global_launches or [])
    throttled = {_account_tag(k): bool(v) for k, v in (throttled or {}).items()}
    resets = {_account_tag(k): v for k, v in (throttle_resets or {}).items()}

    lanes: list[dict] = []
    reasons: dict[str, int] = {}
    granted = 0
    for account in accounts:
        tag = _account_tag(account)
        v = admit(
            account,
            now=now,
            account_launches=list(acct_hist.get(tag) or []),
            global_launches=list(glob_hist),
            throttled=throttled.get(tag, False) or bool(resets.get(tag)),
            throttle_reset=resets.get(tag),
            max_per_account=max_per_account,
            window_min=window_min,
            global_cap=global_cap,
        )
        if v["verdict"] == VERDICT_ADMIT:
            granted += 1
            # Feed the grant forward: the i-th planned lane must see the i-1 it would
            # already have fired, or a plan of N onto one account "grants" all N.
            acct_hist.setdefault(tag, []).append(now)
            glob_hist.append(now)
        else:
            reasons[v["reason"]] = reasons.get(v["reason"], 0) + 1
        lanes.append({"account": account, "verdict": v["verdict"],
                      "reason": v["reason"], "retry_after": v["retry_after"],
                      "detail": v["detail"]})
    retry_after = _earliest_retry([lane.get("retry_after") for lane in lanes])
    return {
        "schema": PLAN_SCHEMA,
        "requested": len(accounts),
        "granted": granted,
        "shortfall": len(accounts) - granted,
        "retry_after": retry_after,
        "reasons": reasons,
        "lanes": lanes,
        "now": _fmt(now),
        "window_min": window_min,
        "max_per_account": max_per_account,
        "global_cap": global_cap,
    }


def _earliest_retry(values) -> str | None:
    """The soonest parseable retry marker among ``values`` (ISO8601-Z), else the first
    unparseable one verbatim (a throttle reset is a human marker like ``Jun 25, 1pm``
    -- surfacing it beats dropping it), else None."""
    best: datetime | None = None
    raw: str | None = None
    for v in values:
        if not v:
            continue
        ts = parse_ts(v)
        if ts is None:
            raw = raw or str(v)
            continue
        if best is None or ts < best:
            best = ts
    return _fmt(best) if best is not None else raw


def plan_wave(
    accounts,
    *,
    ledger_path: str = DEFAULT_LEDGER,
    now: datetime | None = None,
    throttled=None,
    throttle_resets=None,
    max_per_account: int | None = None,
    window_min: int | None = None,
    global_cap: int | None = None,
) -> dict:
    """The one-call PLANNING seam: read the durable ledger, then plan the wave.

    The dry-run twin of :func:`admit_or_defer` -- same ledger, same ceilings, same
    decision -- so a planner and the launcher it precedes cannot disagree about
    launchability. Decide-only: it never appends to the ledger."""
    if now is None:
        now = datetime.now(timezone.utc)
    ceilings = default_ceilings()
    if max_per_account is not None:
        ceilings["max_per_account"] = max_per_account
    if window_min is not None:
        ceilings["window_min"] = window_min
    if global_cap is not None:
        ceilings["global_cap"] = global_cap
    acct_hist, glob = load_launch_history(ledger_path)
    return plan_lanes([str(a) for a in accounts], now=now,
                      account_launches=acct_hist, global_launches=glob,
                      throttled=throttled, throttle_resets=throttle_resets,
                      **ceilings)


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


def load_launch_history(ledger_path: str) -> tuple[dict[str, list[datetime]], list[datetime]]:
    """One pass over the durable ledger -> (launches bucketed by account TAG, all).

    The multi-account twin of :func:`load_launches`: a wave plan prices N lanes across
    several accounts at once, and re-reading the ledger once per lane would both cost N
    passes and risk the lanes disagreeing if the file changed underneath them. Same
    tolerance contract: a missing file is a fresh node, a malformed line is skipped."""
    by_tag: dict[str, list[datetime]] = {}
    glob: list[datetime] = []
    try:
        fh = open(ledger_path, encoding="utf-8", errors="replace")
    except OSError:
        return by_tag, glob
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
            by_tag.setdefault(_account_tag(target), []).append(ts)
    return by_tag, glob


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


def admit_or_defer(
    account: str,
    *,
    ledger_path: str = DEFAULT_LEDGER,
    now: datetime | None = None,
    throttled: bool = False,
    throttle_reset=None,
    record: bool = False,
    max_per_account: int | None = None,
    window_min: int | None = None,
    global_cap: int | None = None,
    **extra,
) -> dict:
    """The one-call launcher seam: read the durable ledger, run the gate, decide.

    A launcher that fires N spawns in a fan-out calls THIS once per spawn instead of
    re-deriving the count/decision itself. It loads the prior launches onto `account`
    (and across the fleet) from the same ledger the watchdog appends to, runs the pure
    ``admit`` decision, and -- on ADMIT + ``record`` -- appends the launch so the i-th
    spawn in a burst sees the i-1 already recorded (the gate is stateful across a wave).
    Ceiling overrides default to the env-tunable fleet ceilings when left None.

    Returns the same verdict dict ``admit`` returns; the caller self-gates on
    ``verdict`` (DEFER carries a structured ``reason`` + ``retry_after``).
    """
    if now is None:
        now = datetime.now(timezone.utc)
    ceilings = default_ceilings()
    if max_per_account is not None:
        ceilings["max_per_account"] = max_per_account
    if window_min is not None:
        ceilings["window_min"] = window_min
    if global_cap is not None:
        ceilings["global_cap"] = global_cap
    acct, glob = load_launches(ledger_path, account)
    v = admit(
        account,
        now=now,
        account_launches=acct,
        global_launches=glob,
        throttled=bool(throttled or throttle_reset),
        throttle_reset=throttle_reset,
        **ceilings,
    )
    if v["verdict"] == VERDICT_ADMIT and record:
        record_launch(ledger_path, account, now, cause=extra.get("cause") or "admit_or_defer")
    return v


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


def _cmd_plan(args) -> int:
    """Plan-side admission for a whole wave (#6492): the dry-run twin of ``admit``.

    Exit 0 = every proposed lane is launchable right now; 3 = at least one lane would
    DEFER, i.e. the plan's advertised grant is NOT the launchable grant. Records
    nothing, so a planner can ask as often as it likes."""
    now = parse_ts(args.now) if args.now else None
    if args.now and now is None:
        print(json.dumps({"verdict": "ERROR", "detail": f"unparseable --now {args.now!r}"}))
        return 2
    plan = plan_wave(
        args.account,
        ledger_path=args.ledger,
        now=now,
        max_per_account=args.max_per_account,
        window_min=args.window_min,
        global_cap=args.global_cap,
    )
    print(json.dumps(plan, indent=2 if args.pretty else None))
    return 0 if plan["shortfall"] == 0 else 3


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

    p = sub.add_parser("plan", help="plan-side admission for a whole wave (decide-only)")
    p.add_argument("--account", action="append", default=[], required=True,
                   help="one proposed lane's target account; repeat in launch order")
    p.add_argument("--ledger", default=DEFAULT_LEDGER, help="launch ledger path")
    p.add_argument("--now", default=None, help="ISO8601-Z clock override (default: now)")
    p.add_argument("--max-per-account", type=int, default=None, dest="max_per_account")
    p.add_argument("--window-min", type=int, default=None, dest="window_min")
    p.add_argument("--global-cap", type=int, default=None, dest="global_cap")
    p.add_argument("--pretty", action="store_true", help="indent the JSON plan")
    p.set_defaults(func=_cmd_plan)

    r = sub.add_parser("reasons", help="list the structured DEFER tokens this gate emits")
    r.set_defaults(func=_cmd_reasons)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
