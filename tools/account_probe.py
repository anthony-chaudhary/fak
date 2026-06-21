#!/usr/bin/env python3
r"""account_probe -- ACTIVE account health probes for the fleet roster.

The fleet roster (``fleet_accounts.py``, the switcher, the resume watchdog) decides
whether an account is usable by *passively* scanning each account's transcript tails
(``fleet_sessions.py``). An account with no transcript inside the scan window gets no
fresh verdict -- its status is carried forward from the previous registry until a
*newer successful turn* on that same account clears it. A blocked account never
produces a successful turn on its own, so the blocker becomes a one-way latch: it can
only release when a human happens to run a real session on that account.

This module breaks the latch by getting GROUND TRUTH on demand: it launches a tiny
``claude -p "say pong"`` under the account's ``CLAUDE_CONFIG_DIR`` and classifies the
result with the SAME signal regexes the passive scan uses (``fleet_session_signals``),
so a probe and a transcript speak one language. A successful probe re-blesses an account
that had silently recovered (re-login, access re-enabled, throttle expired); a failing
probe re-confirms the wall with a fresh timestamp.

The verdict is projected into a ``fleet_sessions`` ROW (``verdict_to_row``) so it flows
through the existing ``merge_known_auth`` / ``merge_known_throttle`` pipeline UNCHANGED:
a fresh ``OK`` probe row (disp=LIVE, age~0) wins the "newest row" comparison and clears a
stale blocker; a fresh ``AUTH``/``LIMIT`` row sets one. The mergers are not modified --
the probe is just a second SOURCE of fresh rows alongside the transcript scan.

Closed status vocabulary (one of):
    OK         account answered cleanly -> available
    AUTH       needs human re-login (/login); a fresh token fixes it
    ACCESS     org/subscription access disabled; a token will NOT fix it (operator/admin)
    CREDIT     credit/billing balance too low
    LIMIT      usage/session limit; carries the reset window(s)
    APIERR     transient API/transport error from the server
    TRANSPORT  could not even run the probe (spawn failure / timeout) -- local/transport

CLI:
    python tools/account_probe.py                 # probe blocked+stale workers, table
    python tools/account_probe.py --all           # probe every worker account
    python tools/account_probe.py --account gem8  # probe one account (tag/basename)
    python tools/account_probe.py --json          # machine verdicts
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_accounts  # noqa: E402
import fleet_session_signals  # noqa: E402

SCHEMA = "fleet.account-probe.v1"

# Per-account probe history, mirroring the resume_ledger.jsonl / transitions.log pattern.
# One JSON line per probe; the prev_status carry makes a status FLIP (ACCESS->OK, OK->AUTH)
# a first-class greppable event for the status card and any alerting.
def reg_dir() -> str:
    return os.environ.get(
        "FLEET_REG_DIR",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "_registry"))


def probe_ledger_path(rd: str | None = None) -> str:
    return os.path.join(rd or reg_dir(), "probe_ledger.jsonl")


def _read_ledger(path: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    out.append(json.loads(ln))
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def last_probe_by_account(rd: str | None = None) -> dict[str, dict[str, Any]]:
    """Most-recent ledger entry per account (account basename -> entry)."""
    latest: dict[str, dict[str, Any]] = {}
    for e in _read_ledger(probe_ledger_path(rd)):
        acct = e.get("account")
        if acct:
            latest[acct] = e  # file is append-ordered, so the last write wins
    return latest


def recent_probe_age_min(account: str, rd: str | None = None) -> float | None:
    """Minutes since this account was last probed, or None if never / unparseable."""
    e = last_probe_by_account(rd).get(account)
    ts = e.get("ts") if e else None
    if not ts:
        return None
    try:
        when = dt.datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    return (utc_now() - when).total_seconds() / 60.0


def append_probe_ledger(verdicts: list[dict[str, Any]], rd: str | None = None,
                        ) -> list[dict[str, Any]]:
    """Append one line per verdict; stamp prev_status from the prior ledger entry.

    Returns the list of ledger records written (each carries ``flip`` = prev != status),
    so a caller can surface flips without re-reading the file.
    """
    rd = rd or reg_dir()
    prev = last_probe_by_account(rd)
    os.makedirs(rd, exist_ok=True)
    records = []
    for v in verdicts:
        acct = v.get("account")
        prev_status = (prev.get(acct) or {}).get("status") if acct else None
        rec = {
            "ts": v.get("probed_utc") or utc_now().isoformat(),
            "account": acct,
            "tag": v.get("tag"),
            "status": v.get("status"),
            "prev_status": prev_status,
            "flip": bool(prev_status and prev_status != v.get("status")),
            "block_kind": v.get("block_kind"),
            "reset": v.get("reset"),
            "latency_ms": v.get("latency_ms"),
        }
        records.append(rec)
    with open(probe_ledger_path(rd), "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, separators=(",", ":")) + "\n")
    return records

# A cheap, fast model for the liveness ping. Overridable; the point is the auth/limit
# verdict, not the answer, so the smallest model is correct.
DEFAULT_PROBE_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_PROMPT = "say pong"
DEFAULT_TIMEOUT_S = 45.0
DEFAULT_MAX_WORKERS = 4

# Closed status set -> the fleet_sessions disposition its row carries. This is the seam
# that lets a probe verdict ride the existing merge pipeline.
STATUS_TO_DISP = {
    "OK": "LIVE",
    "AUTH": "INFRA_AUTH",
    "ACCESS": "INFRA_AUTH",
    "CREDIT": "INFRA_AUTH",
    "LIMIT": "STOPPED_LIMIT",
    "APIERR": "STOPPED_APIERR",
    "TRANSPORT": "STOPPED_QUIET",
}
KNOWN_STATUSES = tuple(STATUS_TO_DISP)
# Statuses that CAN silently recover and so are always worth re-probing, vs LIMIT which
# just re-confirms until its reset passes (the watchdog cost-guards LIMIT separately).
RECOVERABLE_AUTH_STATUSES = ("AUTH", "ACCESS", "CREDIT")


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


# A line the CLI prints to stderr when it gets no piped stdin; it is NOT a model answer.
# We feed empty stdin to avoid it, but strip it belt-and-suspenders so a run whose only
# output is this warning never reads as a successful "pong".
_STDIN_WARNING = "no stdin data received"


def _strip_noise(text: str) -> str:
    """Drop benign CLI chatter (the stdin warning) so it can't masquerade as an answer."""
    lines = [ln for ln in (text or "").splitlines()
             if _STDIN_WARNING not in ln.lower()]
    return "\n".join(lines).strip()


def _success_text(text: str) -> bool:
    """A clean probe answer. We don't require the literal 'pong' (the model may add
    punctuation/casing); we require the ABSENCE of any block signal plus some content."""
    t = _strip_noise(text)
    if not t:
        return False
    if fleet_session_signals.is_auth_error(t):
        return False
    if fleet_session_signals.limit_resets(t):
        return False
    if fleet_session_signals.is_api_error(t):
        return False
    return True


def classify_probe_output(exit_code: int, stdout: str, stderr: str,
                          *, timed_out: bool = False,
                          spawn_error: str = "") -> dict[str, Any]:
    """Map a raw probe run to a closed-vocabulary verdict.

    Order matters: auth/access/credit walls and usage limits are checked before the
    generic transport bucket, and before declaring success, so a banner on stdout is
    never mistaken for a clean answer.
    """
    text = "\n".join(p for p in (stdout or "", stderr or "") if p).strip()
    if spawn_error:
        return {"status": "TRANSPORT", "block_kind": "transport",
                "block_reason": f"could not launch probe: {spawn_error}",
                "reset": None, "weekly": None}
    if timed_out:
        return {"status": "TRANSPORT", "block_kind": "transport",
                "block_reason": "probe timed out", "reset": None, "weekly": None}

    if fleet_session_signals.is_auth_error(text):
        kind = fleet_session_signals.auth_block_kind(text)  # access | credit | auth
        status = {"access": "ACCESS", "credit": "CREDIT"}.get(kind, "AUTH")
        return {"status": status, "block_kind": kind,
                "block_reason": fleet_session_signals.auth_block_reason(text),
                "reset": None, "weekly": None}

    windows = fleet_session_signals.limit_resets(text)
    if windows:
        reset = windows.get("daily") or windows.get("weekly")
        return {"status": "LIMIT", "block_kind": "usage",
                "block_reason": (f"usage limit; resets {reset}" if reset else "usage limit"),
                "reset": reset, "weekly": windows.get("weekly")}

    if fleet_session_signals.is_api_error(text):
        return {"status": "APIERR", "block_kind": "apierr",
                "block_reason": "API/transport error (transient)",
                "reset": None, "weekly": None}

    if _success_text(text):
        return {"status": "OK", "block_kind": None, "block_reason": "",
                "reset": None, "weekly": None}

    # Non-zero exit with no recognized banner: treat as transient transport, not a hard
    # block -- we don't want an unparsed stderr to permanently sideline an account.
    if exit_code != 0:
        return {"status": "TRANSPORT", "block_kind": "transport",
                "block_reason": f"probe exited {exit_code} with no known signal",
                "reset": None, "weekly": None}
    # Zero exit but empty/odd output: be conservative, call it transport so it re-probes.
    return {"status": "TRANSPORT", "block_kind": "transport",
            "block_reason": "probe produced no classifiable output",
            "reset": None, "weekly": None}


def default_claude_exe() -> str:
    return (os.environ.get("FLEET_CLAUDE_EXE")
            or shutil.which("claude") or shutil.which("claude.exe")
            or os.path.join(os.path.expanduser("~"), ".local", "bin", "claude.exe"))


def _default_runner(argv: list[str], *, config_dir: str, timeout: float,
                    ) -> tuple[int, str, str, bool, str]:
    """Run the probe command. Returns (exit_code, stdout, stderr, timed_out, spawn_error).

    Isolated behind a callable so tests inject a fake and never spawn a real claude.
    stdin is fed an empty string so the CLI does not stall waiting on a pipe (the
    observed "no stdin data received in 3s" warning).
    """
    env = os.environ.copy()
    env["CLAUDE_CONFIG_DIR"] = config_dir
    try:
        proc = subprocess.run(
            argv, env=env, input="", capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout if isinstance(exc.stdout, str) else ""
        err = exc.stderr if isinstance(exc.stderr, str) else ""
        return 124, out, err, True, ""
    except OSError as exc:
        return 127, "", "", False, str(exc)
    return proc.returncode, proc.stdout or "", proc.stderr or "", False, ""


def probe_account(row: dict[str, Any], *, claude_exe: str | None = None,
                  model: str = DEFAULT_PROBE_MODEL, prompt: str = DEFAULT_PROMPT,
                  timeout: float = DEFAULT_TIMEOUT_S,
                  runner: Callable[..., tuple[int, str, str, bool, str]] | None = None,
                  ) -> dict[str, Any]:
    """Actively probe one account row (from fleet_accounts.discover_accounts).

    ``runner`` is injectable for tests; it must accept (argv, *, config_dir, timeout)
    and return (exit_code, stdout, stderr, timed_out, spawn_error).
    """
    exe = claude_exe or default_claude_exe()
    config_dir = str(row.get("dir") or "")
    account = str(row.get("account") or "")
    tag = str(row.get("tag") or fleet_accounts.account_tag(account))
    argv = [exe, "-p", prompt, "--model", model]
    run = runner or _default_runner

    started = utc_now()
    exit_code, stdout, stderr, timed_out, spawn_error = run(
        argv, config_dir=config_dir, timeout=timeout)
    probed = utc_now()
    latency_ms = int((probed - started).total_seconds() * 1000)

    verdict = classify_probe_output(exit_code, stdout, stderr,
                                    timed_out=timed_out, spawn_error=spawn_error)
    tail = "\n".join(p for p in (stdout or "", stderr or "") if p).strip()
    verdict.update({
        "account": account,
        "tag": tag,
        "dir": config_dir,
        "product": str(row.get("product") or "claude"),
        "probed_utc": probed.isoformat(),
        "latency_ms": latency_ms,
        "exit_code": exit_code,
        "raw_tail": tail[-400:],
    })
    return verdict


def verdict_to_row(verdict: dict[str, Any]) -> dict[str, Any]:
    """Project a probe verdict into a fleet_sessions ROW the mergers consume unchanged.

    A probe row is timestamped NOW (age~0) so it wins newest-row comparisons in
    merge_known_auth / merge_known_throttle and decides the carry-forward latch.
    """
    status = str(verdict.get("status") or "TRANSPORT")
    disp = STATUS_TO_DISP.get(status, "STOPPED_QUIET")
    probed = str(verdict.get("probed_utc") or utc_now().isoformat())
    reset = verdict.get("reset")
    weekly = verdict.get("weekly")
    is_limit = status == "LIMIT"
    reason = str(verdict.get("block_reason") or "")
    # `last` carries the human banner so merge_known_auth's _auth_info_from_row reclassifies
    # it identically to a real transcript tail.
    last = verdict.get("raw_tail") or reason
    return {
        "account": verdict.get("account"),
        "project": "_probe",
        "session": f"probe-{verdict.get('tag') or 'acct'}",
        "disp": disp,
        "category": "PROBE",
        "cause": "active_probe",
        "reason": reason or f"probe:{status}",
        "last_kind": "probe",
        "age_min": 0.0,
        "seen_utc": probed,
        "throttle_reset": reset if is_limit else None,
        "throttle_weekly": weekly if is_limit else None,
        "throttle_seen": reset if is_limit else None,
        "throttle_current": bool(is_limit and reset),
        "pending_tool": None,
        "cwd": None,
        "git": None,
        "last": str(last)[:200].replace("\n", " "),
        "path": None,
        "autonomous": False,
        "supervised": False,
        "probe_status": status,
    }


def _should_probe(row: dict[str, Any], selector: str) -> bool:
    """Selector policy. ``row`` is an ANNOTATED worker row (has available/blocked/etc.)."""
    if row.get("kind") != "worker":
        return False
    if selector == "all":
        return True
    # A malformed row missing `available` is treated as blocked -> probed (fail toward
    # checking, never silently skip an account we cannot read).
    blocked = not row.get("available", False)
    if selector == "blocked":
        return blocked
    if selector == "stale":
        # "stale" = blocked OR carrying no live/active session evidence right now.
        return blocked or not (row.get("active_sessions") or row.get("live_sessions"))
    return False


def select_targets(annotated_rows: list[dict[str, Any]], selector: str = "blocked",
                   account: str = "", *, skip_active_throttle: bool = False,
                   min_interval_min: float = 0.0, reg_dir_path: str | None = None,
                   ) -> list[dict[str, Any]]:
    """Pick which annotated worker rows to probe.

    ``account`` (a tag/basename substring) overrides the selector for a single account
    AND the freshness floor (an explicit single-account probe is always honored).
    ``skip_active_throttle`` drops throttled accounts whose reset is still in the future
    (the watchdog cost-guard: re-probing them just re-confirms the limit). Auth/access/
    credit blockers are always kept -- those can silently recover.
    ``min_interval_min`` drops accounts probed within the last N minutes (read from the
    ledger) so a tight watchdog cadence cannot spam an account.
    """
    if account:
        needle = account.lower()
        return [
            r for r in annotated_rows
            if r.get("kind") == "worker" and (
                needle in str(r.get("tag") or "").lower()
                or needle in str(r.get("account") or "").lower())
        ]
    targets = [r for r in annotated_rows if _should_probe(r, selector)]
    if skip_active_throttle:
        kept = []
        for r in targets:
            if r.get("block_kind") == "usage" and r.get("throttled") \
                    and fleet_accounts.throttle_is_active(
                        {"reset": r.get("reset"), "weekly": r.get("weekly")}):
                continue
            kept.append(r)
        targets = kept
    if min_interval_min and min_interval_min > 0:
        kept = []
        for r in targets:
            age = recent_probe_age_min(str(r.get("account") or ""), reg_dir_path)
            if age is not None and age < min_interval_min:
                continue  # probed too recently; skip to avoid spamming the account
            kept.append(r)
        targets = kept
    return targets


def probe_accounts(targets: list[dict[str, Any]], *, claude_exe: str | None = None,
                   model: str = DEFAULT_PROBE_MODEL, timeout: float = DEFAULT_TIMEOUT_S,
                   max_workers: int = DEFAULT_MAX_WORKERS,
                   runner: Callable[..., tuple[int, str, str, bool, str]] | None = None,
                   ) -> list[dict[str, Any]]:
    """Probe a list of (annotated worker) rows concurrently, bounded by max_workers."""
    if not targets:
        return []
    workers = max(1, min(int(max_workers), len(targets)))
    verdicts: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(probe_account, r, claude_exe=claude_exe, model=model,
                        timeout=timeout, runner=runner): r
            for r in targets
        }
        for fut in concurrent.futures.as_completed(futs):
            try:
                verdicts.append(fut.result())
            except Exception as exc:  # never let one bad probe sink the batch
                r = futs[fut]
                verdicts.append({
                    "status": "TRANSPORT", "block_kind": "transport",
                    "block_reason": f"probe raised: {exc}", "reset": None, "weekly": None,
                    "account": r.get("account"), "tag": r.get("tag"),
                    "dir": r.get("dir"), "product": r.get("product"),
                    "probed_utc": utc_now().isoformat(), "latency_ms": 0,
                    "exit_code": -1, "raw_tail": str(exc)[:400],
                })
    verdicts.sort(key=lambda v: str(v.get("tag") or v.get("account") or ""))
    return verdicts


# ----------------------------------------------------------------------------- CLI

def _load_annotated(home: str | None = None) -> list[dict[str, Any]]:
    return fleet_accounts.annotated_roster(home or fleet_accounts.USER)


def _roster_was(row: dict[str, Any]) -> str:
    if row.get("available"):
        return "available"
    return str(row.get("block_kind") or "blocked")


def summarize(verdicts: list[dict[str, Any]],
              ledger_records: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Roll up a batch of verdicts into counts + a one-line string.

    ``flips`` counts status changes from the ledger records (if the ledger was written).
    ``all_blocked`` is True iff at least one account was probed and none came back OK --
    the signal a cron/CI wrapper turns into an alert.
    """
    counts: dict[str, int] = {}
    for v in verdicts:
        s = str(v.get("status") or "TRANSPORT")
        counts[s] = counts.get(s, 0) + 1
    flips = sum(1 for r in (ledger_records or []) if r.get("flip"))
    ordered = [s for s in KNOWN_STATUSES if counts.get(s)]
    parts = " ".join(f"{s.lower()}={counts[s]}" for s in ordered)
    line = f"probed {len(verdicts)}: {parts}".rstrip()
    if flips:
        line += f"  flips={flips}"
    return {
        "probed": len(verdicts),
        "counts": counts,
        "flips": flips,
        "ok": counts.get("OK", 0),
        "all_blocked": bool(verdicts) and counts.get("OK", 0) == 0,
        "line": line,
    }


def _print_table(annotated: list[dict[str, Any]], verdicts: list[dict[str, Any]],
                 ledger_records: list[dict[str, Any]] | None = None) -> None:
    by_acct = {r.get("account"): r for r in annotated}
    flip_by_acct = {r.get("account"): r for r in (ledger_records or []) if r.get("flip")}
    print(f"{'tag':<18} {'was(roster)':<14} {'now(probe)':<10} {'reset':<28} {'ms':>6}  note")
    corrections = 0
    for v in verdicts:
        row = by_acct.get(v.get("account"), {})
        was = _roster_was(row)
        now = str(v.get("status"))
        reset = str(v.get("reset") or "")
        ms = int(v.get("latency_ms") or 0)
        # a correction = roster said one thing, live probe says another actionable thing
        roster_blocked = not row.get("available", False)
        probe_ok = now == "OK"
        differs = (probe_ok and roster_blocked) or (
            not probe_ok and not roster_blocked) or (
            row.get("block_kind") and v.get("block_kind")
            and row.get("block_kind") != v.get("block_kind"))
        note = ""
        if differs:
            corrections += 1
            note = "<< CORRECTION"
        flip = flip_by_acct.get(v.get("account"))
        if flip:
            note = (note + " " if note else "") + f"FLIP {flip.get('prev_status')}->{now}"
        print(f"{str(v.get('tag')):<18} {was:<14} {now:<10} {reset:<28} {ms:>6}  {note}")
    summary = summarize(verdicts, ledger_records)
    print(f"\n{summary['line']}")
    print(f"{len(verdicts)} probed, {corrections} correction(s) vs the stale roster.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Active fleet account health probes.")
    parser.add_argument("--account", default="", help="probe one account (tag/basename)")
    parser.add_argument("--all", action="store_true", help="probe every worker account")
    parser.add_argument("--blocked", action="store_true",
                        help="probe only currently-blocked workers (default selector)")
    parser.add_argument("--stale", action="store_true",
                        help="probe blocked OR idle (no live/active session) workers")
    parser.add_argument("--model", default=DEFAULT_PROBE_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_S)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--skip-active-throttle", action="store_true",
                        help="skip accounts whose throttle reset is still in the future")
    parser.add_argument("--min-interval-min", type=float, default=0.0,
                        help="skip accounts probed within the last N minutes (anti-spam)")
    parser.add_argument("--no-ledger", action="store_true",
                        help="do not append to probe_ledger.jsonl")
    parser.add_argument("--exit-code", action="store_true",
                        help="return non-zero when accounts were probed and none are OK")
    parser.add_argument("--home", default=None)
    parser.add_argument("--json", action="store_true", help="emit JSON verdicts")
    args = parser.parse_args(argv)

    selector = "all" if args.all else "stale" if args.stale else "blocked"
    annotated = _load_annotated(args.home)
    targets = select_targets(annotated, selector=selector, account=args.account,
                             skip_active_throttle=args.skip_active_throttle,
                             min_interval_min=args.min_interval_min)
    verdicts = probe_accounts(targets, model=args.model, timeout=args.timeout,
                              max_workers=args.max_workers)
    ledger_records = None
    if verdicts and not args.no_ledger:
        ledger_records = append_probe_ledger(verdicts)

    if args.json:
        print(json.dumps({"schema": SCHEMA, "now": utc_now().isoformat(),
                          "selector": selector,
                          "summary": summarize(verdicts, ledger_records),
                          "verdicts": verdicts}, indent=1))
    else:
        if not verdicts:
            print(f"no accounts matched selector={selector}"
                  + (f" account={args.account!r}" if args.account else ""))
        else:
            _print_table(annotated, verdicts, ledger_records)

    if args.exit_code and summarize(verdicts, ledger_records)["all_blocked"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
