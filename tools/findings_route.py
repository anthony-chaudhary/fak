#!/usr/bin/env python3
"""Findings-route closure appender — a defect STOP self-routes a pickable queue row.

THE GAP THIS CLOSES
-------------------
Fleet's dispatch/supervisor loops emit typed stop-conditions (DRAIN / BLOCKED /
STALE-STAMP / ESCALATE), but a *defect* stop — the loop dying on a recurring wall —
left no trace a future dispatch would pick up. The deterministic-loop-spine rule is
that a defect STOP routes one pickable findings-queue row **at the moment it is
written, not by a later audit**, because a dead loop writes no more rows. This is the
closure half of the RSI loop (`propose -> witness -> keep/revert`): the revert/escalate
verdicts have to route back into the backlog or the loop spins on the same wall forever
(the `NOT_RATCHETING` stop-condition).

THE CONTRACT (ported, not the source's code)
--------------------------------------------
``route_finding`` appends one pickable row to a markdown queue + an append-only JSONL
history, with four invariants:

  * **Idempotent per ``key``** — re-running the *same* defect-stop does not duplicate
    the row (``key`` is the per-stop identity).
  * **Admission damping** — concurrent instances of the same *cause* fold onto one row
    (``cause_key`` is the stable cause identity; no run/loop ts embedded). ``hits``
    counts the fold.
  * **Recurrence escalation** — a cause that re-fires *after* a "fixed" close reopens
    the row and bumps its severity (P2 -> P1 -> P0), counting ``recurrences``.
  * **Never raises** — a routing failure must not kill the loop. Worst case the finding
    is unrecorded (fail-OPEN on this closure layer; the keep layer stays fail-CLOSED).
    Every public entry point returns a verdict dict and swallows its own errors.

Kill-switch: ``FLEET_FINDINGS_ROUTE_ENABLED=0`` (or false/no/off) turns routing into a
no-op without touching the call sites. Unset/anything-else = enabled.

SINK
----
The default sink lives under ``tools/_registry/findings_route/`` — which is *gitignored*
(see ``.gitignore``: ``tools/_registry/``). The queue is a live, reapable runtime
artifact, not a committed doc, so it never trips the docs index-sync / placement gates;
the issue's ``docs/findings-followup-queue.md`` is one example sink, selectable with
``--queue`` / ``--history``. State is folded from the append-only JSONL (the source of
truth); the markdown is a rendered view. A cross-process O_EXCL + TTL lock (the same
Windows-safe, PID-free discipline as ``tools/release_lock.py``) serialises writers.

USAGE
-----
    python tools/findings_route.py route_finding \\
        --key <idempotency-key> --sev <P1|P2> --pattern <short> --item <one-line> \\
        --source <fanout/run-id> --owning-plan <PLAN> [--cause-key <stable-id>]

    python tools/findings_route.py close --cause-key <id> --sha <sha>   # reap a fixed row
    python tools/findings_route.py status [--json]                       # show the queue

Pure stdlib; off the request path (tooling seam).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import datetime as dt
import json
import os
import time
from pathlib import Path
from typing import Any, Iterator

SCHEMA = "fak.findings-route.v1"
QUEUE_SCHEMA = "fak.findings-route-queue.v1"

DEFAULT_SUBDIR = ("tools", "_registry", "findings_route")
QUEUE_NAME = "findings-followup-queue.md"
HISTORY_NAME = "findings-followup-queue.history.jsonl"
LOCK_NAME = ".findings-route.lock"

# Severity ladder, least -> most severe. Escalation bumps one rung and clamps at P0.
LADDER = ("P3", "P2", "P1", "P0")
ENV_FLAG = "FLEET_FINDINGS_ROUTE_ENABLED"
_DISABLED_VALUES = {"0", "false", "no", "off", ""}

LOCK_TIMEOUT_S = 3.0
LOCK_TTL_S = 30.0


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


@dataclasses.dataclass(frozen=True)
class Paths:
    directory: Path
    queue: Path
    history: Path
    lock: Path


def resolve_paths(
    *,
    root: Path | str | None = None,
    queue: Path | str | None = None,
    history: Path | str | None = None,
) -> Paths:
    """Resolve the sink trio. Explicit ``queue``/``history`` win; else default subdir."""
    if queue is not None:
        q = Path(queue).resolve()
        directory = q.parent
        h = Path(history).resolve() if history is not None else directory / HISTORY_NAME
    else:
        base = Path(root).resolve() if root is not None else repo_root()
        directory = base.joinpath(*DEFAULT_SUBDIR)
        q = directory / QUEUE_NAME
        h = Path(history).resolve() if history is not None else directory / HISTORY_NAME
    return Paths(directory=directory, queue=q, history=h, lock=directory / LOCK_NAME)


def enabled_from_env(env: dict[str, str] | None = None) -> bool:
    raw = (env if env is not None else os.environ).get(ENV_FLAG)
    if raw is None:
        return True
    return raw.strip().lower() not in _DISABLED_VALUES


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat()


def _norm_sev(sev: str | None) -> str:
    s = (sev or "").strip().upper()
    return s if s in LADDER else "P2"


def _bump(sev: str) -> str:
    try:
        i = LADDER.index(sev)
    except ValueError:
        i = LADDER.index("P2")
    return LADDER[min(i + 1, len(LADDER) - 1)]


def _more_severe(a: str, b: str) -> str:
    ia = LADDER.index(a) if a in LADDER else 0
    ib = LADDER.index(b) if b in LADDER else 0
    return a if ia >= ib else b


# --------------------------------------------------------------------------- lock


class _LockBusy(Exception):
    """Could not acquire the queue lock within the timeout — fail open, skip the write."""


def _lock_is_stale(lock_path: Path, ttl: float) -> bool:
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    return age > ttl


@contextlib.contextmanager
def queue_lock(lock_path: Path, *, timeout: float = LOCK_TIMEOUT_S, ttl: float = LOCK_TTL_S) -> Iterator[None]:
    """Single-writer queue lock: atomic O_EXCL create, TTL-stale steal, fail-open on busy.

    PID-free staleness on purpose: ``os.kill(pid, 0)`` *terminates* a process on Windows
    (this is a Windows host), so a crashed writer's lock is reclaimed by TTL, never by a
    liveness signal — the same rationale as tools/release_lock.py.
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    fd: int | None = None
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if _lock_is_stale(lock_path, ttl):
                with contextlib.suppress(OSError):
                    lock_path.unlink()
                continue
            if time.time() >= deadline:
                raise _LockBusy(str(lock_path))
            time.sleep(0.02)
    try:
        with contextlib.suppress(OSError):
            os.write(fd, f"{os.getpid()}@{_utcnow()}".encode("utf-8"))
        yield
    finally:
        with contextlib.suppress(OSError):
            os.close(fd)
        with contextlib.suppress(OSError):
            lock_path.unlink()


# --------------------------------------------------------------------------- state


def _read_events(history: Path) -> list[dict[str, Any]]:
    try:
        text = history.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue  # tolerate a torn trailing write; the JSONL stays reapable
        if isinstance(obj, dict):
            events.append(obj)
    return events


def _fold(events: list[dict[str, Any]]) -> tuple[dict[str, dict[str, Any]], set[str], int]:
    """Fold the append-only log to current rows keyed by cause_key (latest snapshot wins)."""
    causes: dict[str, dict[str, Any]] = {}
    seen_keys: set[str] = set()
    max_n = 0
    for ev in events:
        cause = str(ev.get("cause_key") or ev.get("key") or "")
        if not cause:
            continue
        causes[cause] = ev
        for k in ev.get("keys") or []:
            seen_keys.add(str(k))
        if ev.get("key"):
            seen_keys.add(str(ev["key"]))
        with contextlib.suppress(TypeError, ValueError):
            max_n = max(max_n, int(ev.get("n") or 0))
    return causes, seen_keys, max_n


def _row_for_key(causes: dict[str, dict[str, Any]], key: str) -> dict[str, Any] | None:
    for row in causes.values():
        if key in (row.get("keys") or []) or row.get("key") == key:
            return row
    return None


def _append_event(history: Path, snapshot: dict[str, Any]) -> None:
    history.parent.mkdir(parents=True, exist_ok=True)
    with history.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(snapshot, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- render


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def render_queue(events: list[dict[str, Any]]) -> str:
    causes, _seen, _max = _fold(events)
    rows = sorted(causes.values(), key=lambda r: int(r.get("n") or 0))
    lines = [
        "# Findings follow-up queue",
        "",
        "_Self-routed by `tools/findings_route.py` at defect-STOP time — append-only,"
        " reapable. A closed row is struck through (`~~N~~ CLOSED <sha>`); reap freely._",
        "",
        "| # | sev | status | hits | rec | pattern | item | plan | source | cause |",
        "| - | --- | ------ | ---- | --- | ------- | ---- | ---- | ------ | ----- |",
    ]
    for r in rows:
        n = int(r.get("n") or 0)
        closed = str(r.get("status")) == "closed"
        num = f"~~{n}~~" if closed else str(n)
        sha = str(r.get("closed_sha") or "")
        state = f"CLOSED {sha}".strip() if closed else "open"
        lines.append(
            "| {num} | {sev} | {state} | {hits} | {rec} | {pattern} | {item} | {plan} | {source} | {cause} |".format(
                num=num,
                sev=_md_escape(r.get("sev") or ""),
                state=_md_escape(state),
                hits=int(r.get("hits") or 0),
                rec=int(r.get("recurrences") or 0),
                pattern=_md_escape(r.get("pattern") or ""),
                item=_md_escape(r.get("item") or ""),
                plan=_md_escape(r.get("owning_plan") or ""),
                source=_md_escape(r.get("source") or ""),
                cause=_md_escape(r.get("cause_key") or ""),
            )
        )
    lines.append("")
    return "\n".join(lines)


def _write_queue(paths: Paths, events: list[dict[str, Any]]) -> None:
    paths.queue.parent.mkdir(parents=True, exist_ok=True)
    paths.queue.write_text(render_queue(events), encoding="utf-8")


# --------------------------------------------------------------------------- route


def route_finding(
    *,
    key: str,
    sev: str,
    pattern: str,
    item: str,
    source: str,
    owning_plan: str,
    cause_key: str | None = None,
    root: Path | str | None = None,
    queue: Path | str | None = None,
    history: Path | str | None = None,
    now: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Route a defect-STOP into one pickable queue row. NEVER raises (fail-open).

    Returns a verdict dict: ``{ok, action, routed, n, sev, ...}``. ``action`` is one of
    ``routed`` (new row), ``damped`` (folded onto an open cause), ``escalated``
    (reopened a closed cause, severity bumped), ``noop`` (idempotent re-stop),
    ``disabled`` (kill-switch), ``lock_busy`` / ``error`` (fail-open, unrecorded).
    """
    try:
        if not enabled_from_env(env):
            return {"ok": True, "action": "disabled", "routed": False, "key": key, "n": None}
        paths = resolve_paths(root=root, queue=queue, history=history)
        cause = (cause_key or key or "").strip() or key
        sev = _norm_sev(sev)
        ts = now or _utcnow()
        try:
            with queue_lock(paths.lock):
                events = _read_events(paths.history)
                causes, seen_keys, max_n = _fold(events)

                if key in seen_keys:
                    existing = _row_for_key(causes, key) or {}
                    return {
                        "ok": True,
                        "action": "noop",
                        "routed": False,
                        "key": key,
                        "n": existing.get("n"),
                        "sev": existing.get("sev"),
                    }

                prior = causes.get(cause)
                if prior is None:
                    n = max_n + 1
                    row = {
                        "n": n,
                        "cause_key": cause,
                        "keys": [key],
                        "sev": sev,
                        "base_sev": sev,
                        "pattern": pattern,
                        "item": item,
                        "source": source,
                        "owning_plan": owning_plan,
                        "status": "open",
                        "hits": 1,
                        "recurrences": 0,
                        "closed_sha": None,
                        "first_ts": ts,
                    }
                    action = "routed"
                elif str(prior.get("status")) == "closed":
                    # Re-fired after a "fixed" close -> reopen + escalate.
                    row = dict(prior)
                    row["keys"] = list(prior.get("keys") or []) + [key]
                    row["status"] = "open"
                    row["closed_sha"] = None
                    row["recurrences"] = int(prior.get("recurrences") or 0) + 1
                    row["hits"] = int(prior.get("hits") or 0) + 1
                    row["sev"] = _more_severe(_bump(str(prior.get("sev") or sev)), sev)
                    row["source"] = source
                    row["item"] = item
                    action = "escalated"
                else:
                    # Open cause, concurrent instance -> damping fold onto the one row.
                    row = dict(prior)
                    row["keys"] = list(prior.get("keys") or []) + [key]
                    row["hits"] = int(prior.get("hits") or 0) + 1
                    row["sev"] = _more_severe(str(prior.get("sev") or sev), sev)
                    row["source"] = source
                    action = "damped"

                snapshot = dict(row)
                snapshot["schema"] = SCHEMA
                snapshot["action"] = action
                snapshot["key"] = key
                snapshot["last_ts"] = ts
                _append_event(paths.history, snapshot)
                _write_queue(paths, _read_events(paths.history))
                return {
                    "ok": True,
                    "action": action,
                    "routed": True,
                    "key": key,
                    "n": row["n"],
                    "sev": row["sev"],
                    "hits": row["hits"],
                    "recurrences": row["recurrences"],
                    "queue": str(paths.queue),
                    "history": str(paths.history),
                }
        except _LockBusy as exc:
            return {"ok": False, "action": "lock_busy", "routed": False, "key": key, "n": None, "detail": str(exc)}
    except Exception as exc:  # fail-open: a routing failure must NOT kill the loop
        return {"ok": False, "action": "error", "routed": False, "key": key, "n": None, "detail": repr(exc)}


def close_finding(
    *,
    sha: str,
    cause_key: str | None = None,
    n: int | None = None,
    root: Path | str | None = None,
    queue: Path | str | None = None,
    history: Path | str | None = None,
    now: str | None = None,
) -> dict[str, Any]:
    """Reap a fixed row: mark it ``~~N~~ CLOSED <sha>``. Fail-open like route_finding."""
    try:
        paths = resolve_paths(root=root, queue=queue, history=history)
        ts = now or _utcnow()
        try:
            with queue_lock(paths.lock):
                events = _read_events(paths.history)
                causes, _seen, _max = _fold(events)
                target: dict[str, Any] | None = None
                if cause_key is not None:
                    target = causes.get(str(cause_key))
                if target is None and n is not None:
                    target = next((r for r in causes.values() if int(r.get("n") or 0) == int(n)), None)
                if target is None:
                    return {"ok": False, "action": "not_found", "routed": False, "n": n}
                if str(target.get("status")) == "closed" and str(target.get("closed_sha") or "") == str(sha):
                    return {"ok": True, "action": "noop", "routed": False, "n": target.get("n")}
                snapshot = dict(target)
                snapshot["schema"] = SCHEMA
                snapshot["action"] = "closed"
                snapshot["status"] = "closed"
                snapshot["closed_sha"] = str(sha)
                snapshot["last_ts"] = ts
                _append_event(paths.history, snapshot)
                _write_queue(paths, _read_events(paths.history))
                return {"ok": True, "action": "closed", "routed": True, "n": target.get("n"), "closed_sha": str(sha)}
        except _LockBusy as exc:
            return {"ok": False, "action": "lock_busy", "routed": False, "n": n, "detail": str(exc)}
    except Exception as exc:  # fail-open
        return {"ok": False, "action": "error", "routed": False, "n": n, "detail": repr(exc)}


def load_queue(
    *,
    root: Path | str | None = None,
    queue: Path | str | None = None,
    history: Path | str | None = None,
) -> dict[str, Any]:
    paths = resolve_paths(root=root, queue=queue, history=history)
    events = _read_events(paths.history)
    causes, _seen, _max = _fold(events)
    rows = sorted(causes.values(), key=lambda r: int(r.get("n") or 0))
    open_rows = [r for r in rows if str(r.get("status")) != "closed"]
    return {
        "schema": QUEUE_SCHEMA,
        "queue": str(paths.queue),
        "history": str(paths.history),
        "total": len(rows),
        "open": len(open_rows),
        "closed": len(rows) - len(open_rows),
        "rows": rows,
    }


# --------------------------------------------------------------------------- CLI


def _add_route_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--key", required=True, help="idempotency key (per defect-stop)")
    p.add_argument("--sev", required=True, help="severity, e.g. P1 or P2")
    p.add_argument("--pattern", required=True, help="short recurring-pattern label")
    p.add_argument("--item", required=True, help="one-line pickable work item")
    p.add_argument("--source", required=True, help="originating fanout / run-id")
    p.add_argument("--owning-plan", required=True, dest="owning_plan", help="owning plan id")
    p.add_argument("--cause-key", default=None, dest="cause_key", help="stable cause identity (default: --key)")


def _common_sink_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--root", default=None, help="workspace root (default: repo root)")
    p.add_argument("--queue", default=None, help="override the markdown queue path")
    p.add_argument("--history", default=None, help="override the JSONL history path")
    p.add_argument("--now", default=None, help="ISO timestamp (testing/determinism)")
    p.add_argument("--json", action="store_true", help="emit the verdict as JSON")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Route a defect STOP into a pickable findings-queue row.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_route = sub.add_parser("route_finding", aliases=["route"], help="route one defect-stop")
    _add_route_args(p_route)
    _common_sink_args(p_route)

    p_close = sub.add_parser("close", help="reap a fixed row (mark CLOSED <sha>)")
    p_close.add_argument("--cause-key", default=None, dest="cause_key", help="cause identity to close")
    p_close.add_argument("--n", type=int, default=None, help="row number to close")
    p_close.add_argument("--sha", required=True, help="the shipped fix sha")
    _common_sink_args(p_close)

    p_status = sub.add_parser("status", help="show the current queue")
    p_status.add_argument("--root", default=None, help="workspace root (default: repo root)")
    p_status.add_argument("--queue", default=None, help="override the markdown queue path")
    p_status.add_argument("--history", default=None, help="override the JSONL history path")
    p_status.add_argument("--json", action="store_true", help="emit JSON")

    args = ap.parse_args(argv)

    if args.cmd in ("route_finding", "route"):
        verdict = route_finding(
            key=args.key,
            sev=args.sev,
            pattern=args.pattern,
            item=args.item,
            source=args.source,
            owning_plan=args.owning_plan,
            cause_key=args.cause_key,
            root=args.root,
            queue=args.queue,
            history=args.history,
            now=args.now,
        )
        if args.json:
            print(json.dumps(verdict, indent=2, sort_keys=True))
        else:
            print(f"findings-route: {verdict.get('action')} (n={verdict.get('n')} sev={verdict.get('sev')})")
        return 0  # fail-open: routing never breaks the caller's loop

    if args.cmd == "close":
        verdict = close_finding(
            sha=args.sha,
            cause_key=args.cause_key,
            n=args.n,
            root=args.root,
            queue=args.queue,
            history=args.history,
            now=args.now,
        )
        if args.json:
            print(json.dumps(verdict, indent=2, sort_keys=True))
        else:
            print(f"findings-route: {verdict.get('action')} (n={verdict.get('n')})")
        return 0 if verdict.get("ok") else 1

    # status
    snap = load_queue(root=args.root, queue=args.queue, history=args.history)
    if args.json:
        print(json.dumps(snap, indent=2, sort_keys=True))
    else:
        print(render_queue(_read_events(resolve_paths(root=args.root, queue=args.queue, history=args.history).history)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
