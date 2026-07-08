#!/usr/bin/env python3
r"""fleet_changes — surface what CHANGED about account health, not the same wall each tick.

The fleet-status card restated the full account state on every post: every throttled
account with its reset, every /login wall, every access wall — the SAME rows tick after
tick. An account that has needed /login for six hours is not news; posting it again each
time is the noise that trains an operator to stop reading the card. What *is* news is a
transition: an account that HAD room (or was working) and just hit a blocker, or one that
just came back.

This module is the missing memory that lets the card tell the two apart. Each status tick
records one compact per-account fingerprint (`{tag: state}`) to an append-only JSONL
ledger, and a pure fold diffs the current tick against the immediately-prior one into

    changes vs last post: bravo hit its rate limit · charlie is back · 1 still down (delta)

A steady-state tick — nothing gained, nothing lost — renders NOTHING, so a chronically
blocked or archived account is never repeated; it surfaces once, when it transitions, and
then goes quiet. The first-ever tick has no prior to diff against and is likewise silent
(it would otherwise announce the whole fleet as "newly down").

It is deliberately tiny and dependency-free (stdlib only, same as ``fleet_trend`` /
``slack_post``): the ledger is its own sibling file (``fleet-account-state.jsonl``) so the
aggregate trend ledger's row format is untouched, bounded to a ring of the last ``cap``
rows, and the diff + render are pure given the two state dicts, so they are table-tested
with no clock and no disk.

  python tools/fleet_top.py --json | python tools/fleet_changes.py --append -   # record a tick
  python tools/fleet_changes.py --show                                          # render the change line

``fingerprint`` reads a fleet_top snapshot, but ``append`` / ``tail`` / ``diff`` /
``change_line`` take plain ``{tag: state}`` dicts, so the core stays a pure fold a second
producer could feed the same way.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-changes/1"

# The default per-account state ledger, workspace-relative. A SIBLING of the aggregate
# trend ledger under the same docs/nightrun home — kept separate on purpose so the trend
# ledger's flat {ts, usable, live, ...} row format (mirrored in Go) is never touched.
DEFAULT_LEDGER = os.path.join("docs", "nightrun", "fleet-account-state.jsonl")

# The closed set of account states a fingerprint records, and the human phrasing the
# change line uses for each. Deliberately jargon-free: an operator reads "rate-limited",
# not "throttle verdict". USABLE is the one non-down state (the account has room now).
USABLE = "usable"
# down states -> the terse human label used in a "still down" / "was X" clause.
DOWN_LABEL = {
    "throttled": "rate-limited",
    "login": "needs /login",
    "access": "access wall",
    "blocked": "blocked",
}
# down states -> the verb phrase used when the account JUST transitioned into that state.
DOWN_EVENT = {
    "throttled": "hit its rate limit",
    "login": "needs /login now",
    "access": "hit an access wall",
    "blocked": "is blocked",
}

# Default ring size — how many trailing ticks the ledger keeps. A change diff only ever
# needs the last two rows; the cap just guards against unbounded growth on a busy loop.
DEFAULT_CAP = 500


def fingerprint(snap: dict[str, Any]) -> dict[str, str]:
    """Reduce a fleet_top snapshot's account block to ``{tag: state}`` — the minimal
    per-account fact a transition diff needs. Tolerant of a partial/error snapshot: a
    missing block yields an empty fingerprint, never raises. An account can only be in one
    state; ``available`` wins (an expired-but-cached throttle already reads available in
    the snapshot, so it belongs to USABLE, not a phantom down-row)."""
    acc = snap.get("accounts") or {}
    states: dict[str, str] = {}
    for tag in acc.get("available") or []:
        if tag:
            states[str(tag)] = USABLE
    for t in acc.get("throttled") or []:
        tag = t.get("tag")
        if tag and tag not in states:
            states[str(tag)] = "throttled"
    for b in acc.get("blocked") or []:
        tag = b.get("tag")
        if not tag or tag in states:
            continue
        kind = b.get("kind")
        states[str(tag)] = "login" if kind == "auth" else "access" if kind == "access" else "blocked"
    return states


def diff(prev: dict[str, str], cur: dict[str, str]) -> dict[str, Any]:
    """Classify the transition from ``prev`` (the last recorded tick) to ``cur`` — a pure
    fold, the heart of the module.

    - ``newly_down``: an account that WAS usable last tick and is down now — the genuine
      issue the operator asked to be told about. An account merely *first seen* while
      already down is NOT newly_down (we never saw it have room; it may be an archived
      account just entering the ledger), so it is not announced.
    - ``recovered``: down last tick, usable now — the good-news transition.
    - ``worsened``: down in both ticks but the KIND of block changed (e.g. rate limit →
      /login wall) — a real change worth one line.
    - ``still_down``: down in both ticks, same kind — chronic. Collapsed by the renderer to
      a bare count so it is acknowledged without being restated in full every tick.

    Accounts that vanished from ``cur`` (archived/removed) are intentionally dropped: not
    repeating them IS the point."""
    newly_down: list[dict[str, str]] = []
    recovered: list[str] = []
    worsened: list[dict[str, str]] = []
    still_down: list[dict[str, str]] = []
    for tag, state in cur.items():
        was = prev.get(tag)
        if state == USABLE:
            if was is not None and was != USABLE:
                recovered.append(tag)
            continue
        # state is a down-state
        if was == USABLE:
            newly_down.append({"tag": tag, "state": state})
        elif was is not None and was != USABLE and was != state:
            worsened.append({"tag": tag, "state": state, "was": was})
        elif was == state:
            still_down.append({"tag": tag, "state": state})
        # was is None (first sighting, already down): silent — see docstring.
    return {
        "newly_down": newly_down,
        "recovered": recovered,
        "worsened": worsened,
        "still_down": still_down,
    }


def change_line(d: dict[str, Any]) -> str:
    """Render a transition diff into ONE operator-friendly line, or ``""`` when nothing
    changed. Leads with the transitions that need an operator's eye (newly-down first, then
    recoveries and kind-changes), and folds the chronic ``still_down`` set into a single
    ``N still down (names)`` count rather than restating each blocked account's reason and
    fix — that detail already lives in the card's ATTENTION lines. Empty when there is no
    transition AND nothing chronic, so the caller drops the line entirely on a quiet tick."""
    parts: list[str] = []
    for item in d.get("newly_down") or []:
        parts.append(f"{item['tag']} {DOWN_EVENT.get(item['state'], 'went down')}")
    for tag in d.get("recovered") or []:
        parts.append(f"{tag} is back")
    for item in d.get("worsened") or []:
        now_lbl = DOWN_LABEL.get(item["state"], item["state"])
        was_lbl = DOWN_LABEL.get(item["was"], item["was"])
        parts.append(f"{item['tag']} now {now_lbl} (was {was_lbl})")
    transitions = list(parts)  # everything above is a real, actionable change
    still = d.get("still_down") or []
    if transitions and still:
        # Only a bare COUNT of the chronic set, and only on a tick that ALREADY has news:
        # the identities of long-blocked accounts are exactly the repetition the operator
        # asked to stop, so we give capacity context ("N still down") without re-listing
        # names — the actionable ones are already in the card's ATTENTION lines.
        parts.append(f"{len(still)} still down")
    if not transitions:
        return ""
    return "changes vs last post: " + " · ".join(parts)


def append(path: str, states: dict[str, str], now: str, *, cap: int = DEFAULT_CAP) -> dict[str, Any]:
    """Append one ``{ts, states}`` row to the JSONL ledger and return it. Bounded: after
    appending, the file is trimmed to its last ``cap`` rows so the ledger is a ring, never
    unbounded. Best-effort on read (a torn prior line is dropped, not fatal) — a status
    tick must never crash on its own history. The directory is created if absent."""
    row = {"ts": now, "states": {str(k): str(v) for k, v in (states or {}).items()}}
    rows = _read_rows(path)
    rows.append(row)
    if cap > 0 and len(rows) > cap:
        rows = rows[-cap:]
    p = Path(path)
    if p.parent and not p.parent.exists():
        p.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, separators=(",", ":")) + "\n")
    return row


def tail(path: str, n: int) -> list[dict[str, Any]]:
    """The last ``n`` ledger rows (oldest→newest). Missing/empty ledger → []."""
    rows = _read_rows(path)
    return rows[-n:] if n > 0 else rows


def line_from_ledger(path: str, cur: dict[str, str], now: str, *, record: bool = True) -> str:
    """The end-to-end helper the card uses: optionally record ``cur`` as this tick, then
    render the change line by diffing against the immediately-prior recorded tick. With
    ``record=False`` (a dry-run) the ledger is not written, so a preview never pollutes the
    history. Returns ``""`` on the first-ever tick (no prior to diff) or a quiet tick."""
    if record:
        append(path, cur, now)
        rows = tail(path, 2)
    else:
        rows = tail(path, 1) + [{"ts": now, "states": cur}]
    if len(rows) < 2:
        return ""
    return change_line(diff(rows[-2].get("states") or {}, rows[-1].get("states") or {}))


def _read_rows(path: str) -> list[dict[str, Any]]:
    try:
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
    except OSError:
        return []
    out: list[dict[str, Any]] = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue  # tolerate a torn line rather than brick the history
        if isinstance(row, dict):
            out.append(row)
    return out


def _iso_now() -> str:
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Record + render the fleet account-change line.")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="JSONL per-account state path")
    ap.add_argument("--append", metavar="SNAPSHOT", default="",
                    help="record a tick from a fleet_top --json snapshot (file or - for stdin)")
    ap.add_argument("--show", action="store_true", help="render the change line from the ledger")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help="max ledger rows (ring size)")
    ap.add_argument("--now", default="", help="override the tick timestamp (tests)")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable diff")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    if args.append:
        raw = sys.stdin.read() if args.append == "-" else Path(args.append).read_text(encoding="utf-8")
        snap = json.loads(raw)
        row = append(args.ledger, fingerprint(snap), args.now or _iso_now(), cap=args.cap)
        if not args.show and not args.json:
            print(f"recorded {row}")
            return 0

    rows = tail(args.ledger, 2)
    d = diff(rows[-2].get("states") or {}, rows[-1].get("states") or {}) if len(rows) >= 2 else {}
    if args.json:
        print(json.dumps({"schema": SCHEMA, "ledger": args.ledger, "diff": d}, indent=2))
    else:
        print(change_line(d) if d else "(no change to report)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
