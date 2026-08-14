#!/usr/bin/env python3
r"""fleet_trend — turn the fleet-status card from a one-off snapshot into a trend.

Every fleet-status card today answers only "what is true right now". The operator
cannot tell whether usable accounts are recovering or draining, whether the
escalation count is climbing, or whether live sessions just fell off a cliff — the
number carries no direction. This module is the missing time axis: each status tick
appends one compact row to a JSONL history ledger, and a pure renderer folds the
trailing rows into a sparkline + delta per load-bearing metric, so the card can show

    usable 3→3→2→1 ▇▇▄▁ (-2 vs 6 ticks)   escalate 0→0→1 ▁▁█ (+1)

instead of a bare ``usable 1``. It is deliberately tiny and dependency-free (stdlib
only, same as ``slack_post``): the ledger is append-only JSONL (the idiom the
loop-health fold already reads), bounded to a ring of the last ``cap`` rows so it can
never grow without limit, and the fold is pure given the rows + ``now`` so it is
table-tested with no clock and no disk.

  # append one tick from a fleet_top --json snapshot, then print the trend
  python tools/fleet_top.py --json | python tools/fleet_trend.py --append -
  python tools/fleet_trend.py --show          # render the trend from the ledger
  python tools/fleet_trend.py --show --json    # the machine-readable trend rows

The metrics extractor (``metrics_of``) reads a fleet_top snapshot, but ``append`` /
``tail`` / ``render`` take a plain flat ``{key: number}`` dict, so a second producer
(e.g. the dispatch card's queue depth / closed-per-hour) can feed the SAME ledger
later without touching this module.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-trend/1"

# The default history ledger, workspace-relative. Runtime ticks belong under the
# gitignored state root so a scheduled status poll never dirties the shared tree.
DEFAULT_LEDGER = os.path.join(".fak", "nightrun", "fleet-status-history.jsonl")

# The load-bearing scalars a fleet-status trend is about, in render order: each is a
# (key, label) the renderer shows a sparkline + delta for. Chosen so the line reads as
# "is the fleet recovering or draining" at a glance, not a wall of every field.
METRICS: list[tuple[str, str]] = [
    ("usable", "usable"),      # worker accounts usable right now (capacity)
    ("live", "live"),          # live sessions (throughput in flight)
    # The window's session TOTAL (live + resumable + stuck + terminal-history). The
    # ledger key stays "sessions" for history compat, but the rendered label is the
    # explicit "all-sessions": a bare "sessions N→M" misreads as live fleet scale when
    # the count is terminal-history-dominated, so the trend never shows the unqualified
    # word next to the live series (#4651). "live" above is the actionable series.
    ("sessions", "all-sessions"),
    ("escalate", "escalate"),  # operator-actionable items the lifecycle can't heal
]

# Cumulative throughput counters, persisted alongside the gauges so the goodput /
# HEAD-stall SLO (computed by internal/fleettrend on read) can diff them across the
# window. They are NOT rendered by render_line — only recorded. A snapshot carrying no
# throughput block simply never sets them, so a counter-free history honestly reports
# its rates as absent rather than a fabricated zero. Key names mirror internal/fleettrend
# Counters exactly so the Go reader picks them straight off the ledger rows this writes.
COUNTERS: list[tuple[str, str]] = [
    ("lands", "lands"),                      # git-witnessed lands (trunk commit total)
    ("resumes", "resumes"),                  # auto-resume launches
    ("deaths", "deaths"),                    # worker crash/exhaust exits
    ("lands_witnessed", "lands_witnessed"),  # 1 when lands is git-witnessed, else absent
]

# Sparkline ramp, low→high. Eight levels is the most a proportional font renders
# distinctly on a phone; a flat series collapses to the lowest block.
_BLOCKS = "▁▂▃▄▅▆▇█"

# Default ring size — how many trailing ticks the ledger keeps. A daily/hourly tick
# stays well under this for months; the cap only guards against unbounded growth.
DEFAULT_CAP = 500


def metrics_of(snap: dict[str, Any]) -> dict[str, float]:
    """Extract the trend scalars from a fleet_top snapshot. Tolerant of a partial or
    error snapshot: a missing block reads 0, never raises."""
    sess = snap.get("sessions") or {}
    by_cat = sess.get("by_category") or {}
    acc = snap.get("accounts") or {}
    sysv = snap.get("system") or {}
    m = {
        "usable": float(acc.get("usable") or 0),
        "live": float(by_cat.get("LIVE") or 0),
        "sessions": float(sess.get("total") or 0),
        "escalate": float(sysv.get("escalate") or 0),
    }
    # Cumulative throughput counters ride in an optional "throughput" block; a snapshot
    # that carries none leaves them unset (honest absence). lands_witness is a provenance
    # string the producer stamps "git" when lands was counted from the commit graph — we
    # fold it to a 1 flag under the key internal/fleettrend reads.
    tp = snap.get("throughput")
    if isinstance(tp, dict) and tp:
        for c in ("lands", "resumes", "deaths"):
            if tp.get(c) is not None:
                m[c] = float(tp[c] or 0)
        if tp.get("lands_witness") in ("git", "witnessed"):
            m["lands_witnessed"] = 1.0
    return m


def append(path: str, metrics: dict[str, Any], now: str, *, cap: int = DEFAULT_CAP) -> dict[str, Any]:
    """Append one ``{ts, **metrics}`` row to the JSONL ledger and return it.

    Bounded: after appending, the file is trimmed to its last ``cap`` rows so the ledger
    is a ring, never unbounded. Best-effort and never raises on a read error (a corrupt
    prior line is dropped, not fatal) — a status tick must never crash on its own
    history. The directory is created if absent."""
    row = {"ts": now}
    # Backend-local dispatcher probes are not an aggregate fleet census. Preserve their
    # explicit scope for diagnostics without feeding backend_live into the aggregate
    # ``live`` trend or its net-decline alarm.
    if metrics.get("scope") == "backend":
        row["scope"] = "backend"
        row["backend"] = str(metrics.get("backend") or "unknown")
        row["backend_live"] = _num(metrics.get("backend_live"))
    for k, _ in METRICS:
        if k in metrics:
            row[k] = _num(metrics[k])
    # Persist the cumulative counters too when present, so the goodput / HEAD-stall SLO
    # is computable on read. Absent from `metrics` ⇒ absent from the row (never a 0).
    for k, _ in COUNTERS:
        if k in metrics:
            row[k] = _num(metrics[k])
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


def spark(values: list[float]) -> str:
    """A unicode sparkline for a numeric series. A flat series (all equal, incl. a
    single point) renders as the lowest block per point — no false slope."""
    if not values:
        return ""
    lo, hi = min(values), max(values)
    if hi <= lo:
        return _BLOCKS[0] * len(values)
    span = hi - lo
    return "".join(_BLOCKS[min(len(_BLOCKS) - 1, int((v - lo) / span * (len(_BLOCKS) - 1) + 0.5))]
                   for v in values)


def _fmt(v: float) -> str:
    """Render a metric value: integers without a trailing ``.0``."""
    return str(int(v)) if float(v).is_integer() else f"{v:g}"


def metric_trend(rows: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Fold one metric across the rows into {first, last, delta, spark, n}. Returns
    None when the metric never appears (nothing to trend)."""
    series = [_num(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    if not series:
        return None
    return {
        "key": key,
        "first": series[0],
        "last": series[-1],
        "delta": round(series[-1] - series[0], 3),
        "spark": spark(series),
        "n": len(series),
    }


# Net-worker-decline alarm (#4591): how many CONSECUTIVE trailing appends of the
# `live` series must each strictly decline before the drain alarm fires. Three
# mirrors SPAWN_FAILED_RED_STREAK (tools/issue_resolve_dispatch.py): one dip is
# workers finishing, two can be a wind-down — three in a row with no recovery is
# the drain signature (a dead seat / dying pool bleeding the fleet toward zero).
NET_DECLINE_ALARM_STREAK = 3


def net_decline_alarm(rows: list[dict[str, Any]], *, key: str = "live",
                      streak: int = NET_DECLINE_ALARM_STREAK) -> dict[str, Any]:
    """Alarm when the trailing ``key`` series declines STRICTLY for ``streak``
    consecutive ledger appends (#4591) — the net-worker-decline half of the
    dead-seat drain fix: a fleet whose live count only ever steps down is
    draining, whatever each individual tick verdict said. Pure fold over the
    ledger rows (oldest→newest, the :func:`tail` shape). Returns
    ``{red, declines, key, first, last, n, reason}`` — ``declines`` is the
    length of the trailing strict-decline run, ``reason`` non-empty only when
    red. A short / missing / non-declining series is never red (fail-open)."""
    series = [_num(r[key]) for r in rows if isinstance(r.get(key), (int, float))]
    declines = 0
    for prev, cur in zip(reversed(series[:-1]), reversed(series[1:])):
        if cur < prev:
            declines += 1
        else:
            break
    red = declines >= max(1, int(streak))
    reason = ""
    if red:
        window = series[-(declines + 1):]
        reason = (f"net worker decline: '{key}' fell {declines} consecutive "
                  f"ledger appends ({'→'.join(_fmt(v) for v in window)}) — the "
                  f"fleet is draining, not cycling; check seat health "
                  f"(`fak accounts status`) and the dispatch card (#4591)")
    return {"red": red, "key": key, "declines": declines,
            "first": series[0] if series else None,
            "last": series[-1] if series else None,
            "n": len(series), "reason": reason}


def render_line(rows: list[dict[str, Any]]) -> str:
    """One compact trend line across all METRICS present in the rows: per metric a
    ``label first→last spark (Δ±d over N)``. Empty when there is no history yet, so the
    caller can drop the line entirely on a first-ever tick."""
    if not rows:
        return ""
    parts: list[str] = []
    for key, label in METRICS:
        t = metric_trend(rows, key)
        if t is None:
            continue
        arrow = f"{_fmt(t['first'])}→{_fmt(t['last'])}" if t["n"] > 1 else _fmt(t["last"])
        delta = ""
        if t["n"] > 1 and t["delta"]:
            sign = "+" if t["delta"] > 0 else ""
            delta = f" ({sign}{_fmt(t['delta'])} over {t['n']})"
        parts.append(f"{label} {arrow} {t['spark']}{delta}".rstrip())
    return "trend: " + " · ".join(parts) if parts else ""


def _num(v: Any) -> float:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return int(f) if f.is_integer() else f


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
    ap = argparse.ArgumentParser(description="Append + render the fleet-status trend ledger.")
    ap.add_argument("--ledger", default=DEFAULT_LEDGER, help="JSONL history path")
    ap.add_argument("--append", metavar="SNAPSHOT", default="",
                    help="append a tick from a fleet_top --json snapshot (file or - for stdin)")
    ap.add_argument("--show", action="store_true", help="render the trend from the ledger")
    ap.add_argument("--window", type=int, default=24, help="how many trailing ticks to fold")
    ap.add_argument("--cap", type=int, default=DEFAULT_CAP, help="max ledger rows (ring size)")
    ap.add_argument("--now", default="", help="override the tick timestamp (tests)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable rows")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
    except (AttributeError, ValueError):
        pass

    if args.append:
        raw = sys.stdin.read() if args.append == "-" else Path(args.append).read_text(encoding="utf-8")
        snap = json.loads(raw)
        row = append(args.ledger, metrics_of(snap), args.now or _iso_now(), cap=args.cap)
        print(json.dumps(row) if args.json else f"appended {row}")
        if not args.show:
            return 0

    rows = tail(args.ledger, args.window)
    if args.json:
        print(json.dumps({"schema": SCHEMA, "ledger": args.ledger, "rows": rows}, indent=2))
    else:
        line = render_line(rows)
        print(line or "(no history yet)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
