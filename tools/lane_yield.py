"""Shared low-yield lane fold (#2062) — the turns-spent vs ancestry-closes witness.

READ-ONLY, stdlib-only. This module is the ONE home of ``low_yield_lanes`` and its
``count_lane_ancestry_closes`` git join, so the ``dispatch_status`` card (which reports
the verdict) and ``issue_resolve_dispatch`` (which now ACTS on it — soft-excluding a
low-yield lane from the busiest-pick) fold the same evidence identically. It imports
neither of them, side-stepping the ``dispatch_status <-> issue_resolve_dispatch`` cycle
(``dispatch_status`` lazily imports ``issue_resolve_dispatch``); the small tree/log
primitives are duplicated here rather than imported so the shared fold has no back-edge
to a caller. ``dispatch_status`` re-imports the two public folds for back-compat.

A lane is ``LOW_YIELD`` when its recent (``lookback_min``) resolve sessions spent
``>= turns_floor`` turns yet landed exactly 0 ancestry-closes on its tree — turns burned,
nothing closed. The 180-min sliding lookback IS the self-healing TTL: a lane recovers
when its high-turn sessions age out of the window or a fresh resolving commit lands on
its tree. Fail-open throughout: an unknown tree or an unanswerable git join never flips
a lane LOW_YIELD (the witness never fabricates a verdict it cannot substantiate).
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


# resolve-<issue>-<YYYYMMDD-HHMMSS>.log — the dispatch worker log naming.
_RESOLVE_LOG_RE = re.compile(r"resolve-(\d+)-(\d{8}-\d{6})\.log$")
_LOW_YIELD_SCHEMA = "fleet-low-yield-lanes/1"
_LOW_YIELD_TURNS_FLOOR = 40
_LOW_YIELD_LOOKBACK_MIN = 180
# One kernel-adjudicated turn emits a `fak-turn ...` trace line (cmd/fak/guard.go),
# so counting those lines recovers a resolve log's turn count post-hoc.
_FAK_TURN_RE = re.compile(r"^fak-turn\b")
# The resolving-commit grammar, mirrored from tools/issue_closure_audit._RESOLVE_RE
# (itself a mirror of internal/hooks/commit_issuelink.go) so "what closes an issue"
# agrees across the author gate, the closure audit, and this witness.
_LOW_YIELD_RESOLVE_RE = re.compile(
    r"\b(?:close|fixe?|resolve)[sd]?\s+#(\d+)\b", re.IGNORECASE)


def _string_list(v: Any) -> list[str]:
    if isinstance(v, list):
        return [str(x) for x in v if str(x).strip()]
    if isinstance(v, str) and v.strip():
        return [v]
    return []


def _normalize_tree(t: str) -> str:
    t = str(t or "").strip().replace("\\", "/")
    if t.startswith("./"):
        t = t[2:]
    t = t.rstrip("/")
    for suffix in ("/**", "/*"):
        if t.endswith(suffix):
            t = t[: -len(suffix)]
    return t.rstrip("/")


def _clean_tree(tree: Any) -> list[str]:
    out: list[str] = []
    for t in _string_list(tree):
        n = _normalize_tree(t)
        if n:
            out.append(n)
    return out


def _spawn_lane(log: Path) -> str:
    try:
        first = log.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return ""
    for field in first.split():
        if field.startswith("lane="):
            return field[len("lane="):]
    return ""


def _read_worker_tree(stem: Path) -> list[str]:
    import json
    try:
        obj = json.loads(stem.with_suffix(".lease-tree.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return _string_list(obj)


def _log_turn_count(log: Path) -> int:
    """Turns a ``resolve-*.log`` recorded: one per ``fak-turn ...`` trace line the
    guard emits per kernel-adjudicated turn. Best-effort — an unreadable log is 0."""
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for ln in text.splitlines() if _FAK_TURN_RE.match(ln))


def count_lane_ancestry_closes(root: Path, tree: list[str], *,
                               since_iso: str) -> int | None:
    """Count in-window resolving commits touching ``tree`` — the ancestry-closes a
    lane's recent worker sessions actually landed. A resolving commit is one whose
    subject/body matches the repo's ``close|fix|resolve #N`` grammar AND whose diff
    touched the lane's tree, committed at/after ``since_iso``. Returns ``None`` when
    the tree is unknown or git can't answer, so the caller never flips a lane
    LOW_YIELD on a join it could not make (fail-open, like ``silent_workers`` without
    a liveness oracle)."""
    pathspecs = _clean_tree(tree)
    if not pathspecs:
        return None
    cmd = ["git", "log", f"--since={since_iso}", "--no-merges",
           "--pretty=format:%x1e%s%n%b", "--", *pathspecs]
    try:
        proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=30,
                              creationflags=_win_creationflags())
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    # \x1e (record separator) prefixes each commit's subject+body; count records
    # whose message resolves an issue.
    records = (proc.stdout or "").split("\x1e")
    return sum(1 for rec in records
               if rec.strip() and _LOW_YIELD_RESOLVE_RE.search(rec))


def low_yield_lanes(
    runs_dir: Path,
    *,
    closes_counter: Any,
    turns_floor: int = _LOW_YIELD_TURNS_FLOOR,
    lookback_min: int = _LOW_YIELD_LOOKBACK_MIN,
    now_ts: float | None = None,
    lane_trees: dict[str, list[str]] | None = None,
    turns_of_log: Any | None = None,
    include_log: Callable[[Path], bool] | None = None,
) -> dict[str, Any]:
    """Bind turns-spent to ancestry-closes per lane over the recent window (#2062).

    For each ``resolve-*.log`` whose mtime is within ``lookback_min``, derive its
    lane (spawn-header ``lane=``) and turn count (``fak-turn`` trace lines) and roll
    them up per lane. A lane is a LOW_YIELD candidate when its recent sessions spent
    ``>= turns_floor`` turns; for a candidate whose tree is known, ``closes_counter``
    returns the in-window ancestry-closes touching that tree, and the lane is flagged
    ``LOW_YIELD`` when that count is exactly 0 — turns burned, nothing closed. Every
    other lane (below floor, tree unknown, or with closes) is ``OK``. This never
    flips the card's ``ok``; it is the per-session/per-lane feedback ``pick_lane``
    lacked, so a re-seated low-yield lane is no longer invisible.

    The tree is resolved from ``lane_trees`` (the router's lane→tree map, passed by
    ``collect``) with the worker's ``.lease-tree.json`` sidecar as a fallback; a lane
    with neither is reported with ``tree_known=False`` and never flagged, so the
    witness never fabricates a low-yield verdict it cannot substantiate. Pure given
    ``closes_counter``/``turns_of_log``; git lives only in the default counter.

    ``include_log`` (default keep-all) filters which logs count — the dispatcher
    passes a predicate that drops still-LIVE sessions (a live worker's mid-flight
    turns should not flag its lane before it has had the chance to close anything),
    so the card's read stays byte-identical while the picker folds finished sessions
    only.
    """
    empty = {
        "schema": _LOW_YIELD_SCHEMA,
        "turns_floor": turns_floor,
        "lookback_min": lookback_min,
        "lanes": [],
        "low_yield_count": 0,
    }
    if not runs_dir.is_dir():
        return empty
    now_ts = time.time() if now_ts is None else now_ts
    turns_of = turns_of_log or _log_turn_count
    lane_trees = lane_trees or {}
    horizon = now_ts - lookback_min * 60
    by_lane: dict[str, dict[str, Any]] = {}
    for log in runs_dir.glob("resolve-*.log"):
        if not _RESOLVE_LOG_RE.search(log.name):
            continue
        try:
            if log.stat().st_mtime < horizon:
                continue
        except OSError:
            continue
        if include_log is not None and not include_log(log):
            continue
        lane = _spawn_lane(log)
        if not lane:
            continue
        turns = turns_of(log)
        row = by_lane.setdefault(lane, {
            "lane": lane, "sessions": 0, "turns": 0, "max_session_turns": 0,
            "tree": [], "evidence_logs": []})
        row["sessions"] += 1
        row["turns"] += turns
        row["max_session_turns"] = max(row["max_session_turns"], turns)
        for t in _clean_tree(_read_worker_tree(log.with_suffix(""))):
            if t not in row["tree"]:
                row["tree"].append(t)
        if len(row["evidence_logs"]) < 5:
            row["evidence_logs"].append(log.name)

    lanes_out: list[dict[str, Any]] = []
    for lane, row in by_lane.items():
        tree = _clean_tree(lane_trees.get(lane)) or row["tree"]
        row["tree"] = tree
        row["tree_known"] = bool(tree)
        candidate = row["turns"] >= turns_floor
        # Only pay the git join for a candidate lane with a tree to join on — a
        # below-floor lane is never LOW_YIELD regardless of closes.
        closes = closes_counter(lane, tree) if (candidate and tree) else None
        row["closes"] = closes
        row["verdict"] = "LOW_YIELD" if (candidate and closes == 0) else "OK"
        lanes_out.append(row)

    lanes_out.sort(key=lambda r: (r["verdict"] != "LOW_YIELD",
                                  -int(r["turns"]), str(r["lane"])))
    return {
        "schema": _LOW_YIELD_SCHEMA,
        "turns_floor": turns_floor,
        "lookback_min": lookback_min,
        "lanes": lanes_out,
        "low_yield_count": sum(1 for r in lanes_out if r["verdict"] == "LOW_YIELD"),
    }
