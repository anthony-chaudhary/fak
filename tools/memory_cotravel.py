#!/usr/bin/env python3
r"""memory_cotravel.py -- carry a session's slug-scoped agent-memory store along
with its transcript when the resume layer re-homes it onto a healthy account.

THE SEAM THIS CLOSES. Each ``~/.claude*`` account dir holds, PER PROJECT SLUG, two
parallel stores: the conversation transcript ``projects/<slug>/<sid>.jsonl`` AND the
agent auto-memory store ``projects/<slug>/memory/*.md`` (the same store
:mod:`sync_memory` mirrors home<->repo, resolved by ``default_home_memory`` as
``$CLAUDE_CONFIG_DIR/projects/<slug>/memory``). The switcher's resume path re-homes a
session from a throttled account A onto a healthy account B by copying the transcript
(``fleet_resume_watchdog.rehome_transcript``) and pinning ``CLAUDE_CONFIG_DIR=B``. But
the memory store did NOT travel: the resumed session reads/writes memory under B's
slug store, which may be empty or stale, while everything it learned lives under A.
The session lands memory-amnesic. This module is the missing co-travel.

DESIGNED AS AN EXPERIMENTATION SURFACE, not a frozen one-shot:

  * The per-file MERGE decision is a NAMED, PLUGGABLE strategy (the ``STRATEGIES``
    registry). Adding a new way to reconcile a conflict is ~3 lines + a unit test;
    NO core re-home code changes. Default ``additive`` (never clobber a dest memory).

  * The whole co-travel runs behind a SHADOW gate (``FAK_MEMORY_COTRAVEL``), mirroring
    :mod:`switcher_shadow`: in ``shadow`` it COMPUTES every per-file decision and
    appends it to a host-local, off-repo, append-only ledger but copies NOTHING -- so a
    real re-home can be OBSERVED (how often memory would move, how many conflicts each
    strategy hits, whether ``additive``'s skips ever drop a needed memory) with zero
    risk. ``live`` actually copies. ``off`` is the original transcript-only behavior.

Gate / strategy (env, overridable per-call):
    FAK_MEMORY_COTRAVEL = shadow (default) | live | off
    FAK_MEMORY_MERGE    = additive (default) | source_wins | newest_mtime

Ledger (host-local, off-repo, append-only, atomic, size-capped) -- same discipline as
switcher_shadow's ledger: ``~/.claude/fak-memory-cotravel-ledger.jsonl``.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Gate + ledger location (env-overridable; read at call time so tests can patch).
# --------------------------------------------------------------------------- #
HOME = os.environ.get("FLEET_USER_HOME", os.path.expanduser("~"))
LEDGER_PATH = os.environ.get(
    "FAK_MEMORY_COTRAVEL_LEDGER",
    os.path.join(HOME, ".claude", "fak-memory-cotravel-ledger.jsonl"),
)
LEDGER_CAP_BYTES = int(os.environ.get("FAK_MEMORY_COTRAVEL_LEDGER_CAP", str(8 * 1024 * 1024)))

# The default gate is SHADOW: the feature observes real re-homes and copies nothing
# until an operator flips it to ``live``. This is the prove-before-trust default the
# stability doctrine wants -- the amnesia gap is measured immediately and closed on the
# flip, never opened blind.
_DEFAULT_GATE = "shadow"
_DEFAULT_STRATEGY = "additive"
_VALID_GATES = ("shadow", "live", "off")


def gate() -> str:
    g = (os.environ.get("FAK_MEMORY_COTRAVEL") or _DEFAULT_GATE).strip().lower()
    return g if g in _VALID_GATES else _DEFAULT_GATE


def strategy_name() -> str:
    s = (os.environ.get("FAK_MEMORY_MERGE") or _DEFAULT_STRATEGY).strip().lower()
    return s if s in STRATEGIES else _DEFAULT_STRATEGY


# --------------------------------------------------------------------------- #
# Pluggable per-file merge strategies: (src_path, dst_path) -> "copy" | "skip".
# Pure functions of the two file paths -- no I/O beyond stat/read for the compare,
# trivially unit-testable, and NONE of them ever prune a dest-only file (that policy
# lives in the caller, which only ever iterates SOURCE files).
# --------------------------------------------------------------------------- #
def _differ(src: str, dst: str) -> bool:
    """True when dst is absent or its bytes differ from src. Mirrors
    sync_memory.differ -- the same byte-compare the repo<->home mirror uses."""
    if not os.path.exists(dst):
        return True
    try:
        with open(src, "rb") as a, open(dst, "rb") as b:
            return a.read() != b.read()
    except OSError:
        return True


def _additive(src: str, dst: str) -> str:
    """Never clobber. Copy only when the dest file is MISSING; a same-name dest with
    different bytes is kept (a peer / newer memory under the target must survive). The
    safe, regression-proof default."""
    return "copy" if not os.path.exists(dst) else "skip"


def _source_wins(src: str, dst: str) -> str:
    """Source authoritative: copy whenever the bytes differ (overwrite a conflicting
    dest). Guarantees the resumed session sees exactly A's memory -- at the cost of
    clobbering a newer/peer memory already under B."""
    return "copy" if _differ(src, dst) else "skip"


def _newest_mtime(src: str, dst: str) -> str:
    """Keep whichever file is newer by mtime. FLAGGED UNRELIABLE: shutil.copy2 preserves
    the source mtime, so a re-homed copy ties its origin -- the same mtime trap
    resume_resolver documents for transcript owner selection. Present for experiments,
    NOT the default."""
    if not os.path.exists(dst):
        return "copy"
    try:
        return "copy" if os.path.getmtime(src) > os.path.getmtime(dst) else "skip"
    except OSError:
        return "skip"


STRATEGIES = {
    "additive": _additive,
    "source_wins": _source_wins,
    "newest_mtime": _newest_mtime,
}


# --------------------------------------------------------------------------- #
# The co-travel itself -- decide per *.md, then act per the gate.
# --------------------------------------------------------------------------- #
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def plan_one_dir(src_mem: str, dst_mem: str, decide) -> list[dict]:
    """Per-file decisions for carrying *.md from src_mem -> dst_mem under ``decide``.

    Iterates SOURCE files only (so a dest-only memory is never even considered for a
    prune -- co-travel is strictly additive in surface). Returns a list of
    {name, action, dst_exists} records; pure (no copy)."""
    plan: list[dict] = []
    if not os.path.isdir(src_mem):
        return plan
    for src in sorted(glob.glob(os.path.join(src_mem, "*.md"))):
        name = os.path.basename(src)
        dst = os.path.join(dst_mem, name)
        # Never act on a self-copy (mirroring within the owner account resolves dst==src).
        if os.path.abspath(dst) == os.path.abspath(src):
            plan.append({"name": name, "action": "skip", "dst_exists": True})
            continue
        plan.append({"name": name, "action": decide(src, dst),
                     "dst_exists": os.path.exists(dst)})
    return plan


def cotravel_memory(src_cfg: str, dst_cfg: str, slug: str, sid: str,
                    *, dst_slug: str | None = None,
                    gate_value: str | None = None,
                    strategy: str | None = None) -> dict:
    """Carry ``projects/<slug>/memory/*.md`` from src_cfg -> dst_cfg for one re-home.

    The SOURCE memory is always read from the session's owner slug (``slug``). The
    DESTINATION defaults to that same slug, but ``dst_slug`` lets the caller land it
    under a DIFFERENT slug -- the cross-directory resume case: a session born under
    ``C--work-fak`` whose memory lives under that slug must be carried into the LAUNCHING
    cwd's slug (e.g. ``C--work-slack-helpers``) where ``claude --resume`` will look from
    that folder. So src memory = owner slug, dst memory = dst_slug (or owner slug).

    Honors the gate: ``off`` -> no-op; ``shadow`` -> decide + ledger, copy nothing;
    ``live`` -> decide + actually copy the ``copy`` files. Returns a record describing
    what happened (also the ledger row in shadow/live). Fail-open by construction: any
    single file copy that raises (e.g. a Windows mandatory lock) is swallowed, never
    crashing the resolver -- the launcher's contract."""
    g = (gate_value or gate())
    strat = (strategy or strategy_name())
    decide = STRATEGIES.get(strat, _additive)
    dst_slug = dst_slug or slug
    src_mem = os.path.join(src_cfg, "projects", slug, "memory")
    dst_mem = os.path.join(dst_cfg, "projects", dst_slug, "memory")

    rec = {
        "ts": _now_iso(), "session": sid, "slug": slug, "dst_slug": dst_slug,
        "gate": g, "strategy": strat,
        "src_memory": src_mem, "dst_memory": dst_mem,
        "src_has_memory": os.path.isdir(src_mem),
    }
    if g == "off":
        rec.update({"plan": [], "copied": [], "skipped": [], "note": "gate=off (no-op)"})
        return rec

    plan = plan_one_dir(src_mem, dst_mem, decide)
    rec["plan"] = plan
    to_copy = [p["name"] for p in plan if p["action"] == "copy"]
    rec["skipped"] = [p["name"] for p in plan if p["action"] != "copy"]

    if g == "shadow":
        rec["copied"] = []
        rec["would_copy"] = to_copy
        _append_ledger(rec)
        return rec

    # g == "live": actually copy the chosen files (fail-open per file).
    copied: list[str] = []
    if to_copy:
        try:
            os.makedirs(dst_mem, exist_ok=True)
        except OSError:
            pass
    for name in to_copy:
        try:
            shutil.copy2(os.path.join(src_mem, name), os.path.join(dst_mem, name))
            copied.append(name)
        except OSError:
            continue  # locked / vanished -- skip this one, keep going
    rec["copied"] = copied
    _append_ledger(rec)
    return rec


# --------------------------------------------------------------------------- #
# Host-local ledger (off-repo, append-only, atomic, size-capped) -- switcher_shadow shape.
# --------------------------------------------------------------------------- #
def _ledger_path() -> str:
    return os.environ.get("FAK_MEMORY_COTRAVEL_LEDGER", LEDGER_PATH)


def _rotate_if_needed(path: str) -> None:
    try:
        if os.path.getsize(path) > LEDGER_CAP_BYTES:
            os.replace(path, path + ".1")  # keep one prior generation; bounded
    except OSError:
        pass


def _append_ledger(row: dict) -> None:
    """Append one co-travel row. Atomic temp-file + os.replace rewrite (O_APPEND is not
    safe across concurrent re-homes on Windows). Best-effort -- a ledger write must never
    crash the re-home."""
    path = _ledger_path()
    try:
        _rotate_if_needed(path)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        existing = []
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                existing = [ln for ln in f.read().splitlines() if ln.strip()]
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            for ln in existing:
                f.write(ln + "\n")
            f.write(json.dumps(row) + "\n")
        os.replace(tmp, path)
    except OSError:
        pass


def read_ledger() -> list[dict]:
    rows: list[dict] = []
    try:
        with open(_ledger_path(), encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        rows.append(json.loads(ln))
                    except ValueError:
                        continue
    except OSError:
        pass
    return rows
