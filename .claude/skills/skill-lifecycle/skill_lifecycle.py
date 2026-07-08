#!/usr/bin/env python3
"""skill_lifecycle.py — witnessed lifecycle for the project skill pack (#2941).

Hermes' curator keeps a skill library from rotting: per-skill usage telemetry in a
`.usage.json` sidecar, auto-archive of stale agent-created skills, NEVER delete
(max action = archive, restorable), pinned skills exempt, snapshots before any
move. This is that lifecycle for `.claude/skills/`, with fak's twist: transitions
are driven by *witnessed value* (usage that proves the skill earned its keep), not
idle time alone, and every transition is journaled + reversible.

Layout (all under the skills root, default `.claude/skills/`):

  .usage.json               — the telemetry sidecar. Per skill: use_count,
                              view_count, patch_count, last_activity_at, state
                              (live|archived), pinned, origin (bundled|agent).
  .lifecycle-journal.jsonl  — append-only transition log. One JSON object per
                              archive/restore event; the restore path reads the
                              same layout the archive path wrote, so every entry
                              is reversible by construction.
  .archive/<name>/          — where an archived skill LIVES (moved, never
                              deleted). `restore` moves it back.
  .archive/.snapshots/      — pre-transition tar.gz backups, one per archive
                              event, so even a botched move is recoverable.

The hard rules, in code not prose:

  * NEVER DELETE. There is no deletion call in this file. Archive is a move;
    a name collision at the destination is a structured refusal, not an
    overwrite.
  * PINNED IS EXEMPT. A skill pinned in the sidecar or via `pin: true` in its
    SKILL.md front-matter is never auto-archived, and even a manual `archive`
    refuses it (unpin first — the intent must be explicit).
  * BUNDLED IS EXEMPT from auto-archive. Origin comes from the sidecar when
    set; otherwise git decides (a tracked SKILL.md is bundled/core, an
    untracked one is agent-created). When git cannot answer, the skill is
    treated as bundled — the failure mode is "kept a skill too long", never
    "archived a core skill".
  * VALUE GUARDS IDLE. Auto-archive requires BOTH idle beyond --idle-days AND
    witnessed value below --min-value (value = use_count + patch_count; views
    alone prove nothing). A rarely-invoked but proven-valuable skill —
    the emergency-only case — stays live.

Witnessed value here is the usage-count approximation the issue allows for;
the richer witnessed net-value metric is future work (#2908 siblings).

Usage:
  python skill_lifecycle.py record <skill> [--kind use|view|patch]
  python skill_lifecycle.py pin <skill> | unpin <skill>
  python skill_lifecycle.py status [--json]
  python skill_lifecycle.py sweep [--idle-days N] [--min-value V] [--apply]
  python skill_lifecycle.py archive <skill> --reason "..."
  python skill_lifecycle.py restore <skill>

Common flags: --skills-root PATH (default .claude/skills relative to the repo
root this file sits in), --now ISO8601 (injectable clock for tests/replays).
Exit code: 0 = clean; 1 = a structured refusal or invariant violation.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

# CREATE_NO_WINDOW keeps a synchronously-spawned console child (the git probes
# below) from flashing a visible window when this script runs from background
# automation on Windows; 0 on POSIX. Mirrors dispatch_worker.no_window_creationflags
# so the popup gate (internal/windowgate) sees the opt-in without a cross-dir import.
_CREATE_NO_WINDOW = 0x08000000


def no_window_creationflags() -> int:
    """``creationflags`` that suppress a console-window popup on Windows; 0 on POSIX."""
    return _CREATE_NO_WINDOW if os.name == "nt" else 0

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

SIDECAR = ".usage.json"
JOURNAL = ".lifecycle-journal.jsonl"
ARCHIVE_DIR = ".archive"
SNAPSHOT_DIR = ".snapshots"
COUNTER_KINDS = ("use", "view", "patch")
DEFAULT_IDLE_DAYS = 30
DEFAULT_MIN_VALUE = 1


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def parse_ts(s: str) -> datetime:
    dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Refusal(Exception):
    """A structured no: reason_class + human reason, never a silent overwrite."""

    def __init__(self, reason_class: str, reason: str):
        super().__init__(f"{reason_class}: {reason}")
        self.reason_class = reason_class
        self.reason = reason


class Store:
    """The skills root plus its sidecar, journal, and archive locations."""

    def __init__(self, root: Path):
        self.root = root
        self.sidecar_path = root / SIDECAR
        self.journal_path = root / JOURNAL
        self.archive_root = root / ARCHIVE_DIR
        self.snapshot_root = self.archive_root / SNAPSHOT_DIR

    # -- sidecar ---------------------------------------------------------
    def load(self) -> dict:
        if self.sidecar_path.exists():
            data = json.loads(self.sidecar_path.read_text(encoding="utf-8"))
        else:
            data = {}
        data.setdefault("skills", {})
        return data

    def save(self, data: dict) -> None:
        tmp = self.sidecar_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self.sidecar_path)

    def entry(self, data: dict, name: str) -> dict:
        return data["skills"].setdefault(
            name,
            {
                "use_count": 0,
                "view_count": 0,
                "patch_count": 0,
                "last_activity_at": None,
                "state": "live",
                "pinned": False,
                "origin": None,
            },
        )

    # -- journal ---------------------------------------------------------
    def journal(self, event: dict) -> None:
        line = json.dumps(event, sort_keys=True)
        with self.journal_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    # -- disk facts ------------------------------------------------------
    def live_dir(self, name: str) -> Path:
        return self.root / name

    def archived_dir(self, name: str) -> Path:
        return self.archive_root / name

    def live_skills(self) -> list[str]:
        if not self.root.is_dir():
            return []
        out = []
        for p in sorted(self.root.iterdir()):
            if p.is_dir() and not p.name.startswith(".") and (p / "SKILL.md").exists():
                out.append(p.name)
        return out


def frontmatter_pin(skill_md: Path) -> bool:
    """True when the SKILL.md front-matter block carries `pin: true`."""
    try:
        text = skill_md.read_text(encoding="utf-8")
    except OSError:
        return False
    if not text.startswith("---"):
        return False
    end = text.find("\n---", 3)
    if end < 0:
        return False
    block = text[3:end]
    return re.search(r"^pin:\s*true\s*$", block, re.MULTILINE) is not None


def git_origin(store: Store, name: str) -> str:
    """bundled|agent from git; anything git cannot witness is bundled (fail safe)."""
    skill_md = store.live_dir(name) / "SKILL.md"
    try:
        probe = subprocess.run(
            ["git", "-C", str(store.root), "rev-parse", "--is-inside-work-tree"],
            capture_output=True, text=True, timeout=30,
            creationflags=no_window_creationflags(),
        )
        if probe.returncode != 0 or probe.stdout.strip() != "true":
            return "bundled"
        tracked = subprocess.run(
            ["git", "-C", str(store.root), "ls-files", "--error-unmatch", str(skill_md)],
            capture_output=True, text=True, timeout=30,
            creationflags=no_window_creationflags(),
        )
        return "bundled" if tracked.returncode == 0 else "agent"
    except (OSError, subprocess.SubprocessError):
        return "bundled"


def origin_of(store: Store, rec: dict, name: str) -> str:
    if rec.get("origin") in ("bundled", "agent"):
        return rec["origin"]
    return git_origin(store, name)


def is_pinned(store: Store, rec: dict, name: str) -> bool:
    if rec.get("pinned"):
        return True
    return frontmatter_pin(store.live_dir(name) / "SKILL.md")


def witnessed_value(rec: dict) -> int:
    # Views are excluded on purpose: being looked at proves curiosity, not value.
    return int(rec.get("use_count", 0)) + int(rec.get("patch_count", 0))


def idle_days(rec: dict, now: datetime) -> float | None:
    last = rec.get("last_activity_at")
    if not last:
        return None  # never witnessed active — treated as idle since forever
    return (now - parse_ts(last)).total_seconds() / 86400.0


# ---------------------------------------------------------------------------
# transitions — the only two, both journaled, both moves, never a delete

def snapshot(store: Store, src: Path, name: str, now: datetime) -> Path:
    store.snapshot_root.mkdir(parents=True, exist_ok=True)
    stamp = now.strftime("%Y%m%dT%H%M%SZ")
    dest = store.snapshot_root / f"{name}-{stamp}.tar.gz"
    n = 1
    while dest.exists():
        dest = store.snapshot_root / f"{name}-{stamp}-{n}.tar.gz"
        n += 1
    with tarfile.open(dest, "w:gz") as tar:
        tar.add(src, arcname=name)
    return dest


def do_archive(store: Store, name: str, reason: str, now: datetime, by: str) -> dict:
    src = store.live_dir(name)
    if not src.is_dir():
        raise Refusal("SKILL_NOT_LIVE", f"no live skill directory {src}")
    dest = store.archived_dir(name)
    if dest.exists():
        raise Refusal(
            "ARCHIVE_COLLISION",
            f"{dest} already exists — refusing to overwrite prior archive (never-delete)",
        )
    data = store.load()
    rec = store.entry(data, name)
    if is_pinned(store, rec, name):
        raise Refusal("SKILL_PINNED", f"skill '{name}' is pinned — unpin before archiving")
    snap = snapshot(store, src, name, now)
    store.archive_root.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    rec["state"] = "archived"
    store.save(data)
    event = {
        "ts": iso(now),
        "action": "archive",
        "skill": name,
        "from": src.as_posix(),
        "to": dest.as_posix(),
        "snapshot": snap.as_posix(),
        "reason": reason,
        "by": by,
    }
    store.journal(event)
    return event


def do_restore(store: Store, name: str, now: datetime, by: str, reason: str = "operator restore") -> dict:
    src = store.archived_dir(name)
    if not src.is_dir():
        raise Refusal("SKILL_NOT_ARCHIVED", f"no archived skill directory {src}")
    dest = store.live_dir(name)
    if dest.exists():
        raise Refusal(
            "RESTORE_COLLISION",
            f"{dest} already exists — refusing to overwrite the live tree (never-delete)",
        )
    shutil.move(str(src), str(dest))
    data = store.load()
    rec = store.entry(data, name)
    rec["state"] = "live"
    rec["last_activity_at"] = iso(now)
    store.save(data)
    event = {
        "ts": iso(now),
        "action": "restore",
        "skill": name,
        "from": src.as_posix(),
        "to": dest.as_posix(),
        "snapshot": None,
        "reason": reason,
        "by": by,
    }
    store.journal(event)
    return event


# ---------------------------------------------------------------------------
# sweep — the auto-archive verdict, dry-run by default

def sweep_verdicts(store: Store, now: datetime, idle_cutoff_days: float, min_value: int) -> list[dict]:
    data = store.load()
    verdicts = []
    for name in store.live_skills():
        rec = store.entry(data, name)
        value = witnessed_value(rec)
        idle = idle_days(rec, now)
        idle_exceeded = idle is None or idle > idle_cutoff_days
        if is_pinned(store, rec, name):
            verdict, why = "keep", "pinned — exempt from auto-archive"
        elif origin_of(store, rec, name) == "bundled":
            verdict, why = "keep", "bundled/core — never auto-archived"
        elif not idle_exceeded:
            verdict, why = "keep", f"active {idle:.1f}d ago (cutoff {idle_cutoff_days:g}d)"
        elif value >= min_value:
            verdict, why = "keep", f"witnessed value {value} >= {min_value} guards the idle signal"
        else:
            verdict, why = "archive", (
                f"idle ({'never active' if idle is None else f'{idle:.1f}d'} > {idle_cutoff_days:g}d) "
                f"and witnessed value {value} < {min_value}"
            )
        verdicts.append({"skill": name, "verdict": verdict, "reason": why, "value": value})
    return verdicts


def cmd_sweep(store: Store, args, now: datetime) -> int:
    verdicts = sweep_verdicts(store, now, args.idle_days, args.min_value)
    applied = []
    for v in verdicts:
        if v["verdict"] == "archive" and args.apply:
            event = do_archive(store, v["skill"], "auto-archive: " + v["reason"], now, by="sweep")
            applied.append(event)
    print(json.dumps({
        "mode": "apply" if args.apply else "dry-run",
        "verdicts": verdicts,
        "archived": [e["skill"] for e in applied],
    }, indent=2))
    return 0


def cmd_record(store: Store, args, now: datetime) -> int:
    name = args.skill
    if not store.live_dir(name).is_dir() and not store.archived_dir(name).is_dir():
        raise Refusal("SKILL_UNKNOWN", f"no skill named '{name}' live or archived under {store.root}")
    data = store.load()
    rec = store.entry(data, name)
    rec[f"{args.kind}_count"] = int(rec.get(f"{args.kind}_count", 0)) + 1
    rec["last_activity_at"] = iso(now)
    store.save(data)
    print(json.dumps({name: rec}, indent=2))
    return 0


def cmd_pin(store: Store, args, now: datetime, pinned: bool) -> int:
    data = store.load()
    rec = store.entry(data, args.skill)
    rec["pinned"] = pinned
    store.save(data)
    print(json.dumps({args.skill: rec}, indent=2))
    return 0


def cmd_status(store: Store, args, now: datetime) -> int:
    data = store.load()
    rows = []
    for name in store.live_skills():
        rec = store.entry(data, name)
        rows.append({
            "skill": name, "state": "live",
            "pinned": is_pinned(store, rec, name),
            "origin": origin_of(store, rec, name),
            "value": witnessed_value(rec),
            "last_activity_at": rec.get("last_activity_at"),
        })
    if store.archive_root.is_dir():
        for p in sorted(store.archive_root.iterdir()):
            if p.is_dir() and p.name != SNAPSHOT_DIR:
                rows.append({"skill": p.name, "state": "archived"})
    print(json.dumps(rows, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skills-root", default=None,
                    help="skills directory (default: this file's grandparent, i.e. .claude/skills)")
    ap.add_argument("--now", default=None, help="ISO8601 clock override (tests/replays)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("record", help="witness one activity event for a skill")
    p.add_argument("skill")
    p.add_argument("--kind", choices=COUNTER_KINDS, default="use")

    for verb in ("pin", "unpin"):
        p = sub.add_parser(verb, help=f"{verb} a skill ({'exempt from' if verb == 'pin' else 'eligible for'} auto-archive)")
        p.add_argument("skill")

    sub.add_parser("status", help="one row per skill: state, pin, origin, witnessed value")

    p = sub.add_parser("sweep", help="auto-archive verdicts (dry-run unless --apply)")
    p.add_argument("--idle-days", type=float, default=DEFAULT_IDLE_DAYS)
    p.add_argument("--min-value", type=int, default=DEFAULT_MIN_VALUE)
    p.add_argument("--apply", action="store_true")

    p = sub.add_parser("archive", help="manually archive one skill (still snapshot+journal, still refuses pinned)")
    p.add_argument("skill")
    p.add_argument("--reason", required=True)

    p = sub.add_parser("restore", help="move an archived skill back to the live tree")
    p.add_argument("skill")

    args = ap.parse_args(argv)
    root = Path(args.skills_root) if args.skills_root else Path(__file__).resolve().parent.parent
    store = Store(root)
    now = parse_ts(args.now) if args.now else utc_now()

    try:
        if args.cmd == "record":
            return cmd_record(store, args, now)
        if args.cmd == "pin":
            return cmd_pin(store, args, now, pinned=True)
        if args.cmd == "unpin":
            return cmd_pin(store, args, now, pinned=False)
        if args.cmd == "status":
            return cmd_status(store, args, now)
        if args.cmd == "sweep":
            return cmd_sweep(store, args, now)
        if args.cmd == "archive":
            print(json.dumps(do_archive(store, args.skill, args.reason, now, by="operator"), indent=2))
            return 0
        if args.cmd == "restore":
            print(json.dumps(do_restore(store, args.skill, now, by="operator"), indent=2))
            return 0
    except Refusal as r:
        print(json.dumps({"refused": True, "reason_class": r.reason_class, "reason": r.reason}), file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
