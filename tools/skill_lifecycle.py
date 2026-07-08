#!/usr/bin/env python3
"""Witnessed skill lifecycle for an agent-authored skill library (#2941).

Hermes' curator tracks per-skill usage in a sidecar, auto-archives stale
agent-created skills (never bundled ones), never deletes (max action =
archive, restorable), exempts pinned skills. fak adopts that lifecycle with
the repo's trust rules on top:

- **Every transition is journaled and reversible.** An archive is a rename
  into ``<root>/.archive/`` plus a journal row; ``restore`` is the inverse.
  There is NO delete path anywhere in this tool (the never-delete invariant).
- **Only witnessed evidence drives auto-archive.** ``sweep`` proposes a skill
  whose recorded activity is idle past the window — and only a skill marked
  ``origin=agent``; a bundled (or unmarked) skill is exempt, as is a pinned
  one. DRY-RUN BY DEFAULT; ``--live`` executes.
- **The sidecar is the telemetry, the journal is the truth.** ``record``
  bumps ``use_count``/``view_count``/``patch_count`` + ``last_activity_at``
  in ``<root>/.usage.json``; state transitions live in
  ``<root>/.lifecycle-journal.jsonl`` (append-only).

    python tools/skill_lifecycle.py status
    python tools/skill_lifecycle.py record my-skill --kind use
    python tools/skill_lifecycle.py sweep --idle-days 30          # plan (dry-run)
    python tools/skill_lifecycle.py sweep --idle-days 30 --live   # execute
    python tools/skill_lifecycle.py restore my-skill --live
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

SCHEMA = "fleet-skill-lifecycle/1"
SIDECAR = ".usage.json"
JOURNAL = ".lifecycle-journal.jsonl"
ARCHIVE_DIR = ".archive"
RECORD_KINDS = ("use", "view", "patch")


def default_root(workspace: Path) -> Path:
    return workspace / ".claude" / "skills"


def now_iso(now: str | None) -> str:
    if now:
        return now
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_sidecar(root: Path) -> dict[str, Any]:
    p = root / SIDECAR
    try:
        doc = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {}
    if not isinstance(doc, dict) or not isinstance(doc.get("skills"), dict):
        doc = {"schema": SCHEMA, "skills": {}}
    return doc


def save_sidecar(root: Path, doc: dict[str, Any]) -> None:
    doc["schema"] = SCHEMA
    (root / SIDECAR).write_text(
        json.dumps(doc, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def journal_append(root: Path, row: dict[str, Any]) -> None:
    with (root / JOURNAL).open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True) + "\n")


def skill_dirs(root: Path) -> list[str]:
    if not root.is_dir():
        return []
    return sorted(d.name for d in root.iterdir()
                  if d.is_dir() and d.name not in {ARCHIVE_DIR}
                  and not d.name.startswith("."))


def entry(doc: dict[str, Any], name: str) -> dict[str, Any]:
    return doc["skills"].setdefault(name, {
        "use_count": 0, "view_count": 0, "patch_count": 0,
        "last_activity_at": None, "state": "active",
        "pinned": False, "origin": "bundled"})


def cmd_record(root: Path, name: str, kind: str, now: str | None) -> dict[str, Any]:
    if name not in skill_dirs(root):
        return {"ok": False, "error": f"unknown skill: {name}"}
    doc = load_sidecar(root)
    e = entry(doc, name)
    e[f"{kind}_count"] = int(e.get(f"{kind}_count") or 0) + 1
    e["last_activity_at"] = now_iso(now)
    save_sidecar(root, doc)
    return {"ok": True, "skill": name, "kind": kind, **e}


def stale(e: dict[str, Any], idle_days: int, now: str | None) -> bool:
    last = e.get("last_activity_at")
    if not last:
        return True  # never recorded = idle since forever
    try:
        seen = datetime.strptime(str(last), "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=timezone.utc)
    except ValueError:
        return True
    ref = datetime.strptime(now_iso(now), "%Y-%m-%dT%H:%M:%SZ").replace(
        tzinfo=timezone.utc)
    return ref - seen > timedelta(days=idle_days)


def archive_target(root: Path, name: str) -> Path:
    base = root / ARCHIVE_DIR / name
    if not base.exists():
        return base
    n = 2
    while (root / ARCHIVE_DIR / f"{name}-{n}").exists():
        n += 1
    return root / ARCHIVE_DIR / f"{name}-{n}"


def transition(root: Path, name: str, action: str, reason: str,
               now: str | None) -> dict[str, Any]:
    """Move a skill dir (archive/restore) — a RENAME, never a delete."""
    doc = load_sidecar(root)
    e = entry(doc, name)
    if action == "archive":
        src, dst = root / name, archive_target(root, name)
        if not src.is_dir():
            return {"ok": False, "error": f"unknown skill: {name}"}
        if e.get("pinned"):
            return {"ok": False, "error": f"{name} is pinned — unpin before archive"}
        new_state = "archived"
    elif action == "restore":
        cand = [d for d in sorted((root / ARCHIVE_DIR).glob(f"{name}*"))
                if d.is_dir() and (d.name == name or
                                   d.name.rsplit("-", 1)[0] == name)]
        if not cand:
            return {"ok": False, "error": f"nothing archived under: {name}"}
        src, dst = cand[-1], root / name
        if dst.exists():
            return {"ok": False, "error": f"{name} already active — not overwriting"}
        new_state = "active"
    else:
        return {"ok": False, "error": f"unknown action: {action}"}
    dst.parent.mkdir(parents=True, exist_ok=True)
    src.rename(dst)
    e["state"] = new_state
    save_sidecar(root, doc)
    row = {"ts": now_iso(now), "action": action, "skill": name,
           "from": str(src.relative_to(root)), "to": str(dst.relative_to(root)),
           "reason": reason}
    journal_append(root, row)
    return {"ok": True, **row}


def cmd_sweep(root: Path, idle_days: int, live: bool,
              now: str | None) -> dict[str, Any]:
    doc = load_sidecar(root)
    proposed, exempt = [], []
    for name in skill_dirs(root):
        e = entry(doc, name)
        if e.get("pinned"):
            exempt.append({"skill": name, "why": "pinned"})
            continue
        if str(e.get("origin") or "bundled") != "agent":
            exempt.append({"skill": name, "why": "bundled-or-unmarked"})
            continue
        if not stale(e, idle_days, now):
            exempt.append({"skill": name, "why": "recently-active"})
            continue
        proposed.append(name)
    results = []
    for name in proposed:
        if live:
            results.append(transition(
                root, name, "archive",
                f"sweep: idle > {idle_days}d (agent-origin, unpinned)", now))
        else:
            results.append({"ok": True, "skill": name, "action": "would_archive"})
    return {"schema": SCHEMA, "ok": all(r.get("ok") for r in results),
            "live": live, "idle_days": idle_days,
            "proposed": proposed, "exempt_count": len(exempt),
            "exempt": exempt, "results": results}


def cmd_flag(root: Path, name: str, field: str, value: Any,
             now: str | None) -> dict[str, Any]:
    known = set(skill_dirs(root)) | set(load_sidecar(root)["skills"])
    if name not in known:
        return {"ok": False, "error": f"unknown skill: {name}"}
    doc = load_sidecar(root)
    e = entry(doc, name)
    old = e.get(field)
    e[field] = value
    save_sidecar(root, doc)
    journal_append(root, {"ts": now_iso(now), "action": f"set-{field}",
                          "skill": name, "from": old, "to": value,
                          "reason": "operator flag"})
    return {"ok": True, "skill": name, field: value}


def cmd_status(root: Path, idle_days: int, now: str | None) -> dict[str, Any]:
    doc = load_sidecar(root)
    rows = []
    for name in skill_dirs(root):
        e = entry(doc, name)
        rows.append({"skill": name, **e,
                     "stale": stale(e, idle_days, now)})
    archived = sorted(d.name for d in (root / ARCHIVE_DIR).iterdir()
                      if d.is_dir()) if (root / ARCHIVE_DIR).is_dir() else []
    return {"schema": SCHEMA, "ok": True, "root": str(root),
            "active": rows, "archived": archived}


def render_status(p: dict[str, Any]) -> str:
    lines = [f"skill-lifecycle: {len(p['active'])} active, "
             f"{len(p['archived'])} archived  ({p['root']})"]
    lines.append("  skill                          state     use view patch pin origin   last-activity        stale")
    for r in p["active"]:
        lines.append(f"  {r['skill']:<30} {r['state']:<9} {r['use_count']:>3} "
                     f"{r['view_count']:>4} {r['patch_count']:>5} "
                     f"{'yes' if r['pinned'] else ' - ':<3} {r['origin']:<8} "
                     f"{str(r['last_activity_at'] or '-'):<20} "
                     f"{'STALE' if r['stale'] else 'ok'}")
    if p["archived"]:
        lines.append(f"  archived: {', '.join(p['archived'])}")
    return "\n".join(lines)


def render_sweep(p: dict[str, Any]) -> str:
    lines = [f"skill-lifecycle sweep: live={p['live']} idle_days={p['idle_days']} "
             f"proposed={len(p['proposed'])} exempt={p['exempt_count']}"]
    for r in p["results"]:
        lines.append(f"  {r.get('skill')}: {r.get('action') or r.get('error')}")
    if not p["live"] and p["proposed"]:
        lines.append("  DRY-RUN - re-run with --live to archive (restorable)")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Witnessed skill lifecycle: telemetry, archive (never delete), restore.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--skills-root", default="", help="skills dir (default: .claude/skills)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--now", default=None, help=argparse.SUPPRESS)  # test seam
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("status"); sp.add_argument("--idle-days", type=int, default=30)
    sp = sub.add_parser("record"); sp.add_argument("skill")
    sp.add_argument("--kind", choices=RECORD_KINDS, default="use")
    sp = sub.add_parser("sweep"); sp.add_argument("--idle-days", type=int, default=30)
    sp.add_argument("--live", action="store_true")
    sp = sub.add_parser("archive"); sp.add_argument("skill")
    sp.add_argument("--reason", default="operator archive"); sp.add_argument("--live", action="store_true")
    sp = sub.add_parser("restore"); sp.add_argument("skill")
    sp = sub.add_parser("pin"); sp.add_argument("skill")
    sp = sub.add_parser("unpin"); sp.add_argument("skill")
    sp = sub.add_parser("mark"); sp.add_argument("skill")
    sp.add_argument("--origin", choices=("agent", "bundled"), required=True)
    args = ap.parse_args(argv)

    ws = Path(args.workspace).resolve() if args.workspace else Path(__file__).resolve().parent.parent
    root = Path(args.skills_root).resolve() if args.skills_root else default_root(ws)
    if not root.is_dir():
        print(json.dumps({"ok": False, "error": f"no skills root: {root}"}))
        return 1

    if args.cmd == "status":
        payload = cmd_status(root, args.idle_days, args.now)
        print(json.dumps(payload, indent=2) if args.json else render_status(payload))
    elif args.cmd == "record":
        payload = cmd_record(root, args.skill, args.kind, args.now)
        print(json.dumps(payload, indent=2))
    elif args.cmd == "sweep":
        payload = cmd_sweep(root, args.idle_days, args.live, args.now)
        print(json.dumps(payload, indent=2) if args.json else render_sweep(payload))
    elif args.cmd == "archive":
        if not args.live:
            payload = {"ok": True, "skill": args.skill, "action": "would_archive",
                       "note": "dry-run; re-run with --live"}
        else:
            payload = transition(root, args.skill, "archive", args.reason, args.now)
        print(json.dumps(payload, indent=2))
    elif args.cmd == "restore":
        payload = transition(root, args.skill, "restore", "operator restore", args.now)
        print(json.dumps(payload, indent=2))
    elif args.cmd in {"pin", "unpin"}:
        payload = cmd_flag(root, args.skill, "pinned", args.cmd == "pin", args.now)
        print(json.dumps(payload, indent=2))
    else:  # mark
        payload = cmd_flag(root, args.skill, "origin", args.origin, args.now)
        print(json.dumps(payload, indent=2))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
