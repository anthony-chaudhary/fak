#!/usr/bin/env python3
"""Bump fleet's version-marker file(s) in one shot — the /release Step 4 helper.

Fleet's load-bearing version marker is the bare-semver `VERSION` file at the repo
root (fak/ carries no version constant today). Cold-start install docs ALSO carry
copy-paste version pins (`INSTALL.md`: `FAK_VERSION=`, `VERSION=`, `--build-arg
APP_VERSION=`, the PowerShell `$Version`, the illustrative `e.g. X`); those drift
behind a release if nobody bumps them — exactly what stranded INSTALL.md at 0.33.0
after the 0.34.0 cut. This helper bumps both: the `version` target rewrites
`VERSION`, and the `install_docs` target pin-bumps each doc in `INSTALL_DOC_PINS`
on its install/pin lines only (the same context gate `docs_scorecard.py` reads),
so the bump set matches the gate set and the drift can't recur.

Prints a JSON report — `{new_version, dry_run, targets}` — where `targets` maps
each target to `{ok, changed, ...}` (the `install_docs` target nests a per-file
list), matching the shape `.claude/skills/release/SKILL.md` § "Expected behavior
of helpers.release_bump" documents. Idempotent (new == current is a clean no-op).
Pure stdlib.

**Single-writer gate.** Bumping `VERSION` is the moment the release mutates a
single-writer resource, so by default this refuses unless the caller holds the
release lock (`tools/release_lock.py`) — the mechanical enforcement of
`docs/production-readiness.md` item 2 ("single-writer releases"). The /release
skill acquires the lock first (Step 0.5); within one session owner identity is the
shared `$CLAUDE_CODE_SESSION_ID`, so no token need be threaded by hand. `--no-lock`
bypasses the gate for genuinely solo / CI use; a `--dry-run` never needs the lock.

Usage:
  python tools/release_bump.py 0.2.0                 # requires a held release lock
  python tools/release_bump.py 0.2.0 --owner <tok>   # gate on a specific owner
  python tools/release_bump.py 0.2.0 --dry-run       # no lock needed
  python tools/release_bump.py 0.2.0 --no-lock       # solo/CI escape hatch
  python tools/release_bump.py 0.2.0 --skip version
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:  # same tools/ dir; import works when run as `python tools/release_bump.py`
    import release_lock
except ImportError:  # pragma: no cover - defensive
    release_lock = None  # type: ignore[assignment]

SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+$")

# Cold-start docs that carry copy-paste version pins (FAK_VERSION=, VERSION=,
# --build-arg APP_VERSION=, a `$Version = "X"` PowerShell line, an "e.g. X"
# illustrative pin). A release must bump these too, or a fresh-install user
# reads a stale pin in the first doc they hit. INSTALL.md drifted to 0.33.0
# after the 0.34.0 release because the only thing this helper bumped was
# VERSION; this list closes that root cause. `docs_scorecard.py`'s freshness
# KPI gates the same surface from the read side.
INSTALL_DOC_PINS = ["INSTALL.md"]

# An install/pin LINE — the same context gate `tools/docs_scorecard.py` uses to
# tell "stale install advice" from a benchmark-lineage field or a changelog
# entry. Only version strings on these lines are bumped; a bare semver in prose
# is left alone, so the bump set == the gate set (no over-reach, no drift gap).
_INSTALL_CTX_RE = re.compile(
    r"(FAK_VERSION|[A-Za-z_]*VERSION\s*=|@v?\d+\.\d|releases/(?:download|tag|latest)|"
    r"pip install|brew install|go install|pin a version|--build-arg|prints the installed version)",
    re.IGNORECASE,
)
# A bumpable semver token: bare MAJOR.MINOR.PATCH, optionally v- or "-prefixed.
_PIN_SEMVER_RE = re.compile(r"(?P<pre>v?)(?P<ver>\d+\.\d+\.\d+)")

# Distribution manifests that pin the release version on a machine-read line a
# downstream registry parses - not prose, so they drift silently. server.json is
# the Official MCP Registry manifest (its `version` and the `oci` package
# `identifier`/`version` must match the ghcr image release-container.yml pushes,
# or the registry lists a back-version / fails image validation). CITATION.cff is
# the academic-citation card (its `version`/`date-released`). Both stranded
# behind a release (server.json at 0.33.0, CITATION.cff at 0.30.0 after the
# 0.34.0 cut) because the only thing this helper bumped was VERSION + INSTALL.md;
# this list closes that root cause. docs/fak/mcp-registry.md's "Updating on each
# release" step prescribes the server.json bump by hand - this automates it. The
# whole-line context guard mirrors INSTALL_DOC_PINS: only a semver on a pin line
# is rewritten, so a description that happens to contain a version is left alone.
DIST_MANIFEST_PINS = ["server.json", "CITATION.cff"]

# A pin LINE in a distribution manifest - a JSON `"version":`/`"identifier":`
# field or a YAML `version:`/`date-released:` key. Only semvers (and the
# CITATION date) on these lines are bumped.
_DIST_CTX_RE = re.compile(
    r"(\"version\"\s*:|\"identifier\"\s*:|^version\s*:|^date-released\s*:)",
)
# An ISO date token (CITATION.cff `date-released`).
_ISO_DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bump_plain_file(root: Path, rel: str, new: str, *, dry_run: bool) -> dict:
    path = root / rel
    if not path.exists():
        return {"path": rel, "ok": False, "reason": f"{rel} not found"}
    old = path.read_text(encoding="utf-8").strip()
    target = f"{new}\n"
    changed = path.read_text(encoding="utf-8") != target
    if changed and not dry_run:
        path.write_text(target, encoding="utf-8")
    return {"path": rel, "old": old, "new": new, "changed": changed, "ok": True}


def bump_install_doc(root: Path, rel: str, new: str, *, dry_run: bool) -> dict:
    """Bump version pins on install/pin lines in a cold-start doc.

    Only rewrites a semver that sits on a line `_INSTALL_CTX_RE` recognizes as
    install advice, so a benchmark figure or a release-history mention in the
    same file is never touched. Idempotent: a doc already at `new` reports
    changed=False with an empty `replaced`.
    """
    path = root / rel
    if not path.exists():
        return {"path": rel, "ok": False, "reason": f"{rel} not found"}
    replaced: list[str] = []
    out_lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not _INSTALL_CTX_RE.search(line):
            out_lines.append(line)
            continue

        def _sub(m: "re.Match[str]") -> str:
            if m.group("ver") == new:
                return m.group(0)
            replaced.append(f"{m.group('ver')}->{new}")
            return f"{m.group('pre')}{new}"

        out_lines.append(_PIN_SEMVER_RE.sub(_sub, line))
    new_text = "".join(out_lines)
    changed = new_text != path.read_text(encoding="utf-8")
    if changed and not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return {"path": rel, "new": new, "changed": changed, "replaced": replaced, "ok": True}


def bump_dist_manifest(root: Path, rel: str, new: str, *, date, dry_run: bool) -> dict:
    """Bump the release version (and CITATION date) in a distribution manifest.

    Only rewrites a semver on a line `_DIST_CTX_RE` recognizes as a version/identifier
    pin, so the project description or abstract is never touched. For CITATION.cff,
    a `date-released:` line is updated to `date` when one is supplied (a bump with no
    `--date` leaves the date alone, staying idempotent and pure). Returns the same
    shape as `bump_install_doc`: {path, new, changed, replaced, ok}.
    """
    path = root / rel
    if not path.exists():
        return {"path": rel, "ok": False, "reason": f"{rel} not found"}
    replaced = []
    out_lines = []
    for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
        if not _DIST_CTX_RE.search(line):
            out_lines.append(line)
            continue
        if "date-released" in line:
            if date is not None:
                def _sub_date(m):
                    if m.group(0) == date:
                        return m.group(0)
                    replaced.append(f"{m.group(0)}->{date}")
                    return date
                line = _ISO_DATE_RE.sub(_sub_date, line)
            out_lines.append(line)
            continue

        def _sub(m):
            if m.group("ver") == new:
                return m.group(0)
            replaced.append(f"{m.group('ver')}->{new}")
            return f"{m.group('pre')}{new}"

        out_lines.append(_PIN_SEMVER_RE.sub(_sub, line))
    new_text = "".join(out_lines)
    changed = new_text != path.read_text(encoding="utf-8")
    if changed and not dry_run:
        path.write_text(new_text, encoding="utf-8")
    return {"path": rel, "new": new, "changed": changed, "replaced": replaced, "ok": True}


def main() -> int:
    ap = argparse.ArgumentParser(description="Bump fleet version-marker file(s).")
    ap.add_argument("version", help="New semver, e.g. 0.2.0 (no leading v)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--date", default=None, help="Release date YYYY-MM-DD (updates CITATION.cff date-released; omit to leave dates alone)")
    ap.add_argument("--skip", action="append", default=[], choices=["version", "install_docs", "dist_manifests"], help="Skip a target (repeatable)")
    ap.add_argument("--owner", default=None, help="release-lock owner to gate on (default: session id)")
    ap.add_argument("--no-lock", action="store_true", help="bypass the single-writer release lock (solo/CI only)")
    args = ap.parse_args()

    new_version = args.version.lstrip("v")
    if not SEMVER_RE.match(new_version):
        print(f"ERROR: {args.version!r} is not a valid semver (expected X.Y.Z)", file=sys.stderr)
        return 2

    root = repo_root()

    # Single-writer gate: a real bump mutates VERSION, so require the release lock.
    # A dry-run mutates nothing, so it never needs the lock.
    if not args.no_lock and not args.dry_run:
        if release_lock is None:
            print(json.dumps({"ok": False, "reason": "release_lock helper unavailable; pass --no-lock to override"}, indent=2))
            return 4
        owner = args.owner or release_lock.default_owner()
        held, why = release_lock.held_by(root, owner)
        if not held:
            print(json.dumps({
                "ok": False,
                "reason": f"release lock not held ({why})",
                "hint": "acquire it first: python tools/release_lock.py acquire   (or pass --no-lock for solo/CI)",
                "owner_checked": owner,
            }, indent=2))
            return 4
    if args.date is not None and not _ISO_DATE_RE.fullmatch(args.date):
        print(f"ERROR: --date {args.date!r} is not YYYY-MM-DD", file=sys.stderr)
        return 2

    def _install_docs() -> dict:
        files = [bump_install_doc(root, rel, new_version, dry_run=args.dry_run) for rel in INSTALL_DOC_PINS]
        return {"ok": all(f.get("ok", False) for f in files), "files": files}

    def _dist_manifests() -> dict:
        files = [bump_dist_manifest(root, rel, new_version, date=args.date, dry_run=args.dry_run) for rel in DIST_MANIFEST_PINS]
        return {"ok": all(f.get("ok", False) for f in files), "files": files}

    actions = {
        "version": lambda: bump_plain_file(root, "VERSION", new_version, dry_run=args.dry_run),
        "install_docs": _install_docs,
        "dist_manifests": _dist_manifests,
    }

    report: dict = {"new_version": new_version, "dry_run": args.dry_run, "targets": {}}
    any_error = False
    for key, fn in actions.items():
        if key in args.skip:
            report["targets"][key] = {"skipped": True, "reason": "--skip"}
            continue
        try:
            result = fn()
        except Exception as exc:  # surface but don't crash — the skill can still commit manually
            result = {"ok": False, "reason": f"{type(exc).__name__}: {exc}"}
        report["targets"][key] = result
        if not result.get("ok", True):
            any_error = True

    print(json.dumps(report, indent=2))
    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
