#!/usr/bin/env python3
"""Pull the PRIVATE scan instructions into this public clone (gitignored sidecar).

    python tools/pull_scan_needles.py                       # auto-find sibling ../fleet
    python tools/pull_scan_needles.py --from /path/to/fleet # explicit private repo
    python tools/pull_scan_needles.py --status              # report current sidecar
    python tools/pull_scan_needles.py --check --json        # self-healing loop status
    python tools/pull_scan_needles.py --dump                # emit the canonical needle artifact

Why this exists -- the HARD CUT (see PLAN-hard-cut-private-public-2026-06-20.md):
once the public copy is edited DIRECTLY instead of being regenerated from the
private repo, the public tree needs its own standing leak scan
(``scrub_public_copy.py --audit-tree``). But the public copy's committed
``EXPORT_AUDIT_NEEDLES`` are DE-FANGED -- the export scrub rewrote the real
high-sensitivity values (operator IPs, lab host, Slack ids, SSH password) to
generic CGNAT-range / placeholder values the policy KEEPS in public. A scan
against that de-fanged list would cry wolf on kept placeholders AND miss a
freshly-pasted real secret.

The REAL needle values live only in the PRIVATE repo. This tool reads them and
writes them to ``tools/_registry/scrub_needles.private.json`` -- a path
``tools/_registry`` that is GITIGNORED in both repos, so the real values sit on
disk locally (like a ``.env``) and are never committed. ``--audit-tree`` then
runs in ``full`` mode.

Needle source, in preference order (so a TRIMMED or NEW minimal private repo does
not need the whole scrub script -- the durable artifact is enough):
  1. ``<private>/scrub_needles.json`` -- the canonical, script-independent artifact
     (emitted by ``--dump`` / written by ``tools/init_private_repo.py``);
  2. ``<private>/tools/scrub_public_copy.py`` -- import the module's needle lists.

Idempotent. Pure stdlib. Exit 0 ok, 1 (``--check`` only) needles pullable but not
pulled, 2 precondition (no private repo / no needles), 4 internal.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PUBLIC_ROOT = os.path.dirname(HERE)  # tools/ -> repo root
SIDECAR_REL = os.path.join("tools", "_registry", "scrub_needles.private.json")
CANONICAL_REL = "scrub_needles.json"  # at the private repo ROOT: the durable artifact
SCHEMA = "fleet-scrub-needles/1"


def _default_private_root() -> str:
    """Best-effort sibling private repo: ../fleet next to this public clone."""
    return os.path.join(os.path.dirname(PUBLIC_ROOT), "fleet")


def _load_private_module(private_root: str):
    """Import the private repo's scrub_public_copy.py as a module (no main() run)."""
    src = os.path.join(private_root, "tools", "scrub_public_copy.py")
    if not os.path.isfile(src):
        return None, f"private scrubber not found: {src}"
    spec = importlib.util.spec_from_file_location("private_scrub", src)
    if spec is None or spec.loader is None:
        return None, f"cannot load module spec from {src}"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:  # noqa: BLE001 - report any import failure to the operator
        return None, f"failed importing {src}: {exc}"
    return module, None


def read_private_needles(private_root: str):
    """Return ``(audit, export, source, err)`` -- the REAL needle lists.

    Prefers the canonical ``scrub_needles.json`` artifact at the private repo
    root (script-independent, so a trimmed/new private repo carries just it);
    falls back to importing the private ``scrub_public_copy.py`` module.
    """
    canonical = os.path.join(private_root, CANONICAL_REL)
    if os.path.isfile(canonical):
        try:
            with open(canonical, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            return None, None, None, f"unreadable {canonical}: {exc}"
        return (list(data.get("audit_needles") or []),
                list(data.get("export_audit_needles") or []),
                canonical, None)
    module, err = _load_private_module(private_root)
    if err:
        return None, None, None, err
    return (list(getattr(module, "AUDIT_NEEDLES", []) or []),
            list(getattr(module, "EXPORT_AUDIT_NEEDLES", []) or []),
            os.path.join(private_root, "tools", "scrub_public_copy.py"), None)


def status(public_root: str) -> int:
    path = os.path.join(public_root, SIDECAR_REL)
    if not os.path.isfile(path):
        print(f"scan-needles: NOT PULLED (no {SIDECAR_REL}) -- audit-tree runs shape-only")
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"scan-needles: present but unreadable: {exc}", file=sys.stderr)
        return 4
    print(
        f"scan-needles: PULLED from {data.get('source')!r} -- "
        f"{len(data.get('audit_needles') or [])} audit / "
        f"{len(data.get('export_audit_needles') or [])} export needles"
    )
    return 0


def check(public_root: str, private_opt: str | None, as_json: bool) -> int:
    """Self-healing loop status: is the sidecar pulled, and is it pullable here?

    ``ok`` (exit 0) when the sidecar is present OR no private repo is reachable on
    this box (shape-only is the best available -- do NOT nag). NOT ok (exit 1 ->
    ACTION) only when a private repo IS reachable but the needles are not pulled,
    so the loop's ``auto_recover`` can pull them. This keeps the public leak scan
    in ``full`` mode automatically on boxes that can, with no noise on boxes that
    can't.
    """
    sidecar = os.path.join(public_root, SIDECAR_REL)
    pulled = os.path.isfile(sidecar)
    private_root = os.path.abspath(private_opt or _default_private_root())
    private_reachable = os.path.isdir(private_root)
    ok = pulled or not private_reachable
    mode = ("full" if pulled
            else "shape-only-pullable" if private_reachable
            else "shape-only-no-private")
    reason = ("needles pulled (full mode)" if pulled
              else "private repo reachable but needles not pulled -- run pull"
              if private_reachable
              else "no private repo on this box -- shape-only is best available")
    payload = {
        "schema": "fleet-scan-needles-check/1",
        "ok": ok,
        "pulled": pulled,
        "private_reachable": private_reachable,
        "private_root": private_root,
        "mode": mode,
        "reason": reason,
    }
    if as_json:
        print(json.dumps(payload))
    else:
        print(f"scan-needles check: ok={ok} pulled={pulled} "
              f"private_reachable={private_reachable} -- {reason}")
    return 0 if ok else 1


def dump(private_root: str) -> int:
    """Emit the canonical needle artifact (scrub_needles.json) to stdout."""
    audit, export, source, err = read_private_needles(private_root)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if not export:
        print(f"ERROR: no EXPORT_AUDIT_NEEDLES from {source}", file=sys.stderr)
        return 2
    print(json.dumps({
        "schema": SCHEMA,
        "audit_needles": audit,
        "export_audit_needles": export,
    }, indent=2))
    return 0


def pull(private_root: str, public_root: str) -> int:
    audit, export, source, err = read_private_needles(private_root)
    if err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 2
    if not export:
        print(f"ERROR: no EXPORT_AUDIT_NEEDLES from {source}", file=sys.stderr)
        return 2

    payload = {
        "schema": SCHEMA,
        "source": source,
        "note": ("REAL operator needles pulled from the private repo. Gitignored "
                 "(tools/_registry). NEVER commit. Consumed by "
                 "scrub_public_copy.py --audit-tree."),
        "audit_needles": audit,
        "export_audit_needles": export,
    }
    out = os.path.join(public_root, SIDECAR_REL)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out)
    print(f"pulled {len(audit)} audit / {len(export)} export needles from {source}")
    print(f"  -> {SIDECAR_REL} (gitignored)")
    print("  audit-tree now runs in FULL mode; "
          "verify: python tools/scrub_public_copy.py --audit-tree --root .")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from", dest="private", default=None,
                    help="private canonical repo root (default: sibling ../fleet)")
    ap.add_argument("--public-dir", default=PUBLIC_ROOT,
                    help="public clone root (default: this repo)")
    ap.add_argument("--status", action="store_true",
                    help="report the current sidecar instead of pulling")
    ap.add_argument("--check", action="store_true",
                    help="self-healing loop status (ok unless pullable-but-unpulled)")
    ap.add_argument("--dump", action="store_true",
                    help="emit the canonical needle artifact (scrub_needles.json) to stdout")
    ap.add_argument("--json", action="store_true", help="machine-readable output (for --check)")
    args = ap.parse_args(argv)

    public_root = os.path.abspath(args.public_dir)
    if args.status:
        return status(public_root)
    if args.check:
        return check(public_root, args.private, args.json)

    private_root = os.path.abspath(args.private or _default_private_root())
    if args.dump:
        return dump(private_root)
    if not os.path.isdir(private_root):
        print(f"ERROR: private repo not found: {private_root}\n"
              f"  pass --from <private-repo-root>; on a box without the private "
              f"repo, audit-tree stays shape-only (degraded but honest).",
              file=sys.stderr)
        return 2
    return pull(private_root, public_root)


if __name__ == "__main__":
    raise SystemExit(main())
