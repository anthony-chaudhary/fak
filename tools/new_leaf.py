#!/usr/bin/env python3
"""Stamp a NEW conforming fak leaf — the golden path for modular growth.

fak grows by adding leaves: a new package under `fak/internal/` that depends only
on lower layers and (if it sits on the request path) registers itself against the
frozen ABI from `init()`. The architecture makes that the ONLY extension mechanism
(see fak/ARCHITECTURE.md), and architest mechanically enforces it. This script makes
the conforming shape the path of LEAST resistance: one command produces a leaf that
is green by construction — declared in the tier table, optionally slotted into the
defconfig, and carrying a passing test — so growth can never silently erode the
layered-DAG contract.

It performs three edits, each idempotent and refused if already done:
  1. creates fak/internal/<name>/{doc.go,<name>.go,<name>_test.go}
  2. inserts the package's tier into internal/architest (above the new-leaf marker)
  3. with --register, adds the blank-import line to internal/registrations (defconfig)

Pure stdlib; off the request path (DIRECTION.md tooling seam).

Usage:
  python tools/new_leaf.py canon2 --tier foundation
  python tools/new_leaf.py fedtrust --tier composer --register --summary "federated trust gate"
  python tools/new_leaf.py foo --tier mechanism --dry-run
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z][a-z0-9]*$")
TIERS = {"root": 0, "foundation": 1, "mechanism": 2, "composer": 3, "integrator": 4}
MOD = "github.com/anthony-chaudhary/fak/internal"
TIER_MARKER = "// new-leaf:tier"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def doc_go(name: str, tier: str, n: int, summary: str) -> str:
    return (
        f"// Package {name} is {summary}.\n"
        f"//\n"
        f"// Tier: {tier} ({n}) — see internal/architest. This package may import only\n"
        f"// packages whose tier is <= {n}; an upward import fails the architest gate.\n"
        f"// See fak/GROWTH.md for the layering contract and how a leaf bakes in.\n"
        f"package {name}\n"
    )


def impl_go(name: str, register: bool) -> str:
    head = f"package {name}\n"
    if register:
        head += f'\nimport "{MOD}/abi"\n'
    body = (
        "\n// Ready reports that the leaf is wired. Replace this placeholder with the\n"
        "// real surface (the type/constructor this package exists to provide).\n"
        "func Ready() bool { return true }\n"
    )
    if register:
        body += (
            "\n// init registers this leaf's driver against the frozen ABI before the\n"
            "// kernel boots. Pick the Register* that matches the leaf's role — see the\n"
            "// table in fak/ARCHITECTURE.md (RegisterAdjudicator / RegisterFastPath /\n"
            "// RegisterOp / RegisterReason / RegisterEngine / ...).\n"
            "func init() {\n"
            "\t_ = abi.ABIMinor // TODO: replace with the real abi.Register*(...) for this leaf\n"
            "}\n"
        )
    return head + body


def test_go(name: str) -> str:
    return (
        f"package {name}\n\n"
        'import "testing"\n\n'
        "func TestReady(t *testing.T) {\n"
        "\tif !Ready() {\n"
        '\t\tt.Fatal("Ready() should report true for the generated skeleton")\n'
        "\t}\n"
        "}\n"
    )


def insert_before_marker(text: str, marker: str, line: str) -> str:
    """Insert `line` (with trailing newline) immediately before the line containing
    `marker`, preserving its indentation."""
    out = []
    done = False
    for ln in text.splitlines(keepends=True):
        if not done and marker in ln:
            out.append(line)
            done = True
        out.append(ln)
    if not done:
        raise ValueError(f"marker {marker!r} not found")
    return "".join(out)


def add_registration(text: str, name: str) -> str:
    """Insert the blank-import line before the final closing paren of the import block."""
    imp = f'\t_ "{MOD}/{name}"\n'
    if imp in text:
        return text  # idempotent
    idx = text.rfind("\n)")
    if idx == -1:
        raise ValueError("could not find import block close ')' in registrations")
    return text[: idx + 1] + imp + text[idx + 1 :]


def insert_before_all_markers(text: str, marker: str, line: str) -> str:
    """Insert `line` before EVERY line containing `marker` (the lane marker appears in
    both the concurrent and autopick arrays)."""
    out, hits = [], 0
    for ln in text.splitlines(keepends=True):
        if marker in ln:
            out.append(line)
            hits += 1
        out.append(ln)
    if hits == 0:
        raise ValueError(f"marker {marker!r} not found")
    return "".join(out)


def add_leaf_lane(text: str, name: str) -> str:
    """Add the leaf's per-leaf concurrency lane to dos.toml: its name to the concurrent
    and autopick arrays (the `# new-leaf:lane` markers) and its disjoint tree to
    [lanes.trees] (the `# new-leaf:tree` marker). Idempotent."""
    # The lane tree is matched against the LIVE workspace, whose Go module is the
    # REPOSITORY ROOT (AGENTS.md) -- the real path is internal/<name>/, NOT
    # fak/internal/<name>/. Emitting the `fak/` form makes the glob match zero
    # files, so the arbiter cannot detect a collision on the leaf (dos-effective-
    # usage audit, 2026-06-22). Idempotency keys on the real-layout tree line (and
    # still treats a legacy `fak/internal/<name>/**` entry as already-present).
    if f'["internal/{name}/**"]' in text or f'fak/internal/{name}/**' in text:
        return text
    text = insert_before_all_markers(text, "# new-leaf:lane", f'  "{name}",\n')
    text = insert_before_marker(text, "# new-leaf:tree", f'{name} = ["internal/{name}/**"]\n')
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description="Stamp a new conforming fak leaf.")
    ap.add_argument("name", help="Go package name, lowercase (e.g. fedtrust)")
    ap.add_argument("--tier", required=True, choices=list(TIERS), help="layering tier")
    ap.add_argument("--register", action="store_true", help="add to the defconfig (request-path driver)")
    ap.add_argument("--summary", default="", help="one-line package summary for the doc")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    name, tier = args.name, args.tier
    n = TIERS[tier]
    summary = args.summary or f"a tier-{tier} leaf (describe its single responsibility)"

    if not NAME_RE.match(name):
        print(f"ERROR: {name!r} is not a valid lowercase Go package name", file=sys.stderr)
        return 2
    if tier == "root":
        print("ERROR: 'root' is reserved for internal/abi; pick foundation or higher", file=sys.stderr)
        return 2

    root = repo_root()
    leaf_dir = root / "fak" / "internal" / name
    architest = root / "fak" / "internal" / "architest" / "architest_test.go"
    registrations = root / "fak" / "internal" / "registrations" / "registrations.go"
    dos_toml = root / "dos.toml"

    report: dict = {"name": name, "tier": tier, "register": args.register, "dry_run": args.dry_run, "edits": []}

    if leaf_dir.exists():
        print(f"ERROR: {leaf_dir} already exists — refusing to overwrite", file=sys.stderr)
        return 2

    arch_text = architest.read_text(encoding="utf-8")
    if f'"{name}":' in arch_text:
        print(f"ERROR: tier table already declares {name!r}", file=sys.stderr)
        return 2
    new_arch = insert_before_marker(arch_text, TIER_MARKER, f'\t"{name}": {n},\n')

    new_reg = None
    if args.register:
        reg_text = registrations.read_text(encoding="utf-8")
        new_reg = add_registration(reg_text, name)

    # Every leaf is a disjoint tree -> its own dos concurrency lane (keeps the
    # partition current so the fleet can edit it in parallel). Skipped if dos.toml
    # is absent in this checkout.
    new_dos = None
    if dos_toml.exists():
        new_dos = add_leaf_lane(dos_toml.read_text(encoding="utf-8"), name)

    # --- apply (or report) ---
    files = {
        leaf_dir / "doc.go": doc_go(name, tier, n, summary),
        leaf_dir / f"{name}.go": impl_go(name, args.register),
        leaf_dir / f"{name}_test.go": test_go(name),
    }
    if not args.dry_run:
        leaf_dir.mkdir(parents=True, exist_ok=False)
        for path, content in files.items():
            path.write_text(content, encoding="utf-8")
        architest.write_text(new_arch, encoding="utf-8")
        if new_reg is not None:
            registrations.write_text(new_reg, encoding="utf-8")
        if new_dos is not None:
            dos_toml.write_text(new_dos, encoding="utf-8")

    report["edits"] = [str(p.relative_to(root)) for p in files] + [
        str(architest.relative_to(root)) + " (tier table)"
    ] + ([str(registrations.relative_to(root)) + " (defconfig)"] if args.register else []) + (
        ["dos.toml (concurrency lane)"] if new_dos is not None else []
    )
    report["next_steps"] = [
        f"implement the leaf in fak/internal/{name}/{name}.go",
        f".\\fak\\test.ps1 ./internal/{name}/ ./internal/architest/   # both green by construction",
        "the architest gate now enforces this leaf's tier on every CI run",
    ]
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
