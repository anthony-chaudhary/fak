#!/usr/bin/env python3
"""provision_dos_hook.py -- drop the native ``dos-hook`` binary into the gitignored
``tools/.bin`` so ``tools/dos_hook.py`` takes its fast path (issue #2705).

The native ``dos-hook`` source lives in the SEPARATE dos-kernel repo, not fak -- so
unlike ``repoguard`` (built by ``make build`` from ``./cmd/repoguard``) fak cannot
``go build`` it in-tree. This provisioner instead **copies the right prebuilt** from the
enabled dos plugin's ``claude-plugin/bin`` (which ships one binary per host triple --
``dos-hook-windows-amd64.exe``, ``dos-hook-linux-arm64``, ...), or, failing that,
``go build``s it from a discoverable dos Go module (``<dos-kernel>/go``).

Discovery is env-first, then sibling-checkout, then plugin install -- no host path is
hard-coded:
  * ``DOS_KERNEL`` / ``DOS_PLUGIN_ROOT`` env -> an explicit dos-kernel checkout or plugin root;
  * ``CLAUDE_PLUGIN_ROOT`` env -> the plugin root when run inside a hook;
  * sibling checkouts next to the fak repo (``../dos-kernel*``);
  * installed Claude Code plugins (``~/.claude/plugins/repos/**/dos-kernel*``).

Absent every source this exits 1 with guidance -- NOT a hard failure of the world: the
launcher simply keeps using its Python fallback (``python -m dos.cli hook``). So this is
a convenience/perf provisioner, never a correctness gate; it is deliberately NOT in
``make ci``.
"""
from __future__ import annotations

import argparse
import glob
import os
import platform
import shutil
import subprocess
from pathlib import Path


def _no_window_creationflags() -> int:
    """``creationflags`` that stop the ``go build`` console child below from popping a
    visible window when this provisions windowless (pythonw) from a scheduled dispatch
    tick; ``0`` on POSIX. Mirrors dispatch_worker.no_window_creationflags, kept local so
    this provisioner imports only stdlib."""
    return 0x08000000 if os.name == "nt" else 0


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def host_triple() -> tuple:
    """(os, arch, exe_suffix) in the dos plugin's naming, e.g. ('windows','amd64','.exe')."""
    sysname = {"windows": "windows", "linux": "linux", "darwin": "darwin"}.get(
        platform.system().lower(), platform.system().lower()
    )
    machine = platform.machine().lower()
    arch = "arm64" if machine in ("arm64", "aarch64") else "amd64"
    suffix = ".exe" if sysname == "windows" else ""
    return sysname, arch, suffix


def candidate_roots(env: dict) -> list:
    """Ordered dos-kernel / plugin roots to probe for a bin dir or Go module."""
    roots: list = []
    for key in ("DOS_KERNEL", "DOS_PLUGIN_ROOT", "CLAUDE_PLUGIN_ROOT"):
        v = env.get(key)
        if v:
            roots.append(Path(v))
    # Sibling checkouts next to the fak repo (fak=C:/work/fak, dos=C:/work/dos-kernel*).
    parent = repo_root().parent
    for pat in ("dos-kernel*", "dos", "dos-public", "dos-kernel-public"):
        roots.extend(Path(p) for p in glob.glob(str(parent / pat)))
    # Installed Claude Code plugins.
    home = Path(env.get("USERPROFILE") or env.get("HOME") or os.path.expanduser("~"))
    roots.extend(
        Path(p) for p in glob.glob(str(home / ".claude" / "plugins" / "repos" / "**" / "dos-kernel*"), recursive=True)
    )
    # De-dupe, keep order, existing dirs only.
    seen: set = set()
    out: list = []
    for r in roots:
        rp = str(r)
        if rp not in seen and r.exists():
            seen.add(rp)
            out.append(r)
    return out


def find_prebuilt(root: Path, sysname: str, arch: str, suffix: str) -> Path | None:
    """A shipped per-arch binary under ``<root>/**/bin/dos-hook-<os>-<arch><suffix>``."""
    name = f"dos-hook-{sysname}-{arch}{suffix}"
    for bindir in (root / "claude-plugin" / "bin", root / "bin", root):
        cand = bindir / name
        if cand.is_file():
            return cand
    hits = glob.glob(str(root / "**" / "bin" / name), recursive=True)
    return Path(hits[0]) if hits else None


def find_go_module(root: Path) -> Path | None:
    """A dos Go module (``go.mod`` with a ``cmd/dos-hook`` package) under ``root``."""
    for gm in (root / "go", root):
        if (gm / "go.mod").is_file() and (gm / "cmd" / "dos-hook").is_dir():
            return gm
    return None


def provision(dest: Path, env: dict, log=print) -> int:
    sysname, arch, suffix = host_triple()
    roots = candidate_roots(env)
    if not roots:
        log("dos-hook: no dos-kernel checkout or plugin found; launcher stays on the Python fallback.")
        log("          set DOS_KERNEL=<path to dos-kernel checkout> and re-run 'make dos-hook'.")
        return 1

    dest.parent.mkdir(parents=True, exist_ok=True)

    # 1) Prefer a shipped prebuilt (no toolchain needed).
    for root in roots:
        pre = find_prebuilt(root, sysname, arch, suffix)
        if pre:
            shutil.copy2(pre, dest)
            os.chmod(dest, 0o755)
            log(f"dos-hook: copied {pre} -> {dest}")
            return 0

    # 2) Else build from a dos Go module.
    for root in roots:
        gm = find_go_module(root)
        if gm and shutil.which("go"):
            log(f"dos-hook: building from {gm} ...")
            proc = subprocess.run(
                ["go", "build", "-o", str(dest), "./cmd/dos-hook"], cwd=str(gm),
                creationflags=_no_window_creationflags(),
            )
            if proc.returncode == 0 and dest.is_file():
                log(f"dos-hook: built -> {dest}")
                return 0
            log("dos-hook: go build failed; leaving launcher on the Python fallback.")

    log(f"dos-hook: found dos roots {[str(r) for r in roots]} but no prebuilt for "
        f"{sysname}-{arch} and no buildable Go module; launcher stays on the Python fallback.")
    return 1


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description="Provision the native dos-hook binary into tools/.bin.")
    ap.add_argument("--dest", default=None, help="destination path (default tools/.bin/dos-hook[.exe])")
    args = ap.parse_args(argv)
    _, _, suffix = host_triple()
    dest = Path(args.dest) if args.dest else repo_root() / "tools" / ".bin" / f"dos-hook{suffix}"
    return provision(dest, os.environ)


if __name__ == "__main__":
    raise SystemExit(main())
