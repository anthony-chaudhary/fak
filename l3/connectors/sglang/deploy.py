#!/usr/bin/env python3
"""
deploy.py -- Deploy CAMA module + integration patches into an SGLang tree.

This script does up to three things depending on mode:

  1. MODULE:  Copies cama_module/ into SGLang's storage backend directory
             so SGLang can import CamaStorage.

             cama-connector/cama_module/*
               -> <sglang>/python/sglang/srt/mem_cache/storage/cama/

  2. PATCHES: Overwrites four SGLang source files with patched versions
             that wire CAMA into the backend system (env vars, CLI args,
             factory registration, cache controller zero-copy path).

             cama-connector/patches/<file>
               -> <sglang>/python/sglang/srt/.../<file>

  3. SETUP:  Creates a Python venv, installs SGLang (editable) + KV cache
             client into it, deploys CAMA, and verifies everything imports.

Modes:
    python deploy.py /path/to/sglang                # module + patches (default)
    python deploy.py /path/to/sglang --module       # module only
    python deploy.py /path/to/sglang --patch        # patches only
    python deploy.py /path/to/sglang --diff         # dry-run: show what differs
    python deploy.py /path/to/sglang --zip          # deploy + create zip archive
    python deploy.py /path/to/sglang --setup        # full setup: venv + install + deploy

Setup options (only with --setup):
    --venv-dir DIR       Where to create the venv (default: <sglang>/.venv)
    --client CLIENT      KV cache client: cama-client or priskv (default: cama-client)
    --fresh              Remove existing venv before creating a new one
    --find-links URL     Extra --find-links for pip (default: flashinfer wheel index)
"""

from __future__ import annotations

import argparse
import filecmp
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

import cama_patchlib

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
VERSION_FILE = SCRIPT_DIR / "VERSION"
MODULE_DIR = SCRIPT_DIR / "cama_module"
PATCHES_DIR = SCRIPT_DIR / "patches"

FLASHINFER_FIND_LINKS = "https://flashinfer.ai/whl/cu124/torch2.5/"

# Files copied from cama_module/ -> <sglang>/.../storage/cama/
MODULE_FILES = [
    "cama_storage.py",
    "preflight.py",
    "prewarm.py",
    "profiling.py",
    "__init__.py",
]

# Patch file -> relative path inside the SGLang tree where it belongs.
# Each patch is a complete replacement for the corresponding SGLang file.
#
# This is read from cama-connector/patch_manifest.json (single source of truth)
# rather than hardcoded here. The old hardcoded list silently omitted four
# patched files (schedule_policy.py, scheduler_metrics_mixin.py,
# hicache_storage.py, hiradix_cache.py), so deploy.py would drop those fixes on
# a fresh SGLang tree. Run scripts/find_cama_patches.py to verify the manifest
# stays complete against the pinned upstream base.
PATCH_MAP = cama_patchlib.patch_map()


# ── output helpers ─────────────────────────────────────────────────────
def _c(code: str, msg: str) -> str:
    return f"\033[{code}m{msg}\033[0m"

def info(msg: str)  -> None: print(_c("0;36", f"[info]  {msg}"))
def ok(msg: str)    -> None: print(_c("0;32", f"[ok]    {msg}"))
def warn(msg: str)  -> None: print(_c("1;33", f"[warn]  {msg}"))
def error(msg: str) -> None: print(_c("0;31", f"[error] {msg}"), file=sys.stderr)

def run(cmd: list[str], check: bool = True, **kwargs) -> subprocess.CompletedProcess:
    """Run a command, print it, and optionally check return code."""
    display = " ".join(str(c) for c in cmd)
    # Truncate long commands for display
    if len(display) > 120:
        display = display[:117] + "..."
    info(f"$ {display}")
    result = subprocess.run(cmd, **kwargs)
    if check and result.returncode != 0:
        error(f"Command failed (exit {result.returncode})")
        sys.exit(result.returncode)
    return result


# ── setuptools-scm version detection ───────────────────────────────────
def detect_scm_version(sglang_dir: Path) -> str | None:
    """Try to determine the SGLang version that setuptools-scm would produce.

    setuptools-scm runs `git describe --tags` from the repo root.  This fails
    when the tree is a zip/copy without .git, a shallow clone without tags, or
    a renamed directory that breaks the `root = ".."` relative path in
    pyproject.toml.

    Returns a version string to use as SETUPTOOLS_SCM_PRETEND_VERSION, or
    None if git describe works fine and no workaround is needed.
    """
    # Check if pyproject.toml even uses setuptools-scm
    pyproject = sglang_dir / "python" / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(errors="replace")
        if "setuptools_scm" not in text and "setuptools-scm" not in text:
            return None  # not using scm, no issue

    # Check if git describe works from the sglang root
    result = subprocess.run(
        ["git", "describe", "--tags", "--long", "--match", "v*"],
        cwd=str(sglang_dir),
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return None  # git describe works, no workaround needed

    # Try to get something useful from git tags
    tag_result = subprocess.run(
        ["git", "tag", "-l", "v*"],
        cwd=str(sglang_dir),
        capture_output=True,
        text=True,
    )
    if tag_result.returncode == 0 and tag_result.stdout.strip():
        # Use the last tag as the version
        tags = tag_result.stdout.strip().splitlines()
        return tags[-1].lstrip("v")

    # Check if there's a fallback_version in pyproject.toml
    if pyproject.is_file():
        text = pyproject.read_text(errors="replace")
        if "fallback_version" in text:
            return None  # pyproject.toml has its own fallback

    # No git, no tags, no fallback -- use a sensible default
    # Try to extract from directory name (e.g. "sglang-v0.5.7" -> "0.5.7")
    name = sglang_dir.name.lower()
    for prefix in ("sglang-v", "sglang-", "sg-lang-", "v"):
        if name.startswith(prefix):
            candidate = name[len(prefix):].split("-")[0]
            if candidate and candidate[0].isdigit():
                return candidate

    return "0.0.0"


# ── validation ─────────────────────────────────────────────────────────
def read_version() -> str:
    try:
        return VERSION_FILE.read_text().strip()
    except FileNotFoundError:
        return "unknown"


def validate_sglang_dir(sglang_dir: Path) -> Path:
    """Return the resolved sglang dir, or exit if it doesn't look right."""
    sglang_dir = sglang_dir.resolve()
    srt = sglang_dir / "python" / "sglang" / "srt"
    if not srt.is_dir():
        error(f"{sglang_dir} does not look like an SGLang tree")
        error(f"  expected to find: {srt}")
        sys.exit(1)
    return sglang_dir


def cama_target(sglang_dir: Path) -> Path:
    return sglang_dir / "python" / "sglang" / "srt" / "mem_cache" / "storage" / "cama"


# ── diff ───────────────────────────────────────────────────────────────
def do_diff(sglang_dir: Path) -> int:
    """Show what would change. Returns count of differing files."""
    target = cama_target(sglang_dir)
    changes = 0

    info("=== Module files (cama_module/ -> storage/cama/) ===")
    for name in MODULE_FILES:
        src = MODULE_DIR / name
        dst = target / name
        if not src.is_file():
            warn(f"  SKIP  {name}  (not in cama_module/)")
            continue
        if not dst.is_file():
            print(_c("0;32", f"  + NEW  {name}"))
            changes += 1
        elif not filecmp.cmp(str(src), str(dst), shallow=False):
            print(_c("1;33", f"  ~ MOD  {name}"))
            changes += 1
        else:
            print(_c("0;36", f"  = OK   {name}"))

    print()
    info("=== Integration patches (patches/ -> SGLang source files) ===")
    for patch_name, rel_path in PATCH_MAP.items():
        src = PATCHES_DIR / patch_name
        dst = sglang_dir / rel_path
        if not src.is_file():
            print(_c("0;36", f"  - SKIP  {patch_name}  (no patch file)"))
            continue
        if not dst.is_file():
            print(_c("0;31", f"  ! MISS  {rel_path}  (not found in SGLang)"))
            continue
        if not filecmp.cmp(str(src), str(dst), shallow=False):
            print(_c("1;33", f"  ~ DIFF  {patch_name}  ->  {rel_path}"))
            changes += 1
        else:
            print(_c("0;36", f"  = OK    {patch_name}"))

    print()
    if changes == 0:
        ok("Everything is in sync -- no changes needed.")
    else:
        warn(f"{changes} file(s) differ.")
    return changes


# ── deploy module ──────────────────────────────────────────────────────
def do_module(sglang_dir: Path) -> None:
    """Copy cama_module/ files into the SGLang storage backend directory."""
    target = cama_target(sglang_dir)
    target.mkdir(parents=True, exist_ok=True)

    for name in MODULE_FILES:
        src = MODULE_DIR / name
        if not src.is_file():
            warn(f"Skipping {name} (not found in cama_module/)")
            continue
        shutil.copy2(str(src), str(target / name))
        ok(f"Copied  {name}")

    # Copy VERSION so __init__.py can read it after deployment
    if VERSION_FILE.is_file():
        shutil.copy2(str(VERSION_FILE), str(target / "VERSION"))
        ok(f"Copied  VERSION")

    ok(f"Module deployed -> {target}")


# ── deploy patches ─────────────────────────────────────────────────────
def do_patch(sglang_dir: Path) -> None:
    """Overwrite SGLang source files with patched versions.

    Each patch file is a *complete replacement* for the corresponding
    SGLang source file. Before overwriting, the original is backed up
    to <file>.orig (once only -- subsequent runs do not re-backup).
    """
    for patch_name, rel_path in PATCH_MAP.items():
        src = PATCHES_DIR / patch_name
        dst = sglang_dir / rel_path

        if not src.is_file():
            warn(f"Skipping {patch_name} (not in patches/)")
            continue
        if not dst.is_file():
            warn(f"Target not found: {rel_path} -- skipping")
            continue

        # One-time backup of the original SGLang file
        orig = dst.with_suffix(dst.suffix + ".orig")
        if not orig.exists():
            shutil.copy2(str(dst), str(orig))
            info(f"Backed up  {rel_path} -> .orig")

        shutil.copy2(str(src), str(dst))
        ok(f"Patched  {rel_path}")

    ok(f"Integration patches applied -> {sglang_dir}")


# ── zip ────────────────────────────────────────────────────────────────
def do_zip(sglang_dir: Path, version: str) -> None:
    """Create a zip archive of the SGLang tree."""
    base_name = sglang_dir.name
    zip_name = f"{base_name}-with-cama-{version}"
    zip_path = SCRIPT_DIR / zip_name

    info(f"Creating archive: {zip_name}.zip")
    shutil.make_archive(
        str(zip_path),
        "zip",
        root_dir=str(sglang_dir.parent),
        base_dir=base_name,
    )

    final = zip_path.with_suffix(".zip")
    size_mb = final.stat().st_size / (1024 * 1024)
    ok(f"Archive created: {final.name} ({size_mb:.1f} MB)")


# ── venv + install ─────────────────────────────────────────────────────
def find_venv_python(venv_dir: Path) -> Path:
    """Return path to the venv's python binary (Linux or Windows)."""
    for candidate in [
        venv_dir / "bin" / "python",
        venv_dir / "Scripts" / "python.exe",
    ]:
        if candidate.exists():
            return candidate
    error(f"Could not find python in venv: {venv_dir}")
    sys.exit(1)


def do_setup(
    sglang_dir: Path,
    venv_dir: Path,
    client: str,
    fresh: bool,
    find_links: str,
) -> None:
    """Full setup: create venv, install SGLang + client, deploy CAMA, verify.

    Steps performed:
      1. Create a Python virtual environment
      2. Upgrade pip + install setuptools/wheel
      3. Install SGLang (editable) into the venv
      4. Deploy CAMA module + patches into the SGLang tree
      5. Install the KV cache client (cama-client or priskv)
      6. Verify all critical imports
    """
    # ── Step 1: Create venv ────────────────────────────────────────────
    print()
    info("=" * 60)
    info("Step 1/6: Create virtual environment")
    info("=" * 60)

    if fresh and venv_dir.exists():
        info(f"Removing existing venv: {venv_dir}")
        shutil.rmtree(venv_dir)

    if venv_dir.exists():
        info(f"Venv already exists: {venv_dir}")
    else:
        info(f"Creating venv: {venv_dir}")
        venv.create(str(venv_dir), with_pip=True)
    ok(f"Venv ready: {venv_dir}")

    vpy = find_venv_python(venv_dir)
    info(f"Venv python: {vpy}")

    # ── Step 2: Upgrade pip ────────────────────────────────────────────
    print()
    info("=" * 60)
    info("Step 2/6: Upgrade pip + build tools")
    info("=" * 60)

    run([str(vpy), "-m", "pip", "install", "--upgrade",
         "pip", "setuptools>=68.0", "wheel"])

    # ── Step 3: Install SGLang ─────────────────────────────────────────
    print()
    info("=" * 60)
    info("Step 3/6: Install SGLang (editable) into venv")
    info("=" * 60)
    info(f"This installs torch, sgl-kernel, flashinfer, and all other")
    info(f"SGLang dependencies. May take several minutes on first run.")

    # setuptools-scm needs `git describe --tags` to work.  Pre-packaged
    # trees (zips, copies, renamed dirs) often lack .git or tags, causing:
    #   LookupError: setuptools-scm was unable to detect version for ...
    # Fix: set SETUPTOOLS_SCM_PRETEND_VERSION so scm skips git entirely.
    pip_env = None
    pretend_version = detect_scm_version(sglang_dir)
    if pretend_version is not None:
        warn(f"git describe failed for this SGLang tree (zip/copy/no tags)")
        info(f"Setting SETUPTOOLS_SCM_PRETEND_VERSION={pretend_version}")
        pip_env = {**os.environ, "SETUPTOOLS_SCM_PRETEND_VERSION": pretend_version}

    sglang_pkg = str(sglang_dir / "python")
    pip_cmd = [
        str(vpy), "-m", "pip", "install", "-e", sglang_pkg,
    ]
    if find_links:
        pip_cmd += ["--find-links", find_links]

    run(pip_cmd, env=pip_env)

    # ── Step 4: Deploy CAMA ────────────────────────────────────────────
    print()
    info("=" * 60)
    info("Step 4/6: Deploy CAMA module + patches")
    info("=" * 60)

    do_module(sglang_dir)
    print()
    do_patch(sglang_dir)

    # ── Step 5: Install KV cache client ────────────────────────────────
    print()
    info("=" * 60)
    info(f"Step 5/6: Install KV cache client ({client})")
    info("=" * 60)

    run([str(vpy), "-m", "pip", "install", client])

    # ── Step 6: Verify ─────────────────────────────────────────────────
    print()
    info("=" * 60)
    info("Step 6/6: Verify imports")
    info("=" * 60)

    verify_script = _build_verify_script(client)
    result = subprocess.run(
        [str(vpy), "-c", verify_script],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        for line in result.stdout.strip().splitlines():
            ok(line)
    else:
        warn("Some imports failed:")
        output = result.stderr.strip() or result.stdout.strip()
        for line in output.splitlines()[-5:]:
            warn(f"  {line}")

    # ── Summary ────────────────────────────────────────────────────────
    print()
    info("=" * 60)
    info("Setup complete")
    info("=" * 60)
    print()
    ok(f"Venv:    {venv_dir}")
    ok(f"SGLang:  {sglang_dir}")
    ok(f"Client:  {client}")
    print()

    # Show activation command
    activate = venv_dir / "bin" / "activate"
    if not activate.exists():
        activate = venv_dir / "Scripts" / "activate"
    info(f"To activate:  source {activate}")
    info(f"Then run SGLang:")
    info(f"  python -m sglang.launch_server --model-path <model> \\")
    info(f"    --enable-hierarchical-cache --hicache-storage-backend cama")


def _build_verify_script(client: str) -> str:
    """Build a Python snippet that tests all critical imports."""
    lines = [
        "import sys; ok = True",
        "",
        "try:",
        "    import sglang; print(f'sglang: {sglang.__file__}')",
        "except Exception as e:",
        "    print(f'FAIL sglang: {e}', file=sys.stderr); ok = False",
        "",
        "try:",
        "    import sgl_kernel; print(f'sgl-kernel: {sgl_kernel.__version__}')",
        "except Exception as e:",
        "    print(f'FAIL sgl-kernel: {e}', file=sys.stderr); ok = False",
        "",
        "try:",
        "    import flashinfer; print(f'flashinfer: {flashinfer.__version__}')",
        "except Exception as e:",
        "    print(f'FAIL flashinfer: {e}', file=sys.stderr); ok = False",
        "",
        "try:",
        "    from sglang.srt.mem_cache.storage.cama.cama_storage import CamaStorage",
        "    print(f'CamaStorage: OK')",
        "except Exception as e:",
        "    print(f'FAIL CamaStorage: {e}', file=sys.stderr); ok = False",
        "",
    ]

    if client == "cama-client":
        lines += [
            "try:",
            "    from cama_client import PriskvClient, SGL; print('cama-client: OK')",
            "except Exception as e:",
            "    print(f'FAIL cama-client: {e}', file=sys.stderr); ok = False",
        ]
    else:
        lines += [
            "try:",
            "    from priskv.priskv_client import PriskvClient; print('priskv: OK')",
            "except Exception as e:",
            "    print(f'FAIL priskv: {e}', file=sys.stderr); ok = False",
        ]

    lines += [
        "",
        "sys.exit(0 if ok else 1)",
    ]
    return "\n".join(lines)


# ── verify (standalone, for --all mode) ────────────────────────────────
def do_verify(sglang_dir: Path) -> None:
    """Quick sanity check: can Python import the deployed module?"""
    info("Running import check...")
    sglang_python = sglang_dir / "python"
    result = subprocess.run(
        [
            sys.executable, "-c",
            "from sglang.srt.mem_cache.storage.cama.cama_storage import CamaStorage; "
            "print('CamaStorage imported successfully')",
        ],
        env={**os.environ, "PYTHONPATH": str(sglang_python)},
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        ok(result.stdout.strip())
    else:
        warn("Import check failed (expected if deps not installed in this env):")
        stderr = result.stderr.strip()
        if stderr:
            warn(f"  {stderr.splitlines()[-1]}")


# ── main ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy CAMA module + patches into an SGLang tree.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sglang_dir",
        type=Path,
        help="Path to the SGLang checkout or pre-packaged tree",
    )

    # -- mode (mutually exclusive) --
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--module", action="store_const", dest="mode", const="module",
                      help="Deploy cama_module/ only")
    mode.add_argument("--patch",  action="store_const", dest="mode", const="patch",
                      help="Apply integration patches only")
    mode.add_argument("--diff",   action="store_const", dest="mode", const="diff",
                      help="Dry-run: show what would change")
    mode.add_argument("--zip",    action="store_const", dest="mode", const="zip",
                      help="Deploy + create zip archive")
    mode.add_argument("--setup",  action="store_const", dest="mode", const="setup",
                      help="Full setup: create venv, install SGLang + client, deploy CAMA")
    parser.set_defaults(mode="all")

    # -- setup options --
    setup_group = parser.add_argument_group("setup options (only with --setup)")
    setup_group.add_argument(
        "--venv-dir", type=Path, default=None,
        help="Where to create the venv (default: <sglang>/.venv)",
    )
    setup_group.add_argument(
        "--client", choices=["cama-client", "priskv"], default="cama-client",
        help="KV cache client package to install (default: cama-client)",
    )
    setup_group.add_argument(
        "--fresh", action="store_true",
        help="Remove existing venv before creating a new one",
    )
    setup_group.add_argument(
        "--find-links", default=FLASHINFER_FIND_LINKS,
        help=f"Extra --find-links for pip (default: {FLASHINFER_FIND_LINKS})",
    )

    args = parser.parse_args()
    sglang_dir = validate_sglang_dir(args.sglang_dir)
    version = read_version()

    info(f"CAMA connector v{version}")
    info(f"SGLang tree: {sglang_dir}")
    info(f"Mode: {args.mode}")

    if args.mode == "diff":
        print()
        do_diff(sglang_dir)

    elif args.mode == "module":
        print()
        do_module(sglang_dir)

    elif args.mode == "patch":
        print()
        do_patch(sglang_dir)

    elif args.mode == "zip":
        print()
        do_module(sglang_dir)
        do_patch(sglang_dir)
        print()
        do_zip(sglang_dir, version)

    elif args.mode == "setup":
        venv_dir = args.venv_dir or (sglang_dir / ".venv")
        do_setup(
            sglang_dir=sglang_dir,
            venv_dir=venv_dir.resolve(),
            client=args.client,
            fresh=args.fresh,
            find_links=args.find_links,
        )

    else:  # --all (default)
        print()
        do_module(sglang_dir)
        print()
        do_patch(sglang_dir)
        print()
        do_verify(sglang_dir)

    print()
    ok("Done.")


if __name__ == "__main__":
    main()
