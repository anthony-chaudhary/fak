#!/usr/bin/env python3
"""dos_hook.py -- the fak-side launcher that binds Claude Code's dos hooks to a
*versioned* dos contract instead of reaching into ``dos.cli`` internals (issue #2705,
epic #2702).

Why this file exists
--------------------
fak's committed ``.claude/settings.json`` used to hard-wire the dos hooks two
different broken ways:

  * ``python -c "...subprocess.call([... 'dos.cli','hook', <verb>]); sys.exit(0)"``
    -- a direct dependency on dos's 15,525-line ``cli.py``. ``dos.cli`` exits **code 2
    (Claude Code's *block* code)** on any argparse error -- e.g. a dos flag rename on a
    version bump -- which would **hard-block every fak tool call fleet-wide**. Only the
    ``sys.exit(0)`` wrapper stood between the fleet and that wedge.
  * a hard-coded absolute ``C:/work/fak/tools/.bin/dos-hook.exe`` -- machine-specific
    (breaks any other host / a fresh checkout), with **no Python fallback** when the
    binary is absent and **no exit-code coercion** if the binary itself exits non-zero.

This launcher replaces both with the *same contract the enabled dos plugin commits to*
(``dos-kernel/claude-plugin/hooks/hooks.json``): **native-first, fall back to Python,
always exit 0.**

The contract (byte-for-byte the plugin's semantics)
---------------------------------------------------
  1. Buffer stdin ONCE (the Claude Code hook payload). A native->python delegation must
     re-decide on the *real* payload, not a drained stream -- so we read the bytes up
     front and feed the same buffer to whichever backend runs.
  2. Prefer the native Go binary ``tools/.bin/dos-hook[.exe]`` (``exit 0`` = OWNED /
     decided; any non-zero = DELEGATE, e.g. ``exit 3``, or a crash). It is
     byte-identical to the Python verb and ~10-30x faster (measured ~326 ms pretool /
     ~28 ms posttool vs ~877 / ~478 ms cold Python).
  3. On native OWNED (rc == 0) forward its stdout verbatim and stop -- a real **deny
     travels on stdout JSON** (``permissionDecision: deny``) and is preserved.
  4. On native DELEGATE / crash / **binary absent**, fall back to
     ``python -m dos.cli hook <verb>`` fed the buffered stdin. NO regression when the
     native binary is missing -- exactly today's Python path.
  5. **Always exit 0.** The Python fallback's exit code (including argparse ``exit 2``)
     is DISCARDED. A dos version bump that renames a flag can no longer hard-block a
     fak tool call; a real deny still travels on stdout JSON, unaffected.

Provision the native binary with ``make dos-hook`` (gitignored ``tools/.bin/dos-hook``,
same pattern as ``repoguard``). Absent it, this launcher is a clean pass-through to the
Python path.

Invocation (from ``.claude/settings.json``)::

    python tools/dos_hook.py <verb> --workspace .     # verb: pretool | posttool | stop | ...

Fail-safe by construction: any internal error in this launcher is swallowed and the
process still exits 0. A launcher bug must never wedge a live multi-session fleet.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Callable, Optional, Tuple

# A runner turns an argv + stdin buffer into (returncode, stdout_bytes). The real one
# spawns a subprocess; the test injects a pure fake so the contract is pinned without
# a live dos install. stderr is intentionally passed through to the launcher's stderr
# (diagnostics), never captured into the hook's stdout (which carries the JSON verdict).
Runner = Callable[[list, bytes], Tuple[int, bytes]]

# Windows: keep the synchronously-spawned console child from flashing a visible window.
_CREATE_NO_WINDOW = 0x08000000


def _creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


def repo_root(env: Optional[dict] = None) -> Path:
    """The workspace root. Prefer ``CLAUDE_PROJECT_DIR`` (set by Claude Code for hooks);
    fall back to this file's repo (``tools/`` parent) so the launcher works regardless of
    the hook's cwd."""
    env = os.environ if env is None else env
    root = env.get("CLAUDE_PROJECT_DIR")
    if root:
        return Path(root)
    return Path(__file__).resolve().parent.parent


def native_binary(root: Path) -> Optional[Path]:
    """The provisioned native ``dos-hook`` binary under ``tools/.bin``, or ``None`` when
    absent (then the launcher falls back to the Python path -- no regression)."""
    names = ("dos-hook.exe", "dos-hook") if os.name == "nt" else ("dos-hook", "dos-hook.exe")
    for name in names:
        cand = root / "tools" / ".bin" / name
        if cand.exists():
            return cand
    return None


def _real_runner(argv: list, stdin_data: bytes) -> Tuple[int, bytes]:
    """Spawn ``argv`` feeding ``stdin_data``; return (returncode, captured stdout)."""
    try:
        proc = subprocess.run(
            argv,
            input=stdin_data,
            stdout=subprocess.PIPE,
            stderr=None,  # let diagnostics flow to the launcher's stderr, not stdout
            creationflags=_creationflags(),
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError):
        # Backend could not run at all -> behave like "changed nothing".
        return 1, b""
    return proc.returncode, proc.stdout or b""


def run_hook(
    verb: str,
    workspace: str,
    stdin_data: bytes,
    native: Optional[Path],
    run: Runner,
) -> bytes:
    """Decide the call and return the stdout to forward. Pure w.r.t. process exit -- the
    caller always exits 0. Native-first (rc == 0 = OWNED, else DELEGATE), Python fallback.

    A deny (``permissionDecision: deny`` JSON on stdout) is preserved whichever backend
    emits it. The Python fallback's *exit code* is never consulted -- an argparse
    ``exit 2`` cannot block the tool call."""
    if native is not None:
        rc, out = run([str(native), verb, "--workspace", workspace], stdin_data)
        if rc == 0:
            return out  # OWNED: deny (if any) is on stdout; done.
        # else: DELEGATE / crash -> fall through to the Python path on the SAME stdin.
    _rc, out = run(
        [sys.executable, "-m", "dos.cli", "hook", verb, "--workspace", workspace],
        stdin_data,
    )
    # _rc is DISCARDED on purpose: coerce non-deny non-zero exits (incl. argparse 2) to 0.
    return out


def main(
    argv: Optional[list] = None,
    stdin_bytes: Optional[bytes] = None,
    run: Optional[Runner] = None,
    env: Optional[dict] = None,
) -> int:
    """Read the buffered hook payload, run the contract, forward the verdict, exit 0.

    Always returns 0 -- fail-safe. ``argv`` is ``[verb, ...]`` (the launcher accepts and
    passes through ``--workspace <ws>``); the extra args are tolerated so a caller can
    keep the plugin's flag shape."""
    try:
        args = list(sys.argv[1:] if argv is None else argv)
        verb = args[0] if args else "pretool"
        # Accept "--workspace <ws>" anywhere in the tail; default ".".
        workspace = "."
        if "--workspace" in args:
            i = args.index("--workspace")
            if i + 1 < len(args):
                workspace = args[i + 1]

        data = sys.stdin.buffer.read() if stdin_bytes is None else stdin_bytes
        runner = _real_runner if run is None else run
        native = native_binary(repo_root(env))

        out = run_hook(verb, workspace, data or b"", native, runner)
        if out:
            sys.stdout.buffer.write(out)
            sys.stdout.buffer.flush()
    except Exception:  # noqa: BLE001 -- a launcher bug must never wedge the fleet.
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
