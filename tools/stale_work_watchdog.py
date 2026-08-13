#!/usr/bin/env python3
"""Compatibility launcher for the Go-native ``fak garden watchdog`` verb.

The stale-work scanner, GC policy, overlap lock, bounded garden child, timeout
JSON, and descendant-tree reap live in Go. This tiny shim remains only so old
operator commands do not break while scheduled installs move directly to fak.
"""
from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Callable, Sequence


def command(argv: Sequence[str] | None = None) -> list[str]:
    """Return the exact Go command used by this compatibility path."""
    return [os.environ.get("FAK_BIN", "fak"), "garden", "watchdog", *(argv or [])]


def main(
    argv: Sequence[str] | None = None,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
) -> int:
    kwargs: dict[str, object] = {"check": False}
    # Manual use of the legacy shim should retain the no-popup behavior the old
    # Python implementation installed globally. The scheduled task no longer
    # invokes Python at all.
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    return int(runner(command(argv), **kwargs).returncode)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
