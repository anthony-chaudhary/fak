#!/usr/bin/env python3
"""Regression guard for issue #213: no operator machine-path needle in tools/.

The public-copy scrubber rewrites the BACKSLASH form ``C:\\Users\\<op>\\`` to the
generic ``...\\USER\\`` placeholder, but historically missed the FORWARD-slash and
WSL forms (``C:/Users/<op>/...``, ``/mnt/c/Users/<op>/...``) and the harness slug
form (``...-Users-<op>-...``). So an operator username + OneDrive desktop path
survived verbatim in ``tools/bench_slack.py``. ``PUBLIC-SCRUB-POLICY.md`` classes
an operator username + machine path as Must-REDACT.

This test fails if any tools/ source re-introduces that needle in ANY of those
slash/separator directions. The operator username literal is assembled from parts
so THIS file never carries the needle itself (mirroring the NEEDLE trick in
``scrub_public_copy_test.py``).

Run: ``python -m pytest tools/operator_path_needle_test.py`` (or run directly).
"""
from __future__ import annotations

import re
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
# Operator username, assembled so the literal never appears verbatim in this file.
_OP = "an" + "tho"
# Match Users<sep><op><sep> for sep in / \ - : the redact-needle shapes that the
# operator path takes across POSIX, Windows, WSL, and the harness path slug.
_NEEDLE = re.compile(r"Users[\\/_-]" + re.escape(_OP) + r"[\\/_-]")


def _scanned_files() -> list[Path]:
    me = Path(__file__).name
    return sorted(
        p
        for ext in ("*.py", "*.ps1", "*.sh")
        for p in TOOLS.glob(ext)
        if p.name != me
    )


def test_no_operator_path_needle_in_tools() -> None:
    offenders = []
    for p in _scanned_files():
        text = p.read_text(encoding="utf-8", errors="replace")
        if _NEEDLE.search(text):
            offenders.append(p.name)
    assert not offenders, (
        "operator machine-path needle (#213) present in: " + ", ".join(offenders)
    )


if __name__ == "__main__":
    test_no_operator_path_needle_in_tools()
    print("ok: no operator path needle in tools/")
