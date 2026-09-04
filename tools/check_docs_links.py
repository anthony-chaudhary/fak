#!/usr/bin/env python3
"""Docs link checker: verify all markdown links in front-door docs and docs/notes/ resolve."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
import check_links


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    args = list(argv)
    if not any(a in args for a in ("--audit-tree", "--audit-staged")):
        args.append("--audit-tree")
    if "--include-notes" not in args:
        args.append("--include-notes")
    return check_links.main(args)


if __name__ == "__main__":
    sys.exit(main())
