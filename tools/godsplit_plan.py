#!/usr/bin/env python3
"""godsplit_plan — the boundary + hazard planner for a behavior-preserving Go split.

The ``/modularize`` skill retires the code-quality scorecard's ``architecture`` debt
(god-files > 1500 lines, god-functions > 200 lines) by MOVING top-level declarations
into concern-scoped files in the SAME package — a semantic no-op in Go EXCEPT for the
hazards this tool surfaces.

The error-prone part of a clean split is cutting at the right line: a top-level decl's
doc comment sits ABOVE its ``func``/``type`` keyword, so a naive cut at the keyword
orphans the comment. This tool computes, for every top-level declaration, the
DOC-COMMENT-AWARE block boundaries (the exact ``sed -n 'A,Bp'`` range to extract), and
flags the four things that make code motion NOT a no-op:

* per-file build tags (``//go:build``) — moving a decl changes which file carries them;
* ``func init()`` — init order is filename-ALPHABETICAL across a package, so moving an
  ``init()`` between files can silently reorder initialization;
* aliased imports (``x "path"``) — ``goimports -w`` re-derives plain imports after a move
  but does NOT re-infer a local alias, so an alias must be re-added to the new file by hand;
* multi-line backtick raw strings — this is a LINE parser, not a Go tokenizer; embedded
  column-0 ``func``/``type`` text inside a raw string is correctly SKIPPED, but the
  ``raw_strings`` count warns you to eyeball the plan for such a file.

It is line-based and best-effort: ``decl`` detection ignores raw-string interiors and the
package/import header, and func length uses brace-depth over string/comment-stripped code.
Always sanity-check (see the skill): reject any plan where a decl's ``block_end <
block_start``, or where the decl count disagrees with ``grep -cE '^(func|type|var|const) '``.

Read-only. Run from the repo root::

    python tools/godsplit_plan.py internal/model/kv.go            # human table
    python tools/godsplit_plan.py internal/model/kv.go --json     # machine payload

It never edits, moves, or commits anything — it only plans. The ``/modularize`` skill
turns the plan into the actual sed-extract + goimports + gofmt + verify + commit pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any

FILE_HARD_MAX = 1500  # a Go file longer than this is a god-file (scorecard architecture debt)
FUNC_HARD_MAX = 200   # a function/method longer than this is a god-function

_DECL_RE = re.compile(r"^(func|type|var|const)\b")
_BLANK_RE = re.compile(r"^\s*$")
_PACKAGE_RE = re.compile(r"^package\s+(\w+)")
_METHOD_RE = re.compile(r"^func\s+\([^)]*\)\s+(\w+)")
_FUNC_RE = re.compile(r"^func\s+(\w+)")
_TYPEVARCONST_RE = re.compile(r"^(?:type|var|const)\s+(\w+)")
_BUILDTAG_RE = re.compile(r"^//\s*(?:go:build|\+build)\b")
_INIT_RE = re.compile(r"^func\s+init\s*\(")


def decl_name(line: str) -> tuple[str, str]:
    """Return ``(kind, name)`` for a top-level declaration line. A method
    ``func (r R) Name(`` reports kind ``method`` and name ``Name``; a grouped
    ``var (`` / ``const (`` block reports name ``(group)``."""
    if line.startswith("func "):
        m = _METHOD_RE.match(line)
        if m:
            return "method", m.group(1)
        m = _FUNC_RE.match(line)
        return "func", m.group(1) if m else "?"
    m = _TYPEVARCONST_RE.match(line)
    if m:
        kind = line.split(None, 1)[0]
        return kind, m.group(1)
    kind = line.split(None, 1)[0] if line.strip() else "?"
    return kind, "(group)"


def _scan_line(line: str, in_raw: bool) -> tuple[str, bool]:
    """Return ``(code, in_raw_after)`` where ``code`` is LINE with line-comments and the
    contents of ``"…"`` / ``'…'`` / backtick raw strings removed (braces preserved only
    when they are real code). ``in_raw`` carries a multi-line backtick string across
    lines. Best-effort: ``/* … */`` block comments are not handled (rare inside a decl)."""
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if in_raw:
            if c == "`":
                in_raw = False
            i += 1
            continue
        if c == "`":
            in_raw = True
            i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break  # line comment: rest is not code
        if c in ('"', "'"):
            quote = c
            i += 1
            while i < n:
                if line[i] == "\\":
                    i += 2
                    continue
                if line[i] == quote:
                    i += 1
                    break
                i += 1
            continue
        out.append(c)
        i += 1
    return "".join(out), in_raw


def _header_end(lines: list[str]) -> int:
    """1-based line below which top-level decls legitimately begin — the max of the
    ``package`` line, the closing ``)`` of an ``import ( … )`` block, and any single-line
    ``import "…"``. Used to clamp a decl's block_start so an extract can never swallow the
    package clause (which would duplicate it in the new file → build break)."""
    pkg = imp_end = 0
    in_import = False
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("package "):
            pkg = i + 1
        elif s == "import (":
            in_import = True
        elif in_import and s == ")":
            imp_end = i + 1
            in_import = False
        elif not in_import and s.startswith("import ") and '"' in s:
            imp_end = max(imp_end, i + 1)
    return max(pkg, imp_end)


def doc_start(lines: list[str], decl_idx: int, in_raw_before: list[bool]) -> int:
    """1-based line where DECL_IDX's block begins, INCLUDING the leading doc comment —
    the line after the last REAL (not raw-string-interior) blank line above the decl, or
    line 1 if none. ``decl_idx`` is 0-based into ``lines``."""
    last_blank = -1
    for i in range(decl_idx):
        if _BLANK_RE.match(lines[i]) and not in_raw_before[i]:
            last_blank = i
    return last_blank + 2


def func_span(stripped: list[str], decl_idx: int) -> int:
    """1-based END line of the func/method starting at ``decl_idx`` (0-based), by
    brace-depth over the string/comment-stripped code: the function ends when depth
    returns to 0 after the first ``{``. Robust to one-liners, indented composite
    literals, and braces inside strings/raw strings (which were stripped)."""
    depth = 0
    opened = False
    for j in range(decl_idx, len(stripped)):
        for ch in stripped[j]:
            if ch == "{":
                depth += 1
                opened = True
            elif ch == "}":
                depth -= 1
        if opened and depth <= 0:
            return j + 1
    return len(stripped)


def import_aliases(lines: list[str]) -> list[str]:
    """Aliased imports (``alias "path"`` where alias is a real local name, not ``_``/``.``
    and not just the path's basename) — the ones ``goimports`` will NOT re-create."""
    aliases: list[str] = []
    in_block = False
    for line in lines:
        s = line.strip()
        if s == "import (":
            in_block = True
            continue
        if in_block and s == ")":
            in_block = False
            continue
        target = s
        if not in_block:
            if s.startswith("import ") and '"' in s:
                target = s[len("import "):].strip()
            else:
                continue
        m = re.match(r'^([A-Za-z_]\w*)\s+"([^"]+)"', target)
        if not m:
            continue
        alias, path = m.group(1), m.group(2)
        if alias in ("_", "."):
            continue
        if alias != path.rsplit("/", 1)[-1]:
            aliases.append(f'{alias} "{path}"')
    return aliases


def plan(text: str) -> dict[str, Any]:
    """Pure fold: given Go source TEXT, return the split plan — package, hazards, and the
    decl list with doc-comment-aware block ranges + size flags. Decl/build-tag/init
    detection ignores raw-string interiors; block_start is clamped below the header. No
    file I/O, so the test drives it with a string."""
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]  # a trailing newline yields one phantom empty element
    n = len(lines)

    # Pass 1: strip strings/comments per line, tracking multi-line raw strings.
    stripped: list[str] = []
    in_raw_before: list[bool] = []
    raw_strings = 0
    in_raw = False
    for line in lines:
        in_raw_before.append(in_raw)
        code, after = _scan_line(line, in_raw)
        if not in_raw and after:  # a multi-line raw string just opened
            raw_strings += 1
        stripped.append(code)
        in_raw = after

    header_end = _header_end(lines)

    package = ""
    build_tags: list[str] = []
    init_funcs = 0
    decls: list[dict[str, Any]] = []

    # Pass 2: detect decls/build-tags/init only on REAL code lines (not raw interiors).
    for i, line in enumerate(lines):
        if in_raw_before[i]:
            continue
        if not package:
            pm = _PACKAGE_RE.match(line)
            if pm:
                package = pm.group(1)
        if _BUILDTAG_RE.match(line):
            build_tags.append(line.strip())
        if _INIT_RE.match(line):
            init_funcs += 1
        if _DECL_RE.match(line):
            kind, name = decl_name(line)
            start = max(doc_start(lines, i, in_raw_before), header_end + 1)
            d: dict[str, Any] = {"line": i + 1, "kind": kind, "name": name, "block_start": start}
            if kind in ("func", "method"):
                end = func_span(stripped, i)
                d["func_end"] = end
                d["func_lines"] = end - (i + 1) + 1
                d["god_function"] = d["func_lines"] > FUNC_HARD_MAX
            decls.append(d)

    for idx, d in enumerate(decls):
        d["block_end"] = (decls[idx + 1]["block_start"] - 1) if idx + 1 < len(decls) else n

    return {
        "package": package,
        "line_count": n,
        "god_file": n > FILE_HARD_MAX,
        "hazards": {
            "build_tags": build_tags,
            "init_funcs": init_funcs,
            "import_aliases": import_aliases(lines),
            "raw_strings": raw_strings,
        },
        "decls": decls,
    }


def _render(path: str, p: dict[str, Any]) -> str:
    out: list[str] = []
    flag = "  GOD-FILE (>1500)" if p["god_file"] else ""
    out.append(f"{path}: package {p['package']}  ({p['line_count']} lines){flag}")
    hz = p["hazards"]
    out.append("hazards (these make code motion NOT a no-op):")
    out.append(f"  build_tags     : {hz['build_tags'] or 'none'}")
    init_note = " (init order is filename-alpha — moving one between files can reorder)" if hz["init_funcs"] else ""
    out.append(f"  init_funcs     : {hz['init_funcs']}{init_note}")
    alias_note = " (goimports will NOT re-add these after a move — copy them by hand)" if hz["import_aliases"] else ""
    out.append(f"  import_aliases : {hz['import_aliases'] or 'none'}{alias_note}")
    raw_note = " (line parser may misread embedded code — eyeball the ranges)" if hz["raw_strings"] else ""
    out.append(f"  raw_strings    : {hz['raw_strings']}{raw_note}")
    out.append("")
    out.append("decls (block_start..block_end is the doc-comment-aware sed extract range):")
    for d in p["decls"]:
        span = ""
        if d["kind"] in ("func", "method"):
            mark = "  <<< GOD-FUNCTION (>200)" if d.get("god_function") else ""
            span = f"  [{d['func_lines']}L]{mark}"
        warn = "  !! INVERTED RANGE — do not trust this plan" if d["block_end"] < d["block_start"] else ""
        out.append(f"  {d['block_start']:>5}..{d['block_end']:<5} {d['kind']:>6} {d['name']}{span}{warn}")
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Plan a doc-comment-aware, hazard-checked Go split (read-only).")
    ap.add_argument("file", help="the .go file to plan a split of")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable plan")
    args = ap.parse_args(argv)
    try:
        with open(args.file, "r", encoding="utf-8") as fh:
            text = fh.read()
    except OSError as e:
        print(f"godsplit_plan: {e}", file=sys.stderr)
        return 1
    p = plan(text)
    if args.json:
        print(json.dumps(p, indent=2))
    else:
        print(_render(args.file, p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
