#!/usr/bin/env python3
"""Code-slop scorecard — the measuring stick for *slop the compiler can't see*.

The repo already grades its Go module on classic defects (``code_quality_scorecard.py``:
gofmt, ``go vet``, a god-file, an *untested* package, an untagged claim) and grades its
prose for machine-voice (``doc_appeal_scorecard.py``: clichés, em-dash flood, LLM
scaffolding). Neither catches **code slop** — the low-value churn an LLM-driven
codebase accretes that is *structurally valid*: it compiles, it vets clean, the package
has a ``_test.go``, the file is under the god-file ceiling. Every existing KPI scores it
100. The slop hides *inside* those green checks:

  - a copy-pasted block cloned across three files (each file is fine; the duplication isn't)
  - a ``Test*`` that runs but asserts nothing (the ``tests`` KPI only checks *presence*)
  - an unexported helper defined and never called (dead weight the build keeps)
  - a doc comment that only restates the symbol name, or a commented-out code block
    left behind (comment cruft that reads like documentation)

"Don't let the kernel rot into slop" was an unfalsifiable vibe — there was no number to
move. This is that number. It scores the Go module on six slop axes, folds them into a
weighted **slop_score** (0-100, A-F) AND — the lever that makes "less slop" a checkable
target — a **slop_debt** integer: the count of concrete, re-derivable slop defects you
can drive toward zero. Every axis is static ``.go`` analysis only; the scorecard never
shells ``go`` (so it stays no-network, no-build, gate-safe).

The six KPIs (each 0-100):

  duplication     copy-paste clones: a normalized Go token-window appearing in 2+ places [HARD]
  vacuous_tests   a Test/Benchmark func body that makes zero assertions                 [HARD]
  dead_code       an unexported symbol defined but referenced nowhere else              [HARD]
  comment_slop    tautological doc comments + commented-out code blocks                 [HARD]
  stub_masquerade an exported func whose body is only return-nil / panic-unimplemented  [SOFT]
  churn_bloat     recent commits that ADD .go files without ever removing any           [SOFT]

``stub_masquerade`` is deliberately SOFT in v1: there is no clean machine link from a Go
symbol to a ``[STUB]`` line in ``CLAIMS.md`` (the repo's honesty ledger names *behaviors*,
not function names), so a hard gate here risks false positives. It scores but never gates;
promote it to HARD once the heuristic proves tight. ``churn_bloat`` is SOFT and
HEAD-relative — it grades recent history, not the tree, so its number moves as commits
land (pin ``--range`` for a stable read).

``ok`` is False iff any HARD defect exists. Soft signals are advisory and never gate —
the same FAIL/ADVISORY split the sibling scorecards draw.

Read-only by construction: it reads ``.go`` files, ``CLAIMS.md``, ``VERSION`` (to label
output), and shells out to ``git log`` (a read-only verb) only for the SOFT churn axis;
it edits nothing. Run from the repo ROOT::

    python tools/code_slop_scorecard.py                 # human scorecard
    python tools/code_slop_scorecard.py --json          # machine payload (control-pane)
    python tools/code_slop_scorecard.py --markdown      # docs/CODE-SLOP-SCORECARD.md body
    python tools/code_slop_scorecard.py --check-doc     # fail if the snapshot is stale
    python tools/code_slop_scorecard.py --no-toolchain  # parity flag; this scorecard is already static

The companion process is the slop-to-zero program (a ``/slop-score``-shaped loop): each
HARD defect is one unit of slop-debt to retire, and re-running proves the number moved —
the code-side anti-slop counterpart of ``/quality-score`` (defects) and ``/appeal-score``
(prose voice).
"""
from __future__ import annotations

import argparse
import difflib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-code-slop-scorecard/1"
SCORECARD_DOC = "docs/CODE-SLOP-SCORECARD.md"
STAMP_RE = re.compile(r"<!-- code-slop-scorecard:\s*(?P<stamp>[^·<]+?)\s*·")

CLAIMS_REL = "CLAIMS.md"
VERSION_REL = "VERSION"

# ---------------------------------------------------------------------------
# Thresholds. Generous on purpose — a measuring stick should find real-but-modest
# slop and give a number to drive down, not manufacture debt from legitimate code.
# Each constant is a deliberate threshold with a stated reason.
# ---------------------------------------------------------------------------

# Duplication (token-stream engine, #780): a clone is a window of this many normalized
# Go TOKENS appearing in 2+ places. Literals (strings/runes/numbers) collapse to one `L`
# token; keywords/operators/punctuation/identifiers are kept verbatim. Token windows are
# whitespace/comment/line-break invariant, so a clone reformatted across lines still
# hashes identically (the line-shingle engine missed those) and a literal is one token,
# never a blanked husk that stitches unrelated code into a phantom clone. Identifiers are
# kept (NOT anonymized): full identifier anonymization was measured and collapses
# idiomatic Go — `if x != nil { return ... }` and the like — into phantom mega-clusters
# (a single window matched 5000+ sites), the OPPOSITE of tightening precision. ~34 tokens
# is roughly a 6-line block: long enough that an idiomatic err-wrap is not "duplication",
# short enough to catch a copy-pasted body (≈ the old 6-non-trivial-line window).
CLONE_WINDOW_TOKENS = 34
# A window counts only if its EFFECTIVE logic-token count is at least this many
# (computation / comparison / assignment operators + control-flow keywords). This ONE
# structural rule replaces the old hand-tuned line skips: package/import boilerplate,
# struct/interface field lists and composite-literal `Key: value,` data all carry ZERO
# logic tokens, so they are never mistaken for a copy-pasted block — duplication is
# measured on executable structure, not text. The bare assignment ops `=` / `:=` are
# CONTEXT-GATED (FIX 6): they count toward a window's logic ONLY when the same window
# also carries >= 1 non-assignment logic token (a comparison/compute op or a control
# keyword). A window whose only logic is assignment is a pure declaration/literal block
# (e.g. a run of `x := fs.String(...)` flag-decls) and is demoted to zero — while a real
# copied statement body keeps its full count via the control/compute it co-occurs with.
CLONE_MIN_LOGIC_TOKENS = 2
# A clone group is HARD only if its window appears in >= this many DISTINCT locations
# (a location = (file, start-line)). Two is the threshold for "copy-pasted".
CLONE_MIN_OCCURRENCES = 2
# Upper bound on distinct clone-group defects emitted, a runaway backstop only — set
# far above any real count so the slop_debt integer is the TRUE group count, not a
# silently-capped one (the render layer truncates the printed work-list, not the debt).
CLONE_GROUPS_CAP = 5000
# A clone GROUP counts only if its largest site spans at least this many source lines.
# The window engine matches a 34-token window (~6 lines) but block-merge can report a
# group whose merged span is as small as 3-4 lines — a fragment that short (an err-check,
# a sort closure, a struct-field run, a one-line SSE/header call, a signature line) is
# idiomatic Go, not copy-paste slop: a shared helper for it would cost as much as the
# inline code, so "de-duplicating" it would WORSEN readability. 6 lines matches the
# CLONE_WINDOW_TOKENS docstring's own "~6-line block" definition of a real clone.
CLONE_MIN_GROUP_SPAN = 6

# Dead code: cap per file so one messy file is not unbounded debt (mirrors
# code_quality_scorecard.HYGIENE_CAP_PER_FILE). A symbol is dead if its identifier
# appears EXACTLY ONCE across the whole first-party module (its own definition).
DEAD_CAP_PER_FILE = 5

# Comment slop: a commented-out code block is >= this many consecutive `//` lines
# that parse as Go statements (one stray `// note:` is prose, not dead code).
COMMENTED_CODE_MIN_RUN = 2
COMMENT_SLOP_CAP = 300

# A function body is "trivial" (a candidate stub) if its code-only body is at most
# this many statement lines — a real implementation is longer.
STUB_BODY_MAX_LINES = 2

# stub_masquerade SOFT->HARD promotion gate (#781). The axis stays SOFT until BOTH
# preconditions hold: (1) the symbol<->[STUB]-ledger LINK is tight — shipped in fc7449d
# (suppression keyed on backtick-quoted Go symbols, not lowercased prose) — and (2) the
# detector proves ZERO false positives on the tree across a SOAK window of a few
# releases. The detector first shipped in 53e4d5f, which is contained only in release
# 0.34.0, so the soak window OPENS at 0.34.0. `STUB_SOAK_RELEASES` encodes "a few
# releases" as 3 (the conservative reading); a maintainer may tune it. This gate is
# ADVISORY and self-reporting (`stub_promotion_status`): it computes READINESS, it never
# performs the flip — moving `soft` -> `defects` and bumping the KPI weight stays a
# deliberate maintainer act, taken once the elapsed window is reviewed for zero FP.
STUB_DETECTOR_SHIP_RELEASE = "0.34.0"
STUB_SOAK_RELEASES = 3

# Per-KPI weights for the composite slop_score. The HARD axes that most signal a
# rotting codebase weigh most; the SOFT/heuristic axes weigh least.
KPI_WEIGHTS: dict[str, float] = {
    "duplication": 0.26,
    "dead_code": 0.22,
    "comment_slop": 0.22,
    "vacuous_tests": 0.14,
    "stub_masquerade": 0.10,
    "churn_bloat": 0.06,
}

# Directories whose .go is NOT first-party shipped kernel code (same exclusion the
# code-quality scorecard uses): fixtures, vendored/generated trees.
GO_EXCLUDE_DIRS = {".git", ".claude", ".fak", "node_modules", "testdata", "vendor", "__pycache__"}
# `.claude` and `.fak` both hold full repo CHECKOUTS created by the agent machinery:
# `.claude/worktrees/<wt>/` (the worktree-isolation feature) and `.fak/tmp/issue<N>-clean-<sha>/`
# (a dispatch worker's clean clone). Both are gitignored scratch, not first-party source.
# Walking them counts every copied .go as a phantom clone of the real tree (a `.claude`
# worktree once inflated slop-debt 473 -> 2613; a `.fak/tmp` checkout inflated it ~6x, 488 -> 3029),
# which would make the committed snapshot + the #779 gate flap on a transient checkout.
# A worktree copy is identical to its source by construction, never kernel slop — so the
# gather drops the whole `.claude`/`.fak` subtrees, exactly as it drops `.git`/`vendor`.

_TESTFUNC_RE = re.compile(r"^func\s+(Test|Benchmark|Fuzz)(\w*)\s*\(\s*(\w+)\s+\*?", re.MULTILINE)
# A top-level declaration. Captures kind + name so we can find unexported symbols.
_FUNC_DECL_RE = re.compile(r"^func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*[\(\[]")
_TYPE_DECL_RE = re.compile(r"^type\s+([A-Za-z_]\w*)\b")
_VARCONST_DECL_RE = re.compile(r"^(?:var|const)\s+([A-Za-z_]\w*)\b")
# An identifier token, for the dead-code reference scan.
_IDENT_RE = re.compile(r"[A-Za-z_]\w*")
# A `//` comment that is actually a SHELL/CLI usage example, not commented-out Go —
# common in package doc comments (`// go run ./cmd/x`, `// $ fak serve`, `// make ci`).
# These must NOT count as commented-out code.
_SHELL_EXAMPLE_RE = re.compile(
    r"^\s*(\$|>|#|go\s+(run|test|build|install|vet|mod|get|tool)\b|"
    r"make\b|python\b|fak\b|git\b|curl\b|docker\b|cd\b|\./)")
# Go statement shapes that, when seen behind a `//`, mark a commented-out code LINE.
# Tighter than "starts with a keyword": a real statement ends in `{`, `}`, `)`, a
# semicolon, or is an assignment/call — prose and shell examples don't.
_CODEISH_RE = re.compile(
    r"^\s*(return\b.*|defer\s+\w+\(|func\b.*\{|"
    r"(if|for|switch)\b.*\{\s*$|case\b.*:\s*$|"
    r"var\s+\w+\s+[\w\.\*\[\]]+(\s*=.*)?$|const\s+\w+.*=|type\s+\w+\s+\w+|"
    r"\}\s*(else\b.*)?\{?\s*$|\w[\w\.\[\]]*\s*:=\s*\S|"
    r"\w[\w\.\[\]]*\s*[-+*/]?=[^=].*|"
    r"\w[\w\.]*\([^)]*\)\s*$)"
)


def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Literal/comment stripper (borrowed verbatim in spirit from
# code_quality_scorecard._code_only). Blanks out the CONTENTS of string/rune/
# backtick literals and `//` `/* */` comments so a brace, `:=`, or keyword inside a
# string is never mistaken for code. Returns (code, in_raw, in_block) carrying the
# two cross-line spans. `keep_comment_marker` lets the comment scan see WHERE a `//`
# starts (we need the comment text for comment_slop) without leaking string bytes.
# ---------------------------------------------------------------------------

def _code_only(line: str, in_raw: bool, in_block: bool) -> tuple[str, bool, bool]:
    out: list[str] = []
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if in_block:
            if c == "*" and i + 1 < n and line[i + 1] == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_raw:
            if c == "`":
                in_raw = False
            i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            break  # rest of line is a // comment
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            in_block = True
            i += 2
            continue
        if c == "`":
            in_raw = True
            i += 1
            continue
        if c == '"' or c == "'":
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
    return "".join(out), in_raw, in_block


def code_lines_of(text: str) -> list[str]:
    """Each source line reduced to its code-only form (literals/comments blanked),
    cross-line raw-string and block-comment spans tracked. Index-aligned to the
    original lines."""
    out: list[str] = []
    in_raw = in_block = False
    for raw in text.splitlines():
        code, in_raw, in_block = _code_only(raw, in_raw, in_block)
        out.append(code)
    return out


def line_comment_of(line: str, in_raw: bool, in_block: bool) -> tuple[str, bool, bool]:
    """Return (comment_text, in_raw, in_block): the text of a trailing/whole-line
    ``//`` comment on this line (empty if none), with the cross-line spans advanced.
    A comment inside a string or block comment is not a line comment."""
    i, n = 0, len(line)
    while i < n:
        c = line[i]
        if in_block:
            if c == "*" and i + 1 < n and line[i + 1] == "/":
                in_block = False
                i += 2
                continue
            i += 1
            continue
        if in_raw:
            if c == "`":
                in_raw = False
            i += 1
            continue
        if c == "/" and i + 1 < n and line[i + 1] == "/":
            return line[i + 2:].strip(), in_raw, in_block
        if c == "/" and i + 1 < n and line[i + 1] == "*":
            in_block = True
            i += 2
            continue
        if c == "`":
            in_raw = True
            i += 1
            continue
        if c == '"' or c == "'":
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
        i += 1
    return "", in_raw, in_block


# ---------------------------------------------------------------------------
# Per-KPI pure checks. Each takes already-gathered inputs (so tests need no disk)
# and returns {kpi, score, detail, defects:[str], soft:[str]} where every item in
# `defects` is one HARD unit of slop-debt and every item in `soft` is advisory.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Go token-stream lexer (#780). A dependency-free, pure-Python tokenizer — no
# `go/parser`, no `go` shell — so the scorecard stays static and gate-safe inside the
# `demo-scorecards` target. It is the structural foundation of the clone detector:
# duplication is measured on the NORMALIZED token stream, not on text lines, which is
# both whitespace/comment/line-break invariant (a clone reformatted across lines still
# matches — the line-shingle engine missed those) and string-husk immune (a literal is
# one `L` token, never a blanked husk that stitches unrelated code into a phantom clone).
# ---------------------------------------------------------------------------

_GO_KEYWORDS = frozenset({
    "break", "case", "chan", "const", "continue", "default", "defer", "else",
    "fallthrough", "for", "func", "go", "goto", "if", "import", "interface", "map",
    "package", "range", "return", "select", "struct", "switch", "type", "var",
})

# Operators + punctuation, longest-first so a greedy startswith() match never returns a
# proper prefix (`<<=` before `<<` before `<`; `:=` before `:`; `...` before `.`).
_GO_OPS = (
    "<<=", ">>=", "&^=", "...",
    "<-", ":=", "&&", "||", "++", "--", "==", "!=", "<=", ">=",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "&^",
    "+", "-", "*", "/", "%", "&", "|", "^", "<", ">", "=", "!",
    "(", ")", "[", "]", "{", "}", ",", ";", ".", ":",
)

# LOGIC tokens — the ones that denote computation or control flow, the signal that a
# duplicated window is copy-pasted LOGIC rather than a data/declaration shape. A window
# must carry CLONE_MIN_LOGIC_TOKENS of these to count. Identifiers, literals,
# punctuation and the declaration keywords (package/import/type/struct/...) are NOT
# logic, so an import block, a struct field list and a composite-literal `Key: value,`
# region all score zero logic and are never clones — one rule, no per-line skip list.
_LOGIC_OPS = frozenset({
    "+", "-", "*", "/", "%", "&", "|", "^", "<<", ">>", "&^", "&&", "||", "!",
    "==", "!=", "<", "<=", ">", ">=", "=", ":=",
    "+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<=", ">>=", "&^=", "++", "--",
})
_LOGIC_KEYWORDS = frozenset({"if", "for", "switch", "select", "range"})
# Bare assignment operators. These ARE logic tokens (a copied statement body keeps its
# assignments), but they are CONTEXT-GATED in the duplication window-qualify loop (#FP):
# they only contribute to a window's logic count when the SAME window also carries at
# least one NON-assignment logic token (a comparison/compute op or a control keyword).
# A window whose only logic is `=`/`:=` is a pure declaration/literal/field-init block
# (e.g. a run of `engine := fs.String(...)` flag-decls) — data, not copy-pasted logic.
# A real copied statement body always co-occurs with control/compute, so it keeps its
# full logic count via the non-assignment token it carries (#780 recall guard).
_ASSIGN_OPS = frozenset({"=", ":="})


def go_tokens(text: str, *, normalize_idents: bool = True) -> list[tuple[str, int, bool]]:
    """Lex Go source into a normalized token stream of (sym, line, is_logic).

    Comments and whitespace are dropped; every string/rune/number literal collapses to
    ``L``; identifiers collapse to ``I`` (so a clone survives a variable rename) unless
    ``normalize_idents`` is False; keywords, operators and punctuation are kept verbatim.
    ``is_logic`` marks a computation/control-flow token (see ``_LOGIC_OPS`` /
    ``_LOGIC_KEYWORDS``). Best-effort and forgiving — an unterminated literal or a stray
    byte is consumed without raising, since the scorecard must never crash on odd input."""
    out: list[tuple[str, int, bool]] = []
    i, n, line = 0, len(text), 1
    while i < n:
        c = text[i]
        if c == "\n":
            line += 1
            i += 1
            continue
        if c in " \t\r\f\v":
            i += 1
            continue
        # line comment
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            if j == -1:
                break
            i = j
            continue
        # block comment (may span lines)
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            if j == -1:
                line += text.count("\n", i)
                break
            line += text.count("\n", i, j + 2)
            i = j + 2
            continue
        # raw string literal (may span lines)
        if c == "`":
            j = text.find("`", i + 1)
            if j == -1:
                j = n - 1
            line += text.count("\n", i, j + 1)
            out.append(("L", line, False))
            i = j + 1
            continue
        # interpreted string / rune literal
        if c == '"' or c == "'":
            q = c
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n":
                    break  # unterminated — interpreted strings/runes can't span lines
                if text[j] == q:
                    j += 1
                    break
                j += 1
            out.append(("L", line, False))
            i = j
            continue
        # numeric literal
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "._"):
                if text[j] in "eEpP" and j + 1 < n and text[j + 1] in "+-":
                    j += 2  # exponent sign
                    continue
                j += 1
            out.append(("L", line, False))
            i = j
            continue
        # identifier or keyword
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            word = text[i:j]
            if word in _GO_KEYWORDS:
                out.append((word, line, word in _LOGIC_KEYWORDS))
            else:
                out.append(("I" if normalize_idents else word, line, False))
            i = j
            continue
        # operator / punctuation (greedy, longest-first)
        for op in _GO_OPS:
            if text.startswith(op, i):
                out.append((op, line, op in _LOGIC_OPS))
                i += len(op)
                break
        else:
            i += 1  # unknown byte (e.g. a stray non-ASCII rune) — skip
    return out


def _clone_sample(text: str, lineno: int) -> str:
    """A short, human-readable hint for a clone finding: the source line at `lineno`
    (1-based), trimmed. The token engine matches on structure; this just labels it."""
    lines = text.splitlines()
    if 1 <= lineno <= len(lines):
        return lines[lineno - 1].strip()
    return ""


# --- duplication group-level false-positive gates --------------------------
# These run AFTER clone grouping: a whole GROUP is dropped (or demoted to advisory)
# only when EVERY site in it matches a known not-slop shape, with a RECALL GUARD on
# each so a true clone of the same surface stays counted.

# FIX 3 — flag/CLI-plumbing groups. A run of `fs.String(...)` / `fs.Parse()` /
# err-check / brace lines is CLI argument wiring, not copy-pasted logic. The
# group is dropped iff EVERY site is a flag-plumbing block, where a block is
# flag-plumbing iff (1) >= FLAG_PLUMB_LINE_FRAC of its covered non-blank,
# non-`//` lines match a flag-plumbing line shape AND (2) the span actually
# contains a positive flag-API token. Requirement (2) is the RECALL GUARD:
# without it, the err-check + brace + return tail of ANY function would match.
FLAG_PLUMB_LINE_FRAC = 0.7
_FLAG_API_PRESENT = re.compile(
    r"(?:fs|flag)\.(?:String|Bool|Int|Int64|Uint|Float64|Duration|Var|Func|"
    r"NewFlagSet|Parse)\(")
# A single flag-plumbing line shape: a flag decl (a flag-API call assigned via
# :=/=), `fs.Parse(`, an `if err != nil {`, a bare `}` / `} else {`, or a small
# return (`return <int>` / `return err` / `return nil`).
_FLAG_PLUMBING_LINE = re.compile(
    r"^\s*(?:"
    r"[\w.]+\s*(?::=|=)\s*(?:fs|flag)\.(?:String|Bool|Int|Int64|Uint|Float64|"
    r"Duration|Var|Func|NewFlagSet|Parse)\(|"          # flag decl
    r"(?:fs|flag)\.Parse\(|"                            # fs.Parse()
    r"if\s+err\s*!=\s*nil\s*\{|"                        # err check
    r"\}\s*(?:else\s*\{)?\s*$|"                         # bare } / } else {
    r"return\s+(?:\d+|err|nil)\s*$"                     # small return
    r")")


def _is_flag_plumbing_block(text: str, sline: int, eline: int) -> bool:
    """True iff the source span [sline, eline] is CLI flag-plumbing: at least
    FLAG_PLUMB_LINE_FRAC of its covered code lines (non-blank, non-`//`) match the
    flag-plumbing line shape AND the span carries a positive flag-API token."""
    lines = text.splitlines()
    span = lines[sline - 1:eline]
    covered = [ln for ln in span if ln.strip() and not ln.lstrip().startswith("//")]
    if not covered:
        return False
    if not any(_FLAG_API_PRESENT.search(ln) for ln in span):  # recall guard
        return False
    n_plumb = sum(1 for ln in covered if _FLAG_PLUMBING_LINE.match(ln))
    return n_plumb / len(covered) >= FLAG_PLUMB_LINE_FRAC


# FIX 4 — switch-dispatch-arm boilerplate. A small, same-file group of `case X:`
# arms with no loop inside is dispatch boilerplate (e.g. a `gguf_dequant` type
# switch): real, but advisory, not slop-debt. Counted only as a `soft` signal.
# The NO-LOOP conjunct is the RECALL GUARD — a duplicated computation LOOP stays
# counted as a real clone.
DISPATCH_ARM_MAX_SPAN = 8
_CASE_ARM_RE = re.compile(r"^case\s+\S+\s*:\s*$")
_LOOP_TOKEN_RE = re.compile(r"\b(for|range)\b")


# FIX 5 — sort-scaffold-only windows. A window that calls `sort.Slice` /
# `sort.SliceStable` / `sort.Sort` / `sort.Strings` but carries NO real-logic token
# is just the sort scaffolding (the call boilerplate), not the comparator body — two
# unrelated sort sites share only the scaffold. Dropped at window-keying time.
#
# The RECALL GUARD is two-pronged (both are real-logic tells a copied body carries):
#   (1) a comparison operator (`<`/`!=`/… — a real shared comparator like dojo's
#       `a.CalibErr != b.CalibErr` stays), AND
#   (2) a loop / control keyword (`for`/`range`/`if`/`switch`/`select` — a copied
#       `for k := range m { … }; sort.Strings(…)` helper body like `featStr`/`sortedKeys`
#       carries `range`+`for` and stays; without this prong FIX 5 wrongly suppressed
#       genuinely copy-pasted sort-using helpers).
# A window is dropped ONLY when it has a sort verb and NEITHER prong fires — pure
# scaffolding, e.g. the `}` + `sort.Slice(rows, func(i,j int) bool {` head shared by
# two unrelated comparators.
_SORT_VERBS = frozenset({"Slice", "SliceStable", "Sort", "Strings"})
_COMPARE_TOKENS = frozenset({"<", ">", "<=", ">=", "!=", "=="})
_CONTROL_KEYWORDS = frozenset({"for", "range", "if", "switch", "select"})


# Entry-point scaffold (#FP-entry). A clone group whose EVERY site is a `cmd/*/main.go`
# command entry point is parallel-by-design CLI plumbing (the `flag` parse -> build ->
# WriteFile -> Fprintf(os.Stderr) skeleton each `fak` sub-binary repeats), NOT a hot-path
# logic clone. The CLONE_WINDOW_TOKENS docstring's own rule applies: de-duplicating two
# independent `main()`s into a shared helper couples unrelated binaries and worsens, not
# improves, readability. Demoted to ADVISORY (soft), never silently dropped, so the count
# stays visible. The conjunct is strict: ONE `internal/` site (real shipped kernel code)
# makes the `all(...)` False and the group stays HARD -- so a clone that leaked from a
# benchmark main into a hot path is still caught.
_CMD_MAIN_RE = re.compile(r"^cmd/[^/]+/main\.go$")


def _is_entry_point_only(sites: list[tuple[str, int, int]]) -> bool:
    """True iff every site of a clone group is a `cmd/*/main.go` command entry point --
    parallel-by-design CLI scaffolding, demoted to advisory rather than counted as debt.
    A single non-entry-point (any `internal/` or library) site makes this False."""
    return bool(sites) and all(_CMD_MAIN_RE.match(f) for f, _, _ in sites)


def _is_sort_scaffold_only(key: tuple[str, ...]) -> bool:
    """True iff the token window contains a `sort.<verb>` call (one of _SORT_VERBS)
    but NONE of the real-logic tells (a _COMPARE_TOKENS operator or a _CONTROL_KEYWORDS
    keyword) — pure sort scaffolding, no comparator body and no copied loop."""
    has_sort = False
    n = len(key)
    for i in range(n - 2):
        if key[i] == "sort" and key[i + 1] == "." and key[i + 2] in _SORT_VERBS:
            has_sort = True
            break
    if not has_sort:
        return False
    if any(t in _COMPARE_TOKENS for t in key):  # recall guard (1): comparator
        return False
    if any(t in _CONTROL_KEYWORDS for t in key):  # recall guard (2): copied control
        return False
    return True


def kpi_duplication(files: dict[str, str]) -> dict[str, Any]:
    """Copy-paste clones, measured on the normalized Go token stream (#780). For every
    file, slide a CLONE_WINDOW_TOKENS-token window over ``go_tokens`` output, keep only
    windows carrying >= CLONE_MIN_LOGIC_TOKENS logic tokens (so data/declaration regions
    are skipped without a hand-tuned line list), and key each by its normalized token
    sequence. A key seen at >= CLONE_MIN_OCCURRENCES distinct locations is a clone
    window. Per file the clone windows are merged by token adjacency into blocks, and
    blocks sharing any clone window are unioned into one group naming every site — one
    HARD unit of slop-debt each. `files` maps rel-path -> source text.

    Identifiers are kept (``normalize_idents=False``) so distinct code with distinct names
    does not false-match and idiomatic Go does not collapse into phantom clusters; only
    literals are normalized — the precision/recall sweet spot for this tree (#780)."""
    win_locs: dict[tuple[str, ...], set[tuple[str, int]]] = {}  # key -> {(file, start_line)}
    # file -> [(tok_idx, start_line, end_line, key)] for qualifying windows
    per_file: dict[str, list[tuple[int, int, int, tuple[str, ...]]]] = {}
    for rel in sorted(files):
        toks = go_tokens(files[rel], normalize_idents=False)
        m = len(toks)
        quals: list[tuple[int, int, int, int]] = []
        if m >= CLONE_WINDOW_TOKENS:
            # Per token: is_logic (as before) and is_nonassign_logic (logic AND not a
            # bare assignment op). FIX 6: bare `=`/`:=` are CONTEXT-GATED — a window's
            # EFFECTIVE logic count is its full logic count only when it carries >= 1
            # non-assignment logic token; otherwise the assignments don't count and the
            # window (a pure declaration/literal/field-init block) is demoted to zero.
            logic = [1 if t[2] else 0 for t in toks]
            nonassign = [1 if (t[2] and t[0] not in _ASSIGN_OPS) else 0 for t in toks]
            running = sum(logic[:CLONE_WINDOW_TOKENS])
            running_na = sum(nonassign[:CLONE_WINDOW_TOKENS])
            for start in range(0, m - CLONE_WINDOW_TOKENS + 1):
                if start > 0:
                    running += logic[start + CLONE_WINDOW_TOKENS - 1] - logic[start - 1]
                    running_na += (nonassign[start + CLONE_WINDOW_TOKENS - 1]
                                   - nonassign[start - 1])
                effective = running if running_na >= 1 else 0
                if effective < CLONE_MIN_LOGIC_TOKENS:
                    continue
                key = tuple(toks[j][0] for j in range(start, start + CLONE_WINDOW_TOKENS))
                # FIX 5: a sort-scaffold-only window (a `sort.<verb>` call with NO
                # comparison operator) is shared boilerplate, not a comparator clone.
                if _is_sort_scaffold_only(key):
                    continue
                sline = toks[start][1]
                eline = toks[start + CLONE_WINDOW_TOKENS - 1][1]
                quals.append((start, sline, eline, key))
                win_locs.setdefault(key, set()).add((rel, sline))
        per_file[rel] = quals

    clone_keys = {k for k, locs in win_locs.items() if len(locs) >= CLONE_MIN_OCCURRENCES}

    # Per file, merge clone windows that are adjacent in the token stream (a gap of up to
    # one window of non-clone tokens still merges) into one block. Counting raw windows
    # would inflate a single duplicated function into dozens of "clones".
    blocks: list[tuple[str, int, int, frozenset[tuple[str, ...]]]] = []  # (file, start_line, end_line, keyset)
    for rel in sorted(per_file):
        cw = [(idx, sl, el, k) for (idx, sl, el, k) in per_file[rel] if k in clone_keys]
        cur_idx = cur_sl = cur_el = None
        cur_keys: set[int] = set()
        for (idx, sl, el, k) in cw:  # already ascending in token index
            if cur_idx is None:
                cur_idx, cur_sl, cur_el, cur_keys = idx, sl, el, {k}
            elif idx - cur_idx <= CLONE_WINDOW_TOKENS:
                cur_idx, cur_el = idx, max(cur_el, el)
                cur_keys.add(k)
            else:
                blocks.append((rel, cur_sl, cur_el, frozenset(cur_keys)))
                cur_idx, cur_sl, cur_el, cur_keys = idx, sl, el, {k}
        if cur_idx is not None:
            blocks.append((rel, cur_sl, cur_el, frozenset(cur_keys)))

    # Union blocks that share any clone-window key — the copies of one block, even three
    # imperfectly-aligned copies, land in a single group.
    parent = list(range(len(blocks)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    key_to_blocks: dict[tuple[str, ...], list[int]] = {}
    for bi, (_, _, _, keys) in enumerate(blocks):
        for k in keys:
            key_to_blocks.setdefault(k, []).append(bi)
    for bis in key_to_blocks.values():
        for x in bis[1:]:
            parent[_find(x)] = _find(bis[0])

    comps: dict[int, list[int]] = {}
    for bi in range(len(blocks)):
        comps.setdefault(_find(bi), []).append(bi)

    # one group per component with >= CLONE_MIN_OCCURRENCES distinct sites
    group_sites: list[list[tuple[str, int, int]]] = []
    for bis in comps.values():
        sites = sorted({(blocks[b][0], blocks[b][1], blocks[b][2]) for b in bis})
        if len(sites) >= CLONE_MIN_OCCURRENCES:
            group_sites.append(sites)
    group_sites.sort(key=lambda s: s[0])  # deterministic: by first site

    # code-only lines per file (cached) for the FIX-4 in-span loop scan, so a `for`
    # inside a string/comment never false-hits the no-loop recall guard.
    code_cache: dict[str, list[str]] = {}

    def _code_of(rel: str) -> list[str]:
        if rel not in code_cache:
            code_cache[rel] = code_lines_of(files.get(rel, ""))
        return code_cache[rel]

    defects: list[str] = []
    soft: list[str] = []
    groups = 0
    for sites in group_sites:
        # FIX 1: signature-only func-header FP. When a window's whole 34-token span is
        # on a single line (e == s) that begins with `func `, it is a param-list/type
        # collision between two differently-named signatures — never a copied body.
        if all(e == s and _clone_sample(files.get(f, ""), s).startswith("func ")
               for f, s, e in sites):
            continue
        # FIX 3: flag/CLI-plumbing group. Drop iff EVERY site is a flag-plumbing block
        # (a positive flag-API token + a high fraction of flag-plumbing line shapes).
        # The mega-group of real `engine := fs.String` bodies survives: it has real-body
        # sites that are NOT pure flag-plumbing, so the `all(...)` is False.
        if all(_is_flag_plumbing_block(files.get(f, ""), s, e) for f, s, e in sites):
            continue
        # FIX 4: switch-dispatch-arm boilerplate -> advisory, not debt. Same-file group
        # of small `case X:` arms with NO loop inside (the no-loop conjunct is the recall
        # guard — a duplicated computation loop stays a real clone).
        same_file = len({f for f, _, _ in sites}) == 1
        all_case = all(
            _CASE_ARM_RE.match(_clone_sample(files.get(f, ""), s).lstrip())
            for f, s, e in sites)
        max_span = max(e - s + 1 for _, s, e in sites)
        no_loop = not any(
            _LOOP_TOKEN_RE.search(cl)
            for f, s, e in sites
            for cl in _code_of(f)[s - 1:e])
        if same_file and all_case and max_span <= DISPATCH_ARM_MAX_SPAN and no_loop:
            f0, s0, e0 = sites[0]
            span = max(1, e0 - s0 + 1)
            soft.append(
                f"dispatch-arm boilerplate ({len(sites)} arms, ~{span} lines): "
                f"{f0}:{s0}")
            continue
        # Pure-entry-point group -> advisory, not debt. Every site is a cmd/*/main.go
        # command skeleton; de-duplicating across independent binaries worsens
        # readability. One internal/ site keeps the group HARD (see _is_entry_point_only).
        if _is_entry_point_only(sites):
            f0, s0, e0 = sites[0]
            span = max(1, e0 - s0 + 1)
            soft.append(
                f"entry-point scaffold ({len(sites)} cmd mains, ~{span} lines): "
                f"{f0}:{s0}")
            continue
        if max(e - s + 1 for f, s, e in sites) < CLONE_MIN_GROUP_SPAN:
            continue  # sub-6-line fragment: idiomatic, not extractable slop
        groups += 1
        if len(defects) < CLONE_GROUPS_CAP:
            shown = ", ".join(f"{f}:{s}" for f, s, _ in sites[:4])
            more = f" (+{len(sites) - 4} more)" if len(sites) > 4 else ""
            f0, s0, e0 = sites[0]
            span = max(1, e0 - s0 + 1)
            sample = _clone_sample(files.get(f0, ""), s0)
            defects.append(
                f"clone x{len(sites)} (~{span} lines): {shown}{more} — '{sample[:60]}…'")
    score = _clamp(100 - 2 * groups)
    detail = ("no copy-paste clones" if groups == 0
              else f"{groups} duplicated block(s) (copy-pasted across 2+ sites)")
    return {"kpi": "duplication", "score": score, "detail": detail,
            "defects": defects, "soft": soft}


def _func_bodies(code: list[str]) -> list[tuple[str, int, list[str]]]:
    """Brace-depth split of code-only lines into (header, start_lineno, body_lines)
    for every top-level `func`. body_lines are the code-only lines strictly inside
    the outermost braces."""
    out: list[tuple[str, int, list[str]]] = []
    i, n = 0, len(code)
    while i < n:
        line = code[i]
        if re.match(r"^func\b", line.lstrip()) and line.lstrip() == line:
            header = line
            # Inline one-line func: opens AND closes on the header line. Extract the
            # body from between the first `{` and the last `}` so a `func f() { return
            # nil }` stub is still seen as a (trivial) body, not an empty one.
            if "{" in line and line.count("{") - line.count("}") <= 0 and "{" in line:
                first = line.find("{")
                last = line.rfind("}")
                inner = line[first + 1:last].strip() if last > first else ""
                body = [inner] if inner else []
                out.append((header, i + 1, body))
                i += 1
                continue
            # find the opening brace (may be on a later line for a multi-line sig)
            depth = 0
            opened = False
            body = []
            j = i
            while j < n:
                depth += code[j].count("{") - code[j].count("}")
                if "{" in code[j]:
                    opened = True
                if opened and j > i:
                    body.append(code[j])
                if opened and depth <= 0:
                    break
                j += 1
            # strip the trailing `}` line from body if present
            if body and body[-1].strip().startswith("}"):
                body = body[:-1]
            out.append((header, i + 1, body))
            i = j + 1
            continue
        i += 1
    return out


# --- vacuous_tests false-positive guards (FIX 2) ---------------------------
# Two shapes that assert through a channel the `t.*`/`require.`/`assert.` scan can't
# see, so a no-`t.*` body is NOT vacuous:
#   (a) a re-exec HELPER test — the `TestXHelperProcess` pattern: when invoked as a
#       child via os.Exec with an env guard set, it writes to os.Stdout/os.Stderr and
#       calls os.Exit; the PARENT test asserts on that out-of-band output/exit code.
#   (b) a *DoesNotPanic* test whose single statement is a bare call — the assertion IS
#       "this call returns without panicking"; the test fails (panics) if it doesn't.
_REEXEC_OOB_RE = re.compile(r"\b(os\.Stdout|os\.Stderr|fmt\.Fprint)")
_REEXEC_EXIT_RE = re.compile(r"\bos\.Exit\(")
# RAW source — code_only blanks the quotes, so this matches the original lines.
_REEXEC_ENVGUARD_RE = re.compile(
    r'if\s+os\.Getenv\([^)]*\)\s*(==|!=)\s*"[^"]*"\s*\{')
_NOPANIC_NAME_RE = re.compile(r"(?i)does_?not_?panic|no_?panic")
# a single bare call statement (a composite-literal arg with `{}` is allowed).
_SINGLE_CALL_RE = re.compile(r"^[\w.]+\([^;]*\)$")


def _is_reexec_helper_test(body_blob: str, raw_lines: list[str], lineno: int) -> bool:
    """True iff the func is a re-exec child helper: its code-only body calls os.Exit(
    AND writes out-of-band (os.Stdout/os.Stderr/fmt.Fprint), AND its first statement is
    an env-guard (`if os.Getenv(...) == "..." {`) whose next non-blank line is a bare
    `return` (the not-the-child early-out). `body_blob` is the code-only body joined;
    `raw_lines` is text.splitlines() (the env-guard match needs the RAW quotes)."""
    if not (_REEXEC_EXIT_RE.search(body_blob) and _REEXEC_OOB_RE.search(body_blob)):
        return False
    span = raw_lines[lineno - 1:lineno - 1 + 12]
    for i, rl in enumerate(span):
        if _REEXEC_ENVGUARD_RE.search(rl):
            nxt = next((s.strip() for s in span[i + 1:] if s.strip()), "")
            if nxt.startswith("return"):
                return True
    return False


def _is_does_not_panic_test(fname: str, body: list[str]) -> bool:
    """True iff the func name says *does not panic* / *no panic* AND its code-only body
    is exactly one statement that is a single bare call — the assertion is the absence
    of a panic, made by the call itself, not by a `t.*` observation."""
    if not _NOPANIC_NAME_RE.search(fname):
        return False
    real = [b.strip() for b in body if b.strip()]
    return len(real) == 1 and bool(_SINGLE_CALL_RE.match(real[0]))


def kpi_vacuous_tests(test_files: dict[str, str]) -> dict[str, Any]:
    """A ``Test*`` func whose body makes zero assertions/observations: no ``t.Error``/
    ``Fatal``/``Skip``/``Run``/…, no ``require.``/``assert.`` helper, AND not a
    compile-time interface guard (``var _ Iface = (*T)(nil)``, which fails the *build*
    if unsatisfied — a real assertion). Such a test runs and passes no matter what,
    which is worse than no test (a false green).

    ``Benchmark*`` and ``Fuzz*`` are NOT graded: a benchmark's job is to *time* a hot
    path (``b.N`` loop, ``b.ResetTimer``), and the fuzz engine drives ``Fuzz*`` — neither
    is expected to call ``t.Error``. Grading them vacuous would be a false positive.

    A test that DELEGATES its assertion to a helper — ``assertRefusal(t, …)``,
    ``mustEqual(t, …)`` — is NOT vacuous: passing the ``*testing.T`` into any function
    is the idiomatic way to assert through a shared check (the helper calls
    ``t.Fatal``). The detector treats any ``helper(<t-param>, …)`` / ``helper(<t-param>)``
    call as an assertion, using the receiver's actual parameter name."""
    # a compile-time interface/type guard — `var _ X = ...` — is a build-checked
    # assertion; a body that has one is NOT vacuous.
    guard_re = re.compile(r"\bvar\s+_\s+[\w\.\[\]\*]+\s*=")
    defects: list[str] = []
    n_tests = 0
    # capture the *testing.T parameter NAME so the assertion checks bind to it (a test
    # may name it `t`, `tt`, …).
    hdr_re = re.compile(r"^func\s+Test\w*\s*\(\s*(\w+)\s+\*?testing\.T")
    for rel in sorted(test_files):
        text = test_files[rel]
        code = code_lines_of(text)
        raw_lines = text.splitlines()
        for header, lineno, body in _func_bodies(code):
            m = hdr_re.match(header.lstrip())
            if not m:
                continue
            tname = re.escape(m.group(1))
            assert_re = re.compile(
                rf"\b{tname}\.(Error|Errorf|Fatal|Fatalf|Fail|FailNow|Skip|Skipf|"
                rf"Skipped|Run|Cleanup|Helper|Log|Logf|Setenv|Deadline|Parallel)\b|"
                r"\b(require|assert)\.\w+|"
                # delegated assertion: any helper that takes the t-param as an arg
                rf"\b\w+\(\s*{tname}\s*[,)]")
            n_tests += 1
            body_blob = "\n".join(body)
            if assert_re.search(body_blob) or guard_re.search(body_blob):
                continue
            fn = header.lstrip().split("(")[0].replace("func ", "").strip()
            # FIX 2: two NOT-slop shapes assert through a channel the t.* scan can't see —
            # a re-exec'd `*HelperProcess` child (reports via os.Exit/stdout) and a
            # does-not-panic test (the act of returning IS the assertion). Each has a
            # tight, non-forgeable tell; everything else still grades vacuous.
            if _is_reexec_helper_test(body_blob, raw_lines, lineno) \
                    or _is_does_not_panic_test(fn, body):
                continue
            defects.append(f"vacuous test (no assertion): {rel}:{lineno} {fn}")
    score = _clamp(100 - 10 * len(defects))
    detail = (f"{n_tests} Test func(s), all assert"
              if not defects else f"{len(defects)} vacuous of {n_tests} Test func(s)")
    return {"kpi": "vacuous_tests", "score": score, "detail": detail,
            "defects": defects, "soft": []}


# FIX 7: the author-asserted dead-code opt-out directive (#789). `//slop:keep` on the
# line immediately above an unexported decl marks it intentionally unreferenced (a
# provenance marker, an out-of-tree-witnessed contract const). A directive, not prose —
# unambiguous and not gameable by incidental wording. An optional reason may follow.
_SLOP_KEEP_RE = re.compile(r"^\s*//\s*slop:keep\b")


def kpi_dead_code(files: dict[str, str], test_files: dict[str, str]) -> dict[str, Any]:
    """An UNEXPORTED top-level symbol defined but referenced nowhere else in the
    first-party module (its identifier appears exactly once across all .go — its own
    definition). Exported symbols are part of the package API and not graded here
    (a static scan can't see external callers). Capped per file."""
    # token frequency across the WHOLE module (code + tests), code-only so a token
    # inside a string/comment is not a phantom reference.
    freq: dict[str, int] = {}
    all_text = {**files, **test_files}
    for rel in all_text:
        for c in code_lines_of(all_text[rel]):
            for tok in _IDENT_RE.findall(c):
                freq[tok] = freq.get(tok, 0) + 1

    defects: list[str] = []
    per_file: dict[str, int] = {}
    # only non-test files declare shipped symbols we grade
    for rel in sorted(files):
        code = code_lines_of(files[rel])
        raw = files[rel].splitlines()
        for idx, line in enumerate(code):
            s = line.lstrip()
            name = None
            for rx in (_FUNC_DECL_RE, _TYPE_DECL_RE, _VARCONST_DECL_RE):
                m = rx.match(s)
                if m:
                    name = m.group(1)
                    break
            if not name:
                continue
            if name == "_" or name[0].isupper():
                continue  # exported (API) or blank — skip
            if name in ("init", "main"):
                continue  # runtime entry points, never "referenced"
            if freq.get(name, 0) <= 1:
                # FIX 7: an explicit, conscious, non-gameable author opt-out. A symbol is
                # intentionally unreferenced (a symbol-table provenance marker, a contract
                # const checked only by an out-of-tree witness) when the line immediately
                # above its declaration is `//slop:keep [reason]`. Keyed on RAW source (a
                # directive, not prose) so it is unambiguous and not matched by incidental
                # wording — and it forces the author to state intent.
                if idx >= 1 and idx - 1 < len(raw) and _SLOP_KEEP_RE.match(raw[idx - 1]):
                    continue
                if per_file.get(rel, 0) >= DEAD_CAP_PER_FILE:
                    continue
                per_file[rel] = per_file.get(rel, 0) + 1
                if len(defects) < COMMENT_SLOP_CAP:
                    defects.append(f"dead unexported symbol (defined, never referenced): {rel} :: {name}")
    score = _clamp(100 - 5 * len(defects))
    detail = ("no dead unexported symbols"
              if not defects else f"{len(defects)} dead unexported symbol(s)")
    soft = ([f"dead-code capped at {DEAD_CAP_PER_FILE}/file across {len(per_file)} file(s)"]
            if any(v >= DEAD_CAP_PER_FILE for v in per_file.values()) else [])
    return {"kpi": "dead_code", "score": score, "detail": detail,
            "defects": defects, "soft": soft}


def _tautological(name: str, comment: str) -> bool:
    """A doc comment that only restates the symbol name with no added information:
    `// Foo does foo`, `// Bar bar`, `// Baz is a baz`. Heuristic: strip the leading
    name + a few filler words; if nothing of substance remains, it's tautological."""
    c = comment.strip()
    if not c or not c.lower().startswith(name.lower()):
        return False
    rest = c[len(name):].strip()
    # drop the most common vacuous connectors
    rest = re.sub(r"^(is|are|does|do|returns?|represents?|holds?|the|a|an|of|"
                  r"for|to|that|which|will|can|provides?)\b", "", rest, flags=re.I).strip()
    rest = re.sub(r"[\s.\-:]+", " ", rest).strip()
    # what remains, minus a single repeat of the name (camelCase split), tells us if
    # the comment added anything. <= 1 residual word == "// Foo does Foo".
    words = [w for w in rest.split(" ") if w and w.lower() != name.lower()]
    return len(words) <= 1


def kpi_comment_slop(files: dict[str, str]) -> dict[str, Any]:
    """Two comment defects: (a) a tautological doc comment immediately above an
    exported declaration that only restates the name; (b) a commented-out code block
    — COMMENTED_CODE_MIN_RUN+ consecutive `//` lines whose text parses as Go."""
    defects: list[str] = []
    for rel in sorted(files):
        lines = files[rel].splitlines()
        in_raw = in_block = False
        comments: list[tuple[int, str]] = []  # (lineno, comment_text)
        # whole_line[i] is True iff line i is a PURE `//` comment line (no code before
        # the `//`) whose comment text is NOT indented. A trailing field comment
        # (`X int // note`) or an indented doc-comment code block (`//\tinv_freq = …`,
        # the Go convention for a formatted example) is NOT commented-out code.
        whole_line: list[bool] = []
        codeonly: list[str] = []
        for raw in lines:
            pre_raw, _, _ = _code_only(raw, in_raw, in_block)
            ctext, in_raw, in_block = line_comment_of(raw, in_raw, in_block)
            comments.append((len(codeonly) + 1, ctext))
            co, _, _ = _code_only(raw, False, False)
            codeonly.append(co)
            # the text right after `//` (pre-strip): a leading tab or 2+ spaces is the
            # Go doc-comment formatted-example convention, not commented-out code.
            pos = raw.find("//")
            raw_after = raw[pos + 2:] if pos != -1 else ""
            indented = bool(re.match(r"(\t| {2,})", raw_after))
            is_pure = bool(ctext) and pre_raw.strip() == ""
            whole_line.append(is_pure and not indented)
        # (a) tautological doc comment above an exported decl
        code = code_lines_of(files[rel])
        for idx, line in enumerate(code):
            s = line.lstrip()
            name = None
            for rx in (_FUNC_DECL_RE, _TYPE_DECL_RE, _VARCONST_DECL_RE):
                m = rx.match(s)
                if m:
                    name = m.group(1)
                    break
            if not name or not name[0].isupper():
                continue
            # the immediately-preceding line's comment, but ONLY if it is the START of
            # the doc block (the line above it is NOT itself a comment). A `// Factory.`
            # that is the tail of a wrapped sentence ("...and need no\n// Factory.") is
            # prose continuation, not a tautological one-line doc — checking the
            # single-line-doc case avoids that false positive.
            if idx >= 1:
                _, prev_comment = comments[idx - 1]
                two_up_comment = comments[idx - 2][1] if idx >= 2 else ""
                is_block_start = not two_up_comment  # nothing comment-y above it
                if (prev_comment and is_block_start
                        and _tautological(name, prev_comment)):
                    if len(defects) < COMMENT_SLOP_CAP:
                        defects.append(
                            f"tautological doc comment: {rel}:{idx} '// {prev_comment[:50]}'")
        # (b) commented-out code runs. Only a WHOLE-LINE, non-indented `//` comment
        # counts (whole_line[idx]) — a trailing field comment (`X int // note`) and an
        # indented doc-comment example block are excluded. A shell/CLI usage example
        # (`// go run ./cmd/x`, `// $ fak serve`) is likewise NOT commented-out code.
        run_start = None
        run_len = 0
        for idx2, (ln, ctext) in enumerate(comments):
            pure = idx2 < len(whole_line) and whole_line[idx2]
            is_shell = bool(ctext) and bool(_SHELL_EXAMPLE_RE.match(ctext))
            # prose tell: a code fragment followed by a sentence (". Word") is doc
            # prose, not commented-out code (`x += body(x). PostNorm: ...`).
            is_prose = bool(ctext) and bool(re.search(r"\.\s+[A-Z]", ctext))
            is_codeish = (pure and bool(ctext) and not is_shell and not is_prose
                          and bool(_CODEISH_RE.match(ctext)))
            if is_codeish:
                if run_start is None:
                    run_start = ln
                run_len += 1
            else:
                if run_start is not None and run_len >= COMMENTED_CODE_MIN_RUN:
                    if len(defects) < COMMENT_SLOP_CAP:
                        defects.append(
                            f"commented-out code ({run_len} lines): {rel}:{run_start}")
                run_start = None
                run_len = 0
        if run_start is not None and run_len >= COMMENTED_CODE_MIN_RUN:
            if len(defects) < COMMENT_SLOP_CAP:
                defects.append(f"commented-out code ({run_len} lines): {rel}:{run_start}")
    score = _clamp(100 - 3 * len(defects))
    detail = ("no comment slop" if not defects
              else f"{len(defects)} comment-slop site(s)")
    return {"kpi": "comment_slop", "score": score, "detail": detail,
            "defects": defects, "soft": []}


# A code symbol inside a backtick span — `recall.Page`, `xenginekv.AttachArena`,
# `LookupOp`. CLAIMS.md backticks Go symbols and writes behaviors in prose, so keying
# the stub-suppression set on backtick spans (and matching a func by the LAST dotted
# component, case-sensitive) is the TIGHT symbol<->ledger link the SOFT->HARD promotion
# needs — prose words on a `[STUB]` line no longer grant a false free pass (#781).
_BACKTICK_SPAN_RE = re.compile(r"`([^`]+)`")
_DOTTED_IDENT_RE = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
_STUB_CLAIM_LINE_RE = re.compile(r"^\s*-\s+\[STUB\](?:\s|$)")


def _ledger_stub_symbols(claims_text: str) -> set[str]:
    """The set of Go symbol names a CLAIMS.md ``[STUB]`` line declares — collected ONLY
    from backtick-quoted code spans, taking the last dotted component of each
    ``pkg.Symbol`` / ``Type.Method`` so a func matches by its bare exported name.
    Case-sensitive (Go symbols are). Prose on the line is ignored — that is the tight
    link (vs the v1 any-token-lowercased match that pulled in prose words)."""
    symbols: set[str] = set()
    for line in claims_text.splitlines():
        if not _STUB_CLAIM_LINE_RE.match(line):
            continue
        for span in _BACKTICK_SPAN_RE.findall(line):
            for dotted in _DOTTED_IDENT_RE.findall(span):
                symbols.add(dotted.rsplit(".", 1)[-1])
    return symbols


def _parse_release(v: str) -> tuple[int, int] | None:
    """(major, minor) of a ``MAJOR.MINOR[.PATCH]`` version string (a leading ``v`` is
    tolerated), or ``None`` when unparseable. Callers treat ``None`` as fail-safe: an
    unknown version yields a NOT-met soak, never a false promotion."""
    m = re.match(r"\s*v?(\d+)\.(\d+)", v or "")
    return (int(m.group(1)), int(m.group(2))) if m else None


def stub_promotion_status(current_version: str, n_findings: int) -> dict[str, Any]:
    """Re-derivable readiness of the stub_masquerade SOFT->HARD promotion (#781).

    Gate 1 — the symbol<->[STUB]-ledger LINK is tight — shipped in fc7449d, so it is
    always met. Gate 2 — zero false positives across a SOAK window of a few releases —
    is computed here from objective on-disk facts: the release the detector first
    shipped in (``STUB_DETECTOR_SHIP_RELEASE``), the current ``VERSION``, and the live
    finding count. ``releases_since_ship`` counts minor bumps since the ship release on
    the same major line (a cross-major jump or an unparseable version fails SAFE to 0).

    ``soak_met`` is a NECESSARY-not-sufficient mechanical signal: the window has elapsed
    AND the tree is currently clean. ``promotable`` mirrors it — it tells a maintainer
    the window is up and the tree is clean, so they may REVIEW the elapsed window for
    any false positive and then flip (move ``soft`` -> ``defects`` + bump the weight). It
    does NOT itself flip the axis; the readout is advisory, the flip stays a human act."""
    cur = _parse_release(current_version)
    ship = _parse_release(STUB_DETECTOR_SHIP_RELEASE)
    if cur and ship and cur[0] == ship[0] and cur[1] >= ship[1]:
        releases_since = cur[1] - ship[1]
    else:
        releases_since = 0
    soak_met = releases_since >= STUB_SOAK_RELEASES and n_findings == 0
    if soak_met:
        status = ("READY: soak window elapsed and tree clean — a maintainer may review "
                  "the window for zero FP, then move soft->defects + bump the weight")
    else:
        status = (f"AWAITING SOAK: {releases_since}/{STUB_SOAK_RELEASES} release(s) since "
                  f"the detector shipped ({STUB_DETECTOR_SHIP_RELEASE}); stays SOFT (advisory)")
    return {
        "link_tight": True,
        "ship_release": STUB_DETECTOR_SHIP_RELEASE,
        "current_release": current_version or "unknown",
        "soak_releases_required": STUB_SOAK_RELEASES,
        "releases_since_ship": releases_since,
        "live_findings": n_findings,
        "soak_met": soak_met,
        "promotable": soak_met,
        "status": status,
    }


def kpi_stub_masquerade(files: dict[str, str], claims_text: str,
                        version: str = "") -> dict[str, Any]:
    """SOFT (v1, advisory). An EXPORTED func whose body explicitly declares itself
    UNIMPLEMENTED — ``panic("not implemented")`` / ``panic("unimplemented")`` / a
    body whose only statement is a ``// TODO: implement`` — where the symbol is not
    declared on a ``[STUB]`` line in ``CLAIMS.md`` (the honesty ledger). Deliberately
    NARROW on the DETECTION side: a bare ``return nil`` is NOT counted, because on an
    interface method that is overwhelmingly a legitimate "no capabilities / empty
    result" implementation, not a stub (the broad form produced ~40 false positives on
    this tree — trivial accessors like ``func (m *MMU) Caps() []abi.Capability { return
    nil }``). An explicit unimplemented-panic, by contrast, is unambiguous.

    The symbol<->ledger link (the SUPPRESSION side) is now TIGHT (#781): a ``[STUB]``
    line suppresses a func only when the func's exact (case-sensitive) Go name appears
    as a BACKTICK-quoted code symbol on that line — ``- [STUB] `recall.Page` …``
    suppresses ``Page``; ``- [STUB] `xenginekv.AttachArena` …`` suppresses
    ``AttachArena`` (the last dotted component of a ``pkg.Symbol`` / ``Type.Method``
    span). CLAIMS.md names behaviors in PROSE and Go symbols in BACKTICKS, so keying
    only on the backticked code spans drops the prose-fuzz: a stub named after a common
    word ("Path", "Op", "Result") is no longer granted a false free pass by an
    unrelated sentence, and a declared symbol is reliably matched. (v1 added every
    identifier-shaped token on the line, lowercased — prose included.)

    Still SOFT: the SOFT->HARD promotion (move ``soft`` -> ``defects`` + bump the KPI
    weight) is gated on a SECOND condition the link-tightening alone does not satisfy —
    zero false positives confirmed on the tree across a few releases. The detector first
    shipped in 53e4d5f, contained only in release ``STUB_DETECTOR_SHIP_RELEASE``, so the
    soak window opens there; the tree currently has ZERO exported panic-stubs. The
    readiness of that second gate is now SELF-REPORTING, not a prose vibe: the returned
    ``promotion`` block (``stub_promotion_status``) carries a re-derivable
    ``releases_since_ship`` / ``soak_met`` / ``promotable`` readout so an agent or
    operator can check the soak status (and the eventual flip is evidence-bound) without
    re-deriving it by hand. The readout never gates and never flips the axis on its own.
    See #781 / the #775 Track-B epic."""
    stub_symbols = _ledger_stub_symbols(claims_text)

    todo_only_re = re.compile(r"^\s*//\s*TODO[:\s].*implement", re.I)
    panic_stub_re = re.compile(
        r'panic\(\s*"[^"]*(not implemented|unimplemented|not yet implemented|TODO)',
        re.I)

    soft: list[str] = []
    for rel in sorted(files):
        code = code_lines_of(files[rel])
        raw_lines = files[rel].splitlines()
        for header, lineno, body in _func_bodies(code):
            s = header.lstrip()
            m = _FUNC_DECL_RE.match(s)
            if not m:
                continue
            name = m.group(1)
            if not name or not name[0].isupper():
                continue  # only exported funcs "masquerade"
            # The panic/TODO text lives in a STRING literal, which `code_only` blanks —
            # so scan the ORIGINAL source lines across the func's span (bounded; a stub
            # is short). `lineno` is 1-based at the header.
            real = [b for b in body if b.strip()]
            span = max(len(body) + 2, 6)
            raw_span = "\n".join(raw_lines[lineno - 1:lineno - 1 + span])
            is_panic_stub = bool(panic_stub_re.search(raw_span))
            # a body whose ONLY statement is `// TODO: implement` (no real code).
            is_todo_only = (not real and any(
                todo_only_re.match(rl) for rl in raw_lines[lineno:lineno + 4]))
            if is_panic_stub or is_todo_only:
                if name in stub_symbols:
                    continue  # honestly declared (by backtick symbol) in the ledger
                why = "panic-unimplemented" if is_panic_stub else "TODO-only body"
                soft.append(f"possible stub-masquerade (exported, {why}, not [STUB]): {rel}:{lineno} {name}")
    score = _clamp(100 - 4 * len(soft))
    detail = ("no exported stub-masquerade" if not soft
              else f"{len(soft)} possible stub-masquerade(s) [SOFT]")
    # The `promotion` block is advisory metadata (the #781 SOFT->HARD readiness): an
    # extra key the fold/render ignore for scoring — it changes no score, defect, or
    # snapshot, it only surfaces the soak status.
    return {"kpi": "stub_masquerade", "score": score, "detail": detail,
            "defects": [], "soft": soft,
            "promotion": stub_promotion_status(version, len(soft))}


def kpi_churn_bloat(added: int, removed: int, n_commits: int) -> dict[str, Any]:
    """SOFT, HEAD-relative. Over the recent commit range, how many .go files were
    ADDED vs REMOVED. A healthy tree retires as it grows; a steadily-accreting tree
    (added with ~zero removed) is the bloat signal. Advisory only."""
    soft: list[str] = []
    if n_commits == 0:
        detail = "no commits in range (skipped)"
    elif added == 0:
        detail = f"no .go files added in {n_commits} commit(s)"
    else:
        removed / added if added else 0.0
        detail = f"{added} .go added / {removed} removed over {n_commits} commit(s)"
        if added >= 8 and removed == 0:
            soft.append(f"accretion: {added} .go files added, 0 removed over {n_commits} commits "
                        "(net growth with no retirement)")
    # score nudges down with pure accretion but never zeros (it's a SOFT trend)
    if added and removed == 0 and added >= 8:
        score = 80
    else:
        score = 100
    return {"kpi": "churn_bloat", "score": score, "detail": detail,
            "defects": [], "soft": soft}


# ---------------------------------------------------------------------------
# Fold
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, kpis: list[dict[str, Any]],
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA, "ok": False, "verdict": "AUDIT_ERROR",
            "finding": "tooling_error", "reason": error,
            "next_action": "fix the read (run from repo ROOT), then re-run",
            "workspace": workspace, "corpus": {}, "kpis": [],
        }
    by_name = {k["kpi"]: k for k in kpis}
    score = sum(KPI_WEIGHTS[name] * by_name[name]["score"]
                for name in KPI_WEIGHTS if name in by_name)
    score = round(score, 1)
    slop_debt = sum(len(k["defects"]) for k in kpis)
    n_soft = sum(len(k["soft"]) for k in kpis)
    grade = grade_letter(score)
    breakdown = sorted(
        ({"kpi": k["kpi"], "score": k["score"], "debt": len(k["defects"]),
          "detail": k["detail"]} for k in kpis),
        key=lambda x: (-x["debt"], x["score"]))

    corpus = {
        "score": score,
        "grade": grade,
        "slop_debt": slop_debt,
        "soft_signals": n_soft,
        "kpi_scores": {k["kpi"]: k["score"] for k in kpis},
        "debt_by_kpi": {k["kpi"]: len(k["defects"]) for k in kpis},
        "breakdown": breakdown,
    }
    # Lift the stub_masquerade SOFT->HARD promotion readiness (#781) to the corpus
    # summary the control-pane reads, so the soak status is reachable without walking
    # the kpis array — and a future automated check can act on `promotable` to prompt
    # the (deliberate) flip. Absent on fixtures without the block; never affects scoring.
    stub_promotion = (by_name.get("stub_masquerade") or {}).get("promotion")
    if stub_promotion:
        corpus["stub_masquerade_promotion"] = stub_promotion

    if slop_debt == 0:
        ok, verdict, finding = True, "OK", "code_slop_clean"
        reason = (f"no code slop: score {score}/100 (grade {grade}), zero slop-debt "
                  f"across {len(kpis)} KPIs ({n_soft} advisory signal(s))")
        next_action = "no required edit; re-run after the next code change"
    else:
        ok, verdict, finding = False, "ACTION", "code_slop"
        worst = breakdown[0]
        reason = (f"{slop_debt} unit(s) of slop-debt; score {score}/100 (grade {grade}); "
                  f"heaviest KPI: {worst['kpi']} ({worst['debt']} defect(s))")
        next_action = ("retire slop-debt worst-first (see corpus.breakdown + per-KPI defects): "
                       "de-duplicate clones, delete dead unexported symbols, drop "
                       "commented-out code + tautological doc comments, add assertions to "
                       "vacuous tests; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "corpus": corpus, "kpis": kpis,
    }


# ---------------------------------------------------------------------------
# Disk gathering (the impure shell around the pure KPIs)
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _excluded_go(rel: str) -> bool:
    parts = set(Path(rel).parts)
    return bool(parts & GO_EXCLUDE_DIRS)


def gather_go(root: Path) -> tuple[dict[str, str], dict[str, str]]:
    """(files, test_files): rel-path -> source text for first-party .go, split into
    non-test and _test.go. Walks the tree (not git) so an uncommitted change scores."""
    files: dict[str, str] = {}
    test_files: dict[str, str] = {}
    for p in root.rglob("*.go"):
        rel = p.relative_to(root).as_posix()
        if _excluded_go(rel):
            continue
        text = _safe_read(p)
        if rel.endswith("_test.go"):
            test_files[rel] = text
        else:
            files[rel] = text
    return files, test_files


def git_churn(root: Path, rev_range: str) -> tuple[int, int, int]:
    """(added, removed, n_commits) of .go files over rev_range, via a read-only
    `git log --diff-filter`. Fail-open to (0,0,0) when git/range is unavailable."""
    def _count(diff_filter: str) -> int:
        try:
            p = subprocess.run(
                ["git", "log", rev_range, f"--diff-filter={diff_filter}",
                 "--name-only", "--pretty=format:", "--", "*.go"],
                cwd=str(root), capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.SubprocessError):
            return 0
        if p.returncode != 0:
            return 0
        return len({ln.strip() for ln in p.stdout.splitlines() if ln.strip()})

    try:
        c = subprocess.run(["git", "rev-list", "--count", rev_range],
                           cwd=str(root), capture_output=True, text=True, timeout=30)
        n_commits = int(c.stdout.strip()) if c.returncode == 0 and c.stdout.strip() else 0
    except (OSError, subprocess.SubprocessError, ValueError):
        n_commits = 0
    return _count("A"), _count("D"), n_commits


def collect(workspace: Path, *, run_churn: bool = True,
            churn_range: str = "HEAD~20..HEAD") -> dict[str, Any]:
    try:
        files, test_files = gather_go(workspace)
    except OSError as exc:
        return build_payload(workspace=str(workspace), kpis=[],
                             error=f"failed to read .go files: {exc}")
    if not files and not test_files:
        return build_payload(workspace=str(workspace), kpis=[],
                             error="no first-party .go files found (run from repo ROOT)")
    claims_text = _safe_read(workspace / CLAIMS_REL)
    version_text = _safe_read(workspace / VERSION_REL).strip()

    if run_churn:
        added, removed, n_commits = git_churn(workspace, churn_range)
    else:
        added, removed, n_commits = 0, 0, 0

    kpis = [
        kpi_duplication(files),
        kpi_dead_code(files, test_files),
        kpi_comment_slop(files),
        kpi_vacuous_tests(test_files),
        kpi_stub_masquerade(files, claims_text, version_text),
        kpi_churn_bloat(added, removed, n_commits),
    ]
    return build_payload(workspace=str(workspace), kpis=kpis)


# ---------------------------------------------------------------------------
# Render (human + markdown snapshot + doc-freshness check)
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    lines = [
        f"code-slop-scorecard: {payload.get('verdict')} ({payload.get('finding')})",
        f"  {payload.get('reason')}",
        "",
    ]
    if c:
        lines.append(f"  slop_score {c.get('score')}/100 (grade {c.get('grade')}) · "
                     f"slop-debt {c.get('slop_debt')} · {c.get('soft_signals')} soft signal(s)")
        lines.append("")
        lines.append("  per-KPI (worst-first):")
        for b in c.get("breakdown", []):
            flag = "HARD" if b["debt"] else "ok"
            lines.append(f"    {b['kpi']:<16} {b['score']:>3}/100  debt {b['debt']:<3} [{flag}]  {b['detail']}")
        # name the first few defects worst-first
        debts: list[tuple[str, str]] = []
        for k in payload.get("kpis", []):
            for d in k["defects"]:
                debts.append((k["kpi"], d))
        if debts:
            lines.append("")
            lines.append("  slop-debt work-list (first 20):")
            for kpi, d in debts[:20]:
                lines.append(f"    - [{kpi}] {d}")
            if len(debts) > 20:
                lines.append(f"    … +{len(debts) - 20} more")
        softs: list[tuple[str, str]] = []
        for k in payload.get("kpis", []):
            for s in k["soft"]:
                softs.append((k["kpi"], s))
        if softs:
            lines.append("")
            lines.append("  advisory (SOFT, never gates):")
            for kpi, s in softs[:12]:
                lines.append(f"    · [{kpi}] {s}")
        # stub_masquerade SOFT->HARD promotion readiness (#781) — advisory, never gates.
        for k in payload.get("kpis", []):
            promo = k.get("promotion")
            if promo:
                lines.append("")
                lines.append(f"  stub_masquerade promotion (#781): {promo.get('status')}")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    """The committed docs/CODE-SLOP-SCORECARD.md body — a human-facing snapshot."""
    c = payload.get("corpus") or {}
    out: list[str] = [
        "---",
        'title: "fak Code-Slop Scorecard: the slop the compiler can\'t see"',
        ('description: "fak\'s code-slop scorecard grades the Go module on six '
         'deterministic slop axes into a slop-score (0-100, A-F) and a re-derivable '
         'slop-debt count — clones, vacuous tests, dead code, comment cruft."'),
        "---",
        "",
        "# Code-slop scorecard",
        "",
    ]
    if stamp:
        out.append(f"<!-- code-slop-scorecard: {stamp} · process: tools/code_slop_scorecard.py -->")
        out.append("")
    out.append("> Regenerate: `python tools/code_slop_scorecard.py --markdown --stamp DATE "
               "> docs/CODE-SLOP-SCORECARD.md`")
    out.append("> Verify snapshot freshness: `python tools/code_slop_scorecard.py --check-doc`")
    out.append("")
    out.append("> The measuring stick for **slop the compiler can't see**: code that builds, "
               "vets clean, and has a test present, yet rots the kernel from the inside — "
               "copy-paste clones, tests that assert nothing, dead unexported symbols, "
               "commented-out code and tautological doc comments. Six deterministic axes "
               "(duplication · dead_code · comment_slop · vacuous_tests · stub_masquerade · "
               "churn_bloat), folded into a **slop-score** (0–100, A–F) and a **slop-debt** "
               "integer (the count of concrete, re-derivable slop defects). Every number "
               "below is re-derived from disk by `tools/code_slop_scorecard.py` — no "
               "hand-entry. Drive slop-debt to zero to make \"less slop\" provable.")
    out.append("")
    out.append("## Corpus")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Slop-score | {c.get('score', 0)}/100 (grade {c.get('grade', '?')}) |")
    out.append(f"| **Slop-debt (total HARD defects)** | **{c.get('slop_debt', 0)}** |")
    out.append(f"| Soft signals (advisory) | {c.get('soft_signals', 0)} |")
    out.append("")
    out.append("## Per-KPI (worst-first)")
    out.append("")
    out.append("| KPI | Score | Slop-debt | Detail |")
    out.append("|---|---:|---:|---|")
    for b in c.get("breakdown", []):
        out.append(f"| {b['kpi']} | {b['score']}/100 | {b['debt']} | {b['detail']} |")
    out.append("")
    out.append("## What each axis catches")
    out.append("")
    out.append("- **duplication** — a normalized Go token-window copy-pasted into 2+ places. [HARD]")
    out.append("- **dead_code** — an unexported symbol defined but referenced nowhere else. [HARD]")
    out.append("- **comment_slop** — tautological doc comments + commented-out code blocks. [HARD]")
    out.append("- **vacuous_tests** — a Test/Benchmark func that makes zero assertions. [HARD]")
    out.append("- **stub_masquerade** — an exported func with a trivial/panic body, not `[STUB]`. [SOFT]")
    out.append("- **churn_bloat** — recent commits adding `.go` files without retiring any. [SOFT]")
    # stub_masquerade SOFT->HARD promotion readiness (#781) — advisory, never gates. The
    # --json/corpus and text paths already surface it; the committed doc is the canonical
    # operator-facing artifact, so the readout (and the eventual flip procedure) belongs
    # here too. Rendered only when the promotion block is present (build_payload lifts it
    # from the stub_masquerade KPI), so fixtures without it emit no empty section.
    promo = c.get("stub_masquerade_promotion")
    if promo:
        out.append("")
        out.append("## stub_masquerade SOFT->HARD promotion (#781)")
        out.append("")
        out.append("> Advisory readiness for promoting the `stub_masquerade` axis from SOFT "
                   "(scores, never gates) to HARD (a gating defect). Re-derived from disk; the "
                   "readout never performs the flip — moving the finding from `soft` to "
                   "`defects` and bumping its weight stays a deliberate maintainer act, taken "
                   "once the elapsed soak window is reviewed for zero false positives.")
        out.append("")
        out.append("| Gate | State |")
        out.append("|---|---|")
        out.append(f"| symbol<->`[STUB]`-ledger link tight | {'yes' if promo.get('link_tight') else 'no'} |")
        out.append(f"| zero-FP soak (releases since {promo.get('ship_release')}) | "
                   f"{promo.get('releases_since_ship')}/{promo.get('soak_releases_required')} |")
        out.append(f"| promotable now | {'yes' if promo.get('promotable') else 'no'} |")
        out.append("")
        out.append(f"> {promo.get('status')}")
        out.append("")
        out.append("> When `promotable` is yes: review the elapsed window for any false "
                   "positive, then move the `stub_masquerade` finding from `soft` to `defects` "
                   "and bump `KPI_WEIGHTS[\"stub_masquerade\"]` in "
                   "`tools/code_slop_scorecard.py` — the deliberate flip.")
    out.append("")
    out.append(f"> {payload.get('reason', '')}")
    out.append("")
    out.append(f"> next: {payload.get('next_action', '')}")
    return "\n".join(out)


def markdown_stamp(text: str) -> str:
    m = STAMP_RE.search(text)
    return m.group("stamp").strip() if m else ""


def check_markdown_doc(workspace: Path, payload: dict[str, Any], *,
                       doc: str = SCORECARD_DOC) -> dict[str, Any]:
    path = workspace / doc
    try:
        actual = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "doc": doc, "stamp": "", "reason": f"read {doc}: {exc}", "diff": []}
    stamp = markdown_stamp(actual)
    if not stamp:
        return {"ok": False, "doc": doc, "stamp": "", "reason": "scorecard stamp missing", "diff": []}
    expected = render_markdown(payload, stamp=stamp)
    if actual.rstrip() == expected.rstrip():
        return {"ok": True, "doc": doc, "stamp": stamp,
                "reason": f"{doc} matches generated markdown using stamp {stamp}", "diff": []}
    diff = list(difflib.unified_diff(
        actual.splitlines(), expected.splitlines(),
        fromfile=doc, tofile=f"{doc} (generated)", lineterm=""))
    return {"ok": False, "doc": doc, "stamp": stamp,
            "reason": f"{doc} is stale; regenerate with --markdown --stamp {stamp}",
            "diff": diff[:60]}


def render_doc_check(check: dict[str, Any]) -> str:
    status = "OK" if check.get("ok") else "ACTION"
    lines = [
        f"code-slop scorecard doc: {status} "
        f"({'scorecard_doc_fresh' if check.get('ok') else 'scorecard_doc_drift'})",
        f"  {check.get('reason', '')}",
    ]
    if check.get("diff"):
        lines.append("")
        lines.append("diff:")
        lines.extend("  " + line for line in check["diff"])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Code-slop scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true",
                    help="emit the CODE-SLOP-SCORECARD.md body")
    ap.add_argument("--check-doc", action="store_true",
                    help=f"fail if {SCORECARD_DOC} differs from generated markdown")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--no-toolchain", action="store_true",
                    help="parity flag — this scorecard is already static (no-op)")
    ap.add_argument("--range", dest="churn_range", default="HEAD~20..HEAD",
                    help="git range for the SOFT churn_bloat axis")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    # The committed snapshot must be TREE-deterministic (same tree -> same doc), so the
    # snapshot/doc-check paths exclude the HEAD-relative SOFT churn axis (it would drift
    # the doc every commit). The human/json paths include it — it's a live advisory.
    snapshot_mode = args.markdown or args.check_doc
    payload = collect(workspace, run_churn=not snapshot_mode,
                      churn_range=args.churn_range)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    elif args.check_doc:
        check = check_markdown_doc(workspace, payload)
        print(render_doc_check(check))
        return 0 if check.get("ok") and payload.get("ok") else 1
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
