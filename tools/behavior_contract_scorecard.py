#!/usr/bin/env python3
"""behavior_contract_scorecard — the contract-vs-change-detector testing stick.

Hermes' rubric (the inspiration, epic #2871 / #2900): *behavior contracts over
snapshots* — a test should assert how two pieces of data must RELATE (an
invariant), not FREEZE a current value (a model list, a config version literal,
an enumeration count). A change-detector test breaks on every legitimate change
and asserts nothing real: `len(models) != 7` fails the day a real model is added
and proves nothing about what a model list must satisfy.

fak already has a scorecard culture (docs / code-slop / intent-literal). This is
the testing-quality sibling: a DETECTOR that classifies every `Test*` func in the
Go suite as a **contract** (asserts a relation/invariant) or a **change-detector**
(only pins a literal / count / enumeration) and reports the change-detector DEBT
to retire — the number a `/scorecard`-style loop drives to zero.

The classifier is deliberately CONSERVATIVE — it flags only the shapes Hermes
names, and it EXONERATES a test the moment it also carries a real invariant, so a
disciplined test is never debt:

  contract          the func has >= 1 relational/invariant assertion (see below) [GOOD]
  change_detector   the func has >= 1 literal/count/enum FREEZE and ZERO relations [DEBT]
  unclassified      neither a named freeze nor a relation (a scalar example compare,
                    an err!=nil guard, a helper-delegated assertion) — NOT counted   [neutral]

A CHANGE-DETECTOR freeze (a `pin`) is one of three Hermes-named shapes only:
  count    `len(x) == N` / `len(x) != N` with N >= 2 — an enumeration-count freeze.
           (`== 0` / `!= 0` is emptiness and `== 1` is a singleton — structural bounds,
           not enumeration freezes — so they are NOT pins.)
  version  `x == "v1.2"` / `x != "1.4.0"` — a config/version literal freeze (the RHS
           matches a version shape `v?\\d+\\.\\d+`).
  enum     `reflect.DeepEqual(got, []string{...})` / `slices.Equal(got, map{...})` — a
           frozen literal COLLECTION (a model/enumeration list) on one side.

A RELATION (an invariant — what EXONERATES a test as a contract) is one of:
  both-computed   `got != want` / `a.Total != a.Used+a.Free` / `len(a) != len(b)` — a
                  comparison whose BOTH sides are non-literal (neither is a frozen
                  literal, a locally literal-bound var, nor `nil`).
  deep-equal      `reflect.DeepEqual(a, b)` / `slices.Equal(a, b)` comparing two
                  COMPUTED values (no literal collection on either side).
  ordering        `sort.IsSorted(...)` / `sort.SliceIsSorted(...)` / `slices.IsSorted(...)`
                  — an ordering invariant.

Precision guards (why this stays low-false-positive):
  * Only assertion comparisons count — a comparison is an assertion iff it is the
    condition of an `if` whose block calls `t.Error/Errorf/Fatal/Fatalf/Fail/FailNow`
    on the test's own `*testing.T`. A `for i := 0; i < len(x); i++` loop bound is not
    an assertion, so it never mislabels a func as a contract.
  * A bare scalar compare (`got != 3`, `got != "ok"`) is UNCLASSIFIED, not a pin —
    we cannot tell a frozen enumeration from a meaningful expected value, so an
    ordinary example-based unit test is never flagged. Only the three named freezes
    (count / version / enum-list) are debt.
  * A locally literal-bound var is resolved: `n := 3; if len(x) != n` is a count
    freeze; `want := "v1.2"; if v != want` is a version freeze. Struct fields
    (`tt.want`) are NOT literal-bound, so table-driven tests stay relations.

The stick is static, pure-stdlib Go analysis — no `go` shell, no network, no build —
so it is gate-safe. It grades TRACKED `_test.go` only (git ls-files), reading
working-tree bytes, and drops the scratch checkouts under `.claude`/`.fak`/`.dos`
exactly as the sibling scorecards do. Run from the repo ROOT::

    python tools/behavior_contract_scorecard.py           # human work-list
    python tools/behavior_contract_scorecard.py --json     # machine payload

Exit is 0 at zero change-detector debt, 1 otherwise — the same honest bare-run
signal the code-slop scorecard uses, so CI can gate the unit test HARD and run the
detector ADVISORY over the tree.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

SCHEMA = "fleet-behavior-contract-scorecard/1"

# A collection compared to an exact count of >= this many elements is an
# enumeration-count FREEZE. 0 (emptiness) and 1 (a singleton) are structural
# bounds, not frozen enumerations, so they stay below the floor and are NOT pins.
COUNT_MIN = 2
# Penalty per change-detector defect. Ceiling 100 (zero debt), unbounded BELOW —
# a floor at 0 would hide partial progress on a heavy-debt tree (the same reason
# code_slop_scorecard runs its KPI scores negative).
PENALTY_PER_DEBT = 4
# Runaway backstop on the printed work-list only — never a cap on the debt integer.
DEFECT_WORKLIST_CAP = 200

# The test's own *testing.T failure/observation methods. A comparison guarded by an
# `if` whose block calls one of these on the test param is an ASSERTION.
FAIL_METHODS = frozenset({"Error", "Errorf", "Fatal", "Fatalf", "Fail", "FailNow"})
# Deep-equality calls: a relation when comparing two computed values, an enum-list
# freeze when one side is a literal collection.
DEEPEQUAL_FUNCS = frozenset({"DeepEqual", "Equal"})  # reflect.DeepEqual, slices.Equal, maps.Equal
DEEPEQUAL_PKGS = frozenset({"reflect", "slices", "maps"})
# Ordering-invariant calls — always a relation.
ORDERING_FUNCS = frozenset({"IsSorted", "SliceIsSorted"})  # sort.*, slices.IsSorted

_VERSION_RE = re.compile(r"^v?\d+\.\d+")
_CMP_OPS = frozenset({"==", "!=", "<", "<=", ">", ">="})
_EQ_OPS = frozenset({"==", "!="})

# Directories whose _test.go is NOT first-party shipped kernel code — the same
# scratch/vendored trees the code-slop scorecard drops (agent worktrees, dispatch
# clean-clones, the DOS isolation build, vendored/generated code, fixtures).
GO_EXCLUDE_DIRS = {".git", ".claude", ".fak", ".dos", ".tmp", "node_modules", "testdata", "vendor", "__pycache__"}

# A top-level `func TestXxx(<name> *testing.T` header (col-0 only, so subtests /
# closures are analysed within their outer func, never as their own entry).
_TESTHDR_RE = re.compile(r"(?m)^func\s+(Test\w*)\s*\(\s*(\w+)\s+\*testing\.T\b")
# A locally literal-bound variable: `x := 3`, `want = "v1.2"`, `const n = 5`. The
# RHS is exactly ONE bare literal (a compound `x := a + 1` or a struct field does
# NOT match), so struct-fielded table-driven `want`s stay non-literal (relations).
_LIT_VAR_RE = re.compile(
    r"^\s*(?:const\s+|var\s+)?(\w+)\s*(?::=|=)\s*"
    r"(-?0[xX][0-9a-fA-F]+|-?\d+\.\d+|-?\d+|\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'|true|false)"
    r"\s*(?://.*)?$"
)

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


# ---------------------------------------------------------------------------
# io / gather (the impure shell, kept thin — mirrors the sibling scorecards)
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    return (start or Path(__file__)).resolve().parent.parent


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


def _corrupt_path(rel: str) -> bool:
    return any(0xE000 <= ord(c) <= 0xF8FF or ord(c) < 0x20 for c in rel)


def _excluded(rel: str) -> bool:
    if _corrupt_path(rel):
        return True
    return bool(set(Path(rel).parts) & GO_EXCLUDE_DIRS)


def _git_tracked_test_paths(root: Path) -> list[Path] | None:
    """Tracked `_test.go` paths for a git checkout, or None when git is unavailable."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "-z", "--", "*_test.go"],
            cwd=str(root), capture_output=True, timeout=15,
            creationflags=_win_creationflags(),
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    paths: list[Path] = []
    for raw in proc.stdout.split(b"\0"):
        if not raw:
            continue
        rel = raw.decode("utf-8", "surrogateescape")
        if rel.endswith("_test.go") and not _excluded(rel):
            p = root / rel
            if p.is_file():
                paths.append(p)
    return sorted(paths)


def gather_test_go(root: Path) -> dict[str, str]:
    """rel-path -> source text for every tracked first-party `_test.go`. Falls back
    to an rglob when git is unavailable (still dropping the scratch trees)."""
    paths = _git_tracked_test_paths(root)
    if paths is None:
        paths = [p for p in root.rglob("*_test.go")
                 if not _excluded(p.relative_to(root).as_posix())]
    out: dict[str, str] = {}
    for p in sorted(paths):
        rel = p.relative_to(root).as_posix()
        if _excluded(rel):
            continue
        out[rel] = _safe_read(p)
    return out


# ---------------------------------------------------------------------------
# Go lexer — dependency-free, string/comment aware, enough to classify a test's
# assertion conditions. Strings/runes/numbers keep their VALUE (we need the
# literal to tell a version freeze from a computed compare); comments/whitespace
# are dropped; braces/parens/brackets are their own tokens.
# ---------------------------------------------------------------------------

def go_lex(text: str) -> list[tuple[str, str]]:
    toks: list[tuple[str, str]] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n\f\v":
            i += 1
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == "`":
            j = text.find("`", i + 1)
            j = n - 1 if j == -1 else j
            toks.append(("string", text[i:j + 1]))
            i = j + 1
            continue
        if c == '"' or c == "'":
            q = c
            j = i + 1
            while j < n:
                if text[j] == "\\":
                    j += 2
                    continue
                if text[j] == "\n":
                    break
                if text[j] == q:
                    j += 1
                    break
                j += 1
            toks.append(("string" if q == '"' else "rune", text[i:j]))
            i = j
            continue
        if c.isdigit() or (c == "." and i + 1 < n and text[i + 1].isdigit()):
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] in "._"):
                if text[j] in "eEpP" and j + 1 < n and text[j + 1] in "+-":
                    j += 2
                    continue
                j += 1
            val = text[i:j]
            is_hex = val[:2].lower() == "0x"
            kind = "float" if (not is_hex and ("." in val or "e" in val or "E" in val)) else "int"
            toks.append((kind, val))
            i = j
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            toks.append(("name", text[i:j]))
            i = j
            continue
        matched = False
        for op in ("==", "!=", "<=", ">=", "&&", "||", ":=", "<-", "...",
                   "++", "--", "+=", "-=", "*=", "/=", "%="):
            if text.startswith(op, i):
                toks.append(("op", op))
                i += len(op)
                matched = True
                break
        if matched:
            continue
        singles = {"{": "lbrace", "}": "rbrace", "(": "lparen", ")": "rparen",
                   "[": "lbrack", "]": "rbrack"}
        toks.append((singles.get(c, "op"), c))
        i += 1
    return toks


def _scan_block(text: str, i: int) -> int:
    """text[i] == '{'; return the index of the matching '}', string/comment aware.
    On an unbalanced body returns len(text)-1 (best-effort, never raises)."""
    n = len(text)
    depth = 0
    while i < n:
        c = text[i]
        if c == "/" and i + 1 < n and text[i + 1] == "/":
            j = text.find("\n", i)
            i = n if j == -1 else j
            continue
        if c == "/" and i + 1 < n and text[i + 1] == "*":
            j = text.find("*/", i + 2)
            i = n if j == -1 else j + 2
            continue
        if c == "`":
            j = text.find("`", i + 1)
            i = n if j == -1 else j + 1
            continue
        if c == '"' or c == "'":
            q = c
            i += 1
            while i < n:
                if text[i] == "\\":
                    i += 2
                    continue
                if text[i] == q or text[i] == "\n":
                    i += 1
                    break
                i += 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n - 1


def _iter_test_funcs(text: str):
    """Yield (name, tparam, lineno, body_text) for each top-level Test func."""
    for m in _TESTHDR_RE.finditer(text):
        name, tparam = m.group(1), m.group(2)
        brace = text.find("{", m.end())
        if brace == -1:
            continue
        close = _scan_block(text, brace)
        lineno = text.count("\n", 0, m.start()) + 1
        yield name, tparam, lineno, text[brace + 1:close]


# ---------------------------------------------------------------------------
# Assertion extraction + classification (pure over already-lexed tokens)
# ---------------------------------------------------------------------------

def _literal_vars(body: str) -> dict[str, tuple[str, str]]:
    """Locally literal-bound vars in a func body: name -> (kind, value). Only a bare
    single-literal RHS binds; a compound expression or a struct field does not."""
    out: dict[str, tuple[str, str]] = {}
    for line in body.splitlines():
        m = _LIT_VAR_RE.match(line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        if val in ("true", "false"):
            kind = "bool"
        elif val[0] in "\"'":
            kind = "string"
        elif "." in val and not val[:2].lower().startswith("0x"):
            kind = "float"
        else:
            kind = "int"
        out[name] = (kind, val)
    return out


def _has_fail_call(block: list[tuple[str, str]], tparam: str) -> bool:
    """True iff the token block calls `<tparam>.<FAIL method>(`."""
    for i in range(len(block) - 3):
        if (block[i] == ("name", tparam) and block[i + 1] == ("op", ".")
                and block[i + 2][0] == "name" and block[i + 2][1] in FAIL_METHODS
                and block[i + 3][0] == "lparen"):
            return True
    return False


def _assertion_conditions(toks: list[tuple[str, str]], tparam: str) -> list[list[tuple[str, str]]]:
    """Every `if`-condition whose block asserts via `<tparam>.<FAIL>()`."""
    conds: list[list[tuple[str, str]]] = []
    i, n = 0, len(toks)
    while i < n:
        if toks[i] == ("name", "if"):
            j = i + 1
            pdepth = bdepth = 0
            cond: list[tuple[str, str]] = []
            while j < n:
                k = toks[j][0]
                if k == "lparen":
                    pdepth += 1
                elif k == "rparen":
                    pdepth -= 1
                elif k == "lbrack":
                    bdepth += 1
                elif k == "rbrack":
                    bdepth -= 1
                elif k == "lbrace" and pdepth == 0 and bdepth == 0:
                    break
                cond.append(toks[j])
                j += 1
            # j at the block-opening lbrace; brace-match to find the block.
            depth, e = 0, j
            while e < n:
                if toks[e][0] == "lbrace":
                    depth += 1
                elif toks[e][0] == "rbrace":
                    depth -= 1
                    if depth == 0:
                        break
                e += 1
            if _has_fail_call(toks[j + 1:e], tparam):
                conds.append(cond)
            i = j + 1
            continue
        i += 1
    return conds


def _last_clause(cond: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Drop an `if init; cond` init-statement: keep tokens after the last top-level `;`."""
    depth = 0
    cut = -1
    for idx, (k, v) in enumerate(cond):
        if k in ("lparen", "lbrack", "lbrace"):
            depth += 1
        elif k in ("rparen", "rbrack", "rbrace"):
            depth -= 1
        elif k == "op" and v == ";" and depth == 0:
            cut = idx
    return cond[cut + 1:]


def _split_top(toks: list[tuple[str, str]], seps: frozenset[str]) -> list[list[tuple[str, str]]]:
    """Split a token list on top-level (paren/bracket depth 0) separator ops."""
    out: list[list[tuple[str, str]]] = []
    cur: list[tuple[str, str]] = []
    depth = 0
    for k, v in toks:
        if k in ("lparen", "lbrack", "lbrace"):
            depth += 1
        elif k in ("rparen", "rbrack", "rbrace"):
            depth -= 1
        if k == "op" and v in seps and depth == 0:
            out.append(cur)
            cur = []
            continue
        cur.append((k, v))
    out.append(cur)
    return out


def _strip_parens(toks: list[tuple[str, str]]) -> list[tuple[str, str]]:
    while len(toks) >= 2 and toks[0][0] == "lparen" and toks[-1][0] == "rparen":
        # only strip if the outer parens actually wrap the whole span
        depth = 0
        wraps = True
        for idx, (k, _) in enumerate(toks):
            if k == "lparen":
                depth += 1
            elif k == "rparen":
                depth -= 1
                if depth == 0 and idx != len(toks) - 1:
                    wraps = False
                    break
        if not wraps:
            break
        toks = toks[1:-1]
    return toks


def _is_len_call(toks: list[tuple[str, str]]) -> bool:
    return (len(toks) >= 3 and toks[0] == ("name", "len") and toks[1][0] == "lparen"
            and toks[-1][0] == "rparen")


def _as_int(toks: list[tuple[str, str]], lit_vars: dict[str, tuple[str, str]]) -> int | None:
    if len(toks) == 1 and toks[0][0] == "int":
        try:
            return int(toks[0][1], 0)
        except ValueError:
            return None
    if len(toks) == 1 and toks[0][0] == "name":
        kv = lit_vars.get(toks[0][1])
        if kv and kv[0] == "int":
            try:
                return int(kv[1], 0)
            except ValueError:
                return None
    return None


def _as_str(toks: list[tuple[str, str]], lit_vars: dict[str, tuple[str, str]]) -> str | None:
    if len(toks) == 1 and toks[0][0] == "string":
        return toks[0][1].strip('"`')
    if len(toks) == 1 and toks[0][0] == "name":
        kv = lit_vars.get(toks[0][1])
        if kv and kv[0] == "string":
            return kv[1].strip('"`')
    return None


def _is_frozen(toks: list[tuple[str, str]], lit_vars: dict[str, tuple[str, str]]) -> bool:
    """A side that is a bare literal (int/float/string/rune/bool) or a locally
    literal-bound var — the frozen side of a change-detector pin."""
    if len(toks) == 1:
        k, v = toks[0]
        if k in ("int", "float", "string", "rune"):
            return True
        if k == "name" and v in ("true", "false"):
            return True
        if k == "name" and v in lit_vars:
            return True
    return False


def _is_nil(toks: list[tuple[str, str]]) -> bool:
    return len(toks) == 1 and toks[0] == ("name", "nil")


def _classify_compare(lhs, rhs, op, lit_vars) -> tuple[str, str]:
    """Classify one comparison -> ('pin', kind) | ('relation', '') | ('neutral', '')."""
    lhs, rhs = _strip_parens(lhs), _strip_parens(rhs)
    l_len, r_len = _is_len_call(lhs), _is_len_call(rhs)
    li, ri = _as_int(lhs, lit_vars), _as_int(rhs, lit_vars)
    ls, rs = _as_str(lhs, lit_vars), _as_str(rhs, lit_vars)

    # count freeze: len(x) (==|!=) N, N >= COUNT_MIN
    if op in _EQ_OPS:
        if l_len and ri is not None and abs(ri) >= COUNT_MIN:
            return ("pin", "count")
        if r_len and li is not None and abs(li) >= COUNT_MIN:
            return ("pin", "count")
        # version/config-literal freeze: expr (==|!=) "v1.2"
        if ls is not None and _VERSION_RE.match(ls) and not _is_frozen(rhs, lit_vars) and not _is_nil(rhs):
            return ("pin", "version")
        if rs is not None and _VERSION_RE.match(rs) and not _is_frozen(lhs, lit_vars) and not _is_nil(lhs):
            return ("pin", "version")

    # relation: both sides computed (neither frozen, neither nil)
    if (not _is_frozen(lhs, lit_vars) and not _is_frozen(rhs, lit_vars)
            and not _is_nil(lhs) and not _is_nil(rhs)):
        return ("relation", "")
    return ("neutral", "")


def _find_call_args(toks, start):
    """Given toks[start] opening an argument lparen, return (args_token_list, end_idx)
    where args is the tokens between the matched parens; end_idx is the rparen index."""
    depth = 0
    i = start
    n = len(toks)
    while i < n:
        if toks[i][0] == "lparen":
            depth += 1
        elif toks[i][0] == "rparen":
            depth -= 1
            if depth == 0:
                return toks[start + 1:i], i
        i += 1
    return toks[start + 1:], n - 1


def _classify_calls(cond: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Deep-equal / ordering invariant calls in a condition. reflect.DeepEqual with a
    literal collection is an enum-list PIN; comparing two computed values is a RELATION;
    sort.IsSorted-style calls are RELATIONS."""
    out: list[tuple[str, str]] = []
    i, n = 0, len(cond)
    while i < n - 3:
        if (cond[i][0] == "name" and cond[i + 1] == ("op", ".")
                and cond[i + 2][0] == "name" and cond[i + 3][0] == "lparen"):
            pkg, fn = cond[i][1], cond[i + 2][1]
            if fn in ORDERING_FUNCS or (pkg == "slices" and fn == "IsSorted"):
                out.append(("relation", ""))
            elif fn in DEEPEQUAL_FUNCS and pkg in DEEPEQUAL_PKGS:
                args, _ = _find_call_args(cond, i + 3)
                # a literal collection arg contains a top-level `{...}` composite.
                if any(k == "lbrace" for k, _ in args):
                    out.append(("pin", "enum"))
                else:
                    out.append(("relation", ""))
        i += 1
    return out


def _classify_condition(cond, lit_vars) -> list[tuple[str, str]]:
    """All pin/relation verdicts a single assertion condition yields."""
    cond = _last_clause(cond)
    verdicts: list[tuple[str, str]] = list(_classify_calls(cond))
    for sub in _split_top(cond, frozenset({"&&", "||"})):
        # find the first top-level comparison operator
        depth = 0
        pos = -1
        for idx, (k, v) in enumerate(sub):
            if k in ("lparen", "lbrack", "lbrace"):
                depth += 1
            elif k in ("rparen", "rbrack", "rbrace"):
                depth -= 1
            elif k == "op" and v in _CMP_OPS and depth == 0:
                pos = idx
                break
        if pos == -1:
            continue
        verdicts.append(_classify_compare(sub[:pos], sub[pos + 1:], sub[pos][1], lit_vars))
    return verdicts


def _toks_hint(toks: list[tuple[str, str]]) -> str:
    """A compact one-line hint reconstructed from tokens (for the work-list)."""
    return " ".join(v for _, v in toks).replace(" .", ".").replace(". ", ".").replace(" (", "(").replace("( ", "(").replace(" )", ")")


def classify_func(body: str, tparam: str) -> dict[str, Any]:
    """Classify one Test func body -> {class, pins:[(kind, hint)], relations:int}."""
    lit_vars = _literal_vars(body)
    toks = go_lex(body)
    pins: list[tuple[str, str]] = []
    relations = 0
    for cond in _assertion_conditions(toks, tparam):
        for verdict, kind in _classify_condition(cond, lit_vars):
            if verdict == "relation":
                relations += 1
            elif verdict == "pin":
                pins.append((kind, _toks_hint(_last_clause(cond))))
    if relations > 0:
        cls = "contract"
    elif pins:
        cls = "change_detector"
    else:
        cls = "unclassified"
    return {"class": cls, "pins": pins, "relations": relations}


# ---------------------------------------------------------------------------
# KPI + payload
# ---------------------------------------------------------------------------

def kpi_contract_discipline(test_files: dict[str, str]) -> dict[str, Any]:
    """Classify every Test func across the suite; the change_detector count is the
    debt. Returns counts, score, the defect work-list, and an advisory list of the
    (neutral) unclassified tests to eyeball."""
    n_tests = n_contract = n_change = n_unclassified = 0
    defects: list[str] = []
    soft: list[str] = []
    for rel in sorted(test_files):
        text = test_files[rel]
        for name, tparam, lineno, body in _iter_test_funcs(text):
            n_tests += 1
            res = classify_func(body, tparam)
            cls = res["class"]
            if cls == "contract":
                n_contract += 1
            elif cls == "change_detector":
                n_change += 1
                kinds = ",".join(sorted({k for k, _ in res["pins"]}))
                hint = res["pins"][0][1] if res["pins"] else ""
                defects.append(
                    f"change-detector [{kinds}] (freezes a literal, no invariant): "
                    f"{rel}:{lineno} {name} — '{hint[:70]}'")
            else:
                n_unclassified += 1
                soft.append(f"unclassified (no named freeze, no relation): {rel}:{lineno} {name}")
    debt = n_change
    score = min(100, round(100 - PENALTY_PER_DEBT * debt))
    classified = n_contract + n_change
    ratio = round(n_contract / classified, 3) if classified else 1.0
    detail = (f"{n_tests} Test func(s): {n_contract} contract · {n_change} change-detector · "
              f"{n_unclassified} unclassified")
    return {
        "kpi": "contract_discipline",
        "score": score,
        "debt": debt,
        "detail": detail,
        "contract_ratio": ratio,
        "counts": {"tests": n_tests, "contract": n_contract,
                   "change_detector": n_change, "unclassified": n_unclassified},
        "defects": defects,
        "soft": soft,
    }


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


def build_payload(workspace: str, kpi: dict[str, Any] | None, error: str = "") -> dict[str, Any]:
    if error or kpi is None:
        return {"schema": SCHEMA, "workspace": workspace, "ok": False, "error": error,
                "score": 0, "grade": "F", "change_detector_debt": 0}
    debt = kpi["debt"]
    score = kpi["score"]
    ok = debt == 0
    if ok:
        reason = "every classified Test func asserts an invariant — zero change-detector debt"
        nxt = "hold the line: keep contract_ratio at 1.0 as tests are added"
    else:
        reason = (f"{debt} change-detector test(s) freeze a literal/count/enumeration and assert "
                  f"no invariant (Hermes' rubric, #2900)")
        nxt = ("retire worst-first: replace each frozen count/version/list with an invariant "
               "(a relation between two computed values), then re-run to prove the debt fell")
    return {
        "schema": SCHEMA,
        "workspace": workspace,
        "ok": ok,
        "score": score,
        "grade": grade_letter(score),
        "change_detector_debt": debt,
        "contract_ratio": kpi["contract_ratio"],
        "counts": kpi["counts"],
        "reason": reason,
        "next_action": nxt,
        "kpi": kpi,
    }


def collect(workspace: Path) -> dict[str, Any]:
    try:
        test_files = gather_test_go(workspace)
    except OSError as exc:
        return build_payload(str(workspace), None, error=f"failed to read _test.go files: {exc}")
    if not test_files:
        return build_payload(str(workspace), None,
                             error="no first-party _test.go files found (run from repo ROOT)")
    return build_payload(str(workspace), kpi_contract_discipline(test_files))


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    if payload.get("error"):
        return f"behavior-contract-scorecard: ERROR — {payload['error']}"
    kpi = payload.get("kpi") or {}
    lines = [
        f"behavior-contract-scorecard: {'OK' if payload['ok'] else 'ACTION'} "
        f"(contract_ratio {payload.get('contract_ratio')})",
        f"  {payload.get('reason')}",
        "",
        f"  score {payload['score']}/100 (grade {payload['grade']}) · "
        f"change-detector-debt {payload['change_detector_debt']}",
        f"  {kpi.get('detail', '')}",
    ]
    defects = kpi.get("defects") or []
    if defects:
        lines.append("")
        lines.append(f"  change-detector debt work-list (first {min(len(defects), DEFECT_WORKLIST_CAP)}):")
        for d in defects[:DEFECT_WORKLIST_CAP]:
            lines.append(f"    - {d}")
        if len(defects) > DEFECT_WORKLIST_CAP:
            lines.append(f"    … +{len(defects) - DEFECT_WORKLIST_CAP} more")
    soft = kpi.get("soft") or []
    if soft:
        lines.append("")
        lines.append(f"  advisory — {len(soft)} unclassified test(s) (neither a named freeze nor a "
                     f"relation; not counted as debt)")
    lines.append("")
    lines.append(f"next: {payload.get('next_action')}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Behavior-contract vs change-detector test scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
