#!/usr/bin/env python3
"""Regression witnesses for the #780 token-stream clone detector — the two honesty
properties the line-shingle engine could NOT hold, locked in here so the engine swap
that closed #780 cannot silently regress.

#780 replaced ``kpi_duplication``'s line-shingle + hand-tuned skip heuristics with a
normalized Go *token-stream* engine (``go_tokens`` + a logic-token gate). The issue named
the two failure modes that swap was meant to fix:

  * a FALSE POSITIVE on "structurally-similar-but-distinct code" — two blocks that share a
    control/operator skeleton but mean different things; and
  * a MISS on "a clone reformatted" — a copy whose surface text was changed (rewrapped,
    or its constants retuned) even though its structure is identical.

The sibling ``code_slop_scorecard_test.py`` already witnesses the line-break-invariance
half of recall (``test_duplication_catches_reformatted_clone``) and the skip-list removal
(``test_duplication_ignores_*``). This module adds the two it does NOT cover: the
identifier-keeping precision guard, and literal-normalization recall — each with a control
assertion proving the fixture is clone-eligible, so a clean verdict is the #780 mechanism
at work and never a too-short window. Pure-stdlib + in-memory fixtures (hermetic): runs
under ``gated_tool_tests.py --run`` and standalone.

    python tools/code_slop_clone_precision_test.py
    python -m pytest tools/code_slop_clone_precision_test.py -q
"""
from __future__ import annotations

import code_slop_scorecard as cs


def _skeleton(fn: str, a: str, b: str, c: str, r: str) -> str:
    """A ~50-token arithmetic function whose STRUCTURE is fixed but whose five
    identifiers are parameters. Two instances with disjoint identifier sets share the
    same operator/keyword skeleton yet no normalized token window (identifiers are kept,
    not anonymized), so they must not register as a clone."""
    return (f"func {fn}({a}, {b}, {c} int) int {{\n"
            f"\t{r} := {a} + {b}*{c}\n"
            f"\t{r} = {r} - {a}%{b} + {c}\n"
            f"\tif {r} > {a} {{\n"
            f"\t\t{r} = {r}*{b} - {c}\n"
            f"\t}} else {{\n"
            f"\t\t{r} = {r} + {a} - {b}\n"
            f"\t}}\n"
            f"\treturn {r} + {a}*{b} - {c}\n"
            f"}}\n")


def test_distinct_identifiers_defeat_false_clone():
    # PRECISION (#780): the token engine keeps identifiers (``normalize_idents=False``)
    # precisely so a shared skeleton with DIFFERENT names is not a phantom clone — the
    # false positive the issue set out to remove. Disjoint identifier sets -> no shared
    # normalized window -> zero defects.
    distinct = {"a.go": "package a\n" + _skeleton("f", "p", "q", "s", "r"),
                "b.go": "package b\n" + _skeleton("g", "w", "x", "z", "y")}
    assert cs.kpi_duplication(distinct)["defects"] == []
    # CONTROL: the SAME skeleton with IDENTICAL identifiers, copied verbatim, IS a clone —
    # proving the fixture is long enough to be clone-eligible, so the clean verdict above
    # is the renamed identifiers at work, not a sub-window-length escape.
    identical = {"a.go": "package a\n" + _skeleton("f", "p", "q", "s", "r"),
                 "b.go": "package b\n" + _skeleton("f", "p", "q", "s", "r")}
    k = cs.kpi_duplication(identical)
    assert len(k["defects"]) >= 1
    assert k["score"] < 100


def _scale(threshold: int, mul: int, modulus: int) -> str:
    """A clone-eligible loop body whose three numeric CONSTANTS are parameters. Every
    literal collapses to one ``L`` token, so retuning the constants leaves the normalized
    token stream unchanged."""
    return ("func scale(xs []int) int {\n"
            "\ttotal := 0\n"
            "\tfor _, v := range xs {\n"
            f"\t\tif v > {threshold} {{\n"
            f"\t\t\ttotal += v * {mul}\n"
            f"\t\t\ttotal -= v % {modulus}\n"
            "\t\t}\n"
            "\t}\n"
            "\treturn total\n"
            "}\n")


def test_literal_normalization_catches_retuned_clone():
    # RECALL (#780): a copy-pasted block with its CONSTANTS retuned is still one clone,
    # because string/rune/number literals normalize to ``L`` — duplication is measured on
    # structure, not on the surface constants a paste might tweak.
    retuned = {"a.go": "package a\n" + _scale(0, 2, 3),
               "b.go": "package b\n" + _scale(7, 5, 9)}
    k = cs.kpi_duplication(retuned)
    assert len(k["defects"]) >= 1
    assert k["score"] < 100
    # CONTROL: a single copy (no duplicate) is clean — the defect above needs two sites,
    # not merely a literal appearing in one place.
    assert cs.kpi_duplication({"a.go": "package a\n" + _scale(0, 2, 3)})["defects"] == []


def main() -> int:
    tests = sorted((n, f) for n, f in globals().items()
                   if n.startswith("test_") and callable(f))
    failures = 0
    for name, fn in tests:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"  FAIL {name}: {exc}")
    print(f"{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
