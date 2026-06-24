#!/usr/bin/env python3
"""Hermetic tests for tools/godsplit_plan.py."""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "godsplit_plan.py"


def load():
    spec = importlib.util.spec_from_file_location("godsplit_plan", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# A fixture exercising every signal: a build tag, an aliased import (and a
# look-alike that is NOT an alias), a doc-commented type and func, a one-line
# method, and an init(). Line numbers are referenced explicitly in the tests.
FIXTURE = (
    "//go:build linux\n"            # 1
    "\n"                             # 2
    "// Package demo is a fixture.\n"  # 3
    "package demo\n"                # 4
    "\n"                             # 5
    "import (\n"                    # 6
    '\t"fmt"\n'                      # 7
    '\t_ "embed"\n'                 # 8
    '\tfakmodel "github.com/x/y/internal/model"\n'  # 9 (real alias)
    '\tmodel "github.com/x/y/model"\n'              # 10 (alias == basename: NOT a real alias)
    ")\n"                           # 11
    "\n"                             # 12
    "// Widget is a thing.\n"       # 13
    "// Second doc line.\n"         # 14
    "type Widget struct {\n"        # 15
    "\tN int\n"                      # 16
    "}\n"                            # 17
    "\n"                             # 18
    "// Make builds a Widget.\n"    # 19
    "func Make() *Widget {\n"       # 20
    "\treturn &Widget{}\n"          # 21
    "}\n"                            # 22
    "\n"                             # 23
    "func (w *Widget) Inc() { w.N++ }\n"  # 24 (one-line method, no doc)
    "\n"                             # 25
    "func init() {\n"               # 26
    '\tfmt.Println("init")\n'       # 27
    "}\n"                            # 28
)


class GodSplitPlanTest(unittest.TestCase):
    def test_package_and_size(self) -> None:
        p = load().plan(FIXTURE)
        self.assertEqual(p["package"], "demo")
        self.assertEqual(p["line_count"], 28)
        self.assertFalse(p["god_file"])

    def test_hazards(self) -> None:
        p = load().plan(FIXTURE)
        hz = p["hazards"]
        self.assertEqual(hz["build_tags"], ["//go:build linux"])
        self.assertEqual(hz["init_funcs"], 1)
        # Only the genuine alias; `_ "embed"`, plain `"fmt"`, and the basename-matching
        # `model "..../model"` are all excluded.
        self.assertEqual(hz["import_aliases"], ['fakmodel "github.com/x/y/internal/model"'])

    def test_doc_comment_aware_block_starts(self) -> None:
        p = load().plan(FIXTURE)
        by_name = {d["name"]: d for d in p["decls"]}
        # Block starts at the leading doc comment, not the decl keyword.
        self.assertEqual(by_name["Widget"]["block_start"], 13)
        self.assertEqual(by_name["Make"]["block_start"], 19)
        # A decl with no doc comment starts at itself.
        self.assertEqual(by_name["Inc"]["block_start"], 24)
        self.assertEqual(by_name["init"]["block_start"], 26)

    def test_block_ends_chain_to_next_start_and_eof(self) -> None:
        p = load().plan(FIXTURE)
        by_name = {d["name"]: d for d in p["decls"]}
        # Each block runs up to the line before the next decl's block_start.
        self.assertEqual(by_name["Widget"]["block_end"], 18)
        self.assertEqual(by_name["Make"]["block_end"], 23)
        self.assertEqual(by_name["Inc"]["block_end"], 25)
        # The last decl runs to EOF.
        self.assertEqual(by_name["init"]["block_end"], 28)

    def test_func_spans(self) -> None:
        p = load().plan(FIXTURE)
        by_name = {d["name"]: d for d in p["decls"]}
        self.assertEqual(by_name["Make"]["func_lines"], 3)   # lines 20-22
        self.assertEqual(by_name["Inc"]["func_lines"], 1)    # one-liner
        self.assertEqual(by_name["init"]["func_lines"], 3)   # lines 26-28
        self.assertFalse(by_name["Make"]["god_function"])

    def test_decl_name_kinds(self) -> None:
        m = load()
        self.assertEqual(m.decl_name("func (r *R) Foo(x int) {"), ("method", "Foo"))
        self.assertEqual(m.decl_name("func Bar() {"), ("func", "Bar"))
        self.assertEqual(m.decl_name("type T struct {"), ("type", "T"))
        self.assertEqual(m.decl_name("var x = 1"), ("var", "x"))
        self.assertEqual(m.decl_name("const ("), ("const", "(group)"))

    def test_god_function_flag(self) -> None:
        m = load()
        big = "package p\n\nfunc Big() {\n" + "\t_ = 1\n" * 205 + "}\n"
        p = m.plan(big)
        big_decl = next(d for d in p["decls"] if d["name"] == "Big")
        self.assertGreater(big_decl["func_lines"], m.FUNC_HARD_MAX)
        self.assertTrue(big_decl["god_function"])

    def test_god_file_flag(self) -> None:
        m = load()
        huge = "package p\n" + "// filler\n" * 1600
        p = m.plan(huge)
        self.assertTrue(p["god_file"])

    def test_raw_string_embedded_decls_are_ignored(self) -> None:
        # A backtick raw string holding column-0 `func`/`type` text must NOT produce
        # phantom decls, and the real `var Tmpl` must survive with a valid (non-inverted)
        # range. This is the major bug the reviewers caught.
        raw = (
            "package demo\n"               # 1
            "\n"                            # 2
            "// Tmpl is generated source.\n"  # 3
            "var Tmpl = `\n"               # 4  opens raw string
            "func generated() {}\n"        # 5  INSIDE raw — ignore
            "type Thing struct{}\n"        # 6  INSIDE raw — ignore
            "`\n"                           # 7  closes raw string
            "\n"                            # 8
            "// Real is real.\n"           # 9
            "func Real() {}\n"             # 10
        )
        p = load().plan(raw)
        names = [d["name"] for d in p["decls"]]
        self.assertEqual(names, ["Tmpl", "Real"])  # no generated/Thing phantoms
        self.assertEqual(p["hazards"]["raw_strings"], 1)
        tmpl = p["decls"][0]
        self.assertGreaterEqual(tmpl["block_end"], tmpl["block_start"])  # not inverted

    def test_first_decl_block_start_never_swallows_package(self) -> None:
        # No blank between the package clause and the first decl's doc comment: the
        # block_start must clamp BELOW the package line, never to line 1 (which would
        # duplicate the package clause in the extracted file → build break).
        noblank = "package p\n// Doc\ntype X struct{}\n"  # package=1, //Doc=2, type=3
        p = load().plan(noblank)
        x = p["decls"][0]
        self.assertEqual(x["name"], "X")
        self.assertEqual(x["block_start"], 2)  # the // Doc line, NOT line 1 (package)

    def test_func_span_brace_depth_robustness(self) -> None:
        # Brace depth (over string/comment-stripped code) must walk past an indented
        # inner composite-literal close and a brace inside a string, ending at the real
        # closing brace — and handle a one-liner.
        src = (
            "package p\n"                  # 1
            "\n"                           # 2
            "func F() {\n"                 # 3
            "\tm := map[string]int{\n"     # 4
            '\t\t"a": 1,\n'                # 5
            "\t}\n"                        # 6  indented inner close
            '\tx := "}"\n'                 # 7  brace in string — must NOT end the func
            "\t_ = x\n"                    # 8
            "}\n"                          # 9  real close
            "\n"                           # 10
            "func G() { return }\n"        # 11 one-liner
        )
        by_name = {d["name"]: d for d in load().plan(src)["decls"]}
        self.assertEqual(by_name["F"]["func_lines"], 7)  # lines 3..9, not a false early end
        self.assertEqual(by_name["G"]["func_lines"], 1)


if __name__ == "__main__":
    unittest.main()
