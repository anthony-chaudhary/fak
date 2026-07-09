#!/usr/bin/env python3
"""Hermetic tests for tools/trunk_build_probe.py.

No network, no Go toolchain, no real git — every test drives the pure functions
(`parse_build_errors`, `defines_symbol`, `find_forgotten_files`, `is_go_buildable_path`,
`classify`, `diagnose`) with synthetic inputs modeled on the real trunk break the
probe was built to diagnose.
"""
from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROBE = ROOT / "tools" / "trunk_build_probe.py"


def load():
    spec = importlib.util.spec_from_file_location("trunk_build_probe", PROBE)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


t = load()

# A real cmd/fak build-failure block (Windows `\` paths, every "no definition" shape).
REAL_STDERR = (
    "# github.com/anthony-chaudhary/fak/cmd/fak\n"
    "cmd\\fak\\guard.go:167:41: undefined: guardStopHookSameStopFromEnv\n"
    "cmd\\fak\\guard.go:181:51: undefined: gateway.DefaultVCacheAnchor\n"
    "cmd\\fak\\guard.go:966:3: unknown field VCacheAnchor in struct literal of type gateway.Config\n"
    "cmd\\fak\\guard.go:1093:8: srv.SetLogvaultMetricsProvider undefined "
    "(type *gateway.Server has no field or method SetLogvaultMetricsProvider)\n"
    "cmd\\fak\\main.go:52:16: undefined: parseVerbArgv\n"
)


class ParseBuildErrors(unittest.TestCase):
    def test_real_block(self):
        p = t.parse_build_errors(REAL_STDERR)
        self.assertEqual(
            p["failing_packages"],
            ["github.com/anthony-chaudhary/fak/cmd/fak"],
        )
        syms = {s["symbol"] for s in p["missing_symbols"]}
        self.assertIn("guardStopHookSameStopFromEnv", syms)
        # pkg-qualified `gateway.DefaultVCacheAnchor` must reduce to the bare identifier
        self.assertIn("DefaultVCacheAnchor", syms)
        self.assertNotIn("gateway.DefaultVCacheAnchor", syms)
        self.assertIn("VCacheAnchor", syms)  # unknown field
        self.assertIn("SetLogvaultMetricsProvider", syms)  # has no field or method
        self.assertIn("parseVerbArgv", syms)

    def test_captures_file_and_line(self):
        p = t.parse_build_errors(REAL_STDERR)
        at = {s["symbol"]: s["at"] for s in p["missing_symbols"]}
        self.assertEqual(at["guardStopHookSameStopFromEnv"], "cmd\\fak\\guard.go:167")
        self.assertEqual(at["parseVerbArgv"], "cmd\\fak\\main.go:52")

    def test_associates_symbol_with_current_package(self):
        p = t.parse_build_errors(REAL_STDERR)
        for s in p["missing_symbols"]:
            self.assertEqual(s["referenced_in"], "github.com/anthony-chaudhary/fak/cmd/fak")

    def test_empty(self):
        p = t.parse_build_errors("")
        self.assertEqual(p["failing_packages"], [])
        self.assertEqual(p["missing_symbols"], [])

    def test_dedupes_symbol_repeated_in_same_package(self):
        stderr = (
            "# example.com/p\n"
            "p\\a.go:1:1: undefined: Foo\n"
            "p\\b.go:2:2: undefined: Foo\n"
        )
        p = t.parse_build_errors(stderr)
        foos = [s for s in p["missing_symbols"] if s["symbol"] == "Foo"]
        self.assertEqual(len(foos), 1)

    def test_same_symbol_distinct_across_packages(self):
        stderr = (
            "# example.com/p\n"
            "p\\a.go:1:1: undefined: Foo\n"
            "# example.com/q\n"
            "q\\a.go:1:1: undefined: Foo\n"
        )
        p = t.parse_build_errors(stderr)
        foos = [s for s in p["missing_symbols"] if s["symbol"] == "Foo"]
        self.assertEqual(len(foos), 2)
        self.assertEqual(p["failing_packages"], ["example.com/p", "example.com/q"])


class DefinesSymbol(unittest.TestCase):
    def test_true_shapes(self):
        cases = [
            ("func parseVerbArgv(argv []string) verb {", "parseVerbArgv"),
            ("func (s *Server) SetLogvaultMetricsProvider(p P) {", "SetLogvaultMetricsProvider"),
            ("type VCacheAnchor struct {", "VCacheAnchor"),
            ("\tVCacheAnchor string", "VCacheAnchor"),          # struct field
            ("const logvaultMetricsVerifySample = 256", "logvaultMetricsVerifySample"),
            ("var DefaultVCacheAnchor = VCacheAnchor{}", "DefaultVCacheAnchor"),
            ("\tresolveLogvaultDir = \"x\"", "resolveLogvaultDir"),  # untyped block member
            ("func Foo[T any](x T) T {", "Foo"),                # generic func
        ]
        for content, sym in cases:
            self.assertTrue(t.defines_symbol(content, sym), f"should define: {content!r}")

    def test_false_uses_not_definitions(self):
        cases = [
            ("\tfoo := parseVerbArgv(argv)", "parseVerbArgv"),   # call
            ("\tVCacheAnchor: cfg.Anchor,", "VCacheAnchor"),     # struct-literal key
            ("\tsrv.VCacheAnchor = x", "VCacheAnchor"),          # field assignment
            ("\treturn gateway.DefaultVCacheAnchor", "DefaultVCacheAnchor"),
        ]
        for content, sym in cases:
            self.assertFalse(t.defines_symbol(content, sym), f"should NOT define: {content!r}")

    def test_empty_symbol_is_false(self):
        self.assertFalse(t.defines_symbol("anything at all", ""))


class IsGoBuildablePath(unittest.TestCase):
    def test_buildable(self):
        self.assertTrue(t.is_go_buildable_path("cmd/fak/main.go"))
        self.assertTrue(t.is_go_buildable_path("internal/gateway/gateway.go"))
        self.assertTrue(t.is_go_buildable_path("cmd\\fak\\main.go"))  # backslash sep

    def test_excluded(self):
        for p in [
            ".head_build_check/cmd/fak/guard.go",  # dot-dir build artifact
            "internal/_scratch/x.go",              # underscore dir
            "pkg/testdata/y.go",                   # testdata tree
            "vendor/z/a.go",                       # vendored
            "README.md",                           # not .go
            ".head_build_check\\cmd\\fak\\guard.go",  # backslash dot-dir
        ]:
            self.assertFalse(t.is_go_buildable_path(p), f"should be excluded: {p}")


class FindForgottenFiles(unittest.TestCase):
    def test_only_real_go_definer_returned(self):
        missing = [{"symbol": "parseVerbArgv", "referenced_in": "cmd/fak", "at": ""}]
        uncommitted = {
            "cmd/fak/main_preamble.go": "func parseVerbArgv(a []string) verb {}\n",
            "docs/notes.md": "parseVerbArgv is mentioned here\n",           # non-.go
            ".head_build_check/cmd/fak/x.go": "func parseVerbArgv() {}\n",  # dotdir artifact...
        }
        # ...but a dotdir file only reaches here if it slipped past is_go_buildable_path;
        # find_forgotten_files itself filters non-.go and _test.go. The real definer wins.
        got = t.find_forgotten_files(missing, uncommitted)
        paths = {f["path"] for f in got}
        self.assertIn("cmd/fak/main_preamble.go", paths)
        self.assertNotIn("docs/notes.md", paths)

    def test_test_files_never_definers(self):
        # go build ignores _test.go, so a build-time reference can never resolve there
        missing = [{"symbol": "VCacheAnchor", "referenced_in": "cmd/fak", "at": ""}]
        uncommitted = {"cmd/fak/vcache_wiring_test.go": "type VCacheAnchor struct{}\n"}
        self.assertEqual(t.find_forgotten_files(missing, uncommitted), [])

    def test_symbol_with_no_definer_absent(self):
        missing = [{"symbol": "NeverDefined", "referenced_in": "cmd/fak", "at": ""}]
        uncommitted = {"cmd/fak/a.go": "func Something() {}\n"}
        self.assertEqual(t.find_forgotten_files(missing, uncommitted), [])

    def test_groups_multiple_symbols_per_file(self):
        missing = [
            {"symbol": "parseVerbArgv", "referenced_in": "cmd/fak", "at": ""},
            {"symbol": "recoverUsage", "referenced_in": "cmd/fak", "at": ""},
        ]
        uncommitted = {
            "cmd/fak/main_preamble.go": "func parseVerbArgv() {}\nfunc recoverUsage() {}\n"
        }
        got = t.find_forgotten_files(missing, uncommitted)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["path"], "cmd/fak/main_preamble.go")
        self.assertEqual(got[0]["defines"], ["parseVerbArgv", "recoverUsage"])


class ClassifyAndDiagnose(unittest.TestCase):
    def test_build_ok(self):
        d = t.diagnose(True, "", {}, head="abc123")
        self.assertEqual(d["verdict"], "BUILD_OK")
        self.assertTrue(d["builds"])
        self.assertEqual(d["forgotten_files"], [])
        self.assertEqual(d["missing_symbols"], [])
        self.assertIn("not a build break", d["summary"])

    def test_build_broken_coherence(self):
        uncommitted = {"cmd/fak/main_preamble.go": "func parseVerbArgv() {}\n"}
        d = t.diagnose(False, REAL_STDERR_MINIMAL, uncommitted)
        self.assertEqual(d["verdict"], "BUILD_BROKEN_COHERENCE")
        self.assertTrue(any(f["path"] == "cmd/fak/main_preamble.go" for f in d["forgotten_files"]))
        self.assertIn("main_preamble.go", d["summary"])
        self.assertIn("git add", d["summary"])

    def test_build_broken_other_when_no_uncommitted_definer(self):
        d = t.diagnose(False, REAL_STDERR_MINIMAL, {})  # nothing uncommitted defines it
        self.assertEqual(d["verdict"], "BUILD_BROKEN_OTHER")
        self.assertEqual(d["forgotten_files"], [])
        self.assertIn("genuine", d["summary"])

    def test_classify_direct(self):
        self.assertEqual(t.classify(True, [], []), "BUILD_OK")
        self.assertEqual(t.classify(False, [{"path": "x"}], [{"symbol": "S"}]),
                         "BUILD_BROKEN_COHERENCE")
        self.assertEqual(t.classify(False, [], [{"symbol": "S"}]), "BUILD_BROKEN_OTHER")


REAL_STDERR_MINIMAL = (
    "# github.com/anthony-chaudhary/fak/cmd/fak\n"
    "cmd\\fak\\main.go:52:16: undefined: parseVerbArgv\n"
)


if __name__ == "__main__":
    unittest.main()
