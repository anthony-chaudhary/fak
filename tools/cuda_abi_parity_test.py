#!/usr/bin/env python3
"""Hermetic unit tests for tools/cuda_abi_parity.py.

Pure synthetic inputs — no disk, no git, no CUDA — so the parity logic is pinned
independent of the real tree. A final suite runs the checker against the actual repo
seam and asserts it is in parity (the regression sentinel the gate relies on).
"""
from __future__ import annotations

import unittest
from pathlib import Path

import cuda_abi_parity as m


class TestStripComments(unittest.TestCase):
    def test_block_and_line_comments_removed(self):
        src = "void real();\n// fcuda_ghost in a line comment\n/* fcuda_block mention */\nint x;"
        stripped = m.strip_comments(src)
        self.assertNotIn("fcuda_ghost", stripped)
        self.assertNotIn("fcuda_block", stripped)
        self.assertIn("void real()", stripped)

    def test_extern_c_string_literal_preserved(self):
        # The "C" in extern "C" MUST survive — it is the token that marks a definition.
        # A `//` inside a string must NOT open a comment (it is scanned through verbatim).
        src = 'extern "C" void fcuda_x(int n);  // see http://x\nint y;'
        stripped = m.strip_comments(src)
        self.assertIn('extern "C" void fcuda_x(int n);', stripped)
        self.assertNotIn("see http", stripped)  # the trailing // comment is gone
        self.assertIn("int y;", stripped)        # the // inside the string did not eat the line

    def test_newlines_preserved(self):
        src = "a\n/* two\nline */\nb"
        # line count is preserved so the multi-line def window stays aligned.
        self.assertEqual(m.strip_comments(src).count("\n"), src.count("\n"))


class TestSymbolExtraction(unittest.TestCase):
    def test_decls_capture_uppercase_suffix(self):
        # The _T suffix is the trap: a lowercase-only class truncates fcuda_f32_to_f16_T.
        hdr = (
            "void fcuda_matmul_f32(const float *dW, int n);\n"
            "void fcuda_f32_to_f16_T(void *d, const float *s, int out, int in);\n"
            "/* a comment mentioning fcuda_matmul_f32 must not double-count */\n"
        )
        self.assertEqual(m.header_decls(hdr), {"fcuda_matmul_f32", "fcuda_f32_to_f16_T"})

    def test_comment_only_symbol_is_not_a_decl(self):
        # A symbol that appears ONLY in a comment must NOT count as declared.
        hdr = "void fcuda_real(int n);\n/* fcuda_phantom is described but never declared */\n"
        self.assertEqual(m.header_decls(hdr), {"fcuda_real"})

    def test_defs_require_extern_c_not_callsite(self):
        cu = (
            'extern "C" void fcuda_matmul_f32(const float *dW) { fcuda_free(tmp); }\n'
            "__global__ void k_internal_kernel(float *x) { x[0] = 0; }\n"   # NOT part of the ABI
            'static void fcuda_not_exported(void) {}\n'                     # no extern "C": ignored
        )
        # fcuda_free is only a call-site inside the body, NOT a definition.
        self.assertEqual(m.kernel_defs(cu), {"fcuda_matmul_f32"})

    def test_multiline_extern_c_definition_matches(self):
        cu = 'extern "C" void\nfcuda_wrapped_def(const float *x,\n                  int n) {\n  body();\n}\n'
        self.assertEqual(m.kernel_defs(cu), {"fcuda_wrapped_def"})

    def test_calls_match_cgo_form(self):
        go = (
            "C.fcuda_matmul_f32(a, b)\n"
            "x := C.fcuda_argmax_f32(p, n)\n"
            "// C.fcuda_matmul_f32 in a comment is not a call\n"
        )
        self.assertEqual(m.binding_calls(go), {"fcuda_matmul_f32", "fcuda_argmax_f32"})


class TestParity(unittest.TestCase):
    def test_clean_tree_is_ok(self):
        p = m.parity({"fcuda_a", "fcuda_b"}, {"fcuda_a", "fcuda_b"}, {"fcuda_a", "fcuda_b"})
        self.assertEqual(p["hard"], [])
        self.assertEqual(p["uncalled"], [])

    def test_prototype_without_definition_is_hard(self):
        p = m.parity({"fcuda_a"}, set(), {"fcuda_a"})
        self.assertEqual(p["undefined"], ["fcuda_a"])
        self.assertEqual(len(p["hard"]), 1)
        self.assertIn("no definition", p["hard"][0])

    def test_call_without_prototype_is_hard(self):
        p = m.parity(set(), set(), {"fcuda_ghost"})
        self.assertEqual(p["undeclared_calls"], ["fcuda_ghost"])
        self.assertEqual(len(p["hard"]), 1)
        self.assertIn("no prototype", p["hard"][0])

    def test_definition_without_prototype_is_hard(self):
        p = m.parity(set(), {"fcuda_orphan"}, set())
        self.assertEqual(p["undeclared_defs"], ["fcuda_orphan"])
        self.assertEqual(len(p["hard"]), 1)

    def test_uncalled_is_soft_and_allowlist_marks_ok(self):
        p = m.parity({"fcuda_sync", "fcuda_mystery"}, {"fcuda_sync", "fcuda_mystery"}, set())
        self.assertEqual(set(p["uncalled"]), {"fcuda_sync", "fcuda_mystery"})
        self.assertEqual(p["hard"], [])
        joined = "\n".join(p["soft"])
        self.assertIn("OK:", joined)            # fcuda_sync carries its reason
        self.assertIn("fcuda_mystery", joined)  # the unknown one is still surfaced

    def test_payload_documented_ok_not_counted_advisory(self):
        pay = m.build_payload(workspace="/x", decls={"fcuda_sync"}, defs={"fcuda_sync"},
                              calls=set())
        self.assertTrue(pay["ok"])
        self.assertEqual(pay["corpus"]["soft_signals"], 0)  # documented-OK is not advisory debt
        self.assertEqual(pay["corpus"]["hard_mismatches"], 0)


class TestAgainstRealTree(unittest.TestCase):
    """The regression sentinel: the actual fak CUDA seam must be in parity."""

    def test_real_seam_in_parity(self):
        root = Path(__file__).resolve().parent.parent
        if not (root / m.HEADER).exists():
            self.skipTest("not in the fak tree")
        payload = m.collect(root)
        self.assertEqual(payload["verdict"], "OK",
                         msg=f"CUDA ABI drifted: {payload.get('corpus', {}).get('hard')}")
        self.assertEqual(payload["corpus"]["hard_mismatches"], 0)
        self.assertGreaterEqual(payload["corpus"]["n_declared"], 31)


if __name__ == "__main__":
    unittest.main()
