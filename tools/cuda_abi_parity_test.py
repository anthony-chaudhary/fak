#!/usr/bin/env python3
"""Hermetic unit tests for tools/cuda_abi_parity.py.

Synthetic inputs and temporary seam files keep the parity logic independent of
the real tree, with no git or CUDA required. A final suite runs the checker against
the actual repo seam and asserts parity (the regression sentinel the gate relies on).
"""
from __future__ import annotations

import tempfile
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
            'extern "C" int fcuda_declared_only(void);\n'                 # no body: ignored
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


class TestHeaderPortability(unittest.TestCase):
    def test_uint8_t_requires_direct_stdint_include(self):
        hdr = "#include <stddef.h>\nvoid fcuda_q4k(const uint8_t *p, size_t n);\n"
        p = m.build_payload(workspace="/x", decls={"fcuda_q4k"}, defs={"fcuda_q4k"},
                            calls={"fcuda_q4k"}, header_text=hdr)
        self.assertFalse(p["ok"])
        self.assertEqual(p["corpus"]["header_portability"]["missing_includes"],
                         {"stdint.h": ["uint8_t"]})
        self.assertIn("standalone header parse would fail", p["corpus"]["hard"][0])

    def test_current_fixed_header_shape_passes_portability(self):
        hdr = (
            "#include <stddef.h>\n"
            "#include <stdint.h>\n"
            "void fcuda_q4k(const uint8_t *p, size_t n);\n"
        )
        p = m.build_payload(workspace="/x", decls={"fcuda_q4k"}, defs={"fcuda_q4k"},
                            calls={"fcuda_q4k"}, header_text=hdr)
        self.assertTrue(p["ok"])
        self.assertEqual(p["corpus"]["header_portability"]["missing_includes"], {})


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


class TestCollectIncludes(unittest.TestCase):
    """Exercise the reachable CUDA source closure without requiring a compiler."""

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        for rel in (*m.KERNELS, *m.BINDINGS):
            self.write(rel, "// seam fixture\n")
        self.write(m.HEADER, 'extern "C" int fcuda_test(void);\n')
        self.write(m.BINDINGS[0], "C.fcuda_test()\n")

    def write(self, rel, text):
        path = self.root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_reachable_quoted_chain_and_cycle_are_in_parity(self):
        self.write(m.KERNELS[0], (
            '#include "cuda_backend.h"\n'
            '#include "parts/entry.cuh"\n'
            '#include "parts/entry.cuh"\n'
            '#include <not_local.cuh>\n'
            '// #include "not_source.cu"\n'
        ))
        self.write("internal/compute/parts/entry.cuh", '#include "../leaf.cu"\n')
        self.write("internal/compute/leaf.cu", (
            '#include "parts/entry.cuh"\n'
            'extern "C" int fcuda_test(void) { return 0; }\n'
        ))
        self.write("internal/compute/unreachable.cu",
                   'extern "C" int fcuda_unreachable(void) { return 0; }\n')
        payload = m.collect(self.root)
        self.assertEqual(payload["verdict"], "OK", payload)
        self.assertEqual(payload["corpus"]["n_defined"], 1)
        self.assertEqual(payload["corpus"]["hard_mismatches"], 0)
        self.assertEqual(payload, m.collect(self.root))

    def test_reachable_declaration_without_definition_remains_hard(self):
        self.write(m.KERNELS[0], '#include "declarations.cuh"\n')
        self.write("internal/compute/declarations.cuh",
                   'extern "C" int fcuda_test(void);\n')
        # A definition outside the include closure must not hide the missing body.
        self.write("internal/compute/unreachable.cu",
                   'extern "C" int fcuda_test(void) { return 0; }\n')
        payload = m.collect(self.root)
        self.assertEqual(payload["verdict"], "ACTION", payload)
        self.assertEqual(payload["corpus"]["undefined"], ["fcuda_test"])
        self.assertEqual(payload["corpus"]["n_defined"], 0)
        self.assertEqual(payload["corpus"]["hard_mismatches"], 1)

    def test_empty_root_reached_by_alias_remains_an_audit_error(self):
        (self.root / "internal/compute/parts").mkdir()
        self.write(m.KERNELS[1], "")
        self.write(m.KERNELS[0], (
            '#include "parts/../cuda_nccl.cu"\n'
            'extern "C" int fcuda_test(void) { return 0; }\n'
        ))
        payload = m.collect(self.root)
        self.assertEqual(payload["verdict"], "AUDIT_ERROR", payload)
        self.assertFalse(payload["ok"])
        self.assertIn(m.KERNELS[1], payload["reason"])

    def test_unreadable_quoted_include_path_is_an_audit_error(self):
        self.write("internal/compute/leaf.cu",
                   'extern "C" int fcuda_test(void) { return 0; }\n')
        self.write("internal/compute/notdir", "not a directory\n")
        for include in ("missing/../leaf.cu", "notdir/../leaf.cu"):
            with self.subTest(include=include):
                # A readable alias must not hide a later ENOENT or ENOTDIR path.
                self.write(m.KERNELS[0],
                           f'#include "leaf.cu"\n#include "{include}"\n')
                payload = m.collect(self.root)
                self.assertEqual(payload["verdict"], "AUDIT_ERROR", payload)
                self.assertFalse(payload["ok"])
                self.assertIn(include, payload["reason"])

    def test_missing_reachable_include_is_an_audit_error(self):
        self.write(m.KERNELS[0], (
            '#include "parts/entry.cuh"\n'
            'extern "C" int fcuda_test(void) { return 0; }\n'
        ))
        self.write("internal/compute/parts/entry.cuh", '#include "missing.cuh"\n')
        payload = m.collect(self.root)
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["verdict"], "AUDIT_ERROR", payload)
        self.assertIn("missing.cuh", payload["reason"])


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
