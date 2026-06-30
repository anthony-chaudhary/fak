#!/usr/bin/env python3
"""Hermetic tests for tools/check_hardware_tells.py — the COMMIT-TIME doc-content gate.

The gate moves the scrubber's `residual_hits` doc-content scan from POST-commit (make ci)
to COMMIT time, so the author who adds a prose hardware tell is refused before it lands and
reds the whole fleet trunk (issue #1455). These tests pin the three load-bearing properties:

  * a staged *.md that ADDS a prose tell (`on DGX`, `dgx3`, `da33`, `SXM4`) is BLOCKED (exit 1);
  * the known filename-link-text FALSE POSITIVE (`[DGX-OVERNIGHT-PLAN](…)`) is NOT blocked;
  * the gate REUSES the scrubber's patterns/masking (no second pattern list) — proven by
    importing both and asserting the gate finds exactly what the scrubber's residual_hits does.

Each gate run is a real `git commit`-shaped scenario: a throwaway repo, files staged, the
checker invoked exactly as the pre-commit hook invokes it (`--audit-staged --root <root>`).
"""
from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "check_hardware_tells.py"
SCRUB = HERE / "scrub_hardware_names.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class GateRepoHarness:
    """A throwaway git repo with the two tools copied in, so the gate's import + git calls work."""

    def __init__(self, tmp: str):
        self.root = Path(tmp)
        self._git("init", "-q")
        self._git("config", "user.email", "t@example.lab")
        self._git("config", "user.name", "t")
        # The gate imports scrub_hardware_names.py from ITS OWN sibling dir (tools/), so copy
        # both tools into tools/ inside the throwaway repo.
        tools = self.root / "tools"
        tools.mkdir()
        (tools / "check_hardware_tells.py").write_bytes(GATE.read_bytes())
        (tools / "scrub_hardware_names.py").write_bytes(SCRUB.read_bytes())

    def _git(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.root), *args],
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )

    def write(self, rel: str, content: str) -> None:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8", newline="")

    def commit_all(self) -> None:
        self._git("add", "-A")
        self._git("commit", "-q", "-m", "base")

    def stage(self, rel: str) -> None:
        self._git("add", "--", rel)

    def run_gate(self, env_extra=None) -> tuple[int, str]:
        env = dict(os.environ)
        env.pop("FLEET_HW_GUARD", None)
        env.pop("FLEET_ALLOW_HW", None)
        if env_extra:
            env.update(env_extra)
        r = subprocess.run(
            [sys.executable, str(self.root / "tools" / "check_hardware_tells.py"),
             "--audit-staged", "--root", str(self.root)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", env=env,
        )
        return r.returncode, r.stdout + r.stderr


class StagedTellBlockedTest(unittest.TestCase):
    def test_added_prose_dgx_tell_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/note.md", "intro line\n")
            h.commit_all()
            # the author ADDS a prose tell on a new line.
            h.write("docs/note.md", "intro line\nwe ran the eval on the DGX box\n")
            h.stage("docs/note.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 1, out)
            self.assertIn("HARDWARE_TELL", out)
            self.assertIn("--apply", out)

    def test_added_dgxn_tell_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/new.md", "ran the GLM-5.2 decode on dgx3 overnight\n")
            h.stage("docs/new.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 1, out)
            self.assertIn("dgx3", out)

    def test_added_da33_tell_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/cpu.md", "the CPU baseline ran on da33 at 0.063 GB/s\n")
            h.stage("docs/cpu.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 1, out)

    def test_added_sxm4_tell_blocks(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/sku.md", "the box is an SXM4 part\n")
            h.stage("docs/sku.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 1, out)


class FalsePositiveHeldTest(unittest.TestCase):
    def test_filename_link_text_not_blocked(self):
        # the #1455 FP: a link whose VISIBLE text is a filename is an identifier, not prose.
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write(
                "docs/plan.md",
                "see ([DGX-OVERNIGHT-PLAN](../nightrun/DGX-OVERNIGHT-PLAN-2026-06-28.md)). done\n",
            )
            h.stage("docs/plan.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 0, out)

    def test_clean_doc_passes(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/ok.md", "ran the eval on the GPU server overnight\n")
            h.stage("docs/ok.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 0, out)

    def test_identifier_forms_pass(self):
        # code spans / channel names / FQDN shortnames are identifiers, never prose tells.
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write(
                "docs/ids.md",
                "use `cmd/dgxbridge`; the dgx3-control channel; host dgx1.example.lab\n",
            )
            h.stage("docs/ids.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 0, out)


class AddedLinesOnlyTest(unittest.TestCase):
    def test_preexisting_tell_on_untouched_line_not_blocked(self):
        # a peer-authored tell already in the file, on a line THIS commit does not touch, is
        # NOT the author's to fix — refusing it would re-create the "red on a leak you didn't
        # author" friction this gate exists to remove.
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/legacy.md", "old line ran on dgx3 here\nsecond line\n")
            h.commit_all()
            # author appends a CLEAN line; the pre-existing dgx3 line is untouched.
            h.write("docs/legacy.md", "old line ran on dgx3 here\nsecond line\nnew clean line\n")
            h.stage("docs/legacy.md")
            rc, out = h.run_gate()
            self.assertEqual(rc, 0, out)

    def test_non_md_staged_file_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            # a tell in a .go file is the commit-message/identifier gate's job, not this one.
            h.write("pkg/x.go", "// ran on dgx3\npackage x\n")
            h.stage("pkg/x.go")
            rc, out = h.run_gate()
            self.assertEqual(rc, 0, out)


class EscapeAndModeTest(unittest.TestCase):
    def test_allow_env_escapes(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/meta.md", "the scrubber rewrites dgx3 out of prose\n")
            h.stage("docs/meta.md")
            rc, _ = h.run_gate({"FLEET_ALLOW_HW": "1"})
            self.assertEqual(rc, 0)

    def test_off_mode_skips(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/x.md", "ran on dgx3\n")
            h.stage("docs/x.md")
            rc, _ = h.run_gate({"FLEET_HW_GUARD": "off"})
            self.assertEqual(rc, 0)

    def test_warn_mode_does_not_block(self):
        with tempfile.TemporaryDirectory() as d:
            h = GateRepoHarness(d)
            h.write("docs/x.md", "ran on dgx3\n")
            h.stage("docs/x.md")
            rc, out = h.run_gate({"FLEET_HW_GUARD": "warn"})
            self.assertEqual(rc, 0, out)
            self.assertIn("HARDWARE_TELL", out)  # surfaced as advisory, not blocking


class PatternReuseTest(unittest.TestCase):
    """The no-divergence property: the gate does NOT define its own pattern list — it imports
    the scrubber and reuses residual_hits. Proven structurally (the gate module has no tells of
    its own) and behaviorally (for a tell line, the gate's finding == scrubber.residual_hits)."""

    def test_gate_imports_scrubber_residual_hits(self):
        gate = _load(GATE, "check_hardware_tells")
        scrub = _load(SCRUB, "scrub_hardware_names")
        loaded = gate._load_scrubber()
        self.assertIsNotNone(loaded)
        # the gate's scrubber IS the same module surface (same RESIDUAL_TELLS source of truth).
        self.assertEqual(loaded.RESIDUAL_TELLS, scrub.RESIDUAL_TELLS)
        self.assertTrue(hasattr(loaded, "residual_hits"))

    def test_gate_defines_no_pattern_list(self):
        # the gate must not fork a second HARDWARE_TERMS / tell list.
        gate = _load(GATE, "check_hardware_tells2")
        self.assertFalse(hasattr(gate, "HARDWARE_TERMS"))
        self.assertFalse(hasattr(gate, "RESIDUAL_TELLS"))

    def test_gate_decision_matches_scrubber(self):
        scrub = _load(SCRUB, "scrub_hardware_names3")
        # a tell the scrubber's residual_hits flags is exactly what the gate would block.
        tell = "we ran the eval on dgx3 today\n"
        self.assertEqual(len(scrub.residual_hits(tell)), 1)
        fp = "([DGX-OVERNIGHT-PLAN](../nightrun/DGX-OVERNIGHT-PLAN-2026-06-28.md)). done\n"
        self.assertEqual(scrub.residual_hits(fp), [])


if __name__ == "__main__":
    unittest.main()
