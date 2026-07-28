#!/usr/bin/env python3
"""Tests for provision_dos_hook.py — the native ``dos-hook`` provisioner.

This tool exists because the ``dos-hook`` source lives in a DIFFERENT repo, so it
has to go looking for a binary rather than building one. Two properties make that
search safe, and both are easy to break silently:

* the discovery order is env-first, then sibling checkout, then installed plugin,
  with non-existent paths dropped and duplicates collapsed — an operator who sets
  ``DOS_KERNEL`` must win over whatever happens to be lying next to the repo;
* absence is a SOFT failure. No dos checkout anywhere returns 1 with guidance and
  copies nothing, because the launcher's Python fallback is still correct — this
  provisioner is a perf convenience, never a correctness gate.

Every test drives synthetic trees in a temp directory and stubs the discovery when
it must, so no real dos checkout on the host can change the result and no
``go build`` is ever spawned.
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import provision_dos_hook as M  # noqa: E402


def tmpdir():
    return Path(tempfile.mkdtemp())


def touch(path, body="binary"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


class HostTriple(unittest.TestCase):
    def fake_platform(self, system, machine):
        import platform
        old_s, old_m = platform.system, platform.machine
        platform.system, platform.machine = (lambda: system), (lambda: machine)
        self.addCleanup(lambda: (setattr(platform, "system", old_s),
                                 setattr(platform, "machine", old_m)))

    def test_windows_amd64(self):
        self.fake_platform("Windows", "AMD64")
        self.assertEqual(M.host_triple(), ("windows", "amd64", ".exe"))

    def test_linux_aarch64_is_named_arm64_like_the_plugin_does(self):
        self.fake_platform("Linux", "aarch64")
        self.assertEqual(M.host_triple(), ("linux", "arm64", ""))

    def test_darwin_arm64_has_no_exe_suffix(self):
        self.fake_platform("Darwin", "arm64")
        self.assertEqual(M.host_triple(), ("darwin", "arm64", ""))

    def test_an_unknown_machine_falls_back_to_amd64(self):
        self.fake_platform("Linux", "x86_64")
        self.assertEqual(M.host_triple()[1], "amd64")

    def test_the_live_host_triple_is_self_consistent(self):
        sysname, arch, suffix = M.host_triple()
        self.assertIn(arch, ("amd64", "arm64"))
        self.assertEqual(suffix, ".exe" if sysname == "windows" else "")


class CandidateRoots(unittest.TestCase):
    def test_env_roots_come_first_and_in_declared_precedence(self):
        a, b, c = tmpdir(), tmpdir(), tmpdir()
        roots = M.candidate_roots({"DOS_KERNEL": str(a), "DOS_PLUGIN_ROOT": str(b),
                                   "CLAUDE_PLUGIN_ROOT": str(c), "HOME": str(tmpdir())})
        self.assertEqual([str(r) for r in roots[:3]], [str(a), str(b), str(c)])

    def test_a_path_that_does_not_exist_is_dropped(self):
        real = tmpdir()
        ghost = str(real / "definitely-not-here")
        roots = [str(r) for r in M.candidate_roots(
            {"DOS_KERNEL": ghost, "DOS_PLUGIN_ROOT": str(real), "HOME": str(tmpdir())})]
        self.assertNotIn(ghost, roots)
        self.assertEqual(roots[0], str(real))

    def test_the_same_root_named_twice_appears_once(self):
        d = tmpdir()
        roots = [str(r) for r in M.candidate_roots(
            {"DOS_KERNEL": str(d), "DOS_PLUGIN_ROOT": str(d), "HOME": str(tmpdir())})]
        self.assertEqual(roots.count(str(d)), 1)

    def test_no_env_and_an_empty_home_still_returns_a_list(self):
        # Discovery must degrade to "nothing found", never raise.
        self.assertIsInstance(M.candidate_roots({"HOME": str(tmpdir())}), list)


class FindPrebuilt(unittest.TestCase):
    def test_plugin_bin_is_preferred_over_a_bare_bin(self):
        root = tmpdir()
        touch(root / "bin" / "dos-hook-linux-amd64", "from-bare-bin")
        touch(root / "claude-plugin" / "bin" / "dos-hook-linux-amd64", "from-plugin")
        found = M.find_prebuilt(root, "linux", "amd64", "")
        self.assertEqual(found.read_text(encoding="utf-8"), "from-plugin")

    def test_a_binary_for_another_host_is_not_offered(self):
        root = tmpdir()
        touch(root / "claude-plugin" / "bin" / "dos-hook-darwin-arm64")
        self.assertIsNone(M.find_prebuilt(root, "linux", "amd64", ""))

    def test_the_exe_suffix_is_part_of_the_name(self):
        root = tmpdir()
        touch(root / "bin" / "dos-hook-windows-amd64.exe")
        self.assertIsNone(M.find_prebuilt(root, "windows", "amd64", ""))
        self.assertIsNotNone(M.find_prebuilt(root, "windows", "amd64", ".exe"))

    def test_a_nested_bin_dir_is_found_by_the_recursive_fallback(self):
        root = tmpdir()
        touch(root / "vendor" / "dos" / "bin" / "dos-hook-linux-amd64")
        self.assertIsNotNone(M.find_prebuilt(root, "linux", "amd64", ""))

    def test_an_empty_root_yields_none(self):
        self.assertIsNone(M.find_prebuilt(tmpdir(), "linux", "amd64", ""))


class FindGoModule(unittest.TestCase):
    def test_the_go_subdir_layout_is_recognised(self):
        root = tmpdir()
        touch(root / "go" / "go.mod", "module dos\n")
        (root / "go" / "cmd" / "dos-hook").mkdir(parents=True)
        self.assertEqual(M.find_go_module(root), root / "go")

    def test_a_module_at_the_root_is_recognised(self):
        root = tmpdir()
        touch(root / "go.mod", "module dos\n")
        (root / "cmd" / "dos-hook").mkdir(parents=True)
        self.assertEqual(M.find_go_module(root), root)

    def test_a_go_mod_without_the_dos_hook_command_is_not_a_match(self):
        # Some OTHER Go module sitting next to fak must not be built as dos-hook.
        root = tmpdir()
        touch(root / "go.mod", "module something-else\n")
        self.assertIsNone(M.find_go_module(root))


class Provision(unittest.TestCase):
    def stub_roots(self, roots):
        original = M.candidate_roots
        M.candidate_roots = lambda env: list(roots)
        self.addCleanup(lambda: setattr(M, "candidate_roots", original))

    def test_no_dos_checkout_is_a_soft_failure_with_guidance(self):
        self.stub_roots([])
        lines = []
        dest = tmpdir() / "bin" / "dos-hook"
        self.assertEqual(M.provision(dest, {}, log=lines.append), 1)
        self.assertFalse(dest.exists())         # nothing written
        self.assertFalse(dest.parent.exists())  # not even the directory
        joined = " ".join(lines)
        self.assertIn("Python fallback", joined)
        self.assertIn("DOS_KERNEL", joined)

    def test_a_matching_prebuilt_is_copied_and_the_dest_dir_created(self):
        sysname, arch, suffix = M.host_triple()
        root = tmpdir()
        touch(root / "claude-plugin" / "bin" / f"dos-hook-{sysname}-{arch}{suffix}",
              "the-real-binary")
        self.stub_roots([root])
        dest = tmpdir() / "nested" / f"dos-hook{suffix}"
        lines = []
        self.assertEqual(M.provision(dest, {}, log=lines.append), 0)
        self.assertEqual(dest.read_text(encoding="utf-8"), "the-real-binary")
        self.assertIn("copied", " ".join(lines))

    def test_the_first_root_that_has_a_prebuilt_wins(self):
        sysname, arch, suffix = M.host_triple()
        name = f"dos-hook-{sysname}-{arch}{suffix}"
        empty, first, second = tmpdir(), tmpdir(), tmpdir()
        touch(first / "bin" / name, "first")
        touch(second / "bin" / name, "second")
        self.stub_roots([empty, first, second])
        dest = tmpdir() / f"dos-hook{suffix}"
        self.assertEqual(M.provision(dest, {}, log=lambda *_: None), 0)
        self.assertEqual(dest.read_text(encoding="utf-8"), "first")

    def test_roots_with_nothing_usable_report_what_was_searched(self):
        root = tmpdir()          # exists, but holds no binary and no Go module
        self.stub_roots([root])
        lines = []
        dest = tmpdir() / "dos-hook"
        self.assertEqual(M.provision(dest, {}, log=lines.append), 1)
        self.assertFalse(dest.exists())
        joined = " ".join(lines)
        self.assertIn(root.name, joined)         # names the roots it did search
        self.assertIn("Python fallback", joined)

    def test_the_copied_binary_is_executable_on_posix(self):
        if os.name == "nt":
            self.skipTest("POSIX permission bits are not meaningful on Windows")
        sysname, arch, suffix = M.host_triple()
        root = tmpdir()
        touch(root / "bin" / f"dos-hook-{sysname}-{arch}{suffix}")
        self.stub_roots([root])
        dest = tmpdir() / f"dos-hook{suffix}"
        self.assertEqual(M.provision(dest, {}, log=lambda *_: None), 0)
        self.assertTrue(os.access(dest, os.X_OK))


class NoWindowFlags(unittest.TestCase):
    def test_only_windows_gets_the_no_window_creation_flag(self):
        # A dispatch tick can provision windowless; a popped console there is a
        # visible regression, so the flag must be set on nt and 0 elsewhere.
        expected = 0x08000000 if os.name == "nt" else 0
        self.assertEqual(M._no_window_creationflags(), expected)


if __name__ == "__main__":
    unittest.main()
