#!/usr/bin/env python3
"""Hermetic tests for tools/repo_guard.py (no filesystem / no subprocess)."""
from __future__ import annotations

import importlib.util
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "repo_guard.py"

WS = "C:/Users/u/work/fak"
SAFE = ("/tmp", "/var/tmp", "C:/Users/u/.cache", "C:/Users/u/Downloads")


def load():
    spec = importlib.util.spec_from_file_location("repo_guard", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    def _v(self, tool, ti):
        return self.mod.evaluate(tool, ti, workspace_root=WS, safe_roots=SAFE)

    def test_incident_relative_escape_denied(self):
        # The build-output escape that seeded the incident.
        v = self._v("Bash", {"command": "go build -o ../tools/.bin/fak.exe ./cmd/fak"})
        self.assertTrue(v)
        self.assertEqual(v[0]["reason"], "OUT_OF_TREE_WRITE")

    def test_incident_absolute_sibling_rm_denied(self):
        # The exact rm that destroyed the sibling repo (absolute path a regex can't judge).
        v = self._v("Bash", {"command": "rm -rf /c/Users/u/work/tools"})
        self.assertTrue(v)
        self.assertEqual(v[0]["why"], "sibling of workspace")

    def test_write_tool_escape_denied(self):
        self.assertTrue(self._v("Write", {"file_path": "../tools/poison.txt"}))

    def test_in_repo_ops_allowed(self):
        for cmd in (
            "go build -o fak.exe ./cmd/fak",
            "go build -o tools/.bin/fak.exe ./cmd/fak",
            "rm -rf ./build",
            "rm -rf internal/model/.cache",
            "mv internal/a internal/b",
        ):
            self.assertEqual(self._v("Bash", {"command": cmd}), [], cmd)

    def test_scratch_roots_allowed(self):
        for cmd in (
            "echo x > /tmp/log.txt",
            "cp a.txt /var/tmp/b.txt",
            "cp a.txt ~/.cache/b.txt",  # ~ is unresolvable but not a textual escape -> allow
        ):
            self.assertEqual(self._v("Bash", {"command": cmd}), [], cmd)

    def test_claude_state_dir_is_a_safe_root(self):
        # The agent's own memory/state tree (~/.claude/...) is on the production
        # allow-list, so writing memory is never denied as a cross-project leak.
        self.assertIn("/.claude", "".join(self.mod.default_safe_roots()))

    def test_agent_memory_write_allowed(self):
        # An absolute Write into the agent-memory tree, with .claude on safe_roots,
        # must pass (it would otherwise be an out-of-workspace denial).
        roots = SAFE + ("C:/Users/u/.claude",)
        fp = "C:/Users/u/.claude/projects/C--Users-u-work-fak/memory/note.md"
        v = self.mod.evaluate("Write", {"file_path": fp}, workspace_root=WS, safe_roots=roots)
        self.assertEqual(v, [], fp)

    def test_agent_state_roots_admits_account_variant_not_lookalike(self):
        # Per-account memory tree (~/.claude-gem8-netra) is the agent's own state;
        # a .claude look-alike (.claudex) is some other dir and must be excluded.
        roots = self.mod.agent_state_roots(
            "C:/Users/u", entries=[".claude", ".claude-gem8-netra", ".claude.json", ".claudex", "Documents"]
        )
        self.assertIn("C:/Users/u/.claude", roots)
        self.assertIn("C:/Users/u/.claude-gem8-netra", roots)
        self.assertIn("C:/Users/u/.claude.json", roots)
        self.assertNotIn("C:/Users/u/.claudex", roots)
        self.assertNotIn("C:/Users/u/Documents", roots)

    def test_account_variant_memory_write_allowed(self):
        # The exact path the live hook was denying before this fix.
        roots = SAFE + tuple(self.mod.agent_state_roots("C:/Users/u", entries=[".claude-gem8-netra"]))
        fp = "C:/Users/u/.claude-gem8-netra/projects/C--Users-u-work-fak/memory/note.md"
        self.assertEqual(self.mod.evaluate("Write", {"file_path": fp}, workspace_root=WS, safe_roots=roots), [], fp)

    def test_private_companion_is_the_same_named_sibling_only(self):
        self.assertEqual(self.mod.private_companion_roots(WS), ("C:/Users/u/work/fak-private",))
        self.assertEqual(self.mod.private_companion_roots("/c/work/fak"), ("C:/work/fak-private",))

    def test_private_companion_write_allowed(self):
        roots = SAFE + self.mod.private_companion_roots(WS)
        fp = "C:/Users/u/work/fak-private/MEMORY-glm52-2026-06-21.md"
        self.assertEqual(self.mod.evaluate("Write", {"file_path": fp}, workspace_root=WS, safe_roots=roots), [], fp)

    def test_private_companion_does_not_leak_to_lookalike_or_siblings(self):
        # Even WITH the companion + state roots on the allow-list, an unrelated
        # sibling and a `-private` look-alike must STILL deny (no broadening leak).
        roots = SAFE + self.mod.private_companion_roots(WS) + ("C:/Users/u/.claude-gem8-netra",)
        for fp in (
            "C:/Users/u/work/fak-private-evil/x.md",   # look-alike component, not the companion
            "C:/Users/u/work/fak-ci/x.md",             # a real but unrelated sibling
            "C:/Users/u/work/tools/poison.txt",        # the incident sibling
            "C:/Users/u/.claudex/leak.md",             # .claude look-alike, not the state tree
        ):
            self.assertTrue(self.mod.evaluate("Write", {"file_path": fp}, workspace_root=WS, safe_roots=roots), fp)

    def test_null_devices_allowed(self):
        # The universal "discard output" sinks are never a cross-project escape, so the
        # guard must not deny them (denying /dev/null only teaches operators to set
        # FAK_REPO_GUARD=off, disabling the whole gate). Windows NUL is in-tree already.
        for cmd in (
            "make ci > /dev/null 2>&1",
            "go test ./... > /dev/null",
            "echo done >> /dev/stderr",
        ):
            self.assertEqual(self._v("Bash", {"command": cmd}), [], cmd)
        self.assertEqual(self._v("Write", {"file_path": "/dev/null"}), [])

    def test_heredoc_program_comparisons_are_not_redirects(self):
        command = "python - file <<'EOF'\nif depth > 3:\n    return depth\nEOF\n"
        self.assertEqual(self._v("Bash", {"command": command}), [])

    def test_interpreter_program_comparison_is_not_bare_drive_redirect(self):
        command = "python -c 'if depth > 3: print(depth)'"
        self.assertEqual(self._v("Bash", {"command": command}), [])

    def test_out_of_tree_redirect_still_denied(self):
        self.assertTrue(self._v("Bash", {"command": "echo bad > /c/Users/u/work/tools/log.txt"}))

    def test_private_companion_empty_when_workspace_is_private(self):
        # When the workspace IS the private repo, there is no <ws>-private-private
        # companion; the function must return () rather than a nonexistent path.
        self.assertEqual(self.mod.private_companion_roots("C:/Users/u/work/fak-private"), ())
        self.assertEqual(self.mod.private_companion_roots("/c/work/fak-private"), ())
        # the public side still gets its paired private companion.
        self.assertEqual(self.mod.private_companion_roots(WS), ("C:/Users/u/work/fak-private",))

    def test_grep_dash_o_is_not_an_output_path(self):
        # -o is overloaded: grep -o is only-matching, not a build output file.
        self.assertEqual(self._v("Bash", {"command": "grep -o ../foo internal/policy/x.go"}), [])

    def test_reads_are_never_flagged(self):
        self.assertEqual(self._v("Bash", {"command": "cat ../README.md"}), [])

    def test_selftest_passes(self):
        self.assertEqual(self.mod._selftest(), 0)


class HookTests(unittest.TestCase):
    def setUp(self):
        self.mod = load()

    def _run(self, payload, env=None):
        import os
        old = {k: os.environ.get(k) for k in ("FAK_REPO_GUARD",)}
        if env:
            os.environ.update(env)
        else:
            os.environ.pop("FAK_REPO_GUARD", None)
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                rc = self.mod.run_hook(json.dumps(payload))
            return rc, buf.getvalue()
        finally:
            for k, val in old.items():
                if val is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = val

    def test_hook_denies_out_of_tree(self):
        rc, out = self._run(
            {"tool_name": "Bash", "cwd": WS, "tool_input": {"command": "rm -rf ../tools"}}
        )
        self.assertEqual(rc, 0)
        decision = json.loads(out)["hookSpecificOutput"]["permissionDecision"]
        self.assertEqual(decision, "deny")

    def test_hook_allows_in_repo(self):
        rc, out = self._run(
            {"tool_name": "Bash", "cwd": WS, "tool_input": {"command": "rm -rf ./build"}}
        )
        self.assertEqual((rc, out.strip()), (0, ""))

    def test_warn_mode_allows(self):
        rc, out = self._run(
            {"tool_name": "Bash", "cwd": WS, "tool_input": {"command": "rm -rf ../tools"}},
            env={"FAK_REPO_GUARD": "warn"},
        )
        self.assertEqual((rc, out.strip()), (0, ""))  # no deny JSON on stdout

    def test_off_mode_disables(self):
        rc, out = self._run(
            {"tool_name": "Bash", "cwd": WS, "tool_input": {"command": "rm -rf ../tools"}},
            env={"FAK_REPO_GUARD": "off"},
        )
        self.assertEqual((rc, out.strip()), (0, ""))


if __name__ == "__main__":
    unittest.main()
