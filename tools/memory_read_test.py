#!/usr/bin/env python3
"""Hermetic tests for tools/memory_read.py.

render_digest() is exercised over a synthetic store built in a tempdir — no repo
mirror, no network. We assert: an absent store degrades to a harmless one-line note
(exit-0 contract), the index is emitted verbatim, per-fact bodies are expanded with
frontmatter stripped, non-fact docs are never expanded, --index-only skips bodies,
and --max-bytes bounds the per-fact output and reports the omission.

The lessons-ledger contract (#2141) is pinned the same way: a published lesson under
<store>/lessons/ is injected ONLY when its trigger matches the session context AND
its read-time verify probe passes AND the gate is live; shadow reports the would-
inject set without bodies; a failing verify withholds with a note; and a store with
no lessons/ dir renders byte-identical to the pre-ledger digest.
"""
from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "memory_read.py"


def load():
    spec = importlib.util.spec_from_file_location("memory_read", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load()


def _fact(name: str, desc: str, body: str) -> str:
    return (f"---\nname: {name}\ndescription: {desc}\nmetadata:\n  type: project\n"
            f"---\n\n{body}\n")


def build_store(d: Path) -> None:
    (d / "MEMORY.md").write_text(
        "- [First fact](first-fact.md) — the hook one\n"
        "- [Second fact](second-fact.md) — the hook two\n"
        "- [Archive index](MEMORY_archive.md) — cold tier\n",
        encoding="utf-8")
    (d / "first-fact.md").write_text(
        _fact("first-fact", "desc one", "BODY-ONE is the durable fact."), encoding="utf-8")
    (d / "second-fact.md").write_text(
        _fact("second-fact", "desc two", "BODY-TWO is the other fact."), encoding="utf-8")
    (d / "MEMORY_archive.md").write_text("- [old](old.md) — should not expand\n",
                                         encoding="utf-8")


_CTX = {"host": "test-node", "host_os": "windows", "path": "cmd/fak"}


def _lesson(name: str, trigger: dict, verify: dict, body: str) -> str:
    lines = ["---", f"name: {name}", f"description: desc of {name}"]
    if trigger:
        lines.append("trigger:")
        lines += [f"  {k}: {v}" for k, v in trigger.items()]
    if verify:
        lines.append("verify:")
        lines += [f"  {k}: {v}" for k, v in verify.items()]
    lines += ["---", "", body, ""]
    return "\n".join(lines)


def build_lessons_root(t: Path) -> Path:
    """Lay out <root>/.claude/memory with facts + a lessons/ ledger; returns the
    STORE dir. Root carries anchor.txt so verify probes have a real target."""
    store = t / ".claude" / "memory"
    (store / "lessons").mkdir(parents=True)
    build_store(store)
    (t / "anchor.txt").write_text("STILL-TRUE anchor content\n", encoding="utf-8")
    (store / "lessons" / "match-fresh.md").write_text(_lesson(
        "match-fresh", {"host_os": "windows"},
        {"path": "anchor.txt", "contains": "STILL-TRUE"},
        "LESSON-FRESH: use PowerShell here."), encoding="utf-8")
    (store / "lessons" / "match-stale.md").write_text(_lesson(
        "match-stale", {"host_os": "windows"},
        {"path": "gone.txt"},
        "LESSON-STALE: should never be asserted."), encoding="utf-8")
    (store / "lessons" / "other-host.md").write_text(_lesson(
        "other-host", {"host_os": "linux"}, {},
        "LESSON-OTHER: wrong context."), encoding="utf-8")
    return store


class ParseIndex(unittest.TestCase):
    def test_links_extracted_dedup_and_nonfact_dropped(self):
        text = ("- [A](a.md) — h\n- [B](b.md) — h\n- [A again](a.md) — dup\n"
                "- [Idx](MEMORY.md) — self\n- [Sub](sub/c.md) — path\n")
        self.assertEqual(M.parse_index(text), [("A", "a.md"), ("B", "b.md")])


class StripFrontmatter(unittest.TestCase):
    def test_removes_leading_block(self):
        self.assertEqual(
            M.strip_frontmatter("---\nname: x\n---\n\nhello\n"), "hello\n")

    def test_no_frontmatter_passthrough(self):
        self.assertEqual(M.strip_frontmatter("just text\n"), "just text\n")


class RenderDigest(unittest.TestCase):
    def test_absent_store_is_harmless_note(self):
        with tempfile.TemporaryDirectory() as t:
            out = M.render_digest(Path(t) / "nope")
            self.assertIn("no committed memory mirror", out)
            self.assertNotIn("BODY-ONE", out)

    def test_full_digest_expands_facts(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            build_store(d)
            out = M.render_digest(d)
            self.assertIn("committed mirror", out)
            self.assertIn("First fact", out)            # index line present
            self.assertIn("BODY-ONE is the durable fact.", out)
            self.assertIn("BODY-TWO is the other fact.", out)
            self.assertNotIn("name: first-fact", out)   # frontmatter stripped
            self.assertNotIn("should not expand", out)   # non-fact doc not expanded

    def test_index_only_skips_bodies(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            build_store(d)
            out = M.render_digest(d, index_only=True)
            self.assertIn("First fact", out)
            self.assertNotIn("BODY-ONE", out)

    def test_max_bytes_bounds_and_reports_omission(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            build_store(d)
            out = M.render_digest(d, max_bytes=1)   # only the first fact fits
            self.assertIn("BODY-ONE", out)
            self.assertNotIn("BODY-TWO", out)
            self.assertIn("omitted", out)


class ParseLessonMeta(unittest.TestCase):
    def test_trigger_verify_and_scalars_extracted(self):
        meta = M.parse_lesson_meta(_lesson(
            "l1", {"host_os": "windows", "tool": "bash"},
            {"path": "a.txt", "contains": "X"}, "body"))
        self.assertEqual(meta["trigger"], {"host_os": "windows", "tool": "bash"})
        self.assertEqual(meta["verify"], {"path": "a.txt", "contains": "X"})
        self.assertEqual(meta["meta"]["name"], "l1")

    def test_no_frontmatter_is_empty(self):
        self.assertEqual(M.parse_lesson_meta("just a body\n"), {})


class TriggerMatches(unittest.TestCase):
    def test_and_semantics_and_case_insensitive(self):
        self.assertTrue(M.trigger_matches(
            {"host_os": "Windows", "host": "TEST-NODE"}, _CTX))
        self.assertFalse(M.trigger_matches(
            {"host_os": "windows", "tool": "bash"}, _CTX))  # tool not in ctx

    def test_empty_trigger_never_matches(self):
        self.assertFalse(M.trigger_matches({}, _CTX))

    def test_unknown_key_fails_closed(self):
        self.assertFalse(M.trigger_matches({"moon_phase": "full"}, _CTX))

    def test_path_glob(self):
        self.assertTrue(M.trigger_matches({"path_glob": "cmd/fak/**"}, _CTX))
        self.assertTrue(M.trigger_matches({"path_glob": "cmd/*"}, _CTX))
        self.assertFalse(M.trigger_matches({"path_glob": "internal/**"}, _CTX))


class VerifyLesson(unittest.TestCase):
    def test_probe_outcomes(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            (root / "a.txt").write_text("hello NEEDLE", encoding="utf-8")
            self.assertEqual(M.verify_lesson({}, root), (True, ""))
            self.assertTrue(M.verify_lesson(
                {"path": "a.txt", "contains": "NEEDLE"}, root)[0])
            ok, why = M.verify_lesson({"path": "a.txt", "contains": "GONE"}, root)
            self.assertFalse(ok)
            self.assertIn("contains", why)
            ok, why = M.verify_lesson({"path": "missing.txt"}, root)
            self.assertFalse(ok)
            self.assertIn("missing", why)


class RenderLessons(unittest.TestCase):
    def test_live_injects_only_matched_fresh_before_facts(self):
        with tempfile.TemporaryDirectory() as t:
            store = build_lessons_root(Path(t))
            out = M.render_digest(store, ctx=dict(_CTX), lessons="live")
            self.assertIn("LESSON-FRESH", out)
            self.assertNotIn("LESSON-STALE", out)      # withheld, not asserted
            self.assertIn("withheld stale lesson", out)
            self.assertIn("gone.txt", out)
            self.assertNotIn("LESSON-OTHER", out)      # trigger did not match
            self.assertLess(out.index("LESSON-FRESH"), out.index("BODY-ONE"))

    def test_shadow_default_reports_without_injecting(self):
        with tempfile.TemporaryDirectory() as t:
            store = build_lessons_root(Path(t))
            out = M.render_digest(store, ctx=dict(_CTX), lessons="shadow")
            self.assertIn("lessons ledger (shadow)", out)
            self.assertIn("1 of 3", out)
            self.assertIn("match-fresh.md", out)
            self.assertNotIn("LESSON-FRESH", out)      # no body in shadow
            self.assertIn("withheld stale lesson", out)

    def test_off_gate_is_silent(self):
        with tempfile.TemporaryDirectory() as t:
            store = build_lessons_root(Path(t))
            out = M.render_digest(store, ctx=dict(_CTX), lessons="off")
            self.assertNotIn("lesson", out.lower())

    def test_no_lessons_dir_renders_identically(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            build_store(d)
            base = M.render_digest(d)
            gated = M.render_digest(d, ctx=dict(_CTX), lessons="live")
            self.assertEqual(base, gated)

    def test_lessons_render_even_index_only(self):
        with tempfile.TemporaryDirectory() as t:
            store = build_lessons_root(Path(t))
            out = M.render_digest(store, ctx=dict(_CTX), lessons="live",
                                  index_only=True)
            self.assertIn("LESSON-FRESH", out)
            self.assertNotIn("BODY-ONE", out)


if __name__ == "__main__":
    unittest.main()
