#!/usr/bin/env python3
"""Hermetic tests for tools/idea_scout.py.

NOTHING live runs: arXiv/GitHub fetches and `gh` are never called. The pure
logic — Atom/JSON parsing, the transparent relevance score, the three dedup
rungs, issue rendering, and the score→dedup→threshold→CAP planner — is exercised
directly with fixtures, plus a real tmp-dir round-trip of the seen-cache.
"""
from __future__ import annotations

import contextlib
import datetime as dt
import importlib.util
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "idea_scout.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("idea_scout", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


M = load()
NOW = dt.datetime(2026, 6, 22, tzinfo=dt.timezone.utc)

ARXIV_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2606.01234v2</id>
    <published>2026-06-10T00:00:00Z</published>
    <title>Defending LLM Agents against Indirect Prompt Injection</title>
    <summary>We present a guardrail that quarantines untrusted tool results
      before they re-enter the agent context.</summary>
    <author><name>A. Researcher</name></author>
    <author><name>B. Coauthor</name></author>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/2512.99999v1</id>
    <published>2025-12-01T00:00:00Z</published>
    <title>An Unrelated Paper on Quantum Foam</title>
    <summary>Nothing to do with agents.</summary>
    <author><name>C. Physicist</name></author>
  </entry>
</feed>"""

GH_FIXTURE = [
    {"fullName": "acme/agent-firewall",
     "description": "A capability gateway and policy adjudicator for LLM tool calls",
     "url": "https://github.com/acme/agent-firewall",
     "stargazersCount": 540, "pushedAt": "2026-06-15T00:00:00Z",
     "createdAt": "2025-01-01T00:00:00Z", "language": "Go"},
    {"fullName": "tiny/nostars",
     "description": "barely related", "url": "https://github.com/tiny/nostars",
     "stargazersCount": 3, "pushedAt": "2020-01-01T00:00:00Z",
     "createdAt": "2019-01-01T00:00:00Z", "language": "Python"},
]

TOPIC = {"key": "prompt-injection-defense",
         "terms": ["prompt injection", "guardrail", "quarantine", "tool",
                   "agent", "capability", "policy", "gateway"],
         "area": "security"}


class TokenTest(unittest.TestCase):
    def test_tokenize_drops_short_and_punct(self) -> None:
        # length≥3 filter (no stopword list): 'a'(1) and 'kv'(2) drop; 'the'(3) stays.
        self.assertEqual(M.tokenize("A KV-cache, the GPU!"),
                         {"cache", "the", "gpu"})

    def test_jaccard(self) -> None:
        self.assertEqual(M.jaccard(set(), {"a"}), 0.0)
        self.assertAlmostEqual(M.jaccard({"a", "b"}, {"b", "c"}), 1 / 3)
        self.assertEqual(M.jaccard({"a", "b"}, {"a", "b"}), 1.0)


class ArxivParseTest(unittest.TestCase):
    def test_parse_strips_version_and_builds_id(self) -> None:
        cands = M.parse_arxiv_atom(ARXIV_FIXTURE, "prompt-injection-defense")
        self.assertEqual(len(cands), 2)
        c = cands[0]
        self.assertEqual(c["source_id"], "arxiv:2606.01234")  # vN stripped
        self.assertEqual(c["url"], "https://arxiv.org/abs/2606.01234")
        self.assertIn("Indirect Prompt Injection", c["title"])
        self.assertEqual(c["extra"]["authors"], ["A. Researcher", "B. Coauthor"])

    def test_malformed_feed_returns_empty(self) -> None:
        self.assertEqual(M.parse_arxiv_atom("<not xml", "k"), [])


class GithubParseTest(unittest.TestCase):
    def test_maps_fields(self) -> None:
        cands = M.parse_github_repos(GH_FIXTURE, "prompt-injection-defense")
        self.assertEqual(cands[0]["source_id"], "github:acme/agent-firewall")
        self.assertEqual(cands[0]["extra"]["stars"], 540)
        self.assertEqual(cands[0]["extra"]["language"], "Go")

    def test_source_id_is_lowercased(self) -> None:
        # GitHub repo names are case-insensitive; the dedup key must normalize so
        # a casing flip can't slip a duplicate past the seen-cache rung.
        items = [{"fullName": "Acme/Agent-Firewall", "description": "x",
                  "url": "https://github.com/Acme/Agent-Firewall",
                  "stargazersCount": 10, "pushedAt": "", "createdAt": ""}]
        c = M.parse_github_repos(items, "k")[0]
        self.assertEqual(c["source_id"], "github:acme/agent-firewall")
        # a prior run that filed the lower-cased id is now caught on the cache rung
        seen = {"github:acme/agent-firewall": {"filed_at": "2026-01-01"}}
        self.assertEqual(
            M.is_duplicate(c, seen, set(), [], "", 0.55), "seen-cache")


class ScoreTest(unittest.TestCase):
    def test_title_hit_beats_body_hit(self) -> None:
        cfg = dict(M.DEFAULTS)
        title_hit = {"title": "A guardrail for agents", "summary": "x",
                     "published": "", "extra": {}}
        body_hit = {"title": "Untitled", "summary": "a guardrail somewhere",
                    "published": "", "extra": {}}
        st, _ = M.score_candidate(title_hit, TOPIC, cfg, NOW)
        sb, _ = M.score_candidate(body_hit, TOPIC, cfg, NOW)
        self.assertGreater(st, sb)
        self.assertGreaterEqual(st, M.W_TITLE_HIT)

    def test_recency_and_stars_bonus(self) -> None:
        cfg = dict(M.DEFAULTS)
        fresh = {"title": "agent guardrail", "summary": "",
                 "published": "2026-06-10T00:00:00Z",
                 "extra": {"stars": 540, "pushed_at": "2026-06-15T00:00:00Z"}}
        score, reasons = M.score_candidate(fresh, TOPIC, cfg, NOW)
        joined = " ".join(reasons)
        self.assertIn("very fresh", joined)
        self.assertIn("stars", joined)
        self.assertIn("pushed", joined)
        # 2 title hits (20) + recent(12) + fresh(22) + 5★(+5) + push(10) = 69
        self.assertGreaterEqual(score, 60)

    def test_old_paper_no_recency_bonus(self) -> None:
        cfg = dict(M.DEFAULTS)
        old = {"title": "agent guardrail", "summary": "",
               "published": "2020-01-01T00:00:00Z", "extra": {}}
        _, reasons = M.score_candidate(old, TOPIC, cfg, NOW)
        self.assertNotIn("recent", " ".join(reasons))


class DedupTest(unittest.TestCase):
    def _index(self, issues):
        return M.existing_issue_index(issues)

    def test_seen_cache_rung(self) -> None:
        cand = {"source_id": "arxiv:2606.01234", "url": "https://arxiv.org/abs/2606.01234",
                "title": "X"}
        seen = {"arxiv:2606.01234": {"filed_at": "2026-01-01"}}
        self.assertEqual(
            M.is_duplicate(cand, seen, set(), [], "", 0.55), "seen-cache")

    def test_issue_body_source_stamp_rung(self) -> None:
        issues = [{"number": 1, "title": "old",
                   "body": "stuff\n<!-- idea-scout-source: arxiv:2606.01234 -->"}]
        stamped, tsets, bodies = self._index(issues)
        cand = {"source_id": "arxiv:2606.01234",
                "url": "https://arxiv.org/abs/2606.01234", "title": "X"}
        self.assertEqual(
            M.is_duplicate(cand, {}, stamped, tsets, bodies, 0.55), "issue-body")

    def test_issue_body_url_rung(self) -> None:
        issues = [{"number": 2, "title": "manual",
                   "body": "see https://github.com/acme/agent-firewall for prior art"}]
        stamped, tsets, bodies = self._index(issues)
        cand = {"source_id": "github:acme/agent-firewall",
                "url": "https://github.com/acme/agent-firewall", "title": "Z"}
        self.assertEqual(
            M.is_duplicate(cand, {}, stamped, tsets, bodies, 0.55), "issue-body")

    def test_title_near_rung(self) -> None:
        issues = [{"number": 3,
                   "title": "Defending LLM Agents against Indirect Prompt Injection",
                   "body": "no stamp here"}]
        stamped, tsets, bodies = self._index(issues)
        cand = {"source_id": "arxiv:9999.00000", "url": "https://arxiv.org/abs/9999.00000",
                "title": "Defending LLM Agents against Indirect Prompt Injection attacks"}
        self.assertEqual(
            M.is_duplicate(cand, {}, stamped, tsets, bodies, 0.55), "title-near")

    def test_genuinely_new_passes(self) -> None:
        cand = {"source_id": "arxiv:1111.22222", "url": "https://arxiv.org/abs/1111.22222",
                "title": "A totally distinct unrelated headline about turtles"}
        self.assertIsNone(M.is_duplicate(cand, {}, set(), [], "", 0.55))


class RenderTest(unittest.TestCase):
    def test_render_stamps_source_and_labels(self) -> None:
        cand = M.parse_arxiv_atom(ARXIV_FIXTURE, "prompt-injection-defense")[0]
        issue = M.render_issue(cand, 70, ["terms: guardrail"], TOPIC, "2026-06-22")
        self.assertTrue(issue["title"].startswith("idea-scout: "))
        self.assertIn("<!-- idea-scout-source: arxiv:2606.01234 -->", issue["body"])
        self.assertIn("https://arxiv.org/abs/2606.01234", issue["body"])
        self.assertIn("dispatchability: `triage_only`", issue["body"])
        self.assertEqual(issue["labels"], ["idea-scout", "needs-triage",
                                           "triage-only", "research", "security"])
        self.assertIn("Authors:", issue["body"])

    def test_long_title_truncated(self) -> None:
        cand = {"source": "arxiv", "source_id": "arxiv:1", "url": "u",
                "title": "x" * 200, "summary": "", "published": "", "extra": {}}
        issue = M.render_issue(cand, 30, [], TOPIC, "2026-06-22")
        # "idea-scout: " + ≤100 chars
        self.assertLessEqual(len(issue["title"]), len("idea-scout: ") + 100)
        self.assertTrue(issue["title"].endswith("…"))


class LabelTest(unittest.TestCase):
    def test_ensure_scout_label_creates_triage_labels_too(self) -> None:
        orig_run = M.subprocess.run
        calls = []

        class Proc:
            returncode = 0
            stderr = ""

        def fake_run(argv, **kwargs):
            calls.append((argv, kwargs))
            return Proc()

        try:
            M.subprocess.run = fake_run
            M.ensure_scout_label()
        finally:
            M.subprocess.run = orig_run

        self.assertEqual([argv[3] for argv, _ in calls],
                         [M.SCOUT_LABEL, M.TRIAGE_LABEL, M.TRIAGE_ONLY_LABEL])
        self.assertTrue(all(kwargs["timeout"] == 30 for _, kwargs in calls))


class PlanTest(unittest.TestCase):
    def _topics(self):
        return {TOPIC["key"]: TOPIC}

    def test_cap_and_sort(self) -> None:
        cfg = dict(M.DEFAULTS, max_issues=2, min_score=1)
        cands = [
            {"source": "arxiv", "source_id": f"arxiv:{i}", "url": f"u{i}",
             "title": "agent guardrail policy capability tool gateway",
             "summary": "", "published": "2026-06-10T00:00:00Z",
             "topic": TOPIC["key"], "extra": {}}
            for i in range(5)]
        # make one clearly top-scored via extra stars
        cands[3]["source"] = "github"
        cands[3]["extra"] = {"stars": 3000, "pushed_at": "2026-06-20T00:00:00Z"}
        to_file, stats = M.plan_issues(
            cands, self._topics(), {}, set(), [], "", cfg, "2026-06-22", NOW)
        self.assertEqual(len(to_file), 2)  # capped
        self.assertEqual(to_file[0]["source_id"], "arxiv:3")  # highest score first
        self.assertGreaterEqual(to_file[0]["score"], to_file[1]["score"])

    def test_below_min_dropped(self) -> None:
        cfg = dict(M.DEFAULTS, min_score=1000)
        cands = [{"source": "arxiv", "source_id": "arxiv:x", "url": "u",
                  "title": "agent guardrail", "summary": "", "published": "",
                  "topic": TOPIC["key"], "extra": {}}]
        to_file, stats = M.plan_issues(
            cands, self._topics(), {}, set(), [], "", cfg, "2026-06-22", NOW)
        self.assertEqual(to_file, [])
        self.assertEqual(stats["below-min"], 1)

    def test_within_run_dedup(self) -> None:
        cfg = dict(M.DEFAULTS, min_score=1)
        cand = {"source": "arxiv", "source_id": "arxiv:dup", "url": "u",
                "title": "agent guardrail policy", "summary": "",
                "published": "2026-06-10T00:00:00Z", "topic": TOPIC["key"],
                "extra": {}}
        to_file, stats = M.plan_issues(
            [cand, dict(cand)], self._topics(), {}, set(), [], "", cfg,
            "2026-06-22", NOW)
        self.assertEqual(len(to_file), 1)
        self.assertEqual(stats["within-run-dup"], 1)

    def test_seen_cache_skips_in_plan(self) -> None:
        cfg = dict(M.DEFAULTS, min_score=1)
        cand = {"source": "arxiv", "source_id": "arxiv:known", "url": "u",
                "title": "agent guardrail policy", "summary": "",
                "published": "2026-06-10T00:00:00Z", "topic": TOPIC["key"],
                "extra": {}}
        to_file, stats = M.plan_issues(
            [cand], self._topics(), {"arxiv:known": {}}, set(), [], "", cfg,
            "2026-06-22", NOW)
        self.assertEqual(to_file, [])
        self.assertEqual(stats["seen-cache"], 1)


class ConfigCacheTest(unittest.TestCase):
    def test_default_config(self) -> None:
        topics, cfg = M.load_config(None)
        self.assertTrue(topics)
        self.assertEqual(cfg["max_issues"], M.DEFAULTS["max_issues"])

    def test_config_override(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "cfg.json"
            p.write_text(json.dumps({
                "topics": [{"key": "only", "arxiv": "abs:x", "terms": ["x"]}],
                "thresholds": {"max_issues": 9, "bogus": 1},
            }), encoding="utf-8")
            topics, cfg = M.load_config(str(p))
            self.assertEqual([t["key"] for t in topics], ["only"])
            self.assertEqual(cfg["max_issues"], 9)
            self.assertNotIn("bogus", cfg)  # unknown keys ignored

    def test_seen_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            ws = Path(d)
            self.assertEqual(M.load_seen(ws), {})
            M.save_seen(ws, {"arxiv:1": {"filed_at": "2026-06-22"}})
            self.assertEqual(M.load_seen(ws), {"arxiv:1": {"filed_at": "2026-06-22"}})
            self.assertTrue(M.cache_path(ws).exists())

    def test_config_rejects_topic_without_terms(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text(json.dumps({
                "topics": [{"key": "x", "arxiv": "abs:y"}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                M.load_config(str(p))

    def test_config_rejects_topic_without_source(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "bad.json"
            p.write_text(json.dumps({
                "topics": [{"key": "x", "terms": ["y"]}]}), encoding="utf-8")
            with self.assertRaises(ValueError):
                M.load_config(str(p))

    def test_default_topics_pass_validation(self) -> None:
        M.load_config(None)  # must not raise — the baked-in defaults are valid


class MainHermeticTest(unittest.TestCase):
    """Drive main() end to end with every network/gh boundary stubbed, to lock
    the load-bearing safety contracts: dry-run mutates nothing; --live files +
    caches; the cap is never exceeded."""

    def setUp(self) -> None:
        self._orig = (M.fetch_arxiv, M.fetch_github, M.fetch_existing_issues,
                      M.create_issue, M.ensure_scout_label)

    def tearDown(self) -> None:
        (M.fetch_arxiv, M.fetch_github, M.fetch_existing_issues,
         M.create_issue, M.ensure_scout_label) = self._orig

    def _stub(self, *, arxiv: str = "", github_items=None, existing=None,
              created_urls=None):
        M.fetch_arxiv = lambda *a, **k: arxiv
        M.fetch_github = lambda *a, **k: list(github_items or [])
        M.fetch_existing_issues = lambda *a, **k: list(existing or [])
        M.ensure_scout_label = lambda: None
        calls: list = []

        def _create(issue, *, milestone=""):
            calls.append({**issue, "_milestone": milestone})
            return (created_urls or {}).get(
                issue["source_id"], "https://github.com/x/y/issues/1")
        M.create_issue = _create
        return calls

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(argv)
        return rc, buf.getvalue()

    def test_dry_run_writes_no_cache_and_files_nothing(self) -> None:
        calls = self._stub(arxiv=ARXIV_FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run(["--workspace", d])
            self.assertEqual(rc, 0)
            self.assertEqual(calls, [])                      # nothing filed
            self.assertFalse(M.cache_path(Path(d)).exists())  # cache untouched
            self.assertIn("dry-run", out)

    def test_live_files_and_caches(self) -> None:
        calls = self._stub(arxiv=ARXIV_FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d, "--live"])
            self.assertEqual(rc, 0)
            self.assertGreaterEqual(len(calls), 1)
            seen = M.load_seen(Path(d))
            # the on-topic prompt-injection paper is the one that clears min-score
            self.assertIn("arxiv:2606.01234", seen)
            self.assertNotIn("arxiv:2512.99999", seen)  # off-topic, below min-score

    def test_live_assigns_milestone_when_set(self) -> None:
        # --milestone threads through to create_issue so scouted work joins the
        # milestone backlog the dispatch fleet selects from.
        calls = self._stub(arxiv=ARXIV_FIXTURE)
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d, "--live", "--milestone",
                               "Fleet observability you can trust"])
            self.assertEqual(rc, 0)
            self.assertTrue(calls)
            self.assertTrue(all(c["_milestone"] == "Fleet observability you can trust"
                                for c in calls))

    def test_live_respects_cap(self) -> None:
        # three distinct, well-starred, on-topic repos; cap of 1 → exactly 1 filed
        items = [
            {"fullName": f"acme/agent-guardrail-defense-{i}",
             "description": "agent tool guardrail defense quarantine",
             "url": f"https://github.com/acme/agent-guardrail-defense-{i}",
             "stargazersCount": 800, "pushedAt": "2026-06-20T00:00:00Z",
             "createdAt": "2025-01-01T00:00:00Z", "language": "Go"}
            for i in range(3)]
        calls = self._stub(github_items=items)
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d, "--live", "--max-issues", "1"])
            self.assertEqual(rc, 0)
            self.assertEqual(len(calls), 1)  # cap holds despite 3 candidates

    def test_refuse_when_issue_fetch_fails_and_no_cache(self) -> None:
        self._stub(arxiv=ARXIV_FIXTURE)

        def _boom(*a, **k):
            raise RuntimeError("gh not authed")
        M.fetch_existing_issues = _boom
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d])
            self.assertEqual(rc, 2)  # refuse rather than risk a blind run


if __name__ == "__main__":
    unittest.main()
