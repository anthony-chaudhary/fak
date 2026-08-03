#!/usr/bin/env python3
"""Hermetic tests for tools/idea_scout.py.

NOTHING live runs: arXiv/GitHub fetches and `gh` are never called — every fetcher
main() can reach, INCLUDING the fresh GitHub lane, is stubbed. The pure logic —
Atom/JSON parsing, the transparent relevance score, the four dedup rungs, issue
rendering, and the score→dedup→threshold→CAP planner — is exercised directly with
fixtures, plus a real tmp-dir round-trip of the seen-cache.

``FiledStampDurabilityTest`` carries the #5543 regression: a source filed once is
never filed again even with the local cache gone and the original issue outside
the recency window.
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

    def test_trending_bonus(self) -> None:
        cfg = dict(M.DEFAULTS)
        # Same stars, same recent push; only repo age differs. The young repo
        # accrued its stars fast (high stars/day) and earns the trending bonus.
        young = {"title": "agent x", "summary": "",
                 "published": "2026-06-02T00:00:00Z",
                 "extra": {"stars": 400, "pushed_at": "2026-06-20T00:00:00Z"}}
        old = {"title": "agent x", "summary": "",
               "published": "2022-06-02T00:00:00Z",
               "extra": {"stars": 400, "pushed_at": "2026-06-20T00:00:00Z"}}
        young_score, young_reasons = M.score_candidate(young, TOPIC, cfg, NOW)
        old_score, _ = M.score_candidate(old, TOPIC, cfg, NOW)
        self.assertGreater(young_score, old_score)
        self.assertIn("trending", " ".join(young_reasons))


class DedupTest(unittest.TestCase):
    def _index(self, issues):
        return M.existing_issue_index(issues)

    def test_seen_cache_rung(self) -> None:
        cand = {"source_id": "arxiv:2606.01234", "url": "https://arxiv.org/abs/2606.01234",
                "title": "X"}
        seen = {"arxiv:2606.01234": {"filed_at": "2026-01-01"}}
        self.assertEqual(
            M.is_duplicate(cand, seen, set(), [], "", 0.55), "seen-cache")

    def test_filed_stamp_rung(self) -> None:
        # The source_id stamp is rung 2 and is reported under its OWN name: it is
        # the complete filing history, not the windowed URL sighting of rung 3.
        issues = [{"number": 1, "title": "old",
                   "body": "stuff\n<!-- idea-scout-source: arxiv:2606.01234 -->"}]
        stamped, tsets, bodies = self._index(issues)
        cand = {"source_id": "arxiv:2606.01234",
                "url": "https://arxiv.org/abs/2606.01234", "title": "X"}
        self.assertEqual(
            M.is_duplicate(cand, {}, stamped, tsets, bodies, 0.55), "filed-stamp")

    def test_filed_stamp_rung_folds_case(self) -> None:
        # GitHub hands back whichever casing it likes; an un-folded compare would
        # let `Acme/Repo` slip past a stamp reading `acme/repo`.
        stamped = M.stamp_index(
            [{"body": "<!-- idea-scout-source: github:acme/agent-firewall -->"}])
        self.assertEqual(stamped, {"github:acme/agent-firewall"})
        cand = {"source_id": "github:Acme/Agent-Firewall",
                "url": "https://github.com/Acme/Agent-Firewall", "title": "X"}
        self.assertEqual(
            M.is_duplicate(cand, {}, stamped, [], "", 0.55), "filed-stamp")

    def test_closed_issue_stamp_still_blocks(self) -> None:
        # The regression #5543 was built on: a source whose issue was triaged and
        # CLOSED came back as a fresh needs-triage ticket. A stamp is a stamp
        # regardless of state — the index is fetched with `--state all`.
        stamped = M.stamp_index(
            [{"body": "closed long ago\n"
                      "<!-- idea-scout-source: github:fu351/doberman-core -->"}])
        cand = {"source_id": "github:fu351/doberman-core",
                "url": "https://github.com/fu351/Doberman-Core", "title": "X"}
        self.assertEqual(
            M.is_duplicate(cand, {}, stamped, [], "", 0.55), "filed-stamp")

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
        to_file, stats, dropped = M.plan_issues(
            cands, self._topics(), {}, set(), [], "", cfg, "2026-06-22", NOW)
        self.assertEqual(len(to_file), 2)  # capped
        self.assertEqual(to_file[0]["source_id"], "arxiv:3")  # highest score first
        self.assertGreaterEqual(to_file[0]["score"], to_file[1]["score"])

    def test_below_min_dropped(self) -> None:
        cfg = dict(M.DEFAULTS, min_score=1000)
        cands = [{"source": "arxiv", "source_id": "arxiv:x", "url": "u",
                  "title": "agent guardrail", "summary": "", "published": "",
                  "topic": TOPIC["key"], "extra": {}}]
        to_file, stats, dropped = M.plan_issues(
            cands, self._topics(), {}, set(), [], "", cfg, "2026-06-22", NOW)
        self.assertEqual(to_file, [])
        self.assertEqual(stats["below-min"], 1)

    def test_within_run_dedup(self) -> None:
        cfg = dict(M.DEFAULTS, min_score=1)
        cand = {"source": "arxiv", "source_id": "arxiv:dup", "url": "u",
                "title": "agent guardrail policy", "summary": "",
                "published": "2026-06-10T00:00:00Z", "topic": TOPIC["key"],
                "extra": {}}
        to_file, stats, dropped = M.plan_issues(
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
        to_file, stats, dropped = M.plan_issues(
            [cand], self._topics(), {"arxiv:known": {}}, set(), [], "", cfg,
            "2026-06-22", NOW)
        self.assertEqual(to_file, [])
        self.assertEqual(stats["seen-cache"], 1)


class GatherTest(unittest.TestCase):
    """gather_candidates walks GitHub on two lanes; the fresh lane admits young
    repos the stars floor drops and tags their provenance. Hermetic: both
    module-level fetchers are monkeypatched, nothing live runs."""

    def test_fresh_lane_admits_young_repo_and_tags_it(self) -> None:
        young = {"fullName": "newco/fresh-agent",
                 "description": "a brand-new agent tool sandbox",
                 "url": "https://github.com/newco/fresh-agent",
                 "stargazersCount": 8, "pushedAt": "2026-06-18T00:00:00Z",
                 "createdAt": "2026-06-10T00:00:00Z", "language": "Go"}
        topics = [{"key": "t", "github": "agent tool", "terms": ["agent", "tool"]}]
        cfg = dict(M.DEFAULTS)  # min_stars=25, fresh_min_stars=3, fresh_per_topic=6
        errors: list[str] = []
        orig_g, orig_f = M.fetch_github, M.fetch_github_fresh
        try:
            # Same repo on both lanes: the stars floor drops it, only fresh admits it.
            M.fetch_github = lambda q, n: [dict(young)]
            M.fetch_github_fresh = lambda q, n: [dict(young)]
            cands = M.gather_candidates(topics, cfg, errors)
        finally:
            M.fetch_github, M.fetch_github_fresh = orig_g, orig_f
        self.assertEqual(errors, [])
        self.assertEqual(len(cands), 1)
        self.assertEqual(cands[0]["source_id"], "github:newco/fresh-agent")
        self.assertEqual(cands[0]["extra"].get("lane"), "fresh")

    def test_fresh_lane_respects_fresh_min_stars(self) -> None:
        toy = {"fullName": "toy/repo", "url": "https://github.com/toy/repo",
               "description": "", "stargazersCount": 1,
               "pushedAt": "2026-06-19T00:00:00Z", "createdAt": "2026-06-15T00:00:00Z"}
        topics = [{"key": "t", "github": "agent tool", "terms": ["agent"]}]
        cfg = dict(M.DEFAULTS)  # fresh_min_stars=3
        errors: list[str] = []
        orig_g, orig_f = M.fetch_github, M.fetch_github_fresh
        try:
            M.fetch_github = lambda q, n: []
            M.fetch_github_fresh = lambda q, n: [dict(toy)]
            cands = M.gather_candidates(topics, cfg, errors)
        finally:
            M.fetch_github, M.fetch_github_fresh = orig_g, orig_f
        self.assertEqual(errors, [])
        self.assertEqual(cands, [])


class ConfigCacheTest(unittest.TestCase):
    def test_default_config(self) -> None:
        topics, cfg = M.load_config(None)
        self.assertTrue(topics)
        self.assertEqual(cfg["max_issues"], M.DEFAULTS["max_issues"])
        # the fresh-lane knobs are part of the default config
        self.assertEqual(cfg["fresh_per_topic"], M.DEFAULTS["fresh_per_topic"])
        self.assertEqual(cfg["fresh_min_stars"], 3)
        self.assertEqual(cfg["fresh_window_days"], 45)

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
        self._orig = (M.fetch_arxiv, M.fetch_github, M.fetch_github_fresh,
                      M.fetch_existing_issues, M.fetch_scout_issues,
                      M.create_issue, M.ensure_scout_label)

    def tearDown(self) -> None:
        (M.fetch_arxiv, M.fetch_github, M.fetch_github_fresh,
         M.fetch_existing_issues, M.fetch_scout_issues,
         M.create_issue, M.ensure_scout_label) = self._orig

    def _stub(self, *, arxiv: str = "", github_items=None, existing=None,
              filed=None, created_urls=None):
        M.fetch_arxiv = lambda *a, **k: arxiv
        M.fetch_github = lambda *a, **k: list(github_items or [])
        # The fresh lane must be stubbed too or main() reaches the real
        # `gh search repos`: the suite is meant to be hermetic, and a live lane
        # silently injects today's real GitHub results into every fixture.
        M.fetch_github_fresh = lambda *a, **k: []
        M.fetch_existing_issues = lambda *a, **k: list(existing or [])
        # `filed` is the label-targeted rung-2 corpus: every issue the scout has
        # EVER filed, deliberately disjoint from the recency window above so a
        # test can prove the guarantee does not come from the window.
        M.fetch_scout_issues = lambda *a, **k: list(filed or [])
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


class FiledStampDurabilityTest(unittest.TestCase):
    """#5543: a source filed once must never be filed again — with NO local cache
    and with the original issue far outside the recency window.

    The bug was that the only two non-local rungs were a git-ignored cache and a
    fixed-size scan of the most recent `issue_scan_limit` issues. Once the tracker
    outgrew that window, an old filed issue became invisible and its source was
    re-filed. These tests pin the replacement: the guarantee comes from a
    label-targeted index of the scout's own filing history, so an EMPTY window and
    an EMPTY cache still block a re-file."""

    # The real re-file from the ticket: #528 filed it, triage closed it, and months
    # later #5298 filed it again because #528 had fallen out of the 800-window.
    OLD_FILED = [{"number": 528, "title": "idea-scout: fu351/Doberman-Core",
                  "body": "auto-filed long ago\n"
                          "<!-- idea-scout-source: github:fu351/doberman-core -->"}]
    REPO = {"fullName": "fu351/Doberman-Core",
            "description": "an agent tool guardrail policy capability gateway",
            "url": "https://github.com/fu351/Doberman-Core",
            "stargazersCount": 900, "pushedAt": "2026-06-20T00:00:00Z",
            "createdAt": "2025-01-01T00:00:00Z", "language": "Go"}

    def setUp(self) -> None:
        self._orig = (M.fetch_arxiv, M.fetch_github, M.fetch_github_fresh,
                      M.fetch_existing_issues, M.fetch_scout_issues,
                      M.create_issue, M.ensure_scout_label)
        M.fetch_arxiv = lambda *a, **k: ""
        M.fetch_github_fresh = lambda *a, **k: []   # hermetic: no live fresh lane
        M.ensure_scout_label = lambda: None
        self.created: list = []

        def _create(issue, *, milestone=""):
            self.created.append(issue)
            return "https://github.com/x/y/issues/1"
        M.create_issue = _create

    def tearDown(self) -> None:
        (M.fetch_arxiv, M.fetch_github, M.fetch_github_fresh,
         M.fetch_existing_issues, M.fetch_scout_issues,
         M.create_issue, M.ensure_scout_label) = self._orig

    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = M.main(argv)
        return rc, buf.getvalue()

    def test_old_filed_source_blocked_with_empty_cache_and_empty_window(self) -> None:
        M.fetch_github = lambda *a, **k: [dict(self.REPO)]
        M.fetch_existing_issues = lambda *a, **k: []   # window has aged it out
        M.fetch_scout_issues = lambda *a, **k: list(self.OLD_FILED)
        with tempfile.TemporaryDirectory() as d:       # no seen.json at all
            rc, out = self._run(["--workspace", d, "--live", "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual(self.created, [], "re-filed an already-filed source")
        self.assertEqual(payload["planned"], [])
        self.assertEqual(payload["skipped"]["filed-stamp"], 1)
        self.assertTrue(payload["dedup_index"]["scout_index_complete"])

    def test_control_unfiled_source_is_still_admitted(self) -> None:
        # Same wiring, a source that is genuinely absent from the filing history:
        # it MUST still be planned. Otherwise the fix is just "refuse everything".
        novel = dict(self.REPO, fullName="brandnew/agent-policy-gateway",
                     url="https://github.com/brandnew/agent-policy-gateway")
        M.fetch_github = lambda *a, **k: [novel]
        M.fetch_existing_issues = lambda *a, **k: []
        M.fetch_scout_issues = lambda *a, **k: list(self.OLD_FILED)
        with tempfile.TemporaryDirectory() as d:
            rc, out = self._run(["--workspace", d, "--json"])
        self.assertEqual(rc, 0)
        payload = json.loads(out)
        self.assertEqual([p["source_id"] for p in payload["planned"]],
                         ["github:brandnew/agent-policy-gateway"])
        self.assertEqual(payload["skipped"]["filed-stamp"], 0)

    def test_refuse_when_scout_index_unavailable(self) -> None:
        # The durable rung is mandatory. A populated seen-cache must NOT buy a pass:
        # that local file is exactly what proved unreliable.
        M.fetch_github = lambda *a, **k: [dict(self.REPO)]
        M.fetch_existing_issues = lambda *a, **k: []

        def _boom(*a, **k):
            raise RuntimeError("gh: label query failed")
        M.fetch_scout_issues = _boom
        with tempfile.TemporaryDirectory() as d:
            M.save_seen(Path(d), {"github:something": {"filed_at": "2026-01-01"}})
            rc, _ = self._run(["--workspace", d, "--live"])
        self.assertEqual(rc, 2)
        self.assertEqual(self.created, [])

    def test_refuse_when_scout_index_saturates_its_limit(self) -> None:
        # A scan that returns exactly the limit is ambiguous — complete, or
        # truncated? Refuse. This is the tripwire the 800-window never had: it
        # degraded silently instead, which is how #5543 happened.
        M.fetch_github = lambda *a, **k: [dict(self.REPO)]
        M.fetch_existing_issues = lambda *a, **k: []
        saturating = [{"number": i, "title": f"idea-scout: r{i}",
                       "body": f"<!-- idea-scout-source: github:acme/r{i} -->"}
                      for i in range(4)]
        cfg_dir = tempfile.TemporaryDirectory()
        self.addCleanup(cfg_dir.cleanup)
        cfg_path = Path(cfg_dir.name) / "cfg.json"
        cfg_path.write_text(json.dumps({"thresholds": {"scout_scan_limit": 4}}),
                            encoding="utf-8")
        M.fetch_scout_issues = lambda *a, **k: list(saturating)
        with tempfile.TemporaryDirectory() as d:
            rc, _ = self._run(["--workspace", d, "--live",
                               "--config", str(cfg_path)])
        self.assertEqual(rc, 2)
        self.assertEqual(self.created, [])

    def test_scout_index_query_is_label_targeted_not_windowed(self) -> None:
        # The whole point: the query is scoped to the population being deduped.
        # If this ever reverts to a bare recency listing the guarantee dies again.
        seen_args: list = []
        orig = M.gh_json
        try:
            M.gh_json = lambda args, **kw: seen_args.append(args) or []
            M.fetch_scout_issues(5000)
        finally:
            M.gh_json = orig
        argv = seen_args[0]
        self.assertIn("--label", argv)
        self.assertEqual(argv[argv.index("--label") + 1], M.SCOUT_LABEL)
        # closed issues are the ones that get re-filed, so they must be in scope
        self.assertEqual(argv[argv.index("--state") + 1], "all")

    def test_scout_scan_limit_is_a_distinct_knob_from_the_window(self) -> None:
        # Guards against the rejected "just raise issue_scan_limit" fix: the
        # durable rung must not be governed by the recency window's size.
        _, cfg = M.load_config(None)
        self.assertIn("scout_scan_limit", cfg)
        self.assertNotEqual(cfg["scout_scan_limit"], cfg["issue_scan_limit"])
        self.assertGreater(cfg["scout_scan_limit"], cfg["issue_scan_limit"])


if __name__ == "__main__":
    unittest.main()
