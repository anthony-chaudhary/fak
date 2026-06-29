#!/usr/bin/env python3
"""Hermetic tests for tools/session_audit.py.

Locks in the de-dup invariant: Claude Code writes MULTIPLE transcript lines per
billed assistant turn (streaming events / retries / sidechain re-serialization),
all carrying the SAME message.usage. The auditor must fold each billed turn ONCE
(keyed on message.id), or every token/cost/turn total runs ~2x high. A regression
here silently doubles every reported number, so this test is the witness that the
fix from 2026-06-20 (heaviest session 093ca0fc: 901->455 turns, $634->$323) holds.
"""
from __future__ import annotations

import importlib.util
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "session_audit.py"


def load():
    spec = importlib.util.spec_from_file_location("session_audit", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assistant(msg_id, *, out, cread, ccreate, inp=0, tool=None, model="claude-opus-4-8"):
    """One assistant transcript record with a given message.id and usage."""
    content = []
    if tool:
        content.append({"type": "tool_use", "name": tool, "input": {}})
    return {
        "type": "assistant",
        "timestamp": "2026-06-20T00:00:00.000Z",
        "uuid": f"uuid-{msg_id}-{out}-{cread}",   # per-LINE, intentionally unique
        "message": {
            "id": msg_id,
            "model": model,
            "usage": {
                "input_tokens": inp,
                "output_tokens": out,
                "cache_read_input_tokens": cread,
                "cache_creation_input_tokens": ccreate,
            },
            "content": content,
        },
    }


def _write_transcript(records):
    tmp = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False, encoding="utf-8")
    for r in records:
        tmp.write(json.dumps(r) + "\n")
    tmp.close()
    return tmp.name


def _write_transcript_in(root, ns, rel, records):
    path = Path(root) / ns / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return str(path)


class DedupTest(unittest.TestCase):
    def test_duplicate_billed_turn_counted_once(self) -> None:
        sa = load()
        # The same billed turn re-serialized 4x, then a distinct second turn 2x.
        recs = (
            [_assistant("msg-A", out=400, cread=50_000, ccreate=6_000)] * 4
            + [_assistant("msg-B", out=500, cread=60_000, ccreate=7_000, tool="Bash")] * 2
        )
        s = sa.analyze(_write_transcript(recs))

        self.assertEqual(s["assistant_turns"], 2, "two distinct message.ids = two turns")
        self.assertEqual(s["dup_assistant_lines"], 4, "the 6 lines hold 4 duplicates")
        self.assertEqual(s["tokens"]["output"], 900, "400 + 500, not multiplied")
        self.assertEqual(s["tokens"]["cache_read"], 110_000)
        self.assertEqual(s["tokens"]["cache_create"], 13_000)
        self.assertEqual(s["n_tool_use"], 1, "the duplicated tool_use is not re-counted")
        self.assertEqual(s["tools"].get("Bash"), 1)

    def test_no_duplicates_is_a_noop(self) -> None:
        sa = load()
        recs = [
            _assistant("msg-1", out=100, cread=10_000, ccreate=1_000),
            _assistant("msg-2", out=200, cread=20_000, ccreate=2_000),
        ]
        s = sa.analyze(_write_transcript(recs))
        self.assertEqual(s["assistant_turns"], 2)
        self.assertEqual(s["dup_assistant_lines"], 0)
        self.assertEqual(s["tokens"]["output"], 300)

    def test_idless_lines_each_count(self) -> None:
        sa = load()
        # Defensive: a record with no message.id must NOT collapse into one bucket.
        r = _assistant("x", out=50, cread=5_000, ccreate=500)
        del r["message"]["id"]
        s = sa.analyze(_write_transcript([dict(r), dict(r)]))
        self.assertEqual(s["assistant_turns"], 2, "id-less lines are counted individually")
        self.assertEqual(s["tokens"]["output"], 100)

    def test_cost_is_per_deduped_turn(self) -> None:
        sa = load()
        # Opus output @ $75/MTok: 1000 out tok = $0.075, regardless of dup lines.
        recs = [_assistant("msg-only", out=1_000, cread=0, ccreate=0)] * 3
        s = sa.analyze(_write_transcript(recs))
        self.assertAlmostEqual(s["cost_usd"], 1_000 * 75.0 / 1e6, places=9)


class WebActivityReportingTest(unittest.TestCase):
    """The machine-wide web line must surface BOTH the server-tool web requests
    (server_tool_use) AND the client WebSearch/WebFetch tool calls. Counting only
    the former printed "0 / 0" even when a session used the client WebFetch tool,
    directly contradicting the tool-mix table (which listed WebFetch). Lock the
    two-mechanism report so the contradiction can't regress."""

    def test_client_webfetch_is_not_hidden_by_zero_server_count(self) -> None:
        sa = load()
        # A session that used the CLIENT WebFetch tool with ZERO server_tool_use reqs.
        recs = [_assistant("msg-1", out=100, cread=1_000, ccreate=100, tool="WebFetch")]
        s = sa.analyze(_write_transcript(recs))
        self.assertEqual(s["tools"].get("WebFetch"), 1)
        self.assertEqual(s["tokens"]["web_fetch"], 0, "server-tool count is genuinely 0")
        self.assertEqual(s["read_only_frac"], 1.0, "WebFetch is a read-only tool")
        md = sa.report_md([s], sa.aggregate([s]))
        self.assertIn("WebFetch 1", md, "client WebFetch must be visible in the report")
        self.assertNotIn("Web search / fetch requests:** 0 / 0", md,
                         "the misleading server-only line must be gone")

    def test_server_tool_web_requests_surfaced(self) -> None:
        sa = load()
        r = _assistant("msg-1", out=100, cread=1_000, ccreate=100)
        r["message"]["usage"]["server_tool_use"] = {
            "web_search_requests": 3, "web_fetch_requests": 2}
        s = sa.analyze(_write_transcript([r]))
        self.assertEqual(s["tokens"]["web_search"], 3)
        self.assertEqual(s["tokens"]["web_fetch"], 2)
        md = sa.report_md([s], sa.aggregate([s]))
        self.assertIn("search 3 / fetch 2", md)


class ReadOnlyClassificationTest(unittest.TestCase):
    def test_observation_tools_are_read_only(self) -> None:
        sa = load()
        # Monitor/TaskGet/etc. poll or query state; they must not count as
        # side-effecting in the read-only fraction.
        for t in ("Monitor", "TaskGet", "TaskList", "TaskOutput", "ReadMcpResourceTool"):
            self.assertIn(t, sa.READ_ONLY_TOOLS)
        # …while the mutating Task tools stay OUT.
        for t in ("TaskCreate", "TaskUpdate", "TaskStop"):
            self.assertNotIn(t, sa.READ_ONLY_TOOLS)

    def test_monitor_counts_as_read_only_fraction(self) -> None:
        sa = load()
        recs = [
            _assistant("m1", out=10, cread=100, ccreate=10, tool="Monitor"),
            _assistant("m2", out=10, cread=100, ccreate=10, tool="Bash"),
        ]
        s = sa.analyze(_write_transcript(recs))
        self.assertEqual(s["read_only_frac"], 0.5, "Monitor read-only, Bash not")


class DiscoverNamespaceDefaultTest(unittest.TestCase):
    def test_default_discovers_all_non_excluded_namespaces(self) -> None:
        sa = load()
        self.assertEqual(sa.NS_INCLUDE_PREFIX, "", "default namespace filter must not be operator-specific")

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for ns in ("-Users-USER-Documents-GitHub-fleet", "C--work-fak", "AppData-Local-Temp-fixture"):
                nsdir = root / ns
                nsdir.mkdir()
                (nsdir / f"{ns}.jsonl").write_text("{}\n", encoding="utf-8")

            found = sa.discover([str(root)])
            names = {r["ns"] for r in found}
            self.assertIn("-Users-USER-Documents-GitHub-fleet", names)
            self.assertIn("C--work-fak", names)
            self.assertNotIn("AppData-Local-Temp-fixture", names)

            narrowed = sa.discover([str(root)], ns_prefix="C--work")
            self.assertEqual({r["ns"] for r in narrowed}, {"C--work-fak"})


class ReportScopeAndMixTest(unittest.TestCase):
    def test_header_names_actual_scope_and_time_window(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            p = _write_transcript_in(
                d, "C--work-fak", "session-a.jsonl",
                [_assistant("a", out=100, cread=1_000, ccreate=100)])
            s = sa.analyze(p)
            md = sa.report_md([s], sa.aggregate([s]), ns_prefix="C--work-fak",
                              since_days=None)

        self.assertIn("# Session-Transcript Audit — active scope", md)
        self.assertIn("1 namespaces folded (C--work-fak)", md)
        self.assertIn("namespace filter: C--work-fak", md)
        self.assertIn("time window: all-time", md)
        self.assertIn("## Scope totals (EXACT token counts)", md)
        self.assertNotIn("recent sessions, this machine", md)
        self.assertNotIn("Machine-wide totals", md)

    def test_default_audit_warns_when_subagents_are_excluded(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(
                d, "C--work-fak", "session-a.jsonl",
                [_assistant("top", out=100, cread=1_000, ccreate=100)])
            _write_transcript_in(
                d, "C--work-fak", "session-a/subagents/worker.jsonl",
                [_assistant("sub", out=2_000, cread=3_000, ccreate=400)])
            args = SimpleNamespace(root=[d], since_days=None, ns_prefix="",
                                   all=True, include_subagents=False,
                                   max=None, md=None, json=None)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                sa.cmd_audit(args)
            md = out.getvalue()

        self.assertIn("NOTE: +1 subagent transcripts uncounted", md)
        self.assertIn("re-run with `--include-subagents`", md)
        self.assertIn("+2,000 output tok", md)

    def test_model_mix_kpi_reports_output_and_cost_shares(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            p = _write_transcript_in(
                d, "C--work-fak", "session-a.jsonl",
                [
                    _assistant("opus", out=850, cread=0, ccreate=0, model="claude-opus-4-8"),
                    _assistant("haiku", out=150, cread=0, ccreate=0, model="claude-haiku-4-5"),
                ])
            s = sa.analyze(p)
            md = sa.report_md([s], sa.aggregate([s]))

        self.assertIn("## Model-mix KPI (tier shares)", md)
        self.assertIn("| opus | 850 | 85.0% |", md)
        self.assertIn("| haiku | 150 | 15.0% |", md)
        self.assertIn("Opus output share", md)
        self.assertIn("| C--work-fak | 1 | 1,000 | 85.0% |", md)


class BillingBucketTest(unittest.TestCase):
    """Claude and Gemini are DIFFERENT invoices. The auditor must (a) never price a
    non-Claude model at Claude rates (no silent Opus default), (b) keep non-Anthropic
    spend OUT of the Anthropic total, (c) treat <synthetic> as non-billed $0, and
    (d) render the per-bucket / per-model split so a blended number is decomposable."""

    def test_price_for_unknown_model_is_none_not_opus(self) -> None:
        sa = load()
        self.assertIsNone(sa.price_for("gemini-2.5-pro"), "no Claude card for Gemini")
        self.assertIsNone(sa.price_for("gpt-5"), "no card for OpenAI")
        self.assertIsNone(sa.price_for("qwen2.5:14b"), "no card for a local model")
        self.assertIsNone(sa.price_for("<synthetic>"), "synthetic is non-billed")
        # …but a real Claude tier still resolves to its card.
        self.assertEqual(sa.price_for("claude-opus-4-8"), sa.PRICING["opus"])
        self.assertEqual(sa.price_for("claude-haiku-4-5-20251001"), sa.PRICING["haiku"])

    def test_cost_usd_never_fabricates_for_unpriced_model(self) -> None:
        sa = load()
        # 1M output tok on Gemini would be ~$75 if mispriced as Opus — must be $0 here.
        self.assertEqual(sa.cost_usd("gemini-2.5-pro", 0, 0, 0, 1_000_000), 0.0)
        self.assertEqual(sa.cost_usd("<synthetic>", 1_000, 1_000, 1_000, 1_000), 0.0)
        # Opus is still priced exactly.
        self.assertAlmostEqual(sa.cost_usd("claude-opus-4-8", 0, 0, 0, 1_000_000), 75.0, places=6)

    def test_provider_bucket_classification(self) -> None:
        sa = load()
        self.assertEqual(sa.provider_bucket("claude-opus-4-8"), "Anthropic (Claude)")
        self.assertEqual(sa.provider_bucket("gemini-2.5-pro"), "Google (Gemini)")
        self.assertEqual(sa.provider_bucket("gpt-5"), "OpenAI")
        self.assertEqual(sa.provider_bucket("qwen2.5:14b"), "local / self-hosted")
        self.assertEqual(sa.provider_bucket("<synthetic>"), "non-billed (harness)")
        self.assertEqual(sa.provider_bucket("some-future-model"), "UNKNOWN (unpriced bucket)")

    def test_non_claude_spend_excluded_from_total_and_flagged(self) -> None:
        sa = load()
        recs = [
            _assistant("c1", out=1_000, cread=0, ccreate=0, model="claude-opus-4-8"),
            _assistant("g1", out=2_000, cread=0, ccreate=0, model="gemini-2.5-pro"),
        ]
        s = sa.analyze(_write_transcript(recs))
        agg = sa.aggregate([s])
        # The total is ONLY the Anthropic spend (1000 opus out tok @ $75/MTok).
        self.assertAlmostEqual(agg["total_cost_usd"], 1_000 * 75.0 / 1e6, places=9)
        self.assertEqual(agg["per_bucket"]["Google (Gemini)"]["output"], 2_000)
        md = sa.report_md([s], agg)
        self.assertIn("Cost by billing bucket", md)
        self.assertIn("Google (Gemini)", md)
        self.assertIn("— (no card)", md, "unpriced bucket must show no fabricated cost")
        self.assertIn("Other billing buckets present", md, "non-Claude spend must be flagged")

    def test_synthetic_turns_are_non_billed_in_per_model(self) -> None:
        sa = load()
        recs = [
            _assistant("a", out=100, cread=0, ccreate=0, model="claude-opus-4-8"),
            _assistant("syn", out=0, cread=0, ccreate=0, model="<synthetic>"),
        ]
        s = sa.analyze(_write_transcript(recs))
        self.assertIn("<synthetic>", s["per_model"])
        agg = sa.aggregate([s])
        self.assertEqual(sa.model_cost("<synthetic>", agg["per_model"]["<synthetic>"]), 0.0)


if __name__ == "__main__":
    unittest.main()
