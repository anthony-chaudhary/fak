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

import argparse
import importlib.util
import collections
import contextlib
import io
import json
import sys
import tempfile
import pathlib
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


def _assistant(msg_id, *, out, cread, ccreate, inp=0, tool=None, model="claude-opus-4-8",
               tool_id=None, tool_input=None):
    """One assistant transcript record with a given message.id and usage."""
    content = []
    if tool:
        blk = {"type": "tool_use", "name": tool, "input": tool_input or {}}
        if tool_id:
            blk["id"] = tool_id
        content.append(blk)
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


def _user_result(tool_use_id, text, is_error=True):
    """One user transcript record carrying a tool_result for a prior tool_use."""
    return {
        "type": "user",
        "timestamp": "2026-06-20T00:00:01.000Z",
        "message": {"content": [{
            "type": "tool_result", "tool_use_id": tool_use_id,
            "is_error": is_error,
            "content": [{"type": "text", "text": text}],
        }]},
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


class StreamingBlockLinesTest(unittest.TestCase):
    """Newer transcripts stream ONE content block per line, all lines sharing the
    billed turn's message.id. Usage must fold once per id (as before), but blocks
    must be deduped by BLOCK identity — skipping dup lines wholesale undercounted
    tool calls ~6x on real 2026-07 transcripts (39 counted vs 247 present)."""

    def _turn_lines(self, mid, blocks, *, out=100):
        rows = []
        for blk in blocks:
            r = _assistant(mid, out=out, cread=1_000, ccreate=100)
            r["message"]["content"] = [blk]
            rows.append(r)
        return rows

    def test_blocks_across_dup_lines_all_counted_usage_once(self) -> None:
        sa = load()
        recs = self._turn_lines("msg-S", [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "text", "text": "doing it"},
            {"type": "tool_use", "id": "tA", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_use", "id": "tB", "name": "Edit", "input": {"file_path": "x.go"}},
            # a re-serialized repeat of an already-seen block must NOT recount
            {"type": "tool_use", "id": "tA", "name": "Bash", "input": {"command": "ls"}},
        ])
        s = sa.analyze(_write_transcript(recs))
        self.assertEqual(s["assistant_turns"], 1, "one billed turn")
        self.assertEqual(s["tokens"]["output"], 100, "usage folded once")
        self.assertEqual(s["n_tool_use"], 2, "blocks deduped by id, not by line")
        self.assertEqual(s["tools"], {"Bash": 1, "Edit": 1})
        self.assertEqual(s["n_thinking"], 1)
        self.assertEqual(s["n_text"], 1)

    def test_error_attribution_works_for_late_line_tool_use(self) -> None:
        sa = load()
        recs = self._turn_lines("msg-S", [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "id": "tA", "name": "Bash",
             "input": {"command": "git fetch"}},
        ]) + [_user_result("tA", "Exit code 143 Command timed out after 2m 0s")]
        s = sa.analyze(_write_transcript(recs))
        self.assertEqual(s["behavior"]["tool_errors"], {"Bash": 1},
                         "the tool_use arriving on a dup line must still map")
        self.assertEqual(s["behavior"]["timeout_kills"], 1)

    def test_trend_scan_counts_late_line_tool_use(self) -> None:
        sa = load()
        recs = self._turn_lines("msg-S", [
            {"type": "thinking", "thinking": "hmm"},
            {"type": "tool_use", "id": "tA", "name": "Bash",
             "input": {"command": "sleep 10"}},
        ])
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(d, "C--work-fak", "sess.jsonl", recs)
            buckets, _ = sa.trend_scan([d], "", "day", False)
        B = buckets["2026-06-20"]
        self.assertEqual(B["assist_turns"], 1)
        self.assertEqual(dict(B["tools"]), {"Bash": 1})
        self.assertEqual(B["beh"]["sleep_polls"], 1)


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

    def test_top_level_only_warns_about_the_spend_it_hides(self) -> None:
        # #3226 flipped the default; the NOTE now belongs to the opt-OUT view and
        # must point at the flag that is actually hiding the spend.
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(
                d, "C--work-fak", "session-a.jsonl",
                [_assistant("top", out=100, cread=1_000, ccreate=100)])
            _write_transcript_in(
                d, "C--work-fak", "session-a/subagents/worker.jsonl",
                [_assistant("sub", out=2_000, cread=3_000, ccreate=400)])
            args = SimpleNamespace(root=[d], since_days=None, ns_prefix="",
                                   all=True, top_level_only=True,
                                   max=None, md=None, json=None)
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                sa.cmd_audit(args)
            md = out.getvalue()

        self.assertIn("NOTE: +1 subagent transcripts uncounted", md)
        self.assertIn("drop `--top-level-only` to fold them in", md)
        self.assertIn("+2,000 output tok", md)
        self.assertIn("top-level session transcripts ONLY", md)

    def test_hang_gate_exits_nonzero_over_threshold(self) -> None:
        # record→view→gate (#2365 d3): --gate-hangs turns the hang counter into a CI
        # regression gate. One transcript carries a no-TTY wedge; gate 0 must fail (exit
        # 3), gate 1 must pass. Absent the flag the gate is inert (other tests unaffected).
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(
                d, "C--work-fak", "session-h.jsonl",
                [_assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1"),
                 _user_result("t1", "INTERACTIVE_HANG: this command waits for a human "
                                    "and this session has no TTY")])
            base = dict(root=[d], since_days=None, ns_prefix="", all=True,
                        include_subagents=False, max=None, md=None, json=None)
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    sa.cmd_audit(SimpleNamespace(**base, gate_hangs=0))
                self.assertEqual(cm.exception.code, 3)
                sa.cmd_audit(SimpleNamespace(**base, gate_hangs=1))  # within budget: no raise

    def test_never_read_gate_exits_nonzero_over_threshold(self) -> None:
        # #3942: --gate-never-read turns the genuine never-read defect counter into
        # a CI regression gate. One transcript edits a file it never Read; gate 0
        # must fail (exit 3), gate 1 must pass. Absent the flag the gate is inert.
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(
                d, "C--work-fak", "session-n.jsonl",
                [_assistant("m1", out=10, cread=0, ccreate=0, tool="Edit", tool_id="e1",
                            tool_input={"file_path": "C:/x/never.go"}),
                 _user_result("e1", "File has not been read yet. Read it first.")])
            base = dict(root=[d], since_days=None, ns_prefix="", all=True,
                        include_subagents=False, max=None, md=None, json=None)
            with contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as cm:
                    sa.cmd_audit(SimpleNamespace(**base, gate_never_read=0))
                self.assertEqual(cm.exception.code, 3)
                sa.cmd_audit(SimpleNamespace(**base, gate_never_read=1))  # within budget

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


def _run_audit(sa, root, **kw):
    """Run `audit` over `root`, returning (json payload, stdout markdown)."""
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    tmp.close()
    args = SimpleNamespace(**{"root": [root], "since_days": None, "ns_prefix": "",
                              "all": True, "max": None, "md": None, "json": tmp.name,
                              "top_level_only": False, **kw})
    out = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
        sa.cmd_audit(args)
    return json.loads(Path(tmp.name).read_text(encoding="utf-8")), out.getvalue()


class SubagentFoldTest(unittest.TestCase):
    """#3226 — subagent / workflow transcripts are ~23% of billed spend and hold EVERY
    delegated turn, yet the rollup counted only top-level sessions: the headline cost
    understated reality and no behavioral detector could see a fan-out stuck in a loop.
    They are folded in by default now; `--top-level-only` preserves the old view. The
    witness is the reconciliation: default total == top-level-only total + the subagent
    delta the NOTE prints, with nothing counted twice."""

    def _fixture(self, d):
        """One top-level session + one subagent under it. The subagent is BOTH the
        heavier spender and the only transcript carrying a stuck retry loop."""
        _write_transcript_in(d, "C--work-fak", "sess-a.jsonl", [
            _assistant("top", out=100, cread=1_000, ccreate=100, tool="Read",
                       tool_id="r1", tool_input={"file_path": "C:/x/a.go"}),
        ])
        recs = [_assistant("sub-0", out=2_000, cread=3_000, ccreate=400)]
        for i in range(3):   # a verbatim retry loop, visible only inside the subagent
            recs += [_assistant(f"sub-{i+1}", out=10, cread=0, ccreate=0, tool="Bash",
                                tool_id=f"t{i}", tool_input={"command": "go build ./..."}),
                     _user_result(f"t{i}", "Exit code 1")]
        _write_transcript_in(d, "C--work-fak", "sess-a/subagents/worker.jsonl", recs)

    def test_default_total_equals_top_level_only_plus_subagent_delta(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            self._fixture(d)
            full, full_md = _run_audit(sa, d)
            only, only_md = _run_audit(sa, d, top_level_only=True)

        # The NOTE the top-level-only view prints IS the delta the default folds in.
        delta = only["excluded_subagents"]
        self.assertEqual(delta["count"], 1)
        self.assertIsNone(full["excluded_subagents"], "nothing is excluded by default")

        for k in ("output", "input", "cache_read", "cache_create"):
            self.assertEqual(
                full["aggregate"]["totals"].get(k, 0),
                only["aggregate"]["totals"].get(k, 0) + delta["tokens"].get(k, 0),
                f"{k}: default == top-level-only + the NOTE's delta, exactly")
        self.assertAlmostEqual(full["aggregate"]["total_cost_usd"],
                               only["aggregate"]["total_cost_usd"] + delta["cost_usd"],
                               places=9)
        # …and nothing is counted twice: one extra transcript, not two, and the
        # per-namespace rollup grows by exactly that one.
        self.assertEqual(full["aggregate"]["n_sessions"],
                         only["aggregate"]["n_sessions"] + 1)
        self.assertEqual(full["aggregate"]["per_namespace"]["C--work-fak"]["sessions"],
                         only["aggregate"]["per_namespace"]["C--work-fak"]["sessions"] + 1)
        self.assertIn("1 top-level + 1 subagent", full_md)
        self.assertNotIn("NOTE: +1 subagent transcripts uncounted", full_md)
        self.assertIn("NOTE: +1 subagent transcripts uncounted", only_md)

    def test_subagent_namespace_is_the_parent_namespace_not_the_subdir(self) -> None:
        # A subagent lives at <ns>/<session>/subagents/<agent>.jsonl, so
        # basename(dirname(path)) is "subagents" — folding on that would invent a
        # bogus namespace and split the rollup.
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            self._fixture(d)
            full, full_md = _run_audit(sa, d)
        self.assertEqual(list(full["aggregate"]["per_namespace"]), ["C--work-fak"])
        self.assertNotIn("subagents", full["aggregate"]["per_namespace"])
        self.assertIn("1 namespaces folded (C--work-fak)", full_md)

    def test_behavioral_lens_sees_subagent_turns_attributed_to_parent(self) -> None:
        # The retry loop exists ONLY in the subagent transcript. Default view must
        # surface it, labelled with the parent session an operator can open; the
        # top-level-only view is blind to it (that is the gap #3226 names).
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            self._fixture(d)
            full, full_md = _run_audit(sa, d)
            only, _ = _run_audit(sa, d, top_level_only=True)

        rows = full["aggregate"]["behavior"]["repeat_failure_sessions"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session"], "sess-a",
                         "a subagent finding is attributed to its PARENT session")
        self.assertEqual(rows[0]["ns"], "C--work-fak")
        self.assertEqual(rows[0]["count"], 3)
        self.assertEqual(full["aggregate"]["behavior"]["per_tool"]["Bash"]["errors"], 3)
        self.assertIn("sess-a→worker", full_md,
                      "a folded subagent row is labelled parent→agent, not as a session")

        self.assertEqual(only["aggregate"]["behavior"]["repeat_failure_sessions"], [],
                         "top-level-only stays blind to the subagent's loop")

    def test_fold_table_reconciles_and_warns_against_adding_it_back(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            self._fixture(d)
            _, md = _run_audit(sa, d)
        self.assertIn("Subagent / workflow fold — INCLUDED in every total above", md)
        self.assertIn("adding it back on would double-count", md)
        self.assertIn("| top-level sessions | 1 |", md)
        self.assertIn("| subagent / workflow | 1 |", md)
        self.assertIn("| **TOTAL (= scope totals above)** | 2 |", md)
        self.assertNotIn("True spend = top-level + this", md,
                         "the old add-them-up framing double-counts now")

    def test_max_caps_top_level_sessions_not_the_interleaved_list(self) -> None:
        # --max has always meant "top-level sessions"; folding subagents into the
        # same list must not silently redefine it as "transcripts".
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            self._fixture(d)
            _write_transcript_in(d, "C--work-fak", "sess-b.jsonl",
                                 [_assistant("b", out=50, cread=0, ccreate=0)])
            full, _ = _run_audit(sa, d, max=2)
        kinds = collections.Counter(s.get("kind") for s in full["sessions"])
        self.assertEqual(kinds["session"], 2, "both top-level sessions kept")
        self.assertEqual(kinds["subagent"], 1, "its parent survived the cap, so it rides along")


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


class BehavioralLensTest(unittest.TestCase):
    """The stuck/churn detectors from #2365 (+ file-mutation churn, the
    edit/rewrite-loop signal no other layer catches). Every detector reads only
    what the transcript carries; the token lens is untouched."""

    def _one(self, recs):
        sa = load()
        s = sa.analyze(_write_transcript(recs))
        self.assertNotIn("error", s)
        return sa, s

    def test_per_tool_errors_attributed_by_tool_use_id(self) -> None:
        sa, s = self._one([
            _assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1"),
            _user_result("t1", "Exit code 1\nboom"),
            _assistant("m2", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t2"),
            _user_result("t2", "clean output", is_error=False),
            _assistant("m3", out=10, cread=0, ccreate=0, tool="Grep", tool_id="t3"),
            _user_result("t3", "Invalid regex"),
        ])
        self.assertEqual(s["behavior"]["tool_errors"], {"Bash": 1, "Grep": 1})
        agg = sa.aggregate([s])
        pt = agg["behavior"]["per_tool"]
        self.assertEqual(pt["Bash"], {"calls": 2, "errors": 1, "error_rate": 0.5})
        self.assertEqual(pt["Grep"]["error_rate"], 1.0)

    def test_timeout_kills_shell_only(self) -> None:
        _, s = self._one([
            _assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1"),
            _user_result("t1", "Command timed out after 2m 0.0s"),
            _assistant("m2", out=10, cread=0, ccreate=0, tool="PowerShell", tool_id="t2"),
            _user_result("t2", "git fetch\nExit code: 143"),
            _assistant("m3", out=10, cread=0, ccreate=0, tool="WebFetch", tool_id="t3"),
            _user_result("t3", "Request timed out"),   # non-shell: not a harness kill
        ])
        self.assertEqual(s["behavior"]["timeout_kills"], 2)

    def test_interactive_hang_keys_on_exact_emission(self) -> None:
        # #2365 d3: the wedge is counted only on the repo-guard's EXACT emission, so
        # prose that merely mentions an editor does NOT inflate it (the loose-substring
        # form counted 68 where the real figure was 15).
        _, s = self._one([
            _assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1"),
            _user_result("t1", "INTERACTIVE_HANG: this command waits for a human and "
                               "this session has no TTY — a silent hang or EOF'd no-op."),
            _assistant("m2", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t2"),
            _user_result("t2", "I opened the editor to fix the commit message"),  # prose only
        ])
        self.assertEqual(s["behavior"]["interactive_hangs"], 1,
                         "exact emission counts; 'opened the editor' prose does not")

    def test_shell_error_classes_split_and_honest_rate(self) -> None:
        # #2365 finding 2: the raw shell error rate conflates guard refusals + no-TTY
        # hangs (neither the shell failing) with genuine command failures. The breakdown
        # sub-classifies by cause; the aggregate's genuine rate strips the two discounts.
        sa, s = self._one([
            _assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1"),
            _user_result("t1", "[fak] refused 1 tool call(s): Bash (POLICY_BLOCK/TERMINAL)"),
            _assistant("m2", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t2"),
            _user_result("t2", "INTERACTIVE_HANG: this command waits for a human and "
                               "this session has no TTY"),
            _assistant("m3", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t3"),
            _user_result("t3", "Command timed out after 2m 0.0s"),
            _assistant("m4", out=10, cread=0, ccreate=0, tool="PowerShell", tool_id="t4"),
            _user_result("t4", "The term 'gti' is not recognized as the name of a cmdlet"),
            _assistant("m5", out=10, cread=0, ccreate=0, tool="PowerShell", tool_id="t5"),
            _user_result("t5", "go test ./...\nExit code: 2\nFAIL"),
            _assistant("m6", out=10, cread=0, ccreate=0, tool="PowerShell", tool_id="t6"),
            _user_result("t6", "ok", is_error=False),   # a clean shell call
        ])
        self.assertEqual(s["behavior"]["shell_error_classes"], {
            "policy_refusal": 1, "interactive_hang": 1, "timeout_kill": 1,
            "not_found": 1, "nonzero_exit": 1})
        se = sa.aggregate([s])["behavior"]["shell_errors"]
        self.assertEqual(se["shell_calls"], 6)
        self.assertEqual(se["raw_errors"], 5)
        # genuine drops the guard refusal + the now-fixed hang: 5 - 2 = 3 over 6 calls.
        self.assertEqual(se["genuine_errors"], 3)
        self.assertEqual(se["genuine_rate"], 0.5)

    def test_sleep_poll_prefix_foreground_only(self) -> None:
        _, s = self._one([
            _assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1",
                       tool_input={"command": "sleep 30 && curl localhost:8080"}),
            _assistant("m2", out=10, cread=0, ccreate=0, tool="PowerShell", tool_id="t2",
                       tool_input={"command": "Start-Sleep -Seconds 5; Get-Item x"}),
            _assistant("m3", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t3",
                       tool_input={"command": "echo sleep"}),
            _assistant("m4", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t4",
                       tool_input={"command": "sleep 60", "run_in_background": True}),
        ])
        self.assertEqual(s["behavior"]["sleep_polls"], 2,
                         "prefix-anchored, foreground shell calls only")

    def test_edit_write_churn_signatures(self) -> None:
        sa, s = self._one([
            _assistant("m1", out=10, cread=0, ccreate=0, tool="Edit", tool_id="t1"),
            _user_result("t1", "File has not been read yet. Read it first before writing to it."),
            _assistant("m2", out=10, cread=0, ccreate=0, tool="Edit", tool_id="t2"),
            _user_result("t2", "File has not been read yet."),
            _assistant("m3", out=10, cread=0, ccreate=0, tool="Write", tool_id="t3"),
            _user_result("t3", "File has been modified since read, either by the user or by a linter."),
        ])
        self.assertEqual(s["behavior"]["edit_churn"], {"not_read": 2, "stale_read": 1})
        agg = sa.aggregate([s])
        self.assertEqual(agg["behavior"]["wasted_mutation_calls"], 3)

    def test_repeat_failure_signatures(self) -> None:
        recs = []
        for i in range(3):
            recs += [_assistant(f"m{i}", out=10, cread=0, ccreate=0,
                                tool="Bash", tool_id=f"t{i}"),
                     _user_result(f"t{i}", "Exit code 1")]
        recs += [_assistant("mx", out=10, cread=0, ccreate=0, tool="Bash", tool_id="tx"),
                 _user_result("tx", "Exit code 2")]
        _, s = self._one(recs)
        b = s["behavior"]
        self.assertEqual(b["max_repeat_failure"], 3)
        self.assertEqual(b["repeat_failures"],
                         [{"tool": "Bash", "sig": "Exit code 1", "count": 3}])

    def test_two_identical_failures_stay_below_threshold(self) -> None:
        _, s = self._one([
            _assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1"),
            _user_result("t1", "Exit code 1"),
            _assistant("m2", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t2"),
            _user_result("t2", "Exit code 1"),
        ])
        self.assertEqual(s["behavior"]["repeat_failures"], [])
        self.assertEqual(s["behavior"]["max_repeat_failure"], 2)

    def test_file_churn_same_file_mutations(self) -> None:
        recs = [_assistant(f"m{i}", out=10, cread=0, ccreate=0, tool="Edit",
                           tool_id=f"t{i}", tool_input={"file_path": "C:/x/hot.go"})
                for i in range(5)]
        recs.append(_assistant("m9", out=10, cread=0, ccreate=0, tool="Edit",
                               tool_id="t9", tool_input={"file_path": "C:/x/cold.go"}))
        _, s = self._one(recs)
        b = s["behavior"]
        self.assertEqual(b["max_file_churn"], 5)
        self.assertEqual(b["file_churn"], [{
            "file": "C:/x/hot.go",
            "count": 5,
            "distinct_regions": 1,
            "reverts": 0,
        }])

    def test_file_churn_distinct_region_buildout_not_flagged(self) -> None:
        # 6 edits, 6 DISTINCT regions, zero reverts — helper-extraction /
        # verb-per-triple build-out must NOT read as a rewrite loop (the exact
        # false alarm the 2026-07-02 forensic pass refuted on 5 real sessions).
        recs = [_assistant(f"m{i}", out=10, cread=0, ccreate=0, tool="Edit",
                           tool_id=f"t{i}",
                           tool_input={"file_path": "C:/x/grow.go",
                                       "old_string": f"region-{i}-before",
                                       "new_string": f"region-{i}-after"})
                for i in range(6)]
        _, s = self._one(recs)
        b = s["behavior"]
        self.assertEqual(b["max_file_churn"], 6, "raw count still reported")
        self.assertEqual(b["file_churn"], [], "distinct-region build-out is not churn")

    def test_file_churn_revert_amid_region_reuse_flagged(self) -> None:
        # #3943: a revert only marks a rewrite loop when regions are ALSO being
        # reused (distinct < count). Here A is edited twice (region reuse) and a
        # B->A edit restores edit 0's pre-state: distinct=4 < n=5, reverts=1 ->
        # genuine thrash, stays flagged. (This replaces the old
        # test_file_churn_revert_pair_flagged, which asserted a LONE revert amid
        # all-distinct regions was a loop — the b72e2808 false positive #3943
        # refutes; see test_file_churn_lone_revert_all_distinct_not_flagged.)
        edits = [("A", "B"), ("C", "D"), ("A", "E"), ("B", "A"), ("G", "H")]
        recs = [_assistant(f"m{i}", out=10, cread=0, ccreate=0, tool="Edit",
                           tool_id=f"t{i}",
                           tool_input={"file_path": "C:/x/flip.go",
                                       "old_string": old, "new_string": new})
                for i, (old, new) in enumerate(edits)]
        _, s = self._one(recs)
        rows = s["behavior"]["file_churn"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["distinct_regions"], 4, "A reused -> 4 distinct")
        self.assertEqual(rows[0]["reverts"], 1, "B->A restores edit 0's pre-state")

    def test_file_churn_lone_revert_all_distinct_not_flagged(self) -> None:
        # #3943 anchor (b72e2808 shape): count>=5, distinct == count, reverts==1
        # — a long linear refactor across DISTINCT regions that happens to
        # restore one earlier snippet once. NOT a rewrite loop; must NOT flag.
        edits = [("A", "x"), ("B", "y"), ("C", "z"), ("D", "A"), ("E", "w")]
        recs = [_assistant(f"m{i}", out=10, cread=0, ccreate=0, tool="Edit",
                           tool_id=f"t{i}",
                           tool_input={"file_path": "C:/x/author.go",
                                       "old_string": old, "new_string": new})
                for i, (old, new) in enumerate(edits)]
        _, s = self._one(recs)
        b = s["behavior"]
        self.assertEqual(b["max_file_churn"], 5, "raw count still reported")
        self.assertEqual(b["file_churn"], [],
                         "lone revert amid all-distinct regions is not a loop")

    def test_file_churn_region_reuse_multi_revert_flagged(self) -> None:
        # #3943 anchor (5c72b8ba shape): count=5, distinct=4, reverts=2 — repeated
        # same-region reverts are genuine thrash and must stay flagged.
        edits = [("A", "B"), ("C", "D"), ("B", "A"), ("D", "C"), ("A", "E")]
        recs = [_assistant(f"m{i}", out=10, cread=0, ccreate=0, tool="Edit",
                           tool_id=f"t{i}",
                           tool_input={"file_path": "C:/x/thrash.go",
                                       "old_string": old, "new_string": new})
                for i, (old, new) in enumerate(edits)]
        _, s = self._one(recs)
        rows = s["behavior"]["file_churn"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["distinct_regions"], 4)
        self.assertEqual(rows[0]["reverts"], 2, "B->A and D->C are two reverts")

    def test_verbatim_vs_failure_mass_split(self) -> None:
        # 3 DIFFERENT commands sharing one error text: a failure CLASS, not a
        # verbatim retry loop (the bea7d4b9/77ec8b67 false-alarm shape).
        recs = []
        for i in range(3):
            recs += [_assistant(f"m{i}", out=10, cread=0, ccreate=0, tool="Bash",
                                tool_id=f"t{i}",
                                tool_input={"command": f"git cmd-{i}"}),
                     _user_result(f"t{i}", "Exit code 143 Command timed out after 2m 0s")]
        _, s = self._one(recs)
        b = s["behavior"]
        self.assertEqual(b["repeat_failures"], [], "args differ -> not verbatim")
        self.assertEqual(b["max_repeat_failure"], 1)
        self.assertEqual(b["failure_mass"],
                         [{"tool": "Bash",
                           "sig": "Exit code 143 Command timed out after 2m 0s",
                           "count": 3}])

    def test_stall_gap_detection(self) -> None:
        r1 = _assistant("m1", out=10, cread=0, ccreate=0)
        r2 = _assistant("m2", out=10, cread=0, ccreate=0)
        r2["timestamp"] = "2026-06-20T00:10:00.000Z"   # 10 min after r1
        _, s = self._one([r1, r2])
        self.assertEqual(s["behavior"]["stall_gaps"], 1)
        self.assertEqual(s["behavior"]["max_gap_s"], 600.0)

    def test_duplicated_assistant_lines_do_not_double_detectors(self) -> None:
        # The same billed turn re-serialized 3x: one sleep-poll, one file write.
        recs = [_assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1",
                           tool_input={"command": "sleep 9"})] * 3
        _, s = self._one(recs)
        self.assertEqual(s["behavior"]["sleep_polls"], 1)

    def test_report_md_behavioral_section(self) -> None:
        recs = []
        for i in range(3):
            recs += [_assistant(f"m{i}", out=10, cread=0, ccreate=0,
                                tool="Bash", tool_id=f"t{i}",
                                tool_input={"command": "sleep 5"}),
                     _user_result(f"t{i}", "Command timed out after 2m 0.0s")]
        sa, s = self._one(recs)
        md = sa.report_md([s], sa.aggregate([s]))
        self.assertIn("## Behavioral lens — stuck/churn detectors", md)
        self.assertIn("| Bash | 3 | 3 | 100.0% |", md)
        self.assertIn("Timeout kills", md)
        self.assertIn("**Foreground sleep-polls (`sleep`/`Start-Sleep` command prefix):** 3", md)
        self.assertIn("VERBATIM retry loop", md)
        self.assertIn("recurring failure CLASS", md)
        self.assertIn("zero-record stall", md)
        self.assertIn("Command timed out after 2m 0.0s", md)

    def test_aggregate_tolerates_pre_lens_sessions(self) -> None:
        sa, s = self._one([_assistant("m1", out=10, cread=0, ccreate=0, tool="Bash")])
        del s["behavior"]   # a session replayed from pre-lens JSON
        agg = sa.aggregate([s])
        self.assertEqual(agg["behavior"]["timeout_kills"], 0)
        self.assertEqual(agg["behavior"]["per_tool"]["Bash"]["errors"], 0)

    def test_trend_scan_folds_behavior_per_bucket(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(d, "C--work-fak", "sess.jsonl", [
                _assistant("m1", out=10, cread=0, ccreate=0, tool="Bash", tool_id="t1",
                           tool_input={"command": "sleep 30"}),
                _user_result("t1", "Command timed out after 2m 0.0s"),
            ])
            buckets, n = sa.trend_scan([d], "", "day", False)
        self.assertEqual(n, 1)
        B = buckets["2026-06-20"]
        self.assertEqual(dict(B["tool_errors"]), {"Bash": 1})
        self.assertEqual(B["beh"]["timeout_kills"], 1)
        self.assertEqual(B["beh"]["sleep_polls"], 1)


def _restart(marker="Resume"):
    """A session-restart / compaction marker record (#2375 d1: post_resume signal)."""
    return {"type": "last-prompt", "lastPrompt": marker,
            "timestamp": "2026-06-20T00:05:00.000Z"}


class NotReadSubclassTest(unittest.TestCase):
    """#2375 detector 1 — the not_read edit-churn counter conflates three
    mechanically distinct causes; only true_never_read is agent misbehavior.
    Sub-classify from signals the transcript already carries."""

    def _one(self, recs):
        sa = load()
        s = sa.analyze(_write_transcript(recs))
        self.assertNotIn("error", s)
        return sa, s

    def test_post_resume_prior_read_then_restart(self) -> None:
        # a.go read (5 in real life), a restart marker, then an edit that fails
        # not-read: the harness read-state reset, not misbehavior.
        _, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Read", tool_id="r1",
                       tool_input={"file_path": "C:/x/a.go"}),
            _user_result("r1", "the file body", is_error=False),
            _restart(),
            _assistant("m2", out=1, cread=0, ccreate=0, tool="Edit", tool_id="e1",
                       tool_input={"file_path": "C:/x/a.go"}),
            _user_result("e1", "File has not been read yet. Read it first."),
        ])
        self.assertEqual(s["behavior"]["edit_churn"], {"not_read": 1})
        self.assertEqual(s["behavior"]["not_read_classes"], {"post_resume": 1})

    def test_self_duplicate_prior_successful_write(self) -> None:
        # a.go written once (ok), a forked branch re-issues a stale Write that the
        # guard fence refuses with not-read: a duplicate, not misbehavior.
        _, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Write", tool_id="w1",
                       tool_input={"file_path": "C:/x/a.go", "content": "v1"}),
            _user_result("w1", "wrote file", is_error=False),
            _assistant("m2", out=1, cread=0, ccreate=0, tool="Write", tool_id="w2",
                       tool_input={"file_path": "C:/x/a.go", "content": "stale draft"}),
            _user_result("w2", "File has not been read yet."),
        ])
        self.assertEqual(s["behavior"]["not_read_classes"], {"self_duplicate": 1})

    def test_true_never_read_is_the_real_defect(self) -> None:
        # No prior read, no prior write, no restart: the genuine never-read edit.
        _, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Edit", tool_id="e1",
                       tool_input={"file_path": "C:/x/never.go"}),
            _user_result("e1", "File has not been read yet."),
        ])
        self.assertEqual(s["behavior"]["not_read_classes"], {"true_never_read": 1})

    def test_self_duplicate_beats_post_resume(self) -> None:
        # Both signals present (read + restart AND a prior write): the concrete
        # prior write wins — precedence self_duplicate > post_resume.
        _, s = self._one([
            _assistant("m0", out=1, cread=0, ccreate=0, tool="Read", tool_id="r0",
                       tool_input={"file_path": "C:/x/a.go"}),
            _user_result("r0", "body", is_error=False),
            _restart(),
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Write", tool_id="w1",
                       tool_input={"file_path": "C:/x/a.go", "content": "v1"}),
            _user_result("w1", "wrote", is_error=False),
            _assistant("m2", out=1, cread=0, ccreate=0, tool="Write", tool_id="w2",
                       tool_input={"file_path": "C:/x/a.go", "content": "dup"}),
            _user_result("w2", "File has not been read yet."),
        ])
        self.assertEqual(s["behavior"]["not_read_classes"], {"self_duplicate": 1})

    def test_no_restart_prior_read_stays_true_never_read(self) -> None:
        # Read happened but NO restart marker: the read-state is intact, so a
        # not-read failure is genuine (post_resume needs the restart signal).
        _, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Read", tool_id="r1",
                       tool_input={"file_path": "C:/x/a.go"}),
            _user_result("r1", "body", is_error=False),
            _assistant("m2", out=1, cread=0, ccreate=0, tool="Edit", tool_id="e1",
                       tool_input={"file_path": "C:/x/a.go"}),
            _user_result("e1", "File has not been read yet."),
        ])
        self.assertEqual(s["behavior"]["not_read_classes"], {"true_never_read": 1})

    def test_subclass_counts_reconcile_with_not_read_total(self) -> None:
        sa, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Write", tool_id="w1",
                       tool_input={"file_path": "C:/x/a.go", "content": "v"}),
            _user_result("w1", "ok", is_error=False),
            _assistant("m2", out=1, cread=0, ccreate=0, tool="Write", tool_id="w2",
                       tool_input={"file_path": "C:/x/a.go", "content": "dup"}),
            _user_result("w2", "File has not been read yet."),
            _assistant("m3", out=1, cread=0, ccreate=0, tool="Edit", tool_id="e3",
                       tool_input={"file_path": "C:/x/b.go"}),
            _user_result("e3", "File has not been read yet."),
        ])
        nrc = s["behavior"]["not_read_classes"]
        self.assertEqual(sum(nrc.values()), s["behavior"]["edit_churn"]["not_read"])
        self.assertEqual(nrc, {"self_duplicate": 1, "true_never_read": 1})
        agg = sa.aggregate([s])
        self.assertEqual(agg["behavior"]["not_read_classes"],
                         {"self_duplicate": 1, "true_never_read": 1})

    def test_true_never_read_path_surfaced_in_behavior(self) -> None:
        # #3942 — the genuine never-read edit records WHICH file was hit, so the
        # count becomes an actionable offender (only true_never_read; the other
        # sub-classes are not misbehavior and carry no offender list).
        _, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Edit", tool_id="e1",
                       tool_input={"file_path": "C:/x/never.go"}),
            _user_result("e1", "File has not been read yet."),
        ])
        b = s["behavior"]
        self.assertEqual(b["not_read_classes"], {"true_never_read": 1})
        self.assertEqual(b["true_never_read_paths"],
                         [{"path": "C:/x/never.go", "count": 1}])

    def test_post_resume_and_self_dup_leave_offenders_empty(self) -> None:
        # #3942 — only true_never_read populates the offender list; a post_resume
        # reset must not be named as a never-read defect.
        _, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Read", tool_id="r1",
                       tool_input={"file_path": "C:/x/a.go"}),
            _user_result("r1", "body", is_error=False),
            _restart(),
            _assistant("m2", out=1, cread=0, ccreate=0, tool="Edit", tool_id="e1",
                       tool_input={"file_path": "C:/x/a.go"}),
            _user_result("e1", "File has not been read yet."),
        ])
        self.assertEqual(s["behavior"]["not_read_classes"], {"post_resume": 1})
        self.assertEqual(s["behavior"]["true_never_read_paths"], [])

    def test_never_read_offender_surfaced_in_aggregate_and_report(self) -> None:
        # #3942 — aggregate rolls each session's genuine never-read edits into a
        # per-session offender row, and the report renders a named table so a
        # reader can jump straight to the session + file.
        sa, s = self._one([
            _assistant("m1", out=1, cread=0, ccreate=0, tool="Edit", tool_id="e1",
                       tool_input={"file_path": "C:/x/never.go"}),
            _user_result("e1", "File has not been read yet."),
        ])
        agg = sa.aggregate([s])
        rows = agg["behavior"]["never_read_sessions"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["count"], 1)
        self.assertEqual(rows[0]["paths"], ["C:/x/never.go"])
        md = sa.report_md([s], agg)
        self.assertIn("Never-read file(s)", md)
        self.assertIn("C:/x/never.go", md)

    def test_restart_record_variants(self) -> None:
        sa = load()
        self.assertTrue(sa._is_restart_record({"type": "last-prompt", "lastPrompt": "Resume"}))
        self.assertTrue(sa._is_restart_record({"type": "summary"}))
        self.assertTrue(sa._is_restart_record({"isCompactSummary": True}))
        self.assertTrue(sa._is_restart_record(
            {"type": "user", "message": {"content": "<command-name>/resume</command-name>"}}))
        # a prompt that merely MENTIONS resume must not read as a restart.
        self.assertFalse(sa._is_restart_record(
            {"type": "user", "message": {"content": "resume the sweep for me"}}))
        self.assertFalse(sa._is_restart_record({"type": "assistant"}))


class SuccessLoopTest(unittest.TestCase):
    """#2375 detector 2 — loops of SUCCESSFUL identical calls (read-loops /
    glob-storms / output-file poll loops) the failure/mutation loop checks
    never see. Count identical (tool, args-digest) calls, subtract errored ones."""

    def _one(self, recs):
        sa = load()
        s = sa.analyze(_write_transcript(recs))
        self.assertNotIn("error", s)
        return sa, s

    def _reads(self, n, path="C:/x/poll.output", start=0):
        return [_assistant(f"m{start+i}", out=1, cread=0, ccreate=0, tool="Read",
                           tool_id=f"r{start+i}", tool_input={"file_path": path})
                for i in range(n)]

    def test_read_loop_at_threshold_flagged(self) -> None:
        _, s = self._one(self._reads(8))
        b = s["behavior"]
        self.assertEqual(b["max_success_loop"], 8)
        self.assertEqual(b["success_loops"],
                         [{"tool": "Read", "target": "C:/x/poll.output", "count": 8}])

    def test_below_threshold_not_flagged(self) -> None:
        _, s = self._one(self._reads(7))
        self.assertEqual(s["behavior"]["success_loops"], [])
        self.assertEqual(s["behavior"]["max_success_loop"], 7)

    def test_errored_calls_subtracted(self) -> None:
        # 10 identical Reads, one of which ERRORED -> 9 successful, still a loop;
        # but if 3 error, 7 successful drops below threshold.
        recs = self._reads(10)
        recs += [_user_result("r0", "Permission denied")]   # r0 errored
        _, s = self._one(recs)
        self.assertEqual(s["behavior"]["success_loops"],
                         [{"tool": "Read", "target": "C:/x/poll.output", "count": 9}])

        recs2 = self._reads(10)
        recs2 += [_user_result("r0", "boom"), _user_result("r1", "boom"),
                  _user_result("r2", "boom")]
        _, s2 = self._one(recs2)
        self.assertEqual(s2["behavior"]["success_loops"], [],
                         "7 successful is below the loop threshold")

    def test_distinct_args_do_not_group(self) -> None:
        # Different paths -> different digests -> not one loop.
        recs = self._reads(4, path="C:/x/a.output") + self._reads(4, path="C:/x/b.output", start=4)
        _, s = self._one(recs)
        self.assertEqual(s["behavior"]["success_loops"], [])
        self.assertEqual(s["behavior"]["max_success_loop"], 4)

    def test_sanctioned_poll_tools_excluded(self) -> None:
        sa = load()
        # Monitor / TaskOutput are the sanctioned poll surface — never loop-flagged.
        self.assertNotIn("Monitor", sa.SUCCESS_LOOP_TOOLS)
        self.assertNotIn("TaskOutput", sa.SUCCESS_LOOP_TOOLS)
        for t in ("Read", "Glob", "Grep", "LS", "Bash", "PowerShell"):
            self.assertIn(t, sa.SUCCESS_LOOP_TOOLS)
        recs = [_assistant(f"m{i}", out=1, cread=0, ccreate=0, tool="Monitor",
                           tool_id=f"t{i}", tool_input={"selector": "x"})
                for i in range(12)]
        s = sa.analyze(_write_transcript(recs))
        self.assertEqual(s["behavior"]["success_loops"], [])

    def test_glob_storm_flagged_by_pattern(self) -> None:
        recs = [_assistant(f"m{i}", out=1, cread=0, ccreate=0, tool="Glob",
                           tool_id=f"g{i}", tool_input={"pattern": "**/*.go"})
                for i in range(9)]
        _, s = self._one(recs)
        self.assertEqual(s["behavior"]["success_loops"],
                         [{"tool": "Glob", "target": "**/*.go", "count": 9}])

    def test_aggregate_and_report_surface_both_detectors(self) -> None:
        sa = load()
        recs = [
            # a self_duplicate not-read pair
            _assistant("w1", out=1, cread=0, ccreate=0, tool="Write", tool_id="w1",
                       tool_input={"file_path": "C:/x/a.go", "content": "v"}),
            _user_result("w1", "ok", is_error=False),
            _assistant("w2", out=1, cread=0, ccreate=0, tool="Write", tool_id="w2",
                       tool_input={"file_path": "C:/x/a.go", "content": "dup"}),
            _user_result("w2", "File has not been read yet."),
        ] + self._reads(8)
        s = sa.analyze(_write_transcript(recs))
        agg = sa.aggregate([s])
        self.assertEqual(agg["behavior"]["not_read_classes"], {"self_duplicate": 1})
        self.assertEqual(len(agg["behavior"]["success_loop_sessions"]), 1)
        md = sa.report_md([s], agg)
        self.assertIn("not-read sub-classes", md)
        self.assertIn("self-duplicate 1", md)
        self.assertIn("SUCCESSFUL-call loop", md)
        self.assertIn("C:/x/poll.output", md)

    def test_trend_scan_folds_success_loops(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(d, "C--work-fak", "sess.jsonl", self._reads(8))
            buckets, _ = sa.trend_scan([d], "", "day", False)
        self.assertEqual(buckets["2026-06-20"]["beh"]["success_loop_files"], 1)


class DeepErrorShapeTest(unittest.TestCase):
    """#3070: `deep` on an unreadable transcript must fail honestly, not KeyError.

    analyze() returns the {"path","error"} shape (not the success shape) when it
    cannot open/parse the file — a missing path, a wrong --root, a transcript under
    a non-default $CLAUDE_CONFIG_DIR. cmd_deep must detect that and exit non-zero
    with a one-line message, instead of dereferencing the absent success keys and
    handing the operator a KeyError traceback.
    """

    def test_deep_missing_path_exits_cleanly(self) -> None:
        sa = load()
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            sa.cmd_deep(SimpleNamespace(session="/no/such/path/does-not-exist.jsonl"))
        self.assertEqual(cm.exception.code, 2)          # non-zero, not a traceback
        self.assertIn("cannot read transcript", err.getvalue())
        self.assertNotIn("Traceback", err.getvalue())

    def test_deep_valid_transcript_unchanged(self) -> None:
        # A readable transcript still reaches the trajectory header — the guard
        # must not swallow the success path (no false positive on "session" absent).
        sa = load()
        path = _write_transcript([_assistant("m1", out=10, cread=5, ccreate=0)])
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            sa.cmd_deep(SimpleNamespace(session=path))
        self.assertIn("# Trajectory:", out.getvalue())


class CacheBurstTest(unittest.TestCase):
    """#3069 — the cache-CREATE burst lens. Per billed turn the provider reports
    the size of the cached prefix it reused (cache_read); it climbs as context
    grows, then SNAPS back to a floor when a cached suffix is invalidated
    mid-session — a cache_create burst the read-share lens is blind to (a burst
    INFLATES read-share next turn). This locks in: per-session cc_share (sharing
    cache_hit_frac's denominator), the suffix-reset detector + modal floor, and the
    high-burst long-session offender table."""

    def _one(self, recs):
        sa = load()
        s = sa.analyze(_write_transcript(recs))
        self.assertNotIn("error", s)
        return sa, s

    def _cread_turns(self, creads, *, ccreate=1_000, start=0, prefix="m"):
        """One billed turn per cache_read value, each a distinct message.id."""
        return [_assistant(f"{prefix}{start+i}", out=10, cread=cr, ccreate=ccreate)
                for i, cr in enumerate(creads)]

    def test_cc_share_shares_cache_hit_denominator(self) -> None:
        # cc_share = cache_create / (cache_read + cache_create + input) — the SAME
        # denominator as cache_hit_frac, so the two shares + the input share sum to 1.
        _, s = self._one([_assistant("m1", out=100, cread=50_000, ccreate=10_000, inp=1_000)])
        denom = 50_000 + 10_000 + 1_000
        self.assertAlmostEqual(s["cc_share"], 10_000 / denom, places=9)
        self.assertAlmostEqual(s["cache_hit_frac"], 50_000 / denom, places=9)
        self.assertAlmostEqual(s["cache_hit_frac"] + s["cc_share"] + 1_000 / denom,
                               1.0, places=9)

    def test_suffix_reset_detects_snapback_records_floor(self) -> None:
        # cread climbs then snaps back below (prev - 20k) twice, to a fixed floor.
        _, s = self._one(self._cread_turns(
            [0, 56_000, 88_000, 56_000, 95_000, 56_000]))
        b = s["behavior"]
        self.assertEqual(b["suffix_resets"], 2, "two snap-backs > 20k")
        self.assertEqual(b["suffix_reset_floor"], 56_000, "modal floor snapped-to")

    def test_first_turn_and_growth_never_trigger(self) -> None:
        # Monotonic growth (context building up) is not a reset; the first turn,
        # with no predecessor, can never trigger either.
        _, s = self._one(self._cread_turns([0, 40_000, 70_000, 90_000, 110_000]))
        self.assertEqual(s["behavior"]["suffix_resets"], 0)
        self.assertIsNone(s["behavior"]["suffix_reset_floor"])

    def test_small_drop_below_threshold_not_a_reset(self) -> None:
        # A 2k dip (shorter user turn) is noise, not a suffix invalidation.
        _, s = self._one(self._cread_turns([0, 90_000, 88_000, 92_000]))
        self.assertEqual(s["behavior"]["suffix_resets"], 0)

    def test_reset_deduped_by_message_id(self) -> None:
        # The snap-back turn re-serialized 3x must fold ONCE (fed only on a new
        # billed turn), so the reset is counted a single time.
        recs = self._cread_turns([0, 56_000, 90_000])
        recs += [_assistant("mreset", out=10, cread=56_000, ccreate=1_000)] * 3
        _, s = self._one(recs)
        self.assertEqual(s["behavior"]["suffix_resets"], 1, "dup lines don't recount")
        self.assertEqual(s["behavior"]["suffix_reset_floor"], 56_000)

    def test_burst_offender_table_ranks_long_sessions_by_cache_create(self) -> None:
        sa = load()
        # A: 8 turns, 3 resets to 56k, heavy cache_create. B: 8 turns, 1 reset to
        # 57k, lighter cache_create. C: 8 turns, monotonic climb — a CLEAN long
        # session that must NOT be flagged (the read-share false-positive #3069 kills).
        recs_a = self._cread_turns(
            [0, 56_000, 90_000, 56_000, 95_000, 56_000, 100_000, 56_000],
            ccreate=5_000, prefix="a")
        recs_b = self._cread_turns(
            [0, 57_000, 88_000, 57_000, 70_000, 85_000, 95_000, 100_000],
            ccreate=2_000, prefix="b")
        recs_c = self._cread_turns(
            [0, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000],
            ccreate=100, prefix="c")
        sA = sa.analyze(_write_transcript(recs_a))
        sB = sa.analyze(_write_transcript(recs_b))
        sC = sa.analyze(_write_transcript(recs_c))
        self.assertEqual(sA["behavior"]["suffix_resets"], 3)
        self.assertEqual(sB["behavior"]["suffix_resets"], 1)
        self.assertEqual(sC["behavior"]["suffix_resets"], 0)

        agg = sa.aggregate([sC, sB, sA])   # unsorted input
        self.assertEqual(agg["behavior"]["suffix_resets"], 4, "3 + 1 machine-wide")
        self.assertEqual(agg["behavior"]["suffix_reset_floor"], 56_000,
                         "56k appears 3x vs 57k 1x -> modal")
        rows = agg["behavior"]["burst_sessions"]
        self.assertEqual([r["session"] for r in rows], [sA["session"], sB["session"]],
                         "long+bursting only, ranked by cache_create desc; clean C absent")
        self.assertGreater(rows[0]["cache_create"], rows[1]["cache_create"])
        self.assertEqual(rows[0]["reset_floor"], 56_000)
        self.assertEqual(rows[0]["suffix_resets"], 3)

    def test_long_clean_session_is_not_a_burst_offender(self) -> None:
        sa = load()
        s = sa.analyze(_write_transcript(self._cread_turns(
            [0, 50_000, 60_000, 70_000, 80_000, 90_000, 100_000, 110_000], ccreate=100)))
        self.assertEqual(sa.aggregate([s])["behavior"]["burst_sessions"], [])

    def test_short_bursting_session_below_long_threshold_not_flagged(self) -> None:
        sa = load()
        # 4 turns with a reset: it bursts, but is not a LONG session, so it stays
        # out of the offender table (turns < BURST_LONG_SESSION_MIN).
        s = sa.analyze(_write_transcript(self._cread_turns([0, 56_000, 90_000, 56_000])))
        self.assertEqual(s["behavior"]["suffix_resets"], 1)
        self.assertLess(s["assistant_turns"], sa.BURST_LONG_SESSION_MIN)
        self.assertEqual(sa.aggregate([s])["behavior"]["burst_sessions"], [])

    def test_report_and_json_surface_burst_and_cc_share(self) -> None:
        sa = load()
        recs = self._cread_turns(
            [0, 56_000, 90_000, 56_000, 95_000, 56_000, 100_000, 56_000],
            ccreate=5_000)
        s = sa.analyze(_write_transcript(recs))
        agg = sa.aggregate([s])
        md = sa.report_md([s], agg)
        # machine-wide burst line in the scope totals
        self.assertIn("Cache-CREATE burst share of all ingested context", md)
        self.assertIn("create:read ratio", md)
        # behavioral-lens suffix-reset row + modal floor
        self.assertIn("Suffix-cache invalidations", md)
        self.assertIn("modal reset floor", md)
        self.assertIn("56,000", md)
        # the high-burst offender table
        self.assertIn("High-burst long sessions", md)
        self.assertIn("Cache-create tok", md)
        # per-session cc-share column in the heaviest table
        self.assertIn("| Session | NS | Turns | Tool calls | Output tok | I:O | "
                      "Cache-hit | cc-share | Est.$ |", md)
        # cc_share rides the session dict -> JSON
        self.assertIn("cc_share", s)
        self.assertIsNotNone(s["cc_share"])

    def test_trend_scan_folds_suffix_resets(self) -> None:
        sa = load()
        with tempfile.TemporaryDirectory() as d:
            _write_transcript_in(d, "C--work-fak", "sess.jsonl",
                                 self._cread_turns([0, 56_000, 90_000, 56_000]))
            buckets, _ = sa.trend_scan([d], "", "day", False)
        self.assertEqual(buckets["2026-06-20"]["beh"]["suffix_resets"], 1)



class HookLensTest(unittest.TestCase):
    def test_attachment_outcomes_failures_and_latency_are_aggregated(self):
        sa = load()
        rows = [
            {"type": "attachment", "timestamp": "2026-08-16T00:00:00Z",
             "attachment": {"type": "hook_success", "hookEvent": "PostToolUse",
                            "durationMs": 12, "stdout": "", "stderr": ""}},
            {"type": "attachment", "timestamp": "2026-08-16T00:00:01Z",
             "attachment": {"type": "hook_non_blocking_error", "hookEvent": "PostToolUse",
                            "durationMs": 80, "stderr": "helper exited 7"}},
            {"type": "attachment", "timestamp": "2026-08-16T00:00:02Z",
             "attachment": {"type": "hook_cancelled", "hookEvent": "UserPromptSubmit",
                            "durationMs": 30000, "stderr": ""}},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = pathlib.Path(td) / "hooks.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            session = sa.analyze(str(path))
        agg = sa.aggregate([session])
        self.assertEqual(agg["hooks"]["failure_total"], 2)
        self.assertEqual(agg["hooks"]["events"]["PostToolUse"]["outcomes"],
                         {"success": 1, "non_blocking_error": 1})
        self.assertEqual(agg["hooks"]["events"]["PostToolUse"]["duration_ms"]["p90"], 80)
        failure = agg["hooks"]["failures"][0]
        self.assertEqual((failure["event"], failure["outcome"], failure["count"], failure["sessions"]),
                         ("PostToolUse", "non_blocking_error", 1, 1))
        self.assertIn("helper exited 7", failure["signature"])
        report = "\n".join(sa._hook_lens(agg))
        self.assertIn("Hook execution lens", report)
        self.assertIn("| PostToolUse | 2 | 1 | 0 | 1 | 0 | 80 ms | 80 ms |", report)
        self.assertIn("Hook failures/cancellations:** 2", report)


class HookStreamTest(unittest.TestCase):
    def test_pairs_hook_events_and_reconciles_transcript_overlap(self):
        sa = load()
        rows = [
            {"type": "system", "subtype": "hook_started", "hook_id": "prompt-1",
             "hook_event": "UserPromptSubmit", "session_id": "session-a",
             "timestamp": "2026-08-17T00:00:00.000Z"},
            {"type": "system", "subtype": "hook_response", "hook_id": "prompt-1",
             "hook_event": "UserPromptSubmit", "session_id": "session-a",
             "timestamp": "2026-08-17T00:00:00.320Z", "outcome": "success",
             "exit_code": 0, "stdout": "", "stderr": ""},
            {"type": "system", "subtype": "hook_started", "hook_id": "start-1",
             "hook_event": "SessionStart", "session_id": "session-a",
             "timestamp": "2026-08-17T00:00:01.000Z"},
            {"type": "system", "subtype": "hook_response", "hook_id": "start-1",
             "hook_event": "SessionStart", "session_id": "session-a",
             "timestamp": "2026-08-17T00:00:01.100Z", "outcome": "success",
             "exit_code": 0, "stdout": "", "stderr": ""},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hooks.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "bad\n",
                            encoding="utf-8")
            stream = sa.load_hook_streams([str(path)])
        self.assertEqual(len(stream["events"]), 2)
        prompt = stream["events"][0]
        self.assertEqual(prompt["event"], "UserPromptSubmit")
        self.assertAlmostEqual(prompt["duration_ms"], 320, places=3)
        agg = {"_sessions": [{"session": "session-a", "hooks": {
            "outcomes": {"SessionStart|success": 1}}}]}
        reconciled = sa._reconcile_hook_stream(agg, stream)
        self.assertEqual(reconciled["suppressed_transcript_overlaps"], 1)
        self.assertEqual([row["event"] for row in reconciled["events"]],
                         ["UserPromptSubmit"])
        report = "\n".join(sa._hook_stream_lens(reconciled))
        self.assertIn("| UserPromptSubmit | 1 | success=1 | 0=1 | 320.0 ms | 320.0 ms |", report)
        self.assertIn("Transcript overlaps suppressed: 1", report)
        self.assertIn("Malformed stream rows skipped: 1", report)

    def test_capture_hooks_persists_receive_timestamps(self):
        sa = load()
        payload = json.dumps({"type": "system", "subtype": "hook_started",
                              "hook_id": "one", "hook_event": "UserPromptSubmit"})
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "capture.jsonl"
            args = argparse.Namespace(
                out=str(out),
                command=[sys.executable, "-c", f"print({payload!r})"],
            )
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(sa.cmd_capture_hooks(args), 0)
            row = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(row["hook_id"], "one")
        self.assertRegex(row["_captured_at"], r"Z$")

    def test_reports_cancelled_and_unmatched_hook_events(self):
        sa = load()
        rows = [
            {"subtype": "hook_started", "hook_id": "cancel-1",
             "hook_event": "UserPromptSubmit", "session_id": "session-b",
             "ts": "2026-08-17T00:00:00Z"},
            {"subtype": "hook_response", "hook_id": "cancel-1",
             "hook_event": "UserPromptSubmit", "session_id": "session-b",
             "ts": "2026-08-17T00:00:30Z", "outcome": "cancelled", "exit_code": None},
            {"subtype": "hook_started", "hook_id": "orphan",
             "hook_event": "Stop", "session_id": "session-b"},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "hooks.jsonl"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            stream = sa.load_hook_streams([str(path)])
        self.assertEqual(stream["unmatched_starts"], 1)
        self.assertEqual(stream["events"][0]["outcome"], "cancelled")
        self.assertEqual(stream["events"][0]["duration_ms"], 30000)
        report = "\n".join(sa._hook_stream_lens(stream))
        self.assertIn("cancelled=1", report)
        self.assertIn("30,000.0 ms", report)
        self.assertIn("starts 1", report)


class DOSHookLedgerTest(unittest.TestCase):
    def test_independent_ledger_reconciles_missing_posttool_attachments(self):
        sa = load()
        rows = [
            {"ts": "2026-08-15T00:00:00Z", "verb": "posttool", "outcome": "passthrough",
             "exit": 0, "latency_ms": 2.5},
            {"ts": "2026-08-15T00:00:00Z", "verb": "posttool", "outcome": "passthrough",
             "exit": 0, "latency_ms": 1500},
            {"ts": "2026-08-15T00:00:02Z", "verb": "stop", "outcome": "no-claims",
             "exit": 1, "latency_ms": 4},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".dos" / "metrics" / "observations.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows) + "bad\n",
                            encoding="utf-8")
            ledger = sa.load_dos_hook_observations(
                td, None
            )
        self.assertEqual(ledger["verbs"]["posttool"]["count"], 2)
        self.assertEqual(ledger["verbs"]["posttool"]["duration_ms"]["p90"], 1500)
        self.assertEqual(ledger["verbs"]["posttool"]["duration_ms"]["over_100ms"], 1)
        self.assertEqual(ledger["verbs"]["posttool"]["duration_ms"]["over_500ms"], 1)
        self.assertEqual(ledger["verbs"]["posttool"]["duration_ms"]["over_1000ms"], 1)
        self.assertEqual(ledger["malformed_rows"], 1)
        agg = {"hooks": {"events": {}}, "dos_hook_ledger": ledger,
               "tool_mix": {"Read": 1}}
        report = "\n".join(sa._dos_hook_lens(agg))
        self.assertIn("independently witnessed 2 `posttool` calls", report)
        self.assertIn("attachment-observability gap", report)
        self.assertIn("2 ledger rows / 1 audited transcript tool calls = 2.00x", report)
        self.assertIn("1 rows are additional calls in an already-occupied one-second bucket", report)
        self.assertIn("| stop | 1 | no-claims=1 | 1=1 |", report)
        self.assertIn("1,500.0 ms | 1 | 1 | 1 |", report)

    def test_exact_ledger_cutoff_overrides_relative_window(self):
        sa = load()
        rows = [
            {"ts": "2026-08-16T23:59:59Z", "verb": "posttool", "outcome": "passthrough",
             "exit": 0, "latency_ms": 1},
            {"ts": "2026-08-17T00:00:00Z", "verb": "posttool", "outcome": "passthrough",
             "exit": 0, "latency_ms": 2},
        ]
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / ".dos" / "metrics" / "observations.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            ledger = sa.load_dos_hook_observations(td, 0, "2026-08-17T00:00:00Z")
            with self.assertRaisesRegex(ValueError, "timezone"):
                sa.load_dos_hook_observations(td, None, "2026-08-17T00:00:00")
        self.assertEqual(ledger["verbs"]["posttool"]["count"], 1)
        self.assertEqual(ledger["verbs"]["posttool"]["duration_ms"]["max"], 2)

if __name__ == "__main__":
    unittest.main()
