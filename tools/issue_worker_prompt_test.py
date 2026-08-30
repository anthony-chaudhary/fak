#!/usr/bin/env python3
"""Hermetic tests for tools/issue_worker_prompt.py.

render_prompt is pure (data in, string out), so the load-bearing invariants —
the #N citation rule, the trunk/by-path git laws, the honest-block clause — are
asserted directly with no gh/claude/dos call. fetch_issue is exercised against
an injected runner so the gh shell-out never runs.
"""
from __future__ import annotations

import importlib.util
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tools" / "issue_worker_prompt.py"


def load():
    sys.path.insert(0, str(SCRIPT.parent))
    spec = importlib.util.spec_from_file_location("issue_worker_prompt", SCRIPT)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class RenderPromptTest(unittest.TestCase):
    ISSUE = {
        "number": 465,
        "title": "obs: arm the DOS verdict-journal auto-emit",
        "body": "The trust floor's own decisions should be observable.",
        "labels": [{"name": "enhancement"}, {"name": "trust-floor"}],
        "state": "OPEN",
    }

    def test_cites_issue_number_as_the_close_link(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "docs", workspace="C:/work/fak")
        # the #N citation must appear AND be called out as the close-binding rule.
        self.assertIn("#465", p)
        self.assertIn("commit subject", p)
        self.assertIn("never closes", p)  # the consequence of omitting it

    def test_embeds_title_body_labels_and_lane(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "gateway", workspace="C:/work/fak")
        self.assertIn("auto-emit", p)               # title
        self.assertIn("observable", p)              # body
        self.assertIn("enhancement, trust-floor", p)  # labels
        self.assertIn("`gateway` lane", p)          # lane routing

    def test_generation_label_shapes_worker_intent(self) -> None:
        mod = load()
        issue = dict(self.ISSUE, labels=[{"name": "generation"}, {"name": "gen/now"}])
        p = mod.render_prompt(issue, "tools", workspace="C:/work/fak")
        self.assertIn("Generation intent: now", p)
        self.assertIn("immediate trunk-safe", p)
        self.assertIn("orthogonal to priority", p)
        self.assertIn("Generation frame: stream=gen/now", p)
        self.assertIn("allowed risk=low", p)
        self.assertIn("proof bar=focused test", p)

    def test_next_generation_frame_routes_prompt_shape(self) -> None:
        mod = load()
        issue = dict(self.ISSUE, labels=[{"name": "generation"}, {"name": "gen/next"}])
        p = mod.render_prompt(issue, "tools", workspace="C:/work/fak")
        self.assertIn("Generation intent: next", p)
        self.assertIn("Generation frame: stream=gen/next", p)
        self.assertIn("allowed risk=moderate only behind a gate", p)
        self.assertIn("proof bar=contract test plus promotion evidence", p)
        self.assertIn("scope width=near-term foundation", p)
        self.assertIn("expected artifact=agent-runnable schema", p)
        self.assertIn("name promotion evidence, demotion/retirement evidence", p)

    def test_unclassified_generation_tells_worker_not_to_guess(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "tools", workspace="C:/work/fak")
        self.assertIn("Generation intent: unclassified", p)
        self.assertIn("avoid guessing", p)
        self.assertIn("Generation frame: stream=unclassified", p)
        self.assertIn("label/milestone repair", p)

    def test_states_the_git_laws(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "docs", workspace="C:/work/fak")
        self.assertIn("main", p)
        self.assertIn("git add -A", p)      # the forbidden form is named
        self.assertIn("commit -s", p)       # sign-off required
        self.assertIn("OFF_TRUNK", p)

    def test_commit_law_models_message_before_pathspec(self) -> None:
        # INTERACTIVE_HANG root cause (#4323): git parses everything after `--`
        # as a pathspec, so a prompt that shows `-- <paths>` with no `-m` (or
        # with `-m` placed after `--`) invites exactly the ordering that opens
        # the commit-message editor and hangs a headless worker. The example
        # must show `-m` BEFORE `--`, and INTERACTIVE_HANG must be named as
        # the consequence of getting the order wrong.
        mod = load()
        p = mod.render_prompt(self.ISSUE, "docs", workspace="C:/work/fak")
        self.assertIn('git commit -s -m "<subject>" -- <explicit paths>', p)
        self.assertIn("INTERACTIVE_HANG", p)
        # no single backtick-delimited command span may show `--` followed by
        # `-m` (that ordering is what git parses as a pathspec, not a message).
        for span in re.findall(r"`([^`]*)`", p):
            self.assertNotRegex(
                span, r"--.*-m ",
                f"commit example models `-m` after `--`: `{span}`",
            )

    def test_pathspec_race_recovery_preserves_explicit_paths(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "tools", workspace="C:/work/fak")
        self.assertIn("PATHSPEC_RACE", p)
        self.assertIn("preserve your working-tree edits", p)
        self.assertIn("refresh from the current trunk witness", p)
        self.assertIn("recommit by the same explicit paths", p)
        self.assertIn("never recover by sweeping the tree", p)

    def test_guidance_blocks_are_witnessed_rules_not_prose(self) -> None:
        # #3220: the `how to work it` / `git laws` blocks render FROM the structured
        # rule table, so EVERY bullet in them is `- <id>: <imperative> - witness `<w>``.
        # A bullet with no witness is the unobservable prose this replaced — it could
        # name a retired gate and nothing would notice.
        mod = load()
        p = mod.render_prompt(self.ISSUE, "tools", workspace="C:/work/fak")
        bullet = re.compile(r"^- (?P<id>[a-z][a-z0-9-]*): .+ - witness `(?P<w>[^`]+)`$")
        for header in ("how to work it (each rule:", "git laws (enforced below"):
            start = p.index(header)
            block = p[start:p.index("\n\n", start)].split("\n")
            rules = block[1:]
            self.assertTrue(rules, f"{header!r} rendered no rules")
            for line in rules:
                self.assertRegex(line, bullet, f"unwitnessed bullet under {header!r}")

    def test_guidance_rules_mirror_the_go_spec(self) -> None:
        # internal/dispatchtick/promptrules.go is the SPEC; this renderer mirrors it.
        # Assert the ids and witnesses match so the two renderers cannot drift apart
        # silently (the Go side asserts the same pairing from its own direction).
        mod = load()
        go_src = (ROOT / "internal" / "dispatchtick" / "promptrules.go").read_text(
            encoding="utf-8")
        rules = mod._work_rules(465, "docs") + mod._git_law_rules(465, "docs")
        for rid, _imperative, witness in rules:
            self.assertIn(f'ID: "{rid}"', go_src,
                          f"rule id {rid!r} is not in the Go spec")
            self.assertIn(witness.split(" --")[0], go_src,
                          f"witness {witness!r} for {rid!r} is not in the Go spec")

    def test_retired_thought_check_rule_is_absent_and_remaining_rules_stay_ordered(self) -> None:
        mod = load()
        rules = mod._work_rules(9568, "issuecheck")
        expected_ids = [
            "lane-lease",
            "refusal-taxonomy",
            "smallest-change",
            "checkpoint-commit",
            "gate-before-done",
            "proof-by-default",
            "browser-display",
            "no-delete",
            "honest-bail",
        ]
        self.assertEqual(expected_ids, [rid for rid, _imperative, _witness in rules])
        for rule in rules:
            for value in rule:
                self.assertNotIn("thought-check", value)

        p = mod.render_prompt(self.ISSUE, "issuecheck", workspace="C:/work/fak")
        for rid in expected_ids:
            self.assertIn(f"- {rid}:", p)
        for stale in ("thought-check", "top-five-thought-check", "fak-issuecheck"):
            self.assertNotIn(stale, p)

    def test_has_an_honest_block_clause(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "docs", workspace="C:/work/fak")
        self.assertIn("durable handoff", p)
        self.assertIn("gh issue comment <N> --body", p)
        self.assertIn("final chat report alone is not durable", p)
        self.assertIn("ignored scratch is not a deliverable", p)
        self.assertIn("fabricate", p)       # do NOT fabricate a pass

    def test_truncates_an_overlong_body(self) -> None:
        mod = load()
        big = dict(self.ISSUE, body="x" * 5000)
        p = mod.render_prompt(big, "docs", workspace="C:/work/fak")
        self.assertIn("truncated", p)
        self.assertNotIn("x" * 2000, p)     # the 5000-char body was truncated (cap is 1800)

    def test_missing_body_still_renders(self) -> None:
        mod = load()
        nobody = {"number": 7, "title": "t", "labels": []}
        p = mod.render_prompt(nobody, "docs", workspace="C:/work/fak")
        self.assertIn("#7", p)
        self.assertIn("no body", p)


class OriginQualityChecksTest(unittest.TestCase):
    """#1987: every dispatch packet names the per-lane origin-quality checks — each
    with its command, expected artifact, and refusal mode — and a QA-dogfood-spine
    issue's packet additionally names the at-origin score control to run before
    handoff (the #1961 spine done condition)."""

    QA = {
        "number": 1987,
        "title": "include origin-quality checks in every fleet dispatch packet",
        "body": "<!-- fak-qa-dogfood-spine:QD-027 -->\nParent: #1961",
        "labels": [{"name": "testing"}, {"name": "track/E-testing-quality"}],
        "state": "OPEN",
    }
    PLAIN = {"number": 500, "title": "t", "body": "b", "labels": [], "state": "OPEN"}

    def test_every_packet_names_the_origin_gate_with_all_three_parts(self) -> None:
        # command + expected artifact + refusal mode are all present in every packet.
        mod = load()
        p = mod.render_prompt(self.PLAIN, "gateway", workspace="C:/work/fak")
        self.assertIn("origin-quality checks", p)
        self.assertIn("BEFORE final handoff", p)
        self.assertIn("expected artifact", p)
        self.assertIn("refusal mode", p)
        self.assertIn("make ci", p)                 # the full gate command
        self.assertIn("go test ./internal/gateway", p)  # the lane gate command
        self.assertIn("ARCH_LAYER_VIOLATION", p)    # the Go-lane refusal mode

    def test_tools_lane_names_the_pythongate_refusal(self) -> None:
        mod = load()
        p = mod.render_prompt(self.PLAIN, "tools", workspace="C:/work/fak")
        self.assertIn("python tools/<touched>_test.py", p)
        # NEW_PYTHON_TOOL is the token pythongate stamps; REASON_NEW_PYTHON_TOOL was
        # only its Go constant name, which no reason registry declares (#3220).
        self.assertIn("NEW_PYTHON_TOOL", p)         # the tools-lane refusal mode
        self.assertNotIn("REASON_NEW_PYTHON_TOOL", p)

    def test_docs_lane_names_the_claims_lint_refusal(self) -> None:
        mod = load()
        p = mod.render_prompt(self.PLAIN, "docs", workspace="C:/work/fak")
        self.assertIn("make claims-lint", p)
        self.assertIn("[SHIPPED]", p)

    def test_core_lane_names_the_self_modify_refusal(self) -> None:
        mod = load()
        p = mod.render_prompt(self.PLAIN, "abi", workspace="C:/work/fak")
        self.assertIn("CORE_SELF_MODIFY", p)

    def test_cmd_lane_targets_the_cmd_package(self) -> None:
        mod = load()
        p = mod.render_prompt(self.PLAIN, "cmd", workspace="C:/work/fak")
        self.assertIn("go test ./cmd/fak", p)

    def test_qa_dogfood_packet_names_the_at_origin_score_control(self) -> None:
        # the #1961 done condition: a QA-dogfood issue's packet names the exact origin
        # checks to run before final handoff.
        mod = load()
        p = mod.render_prompt(self.QA, "tools", workspace="C:/work/fak")
        self.assertIn("at-origin score control", p)
        self.assertIn("QA-dogfood spine", p)
        self.assertIn("BEFORE handoff", p)

    def test_plain_issue_omits_the_at_origin_score_control_line(self) -> None:
        # a non-spine issue still gets the lane+full gate, but not the spine-only line.
        mod = load()
        p = mod.render_prompt(self.PLAIN, "tools", workspace="C:/work/fak")
        self.assertIn("origin-quality checks", p)
        self.assertNotIn("at-origin score control", p)

    def test_track_label_alone_marks_a_qa_dogfood_issue(self) -> None:
        mod = load()
        no_marker = dict(self.QA, body="no spine marker here")
        self.assertTrue(mod._is_qa_dogfood(no_marker))
        self.assertFalse(mod._is_qa_dogfood(self.PLAIN))


class RenderRepairPromptTest(unittest.TestCase):
    ROWS = [
        {"number": 1207, "title": "docs(docs): session-lifecycle one-pager",
         "missing_fields": ["done_condition", "acceptance_gate", "scope"],
         "reasons": ["ISSUE_SCOPE_INCOMPLETE"]},
        {"number": 1381, "title": "perf(metal): fused expert MLP",
         "missing_fields": ["work_unit"], "reasons": ["ISSUE_UNROUTED"]},
    ]

    def test_names_every_issue_and_its_missing_fields(self) -> None:
        mod = load()
        p = mod.render_repair_prompt(self.ROWS, workspace="C:/work/fak")
        self.assertIn("#1207", p)
        self.assertIn("#1381", p)
        self.assertIn("done_condition", p)
        self.assertIn("ISSUE_UNROUTED", p)

    def test_writes_local_overlays_never_github_or_the_repo_tree(self) -> None:
        # The load-bearing boundary: a repair worker's ONLY write is the local
        # overlay file (issue mutations are operator-gated on this host); the
        # repo tree stays read-only and GitHub is never written.
        mod = load()
        p = mod.render_repair_prompt(self.ROWS, workspace="C:/work/fak")
        self.assertIn("contract-overlays/issue-<N>.md", p)
        self.assertIn("NO GitHub writes", p)
        self.assertIn("READ-ONLY", p)
        self.assertIn("no commits", p)
        self.assertNotIn("gh issue edit", p)  # the write path a worker CANNOT land

    def test_demands_verified_pass_and_honest_hold(self) -> None:
        mod = load()
        p = mod.render_repair_prompt(self.ROWS, workspace="C:/work/fak")
        # verify-before-claim runs the overlay-merged review, and un-repairables
        # get NO overlay (named in the final report), never a fabricated pass.
        self.assertIn("issue_contract_repair.py --verify", p)
        self.assertIn("Do not claim a pass you did", p)
        self.assertIn("NO overlay", p)

    def test_build_repair_folds_rows_without_io(self) -> None:
        mod = load()
        rec = mod.build_repair(self.ROWS, workspace=Path("C:/work/fak"))
        self.assertEqual(rec["kind"], "contract-repair")
        self.assertEqual(rec["issues"], [1207, 1381])
        self.assertEqual(rec["prompt_chars"], len(rec["prompt"]))
        # Bounded: the whole batch prompt stays well under the /goal-style caps.
        self.assertLess(rec["prompt_chars"], 8000)


class WindowsShellHintTest(unittest.TestCase):
    """The opencode/glm worker on Windows runs under PowerShell but the repo's prose
    leans on Unix tools — a worker that shells out to grep/wc burns turns on
    "unrecognized command" (live repair-2464 log). The prompt must name the
    PowerShell-native equivalents on Windows and stay byte-identical off-Windows."""

    ISSUE = {"number": 2464, "title": "t", "body": "b", "labels": [], "state": "OPEN"}
    REPAIR_ROWS = [{"number": 2464, "title": "t",
                    "missing_fields": ["done_condition"], "reasons": ["X"]}]

    def test_guidance_lists_powershell_equivalents_on_windows(self) -> None:
        mod = load()
        hint = mod._windows_shell_guidance("nt")
        self.assertIn("PowerShell", hint)
        # the exact equivalents the issue names + the exact commands the log saw fail
        for tok in ("Select-String", "Get-Content", "Get-ChildItem",
                    "Measure-Object", "grep", "wc"):
            self.assertIn(tok, hint)

    def test_guidance_empty_off_windows(self) -> None:
        mod = load()
        self.assertEqual(mod._windows_shell_guidance("posix"), "")

    def test_resolver_prompt_carries_hint_on_windows(self) -> None:
        mod = load()
        p = mod.render_prompt(self.ISSUE, "tools", workspace="C:/work/fak", host_os="nt")
        self.assertIn("PowerShell", p)
        self.assertIn("Select-String", p)
        # the commands that failed in the log are now named as NOT-on-PATH
        self.assertIn("NOT on PATH", p)

    def test_resolver_prompt_has_no_hint_off_windows(self) -> None:
        mod = load()
        p_posix = mod.render_prompt(self.ISSUE, "tools", workspace="C:/work/fak",
                                    host_os="posix")
        self.assertNotIn("PowerShell", p_posix)
        self.assertNotIn("Select-String", p_posix)

    def test_repair_prompt_carries_hint_on_windows(self) -> None:
        mod = load()
        p = mod.render_repair_prompt(self.REPAIR_ROWS, workspace="C:/work/fak",
                                     host_os="nt")
        self.assertIn("PowerShell", p)
        self.assertIn("Select-String", p)

    def test_repair_prompt_has_no_hint_off_windows(self) -> None:
        mod = load()
        p = mod.render_repair_prompt(self.REPAIR_ROWS, workspace="C:/work/fak",
                                     host_os="posix")
        self.assertNotIn("PowerShell", p)

    def test_off_windows_is_byte_identical_to_default_on_posix(self) -> None:
        # On a POSIX host the default-arg render must equal the explicit posix render,
        # i.e. adding the host_os knob changed nothing for the non-Windows path.
        if os.name != "posix":
            self.skipTest("POSIX-host byte-identity check only runs on POSIX")
        mod = load()
        a = mod.render_prompt(self.ISSUE, "tools", workspace="C:/work/fak")
        b = mod.render_prompt(self.ISSUE, "tools", workspace="C:/work/fak",
                              host_os="posix")
        self.assertEqual(a, b)


class FetchIssueDecodeTest(unittest.TestCase):
    """fetch_issue must decode gh output as UTF-8 and never crash — the docstring
    promises a minimal record on ANY failure. A non-cp1252 byte in an issue body
    (em-dash/smart-quote/emoji) used to raise UnicodeDecodeError in subprocess's
    reader thread, leaving proc.stdout None, and `json.loads(None)` raised a
    TypeError the old `except ValueError` did not catch — crashing the dispatcher.
    """

    class _Proc:
        def __init__(self, returncode: int, stdout, stderr=None) -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def test_forces_utf8_decoding(self) -> None:
        mod = load()
        seen: dict = {}

        def fake_run(cmd, **kw):
            seen.update(kw)
            return self._Proc(0, '{"number": 5, "title": "t"}')

        orig = mod.subprocess.run
        mod.subprocess.run = fake_run
        try:
            doc = mod.fetch_issue(5, workspace=Path("."))
        finally:
            mod.subprocess.run = orig
        # gh emits UTF-8; the shell-out must say so explicitly (not inherit cp1252).
        self.assertEqual(seen.get("encoding"), "utf-8")
        self.assertEqual(seen.get("errors"), "replace")
        self.assertEqual(doc.get("title"), "t")

    def test_survives_none_stdout_without_crashing(self) -> None:
        mod = load()

        def fake_run(cmd, **kw):
            # Simulate a reader-thread decode error: rc 0 but stdout never decoded.
            return self._Proc(0, None)

        orig = mod.subprocess.run
        mod.subprocess.run = fake_run
        try:
            doc = mod.fetch_issue(42, workspace=Path("."))
        finally:
            mod.subprocess.run = orig
        # No TypeError — the minimal record is returned, honoring the contract.
        self.assertEqual(doc["number"], 42)
        self.assertIn("_error", doc)

    def test_nonzero_returncode_with_none_streams_does_not_crash(self) -> None:
        mod = load()

        def fake_run(cmd, **kw):
            return self._Proc(1, None, None)

        orig = mod.subprocess.run
        mod.subprocess.run = fake_run
        try:
            doc = mod.fetch_issue(7, workspace=Path("."))
        finally:
            mod.subprocess.run = orig
        self.assertEqual(doc["number"], 7)
        self.assertIn("_error", doc)


class FetchIssueTest(unittest.TestCase):
    def test_build_record_shape_on_fetch_error(self) -> None:
        mod = load()
        # Force fetch to fail by pointing gh at a bogus workspace+number with a
        # monkeypatched subprocess that raises — build must still return a prompt.
        mod.fetch_issue = lambda number, *, workspace, timeout=60: {
            "number": number, "_error": "gh not available"}
        with tempfile.TemporaryDirectory() as d:
            rec = mod.build(999, "docs", workspace=Path(d))
        self.assertEqual(rec["issue"], 999)
        self.assertEqual(rec["fetch_error"], "gh not available")
        self.assertIn("#999", rec["prompt"])         # prompt renders regardless
        self.assertGreater(rec["prompt_chars"], 100)


if __name__ == "__main__":
    unittest.main()
