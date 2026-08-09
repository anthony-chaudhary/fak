#!/usr/bin/env python3
"""Hermetic tests for loops_inventory: the cron/interval fold, the two declaration
parsers, the renderers, and the trend ledger.

Pure/deterministic — sample declaration text and an injected `now`, a temp tree for the
one disk-touching walker, no clock, no network."""
from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import loops_inventory as li  # noqa: E402


# A representative register_*.ps1 header + params — the shape the real scripts use.
SAMPLE_REGISTER = r"""<#
register_fleet_slack_status.ps1 -- install/remove the OS Scheduled Task that posts the
fleet-status rollup to Slack every 30 minutes.
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [string]$TaskName     = 'FleetSlackStatus',
  [int]$EveryMinutes    = 30
)
$childArgs = @($py, (Join-Path $Workspace 'tools\fleet_slack_status.py'), '--slack')
"""

SAMPLE_REGISTER_LOCAL = r"""<#
register_dispatch_status_doc.ps1 -- keep the operator-local dispatch STATUS DOC fresh
(gitignored .dispatch-runs/dispatch-status.md). This task only WRITES the gitignored doc.
#>
param(
  [string]$TaskName    = 'FleetDispatchStatusDoc',
  [string]$DocPath     = '.dispatch-runs\dispatch-status.md',
  [int]$EveryMinutes   = 30
)
$childArgs = @($py, $tick, '--md', $DocPath)
$tick = Join-Path $Workspace 'tools\dispatch_status.py'
"""

SAMPLE_WORKFLOW = """name: Score signal
on:
  schedule:
    # Daily 07:41 UTC — an off-the-hour minute clear of the other jobs.
    - cron: "41 7 * * *"
  workflow_dispatch:
    inputs:
      foo: {}
jobs:
  run: {}
"""

SAMPLE_WORKFLOW_MULTI = """name: Release cadence
on:
  schedule:
    - cron: "37 */2 * * *"
    - cron: "17 8 * * 1"
jobs: {}
"""

SAMPLE_WORKFLOW_NO_CRON = """name: CI
on:
  push:
    branches: [main]
jobs: {}
"""


# The trigger forms real registrars use — each must resolve to the right period.
TRIG_EVERY_MINUTES = (
    r"param([int]$EveryMinutes = 30)" "\n"
    r"$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `" "\n"
    r"  -RepetitionInterval (New-TimeSpan -Minutes $EveryMinutes)")
TRIG_EVERY_MIN = (
    r"param([int]$EveryMin = 5)" "\n"
    r"  -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) `")
TRIG_EVERY_HOURS = (
    r"param([int]$EveryHours = 6)" "\n"
    r"  -RepetitionInterval (New-TimeSpan -Hours $EveryHours) `")
TRIG_LITERAL = r"  -RepetitionInterval (New-TimeSpan -Minutes 10) -RepetitionDuration x"
TRIG_DAILY = (
    r"param([string]$At = '09:50')" "\n"
    r"$trigger = New-ScheduledTaskTrigger -Daily -At $At")
TRIG_DAILY_PLUS_REP = (
    r"param([string]$At = '03:30', [int]$EveryHours = 4)" "\n"
    r"$t = New-ScheduledTaskTrigger -Daily -At $At" "\n"
    r"$r = New-ScheduledTaskTrigger -Once -At $At `" "\n"
    r"  -RepetitionInterval (New-TimeSpan -Hours $EveryHours)")
TRIG_SCHTASKS = r"schtasks /Create /TN X /SC MINUTE /MO 5 /TR $tr /RL LIMITED /F"


class CadenceExtractTest(unittest.TestCase):
    def test_every_minutes_param(self):
        self.assertEqual(li.extract_cadence(TRIG_EVERY_MINUTES), (30, "every 30 min"))

    def test_every_min_variant(self):
        self.assertEqual(li.extract_cadence(TRIG_EVERY_MIN), (5, "every 5 min"))

    def test_every_hours(self):
        self.assertEqual(li.extract_cadence(TRIG_EVERY_HOURS), (360, "every 6h"))

    def test_literal_minutes(self):
        self.assertEqual(li.extract_cadence(TRIG_LITERAL), (10, "every 10 min"))

    def test_daily_at(self):
        self.assertEqual(li.extract_cadence(TRIG_DAILY), (1440, "daily 09:50"))

    def test_daily_plus_repetition_reports_finer_period(self):
        mins, label = li.extract_cadence(TRIG_DAILY_PLUS_REP)
        self.assertEqual(mins, 240)  # every 4h is the effective sub-daily cadence
        self.assertEqual(label, "every 4h (+ daily 03:30)")

    def test_schtasks_sc_mo(self):
        self.assertEqual(li.extract_cadence(TRIG_SCHTASKS), (5, "every 5 min"))

    def test_unknown_is_none(self):
        self.assertEqual(li.extract_cadence("param()\n# no trigger"), (None, "?"))

    def test_every_hours_zero_is_not_a_repetition(self):
        # worktree-doctor documents EveryHours=0 as "daily only" — must not become 0-min.
        text = (r"param([string]$At='03:30',[int]$EveryHours=0)" "\n"
                r"$t = New-ScheduledTaskTrigger -Daily -At $At" "\n"
                r"  -RepetitionInterval (New-TimeSpan -Hours $EveryHours)")
        mins, label = li.extract_cadence(text)
        self.assertEqual(mins, 1440)
        self.assertEqual(label, "daily 03:30")


class CronCadenceTest(unittest.TestCase):
    def test_every_n_minutes(self):
        self.assertEqual(li.cron_cadence("*/5 * * * *"), ("every 5 min", 5))

    def test_every_n_hours(self):
        self.assertEqual(li.cron_cadence("17 */6 * * *"), ("every 6h", 360))
        self.assertEqual(li.cron_cadence("37 */2 * * *"), ("every 2h", 120))

    def test_daily(self):
        self.assertEqual(li.cron_cadence("41 7 * * *"), ("daily 07:41 UTC", 1440))

    def test_weekly(self):
        self.assertEqual(li.cron_cadence("47 9 * * 1"), ("weekly 09:47 UTC", 10080))

    def test_weekdays(self):
        label, mins = li.cron_cadence("0 9 * * 1-5")
        self.assertEqual(label, "weekdays 09:00 UTC")
        self.assertEqual(mins, 2016)

    def test_hourly(self):
        self.assertEqual(li.cron_cadence("15 * * * *"), ("hourly", 60))

    def test_unrecognized_shape_has_no_period(self):
        label, mins = li.cron_cadence("0 0 1 * *")  # monthly — not a recognized form
        self.assertIsNone(mins)
        self.assertEqual(label, "0 0 1 * *")

    def test_malformed_is_passed_through(self):
        self.assertEqual(li.cron_cadence("nonsense"), ("nonsense", None))


class HumanizeMinutesTest(unittest.TestCase):
    def test_table(self):
        self.assertEqual(li.humanize_minutes(5), "every 5 min")
        self.assertEqual(li.humanize_minutes(30), "every 30 min")
        self.assertEqual(li.humanize_minutes(180), "every 3h")
        self.assertEqual(li.humanize_minutes(720), "every 12h")
        self.assertEqual(li.humanize_minutes(1440), "daily")
        self.assertEqual(li.humanize_minutes(2880), "every 2d")
        self.assertEqual(li.humanize_minutes(None), "?")


class ParseRegisterTest(unittest.TestCase):
    def test_extracts_task_cadence_tick_purpose_sink(self):
        rec = li.parse_register_ps1(SAMPLE_REGISTER, "register_fleet_slack_status.ps1")
        self.assertIsNotNone(rec)
        assert rec is not None
        self.assertEqual(rec["surface"], "scheduled-task")
        self.assertEqual(rec["name"], "FleetSlackStatus")
        self.assertEqual(rec["cadence_minutes"], 30)
        self.assertEqual(rec["cadence"], "every 30 min")
        self.assertEqual(rec["tick"], "tools/fleet_slack_status.py")
        self.assertEqual(rec["sink"], li.SINK_SLACK)
        self.assertIn("fleet-status rollup", rec["purpose"])
        self.assertEqual(rec["source"], "tools/register_fleet_slack_status.ps1")

    def test_gitignored_doc_tags_local_not_repo(self):
        rec = li.parse_register_ps1(SAMPLE_REGISTER_LOCAL, "register_dispatch_status_doc.ps1")
        assert rec is not None
        self.assertEqual(rec["name"], "FleetDispatchStatusDoc")
        self.assertEqual(rec["sink"], li.SINK_LOCAL)
        self.assertEqual(rec["tick"], "tools/dispatch_status.py")

    def test_no_taskname_is_skipped(self):
        self.assertIsNone(li.parse_register_ps1("param()\n", "register_noop.ps1"))

    def test_issue_filer_tags_github_not_action(self):
        text = ("<#\nregister_idea_scout.ps1 -- file new hits as GitHub issues.\n"
                "the only side effect is `gh issue create`, gated behind -Live.\n#>\n"
                r"$TaskName = 'FleetIdeaScout'" "\n"
                r"$trigger = New-ScheduledTaskTrigger -Daily -At '09:00'")
        rec = li.parse_register_ps1(text, "register_idea_scout.ps1")
        assert rec is not None
        self.assertEqual(rec["sink"], li.SINK_GITHUB)

    def test_dispatcher_reading_issues_is_not_github(self):
        # a loop that only READS the backlog (no create/reopen/--file-issues) must not
        # be mistaken for an issue reporter.
        text = ("<#\nkeeps the issue dispatcher always-on; spawns workers.\n#>\n"
                r"$TaskName = 'FleetIssueDispatch'" "\n"
                r"$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `" "\n"
                r"  -RepetitionInterval (New-TimeSpan -Minutes 5)")
        rec = li.parse_register_ps1(text, "register_issue_dispatch.ps1")
        assert rec is not None
        self.assertNotEqual(rec["sink"], li.SINK_GITHUB)

    def test_guard_with_docs_prose_tags_action_not_repo(self):
        # a stray "docs/..." mention in prose must not outvote the structural guard signal.
        text = ("<#\nprocess-resource guard; see docs/guards.md for the runaway classes.\n#>\n"
                r"$TaskName = 'FleetProcResourceGuard'" "\n"
                r"  -RepetitionInterval (New-TimeSpan -Minutes $EveryMin)" "\n"
                r"param([int]$EveryMin = 10)")
        rec = li.parse_register_ps1(text, "register_proc_resource_guard.ps1")
        assert rec is not None
        self.assertEqual(rec["sink"], li.SINK_ACTION)

    def test_wrapper_tool_is_not_picked_as_tick(self):
        text = (r"$TaskName = 'FleetX'" "\n"
                r"$wrapper = 'tools\fak_loop_task.ps1'" "\n"
                r"$tick = 'tools\real_work.py'" "\n")
        rec = li.parse_register_ps1(text, "register_x.ps1")
        assert rec is not None
        self.assertEqual(rec["tick"], "tools/real_work.py")


class ParseWorkflowTest(unittest.TestCase):
    def test_single_cron(self):
        rec = li.parse_workflow_yml(SAMPLE_WORKFLOW, "score-signal.yml")
        assert rec is not None
        self.assertEqual(rec["surface"], "github-actions")
        self.assertEqual(rec["name"], "Score signal")
        self.assertEqual(rec["cadence"], "daily 07:41 UTC")
        self.assertEqual(rec["cadence_minutes"], 1440)
        self.assertEqual(rec["sink"], li.SINK_SLACK)  # "signal" -> slack surface
        self.assertEqual(rec["source"], ".github/workflows/score-signal.yml")

    def test_multi_cron_uses_most_frequent(self):
        rec = li.parse_workflow_yml(SAMPLE_WORKFLOW_MULTI, "release-cadence.yml")
        assert rec is not None
        # every-2h (120) beats weekly (10080) for the headline cadence.
        self.assertEqual(rec["cadence_minutes"], 120)
        self.assertEqual(rec["crons"], ["37 */2 * * *", "17 8 * * 1"])

    def test_no_cron_is_skipped(self):
        self.assertIsNone(li.parse_workflow_yml(SAMPLE_WORKFLOW_NO_CRON, "ci.yml"))


class RenderTest(unittest.TestCase):
    def _inv(self):
        loops = [
            li.parse_register_ps1(SAMPLE_REGISTER, "register_fleet_slack_status.ps1"),
            li.parse_register_ps1(SAMPLE_REGISTER_LOCAL, "register_dispatch_status_doc.ps1"),
            li.parse_workflow_yml(SAMPLE_WORKFLOW, "score-signal.yml"),
        ]
        loops.sort(key=li._sort_key)
        return {"schema": li.SCHEMA, "loops": loops, "summary": li.summarize(loops)}

    def test_summary_counts(self):
        s = self._inv()["summary"]
        self.assertEqual(s["total"], 3)
        self.assertEqual(s["tasks"], 2)
        self.assertEqual(s["workflows"], 1)
        self.assertEqual(s["slack"], 2)   # FleetSlackStatus + score-signal
        self.assertEqual(s["local"], 1)

    def test_md_has_both_tables_and_headline(self):
        md = li.render_md(self._inv(), "2026-07-11T00:00:00Z")
        self.assertIn("# Recurring loops inventory", md)
        self.assertIn("**3 loops**", md)
        self.assertIn("## OS Scheduled Tasks", md)
        self.assertIn("## GitHub Actions (cron)", md)
        self.assertIn("FleetSlackStatus", md)
        self.assertIn("`.github/workflows/score-signal.yml`", md)
        # generated-by provenance line names the tool
        self.assertIn("tools/loops_inventory.py", md)

    def test_md_escapes_pipes(self):
        rec = {"surface": "scheduled-task", "name": "X", "cadence": "daily",
               "cadence_minutes": 1440, "sink": "?", "tick": "",
               "purpose": "a | b", "source": "tools/register_x.ps1"}
        inv = {"schema": li.SCHEMA, "loops": [rec], "summary": li.summarize([rec])}
        md = li.render_md(inv, "now")
        self.assertIn(r"a \| b", md)

    def test_slack_is_compact(self):
        card = li.render_slack(self._inv())
        self.assertIn("Recurring loops:", card)
        self.assertIn("3 total", card)
        self.assertIn("fastest:", card)


class TrendTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.ledger = str(Path(self.dir.name) / "sub" / "hist.jsonl")

    def test_row_pulls_metrics(self):
        row = li.trend_row({"total": 5, "tasks": 3, "workflows": 2, "slack": 1, "repo": 1},
                           "2026-07-11T00:00:00Z")
        self.assertEqual(row, {"ts": "2026-07-11T00:00:00Z", "total": 5, "tasks": 3,
                               "workflows": 2, "slack": 1, "repo": 1})

    def test_append_creates_dir_and_is_bounded(self):
        for i in range(5):
            li.trend_append(self.ledger, {"ts": f"t{i}", "total": i}, cap=3)
        rows = li._read_rows(self.ledger)
        self.assertEqual(len(rows), 3)
        self.assertEqual([r["total"] for r in rows], [2, 3, 4])

    def test_line_empty_on_first_tick(self):
        rows = li.trend_append(self.ledger, li.trend_row({"total": 5}, "t0"))
        self.assertEqual(li.trend_line(rows), "")

    def test_line_shows_moved_metric(self):
        li.trend_append(self.ledger, li.trend_row({"total": 5, "tasks": 3}, "t0"))
        rows = li.trend_append(self.ledger, li.trend_row({"total": 7, "tasks": 3}, "t1"))
        line = li.trend_line(rows)
        self.assertIn("total 5→7", line)
        self.assertIn("+2 over 2", line)
        self.assertNotIn("tasks", line)  # unchanged metric is dropped


class DiscoverTest(unittest.TestCase):
    """The one disk-touching walker, over a temp tree mirroring the real layout."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        (self.root / "tools").mkdir()
        (self.root / ".github" / "workflows").mkdir(parents=True)
        (self.root / "tools" / "register_fleet_slack_status.ps1").write_text(
            SAMPLE_REGISTER, encoding="utf-8")
        (self.root / "tools" / "register_dispatch_status_doc.ps1").write_text(
            SAMPLE_REGISTER_LOCAL, encoding="utf-8")
        (self.root / "tools" / "not_a_register.ps1").write_text(
            "$TaskName = 'Nope'", encoding="utf-8")  # not register_*, ignored
        (self.root / ".github" / "workflows" / "score-signal.yml").write_text(
            SAMPLE_WORKFLOW, encoding="utf-8")
        (self.root / ".github" / "workflows" / "ci.yml").write_text(
            SAMPLE_WORKFLOW_NO_CRON, encoding="utf-8")  # no cron, ignored

    def test_discovers_both_surfaces_and_skips_noise(self):
        inv = li.discover(str(self.root))
        names = {lp["name"] for lp in inv["loops"]}
        self.assertEqual(names, {"FleetSlackStatus", "FleetDispatchStatusDoc", "Score signal"})
        self.assertEqual(inv["summary"]["tasks"], 2)
        self.assertEqual(inv["summary"]["workflows"], 1)

    def test_sorted_tasks_before_workflows_then_by_frequency(self):
        inv = li.discover(str(self.root))
        self.assertEqual(inv["loops"][0]["surface"], "scheduled-task")
        self.assertEqual(inv["loops"][-1]["surface"], "github-actions")

    def test_json_roundtrips(self):
        inv = li.discover(str(self.root))
        # the whole inventory must be JSON-serializable (the --json contract)
        json.loads(json.dumps(inv))


class MainCliTest(unittest.TestCase):
    """Exercise main() end-to-end over a temp tree, with slack_post faked in sys.modules so
    no network is touched and the exit-code contract is checkable."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        self.root = Path(self.dir.name)
        (self.root / "tools").mkdir()
        (self.root / "tools" / "register_fleet_slack_status.ps1").write_text(
            SAMPLE_REGISTER, encoding="utf-8")
        # Fake slack_post whose send() echoes a configurable verdict.
        self._verdict = {"posted": True, "dry_run": False, "channel": "C1"}
        fake = types.ModuleType("slack_post")
        fake.send = lambda text, **kw: dict(self._verdict)  # type: ignore[attr-defined]
        self._saved = sys.modules.get("slack_post")
        sys.modules["slack_post"] = fake
        self.addCleanup(self._restore)

    def _restore(self):
        if self._saved is not None:
            sys.modules["slack_post"] = self._saved
        else:
            sys.modules.pop("slack_post", None)

    def _run(self, argv):
        """main() prints a summary; swallow it so the test run stays quiet."""
        with contextlib.redirect_stdout(io.StringIO()):
            return li.main(argv)

    def test_md_and_ledger_written(self):
        doc = self.root / "docs" / "loops.md"
        ledger = self.root / ".fak" / "hist.jsonl"
        rc = self._run(["--workspace", str(self.root), "--md", str(doc),
                        "--ledger", str(ledger), "--now", "2026-07-11T00:00:00Z"])
        self.assertEqual(rc, 0)
        self.assertIn("Recurring loops inventory", doc.read_text(encoding="utf-8"))
        self.assertEqual(len(li._read_rows(str(ledger))), 1)

    def test_live_post_landed_is_rc0(self):
        self._verdict = {"posted": True, "dry_run": False, "channel": "C1"}
        rc = self._run(["--workspace", str(self.root), "--slack"])
        self.assertEqual(rc, 0)

    def test_live_post_skipped_is_rc1(self):
        # a live post that resolves no channel/token is a misconfiguration → non-zero.
        self._verdict = {"posted": False, "dry_run": False, "channel": "",
                         "skipped": "no channel resolved"}
        rc = self._run(["--workspace", str(self.root), "--slack"])
        self.assertEqual(rc, 1)

    def test_dry_run_never_fails(self):
        # dry-run posts nothing by design, so it must not be treated as a failure.
        self._verdict = {"posted": False, "dry_run": True, "channel": "C1", "skipped": "dry-run"}
        rc = self._run(["--workspace", str(self.root), "--slack", "--dry-run"])
        self.assertEqual(rc, 0)


class TaskNameGuardTest(unittest.TestCase):
    """#5572 — the task name is READ from these scripts and published as ground truth, so
    it must be a static literal read from the real param/assignment site.

    Every fixture below is text handed straight to the pure parser (or written into a temp
    tree for the CLI witness). No registrar is executed and no scheduled task is touched —
    reading a registrar as text is the only thing this tool ever does to one.
    """

    # An interpolated default: PowerShell expands this at run time, so the literal body
    # "Fleet$($Suffix)Doctor" names nothing on the host.
    INTERPOLATED = (
        "<#\nregister_flaky.ps1 -- install the task.\n#>\n"
        "param(\n"
        "  [string]$Suffix = 'Doctor',\n"
        "  [string]$TaskName = \"Fleet$($Suffix)Doctor\",\n"
        "  [int]$EveryMinutes = 30\n"
        ")\n")

    # A backtick-escaped default is equally unreadable statically.
    BACKTICKED = "param([string]$TaskName = \"Fleet`tDoctor\", [int]$EveryMinutes = 30)\n"

    # The usage header documents an override BEFORE the param block declares the default —
    # the exact first-match race the unanchored search lost.
    HEADER_DECOY = (
        "<#\nregister_decoy.ps1 -- install the task. Override the name with\n"
        "  .\\register_decoy.ps1 -TaskName 'NotTheRealOne'\n"
        "which sets $TaskName = 'NotTheRealOne' for that run only.\n#>\n"
        "param(\n"
        "  [string]$TaskName = 'FleetRealName',\n"
        "  [int]$EveryMinutes = 15\n"
        ")\n")

    # Same race via a plain line comment recording a rename.
    LINE_COMMENT_DECOY = (
        "param(\n"
        "  # legacy, removed 2026-01: $TaskName = 'OldName'\n"
        "  [string]$TaskName = 'FleetRealName',\n"
        "  [int]$EveryMinutes = 15\n"
        ")\n")

    def test_interpolated_name_is_refused_not_emitted(self):
        rec = li.parse_register_ps1(self.INTERPOLATED, "register_flaky.ps1")
        assert rec is not None
        # The raw body must never reach the doc — that is the whole defect.
        self.assertNotIn("$", rec["name"])
        self.assertNotIn("Suffix", rec["name"])
        self.assertEqual(rec["name"], li.NAME_UNRESOLVED)
        self.assertTrue(rec["name_unresolved"])
        # Still inventoried: the loop exists, only its name is unknowable.
        self.assertEqual(rec["source"], "tools/register_flaky.ps1")
        self.assertEqual(rec["cadence_minutes"], 30)

    def test_backticked_name_is_refused(self):
        rec = li.parse_register_ps1(self.BACKTICKED, "register_bt.ps1")
        assert rec is not None
        self.assertNotIn("`", rec["name"])
        self.assertEqual(rec["name"], li.NAME_UNRESOLVED)

    def test_header_block_decoy_does_not_shadow_param_default(self):
        rec = li.parse_register_ps1(self.HEADER_DECOY, "register_decoy.ps1")
        assert rec is not None
        self.assertEqual(rec["name"], "FleetRealName")
        self.assertFalse(rec["name_unresolved"])

    def test_line_comment_decoy_does_not_shadow_param_default(self):
        rec = li.parse_register_ps1(self.LINE_COMMENT_DECOY, "register_legacy.ps1")
        assert rec is not None
        self.assertEqual(rec["name"], "FleetRealName")

    def test_literal_registrar_parses_unchanged(self):
        # CONTROL: the ordinary shape every register_*.ps1 uses today is untouched.
        rec = li.parse_register_ps1(SAMPLE_REGISTER, "register_fleet_slack_status.ps1")
        assert rec is not None
        self.assertEqual(rec["name"], "FleetSlackStatus")
        self.assertFalse(rec["name_unresolved"])
        self.assertEqual(rec["sink"], li.SINK_SLACK)
        self.assertEqual(rec["cadence"], "every 30 min")

    def test_hash_inside_a_string_is_not_a_comment(self):
        # Comment stripping must not truncate a line at a '#' that lives inside a string.
        text = ("$Doc = 'docs\\a#b.md'\n"
                "param([string]$TaskName = 'FleetHashSafe', [int]$EveryMinutes = 15)\n")
        rec = li.parse_register_ps1(text, "register_hash.ps1")
        assert rec is not None
        self.assertEqual(rec["name"], "FleetHashSafe")

    def test_summary_counts_the_refusal(self):
        loops = [li.parse_register_ps1(self.INTERPOLATED, "register_flaky.ps1"),
                 li.parse_register_ps1(SAMPLE_REGISTER, "register_fleet_slack_status.ps1")]
        self.assertEqual(li.summarize(loops)["unresolved"], 1)
        self.assertEqual(li.summarize([loops[1]])["unresolved"], 0)

    def test_md_and_slack_show_the_refusal(self):
        loops = [li.parse_register_ps1(self.INTERPOLATED, "register_flaky.ps1")]
        inv = {"schema": li.SCHEMA, "loops": loops, "summary": li.summarize(loops)}
        md = li.render_md(inv, "2026-08-02T00:00:00Z")
        self.assertIn("Refused 1 task name(s)", md)
        self.assertIn(li.NAME_UNRESOLVED, md)
        self.assertIn("`tools/register_flaky.ps1`", md)  # names WHICH script drifted
        self.assertIn("refused 1 task name(s)", li.render_slack(inv))

    def test_clean_tree_carries_no_refusal_text(self):
        # CONTROL: nothing about the refusal leaks into the doc when every name is literal.
        loops = [li.parse_register_ps1(SAMPLE_REGISTER, "register_fleet_slack_status.ps1")]
        inv = {"schema": li.SCHEMA, "loops": loops, "summary": li.summarize(loops)}
        md = li.render_md(inv, "2026-08-02T00:00:00Z")
        self.assertNotIn("Refused", md)
        self.assertNotIn("unresolved", md)
        self.assertNotIn("refused", li.render_slack(inv))


class TaskNameGuardCliTest(unittest.TestCase):
    """The refusal has to reach a sink an operator actually watches: the rendered doc and
    the exit code (which the scheduled task stores as LastTaskResult)."""

    def _tree(self, register_text: str) -> Path:
        d = tempfile.TemporaryDirectory()
        self.addCleanup(d.cleanup)
        root = Path(d.name)
        (root / "tools").mkdir()
        (root / "tools" / "register_flaky.ps1").write_text(register_text, encoding="utf-8")
        return root

    def _run(self, argv):
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            return li.main(argv), buf.getvalue()

    def test_interpolated_name_exits_nonzero_and_still_renders_the_doc(self):
        root = self._tree(TaskNameGuardTest.INTERPOLATED)
        doc = root / "docs" / "loops.md"
        rc, out = self._run(["--workspace", str(root), "--md", str(doc),
                             "--now", "2026-08-02T00:00:00Z"])
        self.assertEqual(rc, 1)
        self.assertIn("refused 1 task name(s)", out)
        text = doc.read_text(encoding="utf-8")
        self.assertIn(li.NAME_UNRESOLVED, text)
        self.assertNotIn("Fleet$", text)

    def test_all_literal_names_exit_zero(self):
        # CONTROL: today's tree shape stays rc 0 — the guard adds no new failure.
        root = self._tree(SAMPLE_REGISTER)
        doc = root / "docs" / "loops.md"
        rc, out = self._run(["--workspace", str(root), "--md", str(doc),
                             "--now", "2026-08-02T00:00:00Z"])
        self.assertEqual(rc, 0)
        self.assertNotIn("refused", out)
        self.assertIn("FleetSlackStatus", doc.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
