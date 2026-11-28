#!/usr/bin/env python3
"""Loop config, list, check, audit, scaffold, and catalog tests for fleet_control_pane."""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_control_pane as pane  # noqa: E402


class FleetControlPaneLoopsTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "VERSION").write_text("test\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def _audit_config(self, loops: dict) -> dict:
        return {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": loops,
        }

    def test_loop_config_plan_dry_run_does_not_write_local_config(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": False,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="build-loop",
            status_cmd=[sys.executable, "tools/build_status.py", "--json"],
            recover_cmd=[sys.executable, "tools/build_watchdog.py"],
            auto_recover=True,
            action="restart build loop",
            timeout_s=15,
            apply=False,
        )

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["apply"])
        self.assertFalse((self.root / "tools" / "_registry" / "control_pane.local.json").exists())
        self.assertEqual(doc["spec"]["status_cmd"], [sys.executable, "tools/build_status.py", "--json"])
        self.assertIn("--apply", doc["command"])

    def test_loop_config_plan_apply_preserves_existing_local_config(self) -> None:
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({"schema": pane.CONFIG_SCHEMA, "target": 8}), encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(local)},
            "_local_exists": True,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="build-loop",
            status_cmd=[sys.executable, "tools/build_status.py", "--json"],
            recover_cmd=[sys.executable, "tools/build_watchdog.py"],
            auto_recover=True,
            action="restart build loop",
            timeout_s=15,
            recover_timeout_s=60,
            cwd="tools",
            ok_returncodes=[0, 2],
            apply=True,
        )

        written = json.loads(local.read_text(encoding="utf-8"))
        self.assertTrue(doc["ok"])
        self.assertTrue(doc["apply"])
        self.assertEqual(written["target"], 8)
        self.assertEqual(written["loops"]["build-loop"]["status_cmd"], [sys.executable, "tools/build_status.py", "--json"])
        self.assertEqual(written["loops"]["build-loop"]["recover_cmd"], [sys.executable, "tools/build_watchdog.py"])
        self.assertTrue(written["loops"]["build-loop"]["auto_recover"])
        self.assertEqual(written["loops"]["build-loop"]["recover_timeout_s"], 60)
        self.assertEqual(written["loops"]["build-loop"]["ok_returncodes"], [0, 2])

    def test_loop_config_plan_repo_scope_writes_tracked_catalog_not_local_config(self) -> None:
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        catalog = self.root / "tools" / "control_pane.loops.json"
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(local), "loop_catalog": str(catalog)},
            "_local_exists": False,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="shared-loop",
            scope="repo",
            status_cmd=[sys.executable, "tools/shared_status.py", "--json"],
            recover_cmd=[sys.executable, "tools/shared_watchdog.py"],
            auto_recover=True,
            apply=True,
        )

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["scope"], "repo")
        self.assertEqual(doc["config_path"], str(catalog))
        self.assertIsNone(doc["local_config"])
        self.assertFalse(local.exists())
        written = json.loads(catalog.read_text(encoding="utf-8"))
        self.assertEqual(written["schema"], pane.LOOP_CATALOG_SCHEMA)
        self.assertEqual(written["loops"]["shared-loop"]["status_cmd"], [sys.executable, "tools/shared_status.py", "--json"])
        self.assertIn("--scope repo", doc["command"])
        self.assertEqual(len(doc["followup_commands"]), 1)
        self.assertIn("commit --dirty-group tools/fleet-control-pane", doc["followup_commands"][0])
        self.assertIn("tools: add shared-loop loop", doc["followup_commands"][0])

    def test_loop_config_plan_matching_repo_loop_has_no_commit_followup(self) -> None:
        catalog = self.root / "tools" / "control_pane.loops.json"
        catalog.write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {
                "shared-loop": {
                    "enabled": True,
                    "status_cmd": [sys.executable, "tools/shared_status.py", "--json"],
                    "auto_recover": False,
                },
            },
        }), encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"loop_catalog": str(catalog)},
            "_loop_catalog_exists": True,
        }

        doc = pane.loop_config_plan(
            config,
            loop_name="shared-loop",
            scope="repo",
            status_cmd=[sys.executable, "tools/shared_status.py", "--json"],
            apply=True,
        )

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["changed"])
        self.assertEqual(doc["followup_commands"], [])

    def test_loop_list_reports_sources_and_readiness(self) -> None:
        status_script = self.root / "tools" / "shared_status.py"
        status_script.write_text("print('{}')\n", encoding="utf-8")
        (self.root / "tools" / "control_pane.example.json").write_text(json.dumps({
            "schema": pane.CONFIG_SCHEMA,
            "loops": {
                "example-loop": {
                    "enabled": False,
                    "status_cmd": [sys.executable, "tools/example_status.py"],
                },
            },
        }), encoding="utf-8")
        (self.root / "tools" / "control_pane.loops.json").write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {
                "shared-loop": {
                    "enabled": True,
                    "status_cmd": [sys.executable, "tools/shared_status.py"],
                    "auto_recover": True,
                    "recover_cmd": [sys.executable, "tools/missing_recover.py"],
                },
            },
        }), encoding="utf-8")
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(json.dumps({
            "schema": pane.CONFIG_SCHEMA,
            "loops": {
                "shared-loop": {
                    "recover_cmd": [],
                    "auto_recover": False,
                },
            },
        }), encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_list(config)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["count"], 2)
        shared = [loop for loop in doc["loops"] if loop["name"] == "shared-loop"][0]
        self.assertEqual(shared["source"], "local")
        self.assertTrue(shared["overridden"])
        self.assertEqual([entry["source"] for entry in shared["sources"]], ["repo", "local"])
        self.assertTrue(shared["status_ready"]["ok"])
        self.assertTrue(shared["ready"])
        example = [loop for loop in doc["loops"] if loop["name"] == "example-loop"][0]
        self.assertFalse(example["enabled"])
        self.assertEqual(example["source"], "example")

    def test_loop_list_empty_includes_repo_add_command(self) -> None:
        config = pane.load_config(self.root)

        doc = pane.loop_list(config)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["count"], 0)
        self.assertEqual(len(doc["commands"]), 2)
        self.assertIn("loop-scaffold NAME", doc["commands"][0])
        self.assertIn("loop-add NAME --scope repo", doc["commands"][1])

    def test_loop_check_plan_runs_one_loop_and_plans_recovery(self) -> None:
        bad_script = self.root / "tools" / "bad_loop.py"
        recover_script = self.root / "tools" / "recover_loop.py"
        bad_script.write_text("print('{\"verdict\": \"STALLED\", \"detail\": \"heartbeat stale\"}')\n", encoding="utf-8")
        recover_script.write_text("print('recovering')\n", encoding="utf-8")
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, str(bad_script)],
                    "recover_cmd": [sys.executable, str(recover_script)],
                    "auto_recover": False,
                },
            },
        }

        doc = pane.loop_check_plan(config, "stalled", recover=True, apply=False)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["verdict"], "ACTION")
        self.assertTrue(doc["needs_action"])
        self.assertEqual(doc["check"]["detail"], "verdict=STALLED")
        self.assertEqual(doc["recovery"]["reason"], "dry run")
        self.assertTrue(doc["recovery"]["skipped"])

    def test_loop_check_plan_apply_runs_manual_recovery(self) -> None:
        bad_script = self.root / "tools" / "bad_loop.py"
        recover_script = self.root / "tools" / "recover_loop.py"
        marker = self.root / "recovered.txt"
        bad_script.write_text("print('{\"ok\": false, \"reason\": \"needs restart\"}')\n", encoding="utf-8")
        recover_script.write_text(
            f"from pathlib import Path\nPath({str(marker)!r}).write_text('done', encoding='utf-8')\n",
            encoding="utf-8",
        )
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {
                "stalled": {
                    "status_cmd": [sys.executable, str(bad_script)],
                    "recover_cmd": [sys.executable, str(recover_script)],
                },
            },
        }

        doc = pane.loop_check_plan(config, "stalled", recover=True, apply=True)

        self.assertTrue(doc["ok"])
        self.assertEqual(doc["verdict"], "ACTION")
        self.assertFalse(doc["recovery"]["skipped"])
        self.assertEqual(doc["recovery"]["returncode"], 0)
        self.assertEqual(marker.read_text(encoding="utf-8"), "done")

    def test_loop_check_plan_reports_unknown_loop(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {"known": {"status_cmd": [sys.executable, "-c", "print('{}')"]}},
        }

        doc = pane.loop_check_plan(config, "missing")

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["reason"], "unknown loop")
        self.assertEqual(doc["available"], ["known"])

    def test_loop_audit_action_is_not_broken(self) -> None:
        # The core invariant: a loop that ran fine but surfaces a condition
        # (ok:false in its payload -> verdict ACTION) is the loop WORKING, so it
        # buckets as `action`, NOT `broken`, and does not fail the audit.
        surfacing = self.root / "tools" / "surfacing.py"
        surfacing.write_text("print('{\"ok\": false, \"reason\": \"closure_rate low\"}')\n", encoding="utf-8")
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "surfacer": {"status_cmd": [sys.executable, str(surfacing)]},
            "happy": {"status_cmd": [sys.executable, str(healthy)]},
        })

        doc = pane.loop_audit(config)

        self.assertEqual(doc["schema"], pane.LOOP_AUDIT_SCHEMA)
        self.assertTrue(doc["ok"])  # no broken loop
        self.assertEqual(doc["counts"], {"healthy": 1, "action": 1, "broken": 0, "total": 2})
        by_name = {loop["name"]: loop for loop in doc["loops"]}
        self.assertEqual(by_name["surfacer"]["bucket"], "action")
        self.assertEqual(by_name["happy"]["bucket"], "healthy")
        # detail is a single clean line lifted from the payload, not a blob.
        self.assertEqual(by_name["surfacer"]["detail"], "closure_rate low")
        self.assertNotIn("\n", by_name["surfacer"]["detail"])

    def test_loop_audit_broken_loop_fails_the_audit(self) -> None:
        # A loop whose status command cannot run at all (state UNAVAILABLE) is the
        # only real failure: it buckets `broken` and flips top-level ok to false.
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "happy": {"status_cmd": [sys.executable, str(healthy)]},
            "missing-cmd": {"status_cmd": ["this-binary-does-not-exist-xyz", "--json"]},
        })

        doc = pane.loop_audit(config)

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["counts"]["broken"], 1)
        by_name = {loop["name"]: loop for loop in doc["loops"]}
        self.assertEqual(by_name["missing-cmd"]["bucket"], "broken")
        self.assertEqual(by_name["missing-cmd"]["state"], "UNAVAILABLE")

    def test_loop_audit_names_subsets_and_reports_missing(self) -> None:
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "a": {"status_cmd": [sys.executable, str(healthy)]},
            "b": {"status_cmd": [sys.executable, str(healthy)]},
        })

        doc = pane.loop_audit(config, names=["a", "ghost"])

        self.assertEqual([loop["name"] for loop in doc["loops"]], ["a"])
        self.assertEqual(doc["counts"]["total"], 1)
        self.assertEqual(doc["missing"], ["ghost"])

    def test_loop_audit_skips_disabled_and_its_own_name(self) -> None:
        healthy = self.root / "tools" / "healthy.py"
        healthy.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        config = self._audit_config({
            "a": {"status_cmd": [sys.executable, str(healthy)]},
            "off": {"status_cmd": [sys.executable, str(healthy)], "enabled": False},
            pane.LOOP_AUDIT_SELF_NAME: {"status_cmd": [sys.executable, str(healthy)]},
        })

        doc = pane.loop_audit(config)

        names = {loop["name"] for loop in doc["loops"]}
        self.assertEqual(names, {"a"})  # disabled + self both excluded

    def test_loop_scaffold_plan_dry_run_prints_apply_command_without_writing(self) -> None:
        config = pane.load_config(self.root)

        doc = pane.loop_scaffold_plan(config, "nightly-build")

        self.assertTrue(doc["ok"])
        self.assertFalse(doc["apply"])
        self.assertFalse((self.root / "tools" / "loops" / "nightly-build_status.py").exists())
        self.assertFalse((self.root / "tools" / "control_pane.loops.json").exists())
        self.assertIn("loop-scaffold nightly-build --apply", doc["commands"][0])
        self.assertFalse(doc["spec"]["enabled"])
        self.assertEqual(doc["spec"]["status_cmd"], ["{python}", "tools/loops/nightly-build_status.py", "--json"])

    def test_loop_scaffold_plan_apply_writes_scripts_and_repo_catalog(self) -> None:
        config = pane.load_config(self.root)

        doc = pane.loop_scaffold_plan(config, "nightly-build", apply=True)

        status_path = self.root / "tools" / "loops" / "nightly-build_status.py"
        recover_path = self.root / "tools" / "loops" / "nightly-build_recover.py"
        catalog_path = self.root / "tools" / "control_pane.loops.json"
        self.assertTrue(doc["ok"])
        self.assertTrue(status_path.exists())
        self.assertTrue(recover_path.exists())
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
        loop = catalog["loops"]["nightly-build"]
        self.assertFalse(loop["enabled"])
        self.assertEqual(loop["status_cmd"], ["{python}", "tools/loops/nightly-build_status.py", "--json"])
        self.assertEqual(loop["recover_cmd"], ["{python}", "tools/loops/nightly-build_recover.py"])
        self.assertEqual(len(doc["followup_commands"]), 2)
        self.assertIn("loop-check nightly-build --recover", doc["followup_commands"][0])
        self.assertIn("--path tools/loops/nightly-build_status.py", doc["followup_commands"][1])
        self.assertIn("--path tools/control_pane.loops.json", doc["followup_commands"][1])

    def test_loop_scaffold_plan_refuses_existing_files_without_force(self) -> None:
        status_path = self.root / "tools" / "loops" / "nightly-build_status.py"
        status_path.parent.mkdir(parents=True)
        status_path.write_text("custom\n", encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_scaffold_plan(config, "nightly-build", apply=True)

        self.assertFalse(doc["ok"])
        self.assertIn("refusing to overwrite", doc["reason"])
        self.assertEqual(doc["commands"], [])
        self.assertEqual(status_path.read_text(encoding="utf-8"), "custom\n")

    def test_loop_set_plan_enables_repo_loop_without_retyping_commands(self) -> None:
        catalog = self.root / "tools" / "control_pane.loops.json"
        catalog.write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {
                "nightly-build": {
                    "enabled": False,
                    "status_cmd": ["{python}", "tools/loops/nightly-build_status.py", "--json"],
                    "recover_cmd": ["{python}", "tools/loops/nightly-build_recover.py"],
                    "auto_recover": False,
                },
            },
        }), encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_set_plan(config, "nightly-build", enabled=True, auto_recover=True, apply=True)

        written = json.loads(catalog.read_text(encoding="utf-8"))
        loop = written["loops"]["nightly-build"]
        self.assertTrue(doc["ok"])
        self.assertTrue(doc["changed"])
        self.assertTrue(loop["enabled"])
        self.assertTrue(loop["auto_recover"])
        self.assertEqual(loop["status_cmd"], ["{python}", "tools/loops/nightly-build_status.py", "--json"])
        self.assertEqual(len(doc["followup_commands"]), 2)
        self.assertIn("loop-check nightly-build --recover", doc["followup_commands"][0])
        self.assertIn("--path tools/control_pane.loops.json", doc["followup_commands"][1])

    def test_loop_set_plan_refuses_unknown_loop(self) -> None:
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "loops": {"known": {"status_cmd": ["known"]}},
        }

        doc = pane.loop_set_plan(config, "missing", enabled=True)

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["reason"], "loop is not known in this scope")
        self.assertEqual(doc["available"], ["known"])

    def test_loop_set_plan_refuses_without_setting(self) -> None:
        catalog = self.root / "tools" / "control_pane.loops.json"
        catalog.write_text(json.dumps({
            "schema": pane.LOOP_CATALOG_SCHEMA,
            "loops": {"known": {"status_cmd": ["known"]}},
        }), encoding="utf-8")
        config = pane.load_config(self.root)

        doc = pane.loop_set_plan(config, "known")

        self.assertFalse(doc["ok"])
        self.assertEqual(doc["reason"], "select at least one loop setting")
        self.assertEqual(doc["commands"], [])

    def test_collect_loops_classifies_status_and_recommendations(self) -> None:
        ok_script = self.root / "tools" / "ok_loop.py"
        bad_script = self.root / "tools" / "bad_loop.py"
        release_script = self.root / "tools" / "release_loop.py"
        ok_script.write_text("print('{\"ok\": true}')\n", encoding="utf-8")
        bad_script.write_text("print('{\"verdict\": \"STALLED\", \"detail\": \"heartbeat stale\"}')\n", encoding="utf-8")
        release_script.write_text(
            "print('{\"ok\": false, \"verdict\": \"ACTION\", \"detail\": \"billing needs attention\"}')\n",
            encoding="utf-8",
        )
        config = {
            **pane.normalize_config(pane.default_config(self.root), self.root),
            "_paths": {"local": str(self.root / "tools" / "_registry" / "control_pane.local.json")},
            "_local_exists": True,
            "loops": {
                "healthy": {"status_cmd": [sys.executable, str(ok_script)]},
                "stalled": {
                    "status_cmd": [sys.executable, str(bad_script)],
                    "action": "restart stalled loop",
                },
                "release-status": {"status_cmd": [sys.executable, str(release_script)]},
            },
        }

        loops = pane.collect_loops(config)

        self.assertEqual(loops["count"], 3)
        self.assertEqual(loops["configured"], 3)
        self.assertEqual(loops["enabled"], 3)
        self.assertEqual(loops["disabled"], 0)
        self.assertEqual(loops["states"]["OK"], 1)
        self.assertEqual(loops["states"]["ACTION"], 2)
        stalled = [check for check in loops["checks"] if check["name"] == "stalled"][0]
        self.assertEqual(stalled["detail"], "verdict=STALLED")
        release = [check for check in loops["checks"] if check["name"] == "release-status"][0]
        self.assertEqual(release["detail"], "billing needs attention")

        actions = pane.recommended_actions({
            "config": {"local_config_exists": True},
            "registry": {"exists": True},
            "control_tick": {"task": {}},
            "watchdogs": {},
            "loops": loops,
            "supervisor": {
                "available": True,
                "payload": {"verdict": "OK", "process": {"alive": True}, "diagnose": {"health": "OK"}},
            },
            "git": {"dirty_total": 0, "safe_ff": {"state": "in-sync", "ok": True}},
        }, config)
        self.assertTrue(any("Loop stalled is ACTION" in action for action in actions))
        self.assertTrue(any("restart stalled loop" in action for action in actions))
        self.assertTrue(any("Loop release-status is ACTION" in action for action in actions))
        self.assertTrue(any("billing needs attention" in action for action in actions))

    def test_repo_catalog_tracks_dos_watchdog_as_dry_run_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-watchdog"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_watchdog.py", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Dry-run only", spec["action"])

    def test_repo_catalog_tracks_dos_canary_audit_as_read_only_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-canary-audit"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_canary_audit.py", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Read-only productivity fold", spec["action"])

    def test_repo_catalog_tracks_dos_canary_as_dry_run_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-canary"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_canary.py", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Dry-run only", spec["action"])

    def test_repo_catalog_tracks_dos_canary_history_as_read_only_status_loop(self) -> None:
        catalog_path = Path(__file__).resolve().parent / "control_pane.loops.json"
        catalog = json.loads(catalog_path.read_text(encoding="utf-8"))

        spec = catalog["loops"]["dos-supervisor-canary-history"]

        self.assertTrue(spec["enabled"])
        self.assertEqual(spec["status_cmd"], ["{python}", "tools/dos_supervisor_canary.py", "--history", "--json"])
        self.assertFalse(spec["auto_recover"])
        self.assertNotIn("recover_cmd", spec)
        self.assertIn("Read-only canary ledger", spec["action"])

    def test_dos_watchdog_would_enact_payload_is_ok_loop_detail(self) -> None:
        proc = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=json.dumps({
                "schema": "fleet-dos-supervisor-watchdog/1",
                "ok": True,
                "action": "would_enact",
                "reason": "bounded canary available for 3 spawn lane(s)",
            }),
            stderr="",
        )

        state, payload, detail = pane.classify_loop_status(proc, {})

        self.assertEqual(state, "OK")
        self.assertEqual(payload["action"], "would_enact")
        self.assertEqual(detail, "bounded canary available for 3 spawn lane(s)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
