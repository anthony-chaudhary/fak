#!/usr/bin/env python3
"""Core config, atomic JSON/file I/O, and subprocess helper tests for fleet_control_pane."""
from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fleet_control_pane as pane  # noqa: E402


class FleetControlPaneCoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        (self.root / "tools").mkdir()
        (self.root / "VERSION").write_text("test\n", encoding="utf-8")
        self.addCleanup(self._tmp.cleanup)

    def test_run_via_files_does_not_block_on_detached_grandchild(self) -> None:
        # Regression: a watchdog may relaunch a long-lived DETACHED daemon (the job
        # supervisor / a resumed claude) that inherits this process's stdout. With pipe
        # capture, run() blocks on EOF until that daemon also closes the pipe -- which
        # never happens, hanging the tick, and `timeout` cannot rescue it (the post-timeout
        # drain re-blocks on the same pipe). via_files must return as soon as the immediate
        # child exits, regardless of the still-living grandchild.
        import time

        child = (
            "import subprocess,sys;"
            "subprocess.Popen([sys.executable,'-c','import time;time.sleep(8)']);"
            "print('child-done')"
        )
        # cwd is the system temp (NOT self.root): the detached grandchild inherits the cwd
        # and holds it for 8s, which would otherwise block this test's tmpdir cleanup.
        cwd = Path(tempfile.gettempdir())
        t0 = time.monotonic()
        proc = pane.run([sys.executable, "-c", child], cwd, timeout=20, via_files=True)
        elapsed = time.monotonic() - t0
        self.assertEqual(proc.returncode, 0)
        self.assertIn("child-done", proc.stdout)
        self.assertLess(elapsed, 5.0, f"run() blocked on the detached grandchild ({elapsed:.1f}s)")

    def test_run_via_files_captures_stdout_and_stderr(self) -> None:
        prog = "import sys; print('out-line'); print('err-line', file=sys.stderr)"
        proc = pane.run([sys.executable, "-c", prog], self.root, timeout=20, via_files=True)
        self.assertEqual(proc.returncode, 0)
        self.assertIn("out-line", proc.stdout)
        self.assertIn("err-line", proc.stderr)

    def test_write_text_atomic_retries_windows_replace_lock(self) -> None:
        target = self.root / "tools" / "_registry" / "control_pane.json"
        calls: list[Path] = []
        real_replace = Path.replace

        def flaky_replace(src: Path, dst: Path) -> Path:
            calls.append(src)
            if len(calls) == 1:
                raise PermissionError("target locked")
            return real_replace(src, dst)

        with (
            mock.patch.object(Path, "replace", new=flaky_replace),
            mock.patch.object(pane.time, "sleep") as sleep_mock,
        ):
            pane.write_text_atomic(target, "ok\n")

        self.assertEqual(target.read_text(encoding="utf-8"), "ok\n")
        self.assertEqual(sleep_mock.call_count, 1)
        self.assertEqual(len(calls), 2)
        self.assertNotEqual(calls[0].name, "control_pane.json.tmp")
        self.assertFalse(target.with_name(target.name + ".tmp").exists())
        self.assertEqual(list(target.parent.glob(f".{target.name}.*.tmp")), [])

    def test_init_config_writes_ignored_local_config(self) -> None:
        with mock.patch.object(sys, "executable", "/py"):
            result = pane.init_config(self.root)

        self.assertTrue(result["written"])
        path = Path(result["path"])
        self.assertEqual(path, self.root / "tools" / "_registry" / "control_pane.local.json")
        doc = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(doc["schema"], pane.CONFIG_SCHEMA)
        self.assertIn("registry_dir", doc)

    def test_init_config_writes_machine_overrides(self) -> None:
        result = pane.init_config(
            self.root,
            overrides={
                "job_dir": "../job2",
                "machine_dir": "../shared/machines",
                "machine_id": "dgx-1",
                "target": 8,
                "session_window_h": 12,
            },
        )

        self.assertTrue(result["written"])
        doc = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["job_dir"], "../job2")
        self.assertEqual(doc["machine_dir"], "../shared/machines")
        self.assertEqual(doc["machine_id"], "dgx-1")
        self.assertEqual(doc["target"], 8)
        self.assertEqual(doc["session_window_h"], 12.0)

    def test_init_config_uses_effective_base_config_defaults(self) -> None:
        base = pane.normalize_config(
            {
                **pane.default_config(self.root),
                "machine_dir": "../shared/machines",
                "registry_dir": "../shared/registry",
                "machine_id": "dgx-1",
                "target": 8,
            },
            self.root,
        )

        result = pane.init_config(self.root, base_config=base)

        doc = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
        self.assertEqual(doc["machine_dir"], str((self.root / "../shared/machines").resolve()))
        self.assertEqual(doc["registry_dir"], str((self.root / "../shared/registry").resolve()))
        self.assertEqual(doc["machine_id"], "dgx-1")
        self.assertEqual(doc["target"], 8)

    def test_directory_probe_uses_unique_probe_file(self) -> None:
        target = self.root / "tools" / "_registry"
        target.mkdir(parents=True)
        (target / ".control-pane-write-test").mkdir()

        got = pane.directory_probe(target, create=False)

        self.assertTrue(got["ok"])
        self.assertTrue((target / ".control-pane-write-test").is_dir())

    def test_load_config_merges_example_local_and_env(self) -> None:
        example = {
            "schema": pane.CONFIG_SCHEMA,
            "target": 2,
            "job_dir": "../example-job",
            "registry_dir": "tools/_registry",
            "watchdogs": {"supervisor": {"task_name": "ExampleTask"}},
        }
        local = {
            "target": 7,
            "watchdogs": {"supervisor": {"interval_min": 3}},
        }
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps(example), encoding="utf-8")
        (self.root / "tools" / "_registry").mkdir()
        (self.root / "tools" / "_registry" / "control_pane.local.json").write_text(
            json.dumps(local), encoding="utf-8")

        with mock.patch.dict(os.environ, {"FLEET_JOB_DIR": str(self.root / "env-job")}):
            config = pane.load_config(self.root)

        self.assertEqual(config["target"], 7)
        self.assertEqual(config["job_dir"], str((self.root / "env-job").resolve()))
        self.assertEqual(config["watchdogs"]["supervisor"]["task_name"], "ExampleTask")
        self.assertEqual(config["watchdogs"]["supervisor"]["interval_min"], 3)
        self.assertTrue(config["_local_exists"])

    def test_load_config_merges_repo_loop_catalog_before_local_overrides(self) -> None:
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "loops": {
                    "example-loop": {
                        "enabled": False,
                        "status_cmd": ["example"],
                    },
                },
            }),
            encoding="utf-8",
        )
        (self.root / "tools" / "control_pane.loops.json").write_text(
            json.dumps({
                "schema": pane.LOOP_CATALOG_SCHEMA,
                "loops": {
                    "shared-loop": {
                        "enabled": True,
                        "status_cmd": ["catalog"],
                        "auto_recover": True,
                    },
                },
            }),
            encoding="utf-8",
        )
        (self.root / "tools" / "_registry").mkdir()
        (self.root / "tools" / "_registry" / "control_pane.local.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "loops": {
                    "shared-loop": {
                        "enabled": False,
                        "status_cmd": ["local"],
                    },
                },
            }),
            encoding="utf-8",
        )

        config = pane.load_config(self.root)

        self.assertEqual(config["loops"]["example-loop"]["status_cmd"], ["example"])
        self.assertFalse(config["loops"]["shared-loop"]["enabled"])
        self.assertEqual(config["loops"]["shared-loop"]["status_cmd"], ["local"])
        self.assertTrue(config["loops"]["shared-loop"]["auto_recover"])
        self.assertEqual(config["_paths"]["loop_catalog"], str(self.root / "tools" / "control_pane.loops.json"))

    def test_local_config_default_drift_reports_shadowed_tracked_defaults(self) -> None:
        (self.root / "tools" / "control_pane.example.json").write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "../shared/machines",
                "registry_dir": "../shared/registry",
                "target": 8,
            }),
            encoding="utf-8",
        )
        local = self.root / "tools" / "_registry" / "control_pane.local.json"
        local.parent.mkdir(parents=True)
        local.write_text(
            json.dumps({
                "schema": pane.CONFIG_SCHEMA,
                "machine_dir": "tools/_registry/machines",
                "registry_dir": "tools/_registry",
                "target": 4,
            }),
            encoding="utf-8",
        )
        config = pane.load_config(self.root)

        drift = pane.local_config_default_drift(config)

        by_key = {item["key"]: item for item in drift["items"]}
        self.assertEqual(drift["count"], 3)
        self.assertEqual(by_key["target"]["current"], 4)
        self.assertEqual(by_key["target"]["tracked"], 8)
        self.assertIn("--force", drift["command"])
        self.assertIn("--machine-dir", drift["command"])
        self.assertIn(str((self.root / "../shared/machines").resolve()), drift["command"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
