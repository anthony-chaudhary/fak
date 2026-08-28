#!/usr/bin/env python3
"""Focused, host-safe checks for the macOS native Grafana fallback."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path


HERE = Path(__file__).resolve().parent


class NativeFallbackTest(unittest.TestCase):
    def make_fixture(self) -> Path:
        root = Path(tempfile.mkdtemp(prefix="fak-grafana-native-"))
        self.addCleanup(shutil.rmtree, root, True)
        grafana = root / "tools" / "grafana"
        (grafana / "provisioning" / "datasources").mkdir(parents=True)
        (grafana / "dashboards").mkdir()
        for relative in (
            "up.sh",
            "down.sh",
            "prometheus.yml",
            "prometheus-alerts.yml",
            "provisioning/datasources/datasource.yml",
        ):
            destination = grafana / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(HERE / relative, destination)
        (grafana / "dashboards" / "fak-native-kernel-performance.json").write_text(
            "{}\n", encoding="utf-8"
        )
        return root

    def test_generated_native_configuration_is_loopback_only(self) -> None:
        root = self.make_fixture()
        script = root / "tools" / "grafana" / "up.sh"
        env = os.environ.copy()
        env["FAK_GRAFANA_NATIVE_CONFIG_ONLY"] = "1"
        subprocess.run(["bash", str(script)], check=True, env=env)

        native = root / "tools" / "grafana" / ".run" / "native"
        prometheus = (native / "prometheus.yml").read_text(encoding="utf-8")
        datasource = (
            native / "provisioning" / "datasources" / "datasource.yml"
        ).read_text(encoding="utf-8")
        dashboards = (
            native / "provisioning" / "dashboards" / "dashboards.yml"
        ).read_text(encoding="utf-8")

        self.assertNotIn("host.docker.internal", prometheus)
        self.assertNotIn("\nalerting:\n", prometheus)
        for port in (8080, 9095, 9097, 9098):
            self.assertIn(f'127.0.0.1:{port}', prometheus)
        self.assertIn(
            str(root / "tools" / "grafana" / "prometheus-alerts.yml"), prometheus
        )
        self.assertIn("url: http://127.0.0.1:9091", datasource)
        self.assertIn(str(root / "tools" / "grafana" / "dashboards"), dashboards)
        self.assertTrue(
            (root / "tools" / "grafana" / "dashboards"
             / "fak-native-kernel-performance.json").is_file()
        )

    def test_down_stops_only_process_with_matching_owner(self) -> None:
        root = self.make_fixture()
        run_dir = root / "tools" / "grafana" / ".run"
        run_dir.mkdir()
        (run_dir / "stack.mode").write_text("native\n", encoding="utf-8")

        owner = "native-test-owner"
        owned = subprocess.Popen(
            [
                "bash",
                "-c",
                (
                    'child=""; '
                    "stop_child() { "
                    '[ -z "$child" ] || kill "$child" 2>/dev/null || true; '
                    '[ -z "$child" ] || wait "$child" 2>/dev/null || true; '
                    "exit 0; }; "
                    "trap stop_child TERM INT HUP; "
                    '"$@" & child=$!; wait "$child"'
                ),
                f"fak-grafana-owner={owner}",
                "sleep",
                "30",
            ]
        )
        adopted = subprocess.Popen(["sleep", "30"])
        self.addCleanup(self.stop_process, owned)
        self.addCleanup(self.stop_process, adopted)
        (run_dir / "owned.pid").write_text(
            f"pid={owned.pid}\nowner={owner}\n", encoding="utf-8"
        )
        (run_dir / "adopted.pid").write_text(
            f"pid={adopted.pid}\nowner=not-the-owner\n", encoding="utf-8"
        )

        subprocess.run(
            ["bash", str(root / "tools" / "grafana" / "down.sh")], check=True
        )
        for _ in range(50):
            if owned.poll() is not None:
                break
            time.sleep(0.02)

        self.assertIsNotNone(owned.poll(), "owned supervisor was not stopped")
        self.assertIsNone(adopted.poll(), "adopted process must remain running")
        self.assertFalse((run_dir / "owned.pid").exists())
        self.assertFalse((run_dir / "adopted.pid").exists())

    def test_down_preserves_docker_compose_purge_behavior(self) -> None:
        root = self.make_fixture()
        grafana = root / "tools" / "grafana"
        run_dir = grafana / ".run"
        fake_bin = root / "fake-bin"
        run_dir.mkdir()
        fake_bin.mkdir()
        (run_dir / "stack.mode").write_text("docker\n", encoding="utf-8")
        calls = root / "docker.calls"
        docker = fake_bin / "docker"
        docker.write_text(
            (
                "#!/usr/bin/env bash\n"
                'if [ "${1:-}" = info ]; then exit 0; fi\n'
                f'printf \'%s\\n\' "$*" >>"{calls}"\n'
            ),
            encoding="utf-8",
        )
        docker.chmod(0o755)
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"

        subprocess.run(
            ["bash", str(grafana / "down.sh"), "--purge"], check=True, env=env
        )

        self.assertEqual(calls.read_text(encoding="utf-8"), "compose down -v\n")

    @staticmethod
    def stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
