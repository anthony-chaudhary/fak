#!/usr/bin/env python3
"""Focused, host-safe checks for the macOS native Grafana fallback."""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.request
import uuid
from pathlib import Path


HERE = Path(__file__).resolve().parent


class NativeFallbackTest(unittest.TestCase):
    def run_shell_function(
        self, name: str, setup: str, result: str
    ) -> subprocess.CompletedProcess[str]:
        script = (HERE / "up.sh").read_text(encoding="utf-8")
        start = script.index(f"{name}() {{")
        end = script.index("\n}\n", start) + 3
        return subprocess.run(
            ["bash", "-c", f"{script[start:end]}\n{setup}\n{name}\n{result}"],
            check=True,
            capture_output=True,
            text=True,
        )

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

    def test_native_gateway_serve_is_loopback_only(self) -> None:
        result = self.run_shell_function(
            "configure_gateway_addr",
            "STACK_MODE=native; unset FAK_GATEWAY_ADDR",
            'printf "%s\\n" "$GATEWAY_ADDR"',
        )

        self.assertEqual(result.stdout, "127.0.0.1:8080\n")

    def test_missing_docker_cli_skips_daemon_startup(self) -> None:
        result = self.run_shell_function(
            "select_stack_mode",
            (
                "command() { return 1; }; "
                "uname() { printf 'Darwin\\n'; }; "
                "ensure_docker_daemon() { printf 'called\\n' >&2; return 0; }; "
                "die() { printf 'died\\n' >&2; return 1; }; "
                "STACK_MODE=unset"
            ),
            'printf "%s\\n" "$STACK_MODE"',
        )

        self.assertEqual(result.stdout, "native\n")
        self.assertEqual(result.stderr, "")

    def test_present_docker_cli_preserves_daemon_path(self) -> None:
        result = self.run_shell_function(
            "select_stack_mode",
            (
                "command() { return 0; }; "
                "ensure_docker_daemon() { printf 'called\\n' >&2; return 0; }; "
                "STACK_MODE=unset"
            ),
            'printf "%s\\n" "$STACK_MODE"',
        )

        self.assertEqual(result.stdout, "docker\n")
        self.assertEqual(result.stderr, "called\n")

    def test_non_macos_owned_process_survives_launcher_exit_and_down_stops_it(
        self,
    ) -> None:
        root = self.make_fixture()
        run_dir = root / "tools" / "grafana" / ".run"
        run_dir.mkdir()
        (run_dir / "stack.mode").write_text("native\n", encoding="utf-8")
        script = (root / "tools" / "grafana" / "up.sh").read_text(
            encoding="utf-8"
        )
        start = script.index("start_bg() {")
        end = script.index("\n}\n", start) + 3
        owner = "native-persistence-owner"

        subprocess.run(
            [
                "bash",
                "-c",
                (
                    f"{script[start:end]}\n"
                    "port_live() { return 1; }\n"
                    "log() { :; }\n"
                    f"RUN_DIR={str(run_dir)!r}\n"
                    f"RUN_ID={owner!r}\n"
                    "uname() { printf 'Linux\\n'; }\n"
                    "start_bg owned '65534 /' sleep 30\n"
                ),
            ],
            check=True,
        )

        pid_file = run_dir / "owned.pid"
        metadata = dict(
            line.split("=", 1)
            for line in pid_file.read_text(encoding="utf-8").splitlines()
        )
        pid = int(metadata["pid"])
        self.addCleanup(self.stop_pid, pid)
        self.assertEqual(metadata["owner"], owner)
        command_line = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn(f"fak-grafana-owner={owner}", command_line)

        os.kill(pid, 1)
        time.sleep(0.1)
        os.kill(pid, 0)
        self.assertEqual(
            pid_file.read_text(encoding="utf-8"),
            f"pid={pid}\nowner={owner}\n",
        )
        command_line = subprocess.run(
            ["ps", "-p", str(pid), "-o", "command="],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        self.assertIn(f"fak-grafana-owner={owner}", command_line)

        subprocess.run(
            ["bash", str(root / "tools" / "grafana" / "down.sh")], check=True
        )
        for _ in range(50):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.02)
        else:
            self.fail("owned supervisor was not stopped by down.sh")
        self.assertFalse(pid_file.exists())

    def test_launchctl_submit_preserves_bash_c_argument_positions(self) -> None:
        root = self.make_fixture()
        run_dir = root / "tools" / "grafana" / ".run"
        run_dir.mkdir()
        script = (root / "tools" / "grafana" / "up.sh").read_text(
            encoding="utf-8"
        )
        start = script.index("start_bg() {")
        end = script.index("\n}\n", start) + 3
        owner = "launchctl-argv-owner"
        env = os.environ.copy()
        env.update({"RUN_DIR_UNDER_TEST": str(run_dir), "RUN_ID_UNDER_TEST": owner})

        subprocess.run(
            [
                "/bin/bash",
                "-c",
                (
                    f"{script[start:end]}\n"
                    "port_live() { return 1; }\n"
                    "log() { :; }\n"
                    "die() { printf '%s\\n' \"$*\" >&2; exit 1; }\n"
                    "uname() { printf 'Darwin\\n'; }\n"
                    "launchctl() {\n"
                    "  [ \"${1:-}\" = submit ] || return 0\n"
                    "  shift\n"
                    "  local executable= arg0\n"
                    "  while [ \"${1:-}\" != -- ]; do\n"
                    "    [ \"$1\" != -p ] || executable=\"$2\"\n"
                    "    shift 2\n"
                    "  done\n"
                    "  shift\n"
                    "  if [ -z \"$executable\" ]; then executable=\"$1\"; shift; fi\n"
                    "  arg0=\"$1\"\n"
                    "  shift\n"
                    "  (exec -a \"$arg0\" \"$executable\" \"$@\")\n"
                    "}\n"
                    "RUN_DIR=\"$RUN_DIR_UNDER_TEST\"\n"
                    "RUN_ID=\"$RUN_ID_UNDER_TEST\"\n"
                    "start_bg launchctl_argv '65534 /' /usr/bin/true\n"
                ),
            ],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )

        pid_file = run_dir / "launchctl_argv.pid"
        metadata = dict(
            line.split("=", 1)
            for line in pid_file.read_text(encoding="utf-8").splitlines()
        )
        self.assertEqual(metadata["owner"], owner)
        self.assertEqual(metadata["supervisor"], "launchd")
        self.assertEqual(
            metadata["label"], self.launchd_label("launchctl_argv", owner)
        )

    @unittest.skipUnless(sys.platform == "darwin", "requires real launchd")
    def test_real_up_down_launchd_lifecycle_and_adoption(self) -> None:
        try:
            with socket.socket() as probe:
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                probe.bind(("127.0.0.1", 8080))
        except OSError as error:
            self.skipTest(f"127.0.0.1:8080 is already in use: {error}")

        root = self.make_fixture()
        grafana = root / "tools" / "grafana"
        run_dir = grafana / ".run"
        fake_bin = root / "fake-bin"
        model_dir = root / "model"
        run_dir.mkdir()
        fake_bin.mkdir()
        model_dir.mkdir()
        (model_dir / "weights.f32").write_bytes(b"fixture")

        fake_fak = run_dir / "fak"
        fake_fak.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
[ "${1:-}" = serve ] || exit 64
shift
addr=127.0.0.1:8080
while [ "$#" -gt 0 ]; do
  if [ "$1" = --addr ]; then
    addr="$2"
    shift 2
  else
    shift
  fi
done
host="${addr%:*}"
port="${addr##*:}"
exec /usr/bin/python3 -c '
import http.server
import sys

class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok\\n")

    def log_message(self, _format, *_args):
        pass

http.server.ThreadingHTTPServer((sys.argv[1], int(sys.argv[2])), Handler).serve_forever()
' "$host" "$port" "${FAK_GRAFANA_TEST_TOKEN:-}"
""",
            encoding="utf-8",
        )
        fake_fak.chmod(0o755)

        docker = fake_bin / "docker"
        docker.write_text(
            """#!/usr/bin/env bash
case "${1:-}" in
  info) exit 0 ;;
  compose) exit 0 ;;
esac
exit 64
""",
            encoding="utf-8",
        )
        docker.chmod(0o755)

        curl = fake_bin / "curl"
        curl.write_text(
            """#!/usr/bin/env bash
for arg in "$@"; do
  case "$arg" in
    http://127.0.0.1:8080/*|http://localhost:8080/*)
      exec /usr/bin/curl "$@"
      ;;
  esac
done
exit 0
""",
            encoding="utf-8",
        )
        curl.chmod(0o755)

        token = os.environ.get("FAK_GRAFANA_TEST_TOKEN", uuid.uuid4().hex)
        owner = f"issue-9402-owned-{token}"
        adopted_owner = f"issue-9402-adopted-{token}"
        owned_label = self.launchd_label("fak_gateway", owner)
        adopted_label = self.launchd_label("fak_gateway", adopted_owner)
        self.addCleanup(self.remove_launchd_job, owned_label)
        self.addCleanup(self.remove_launchd_job, adopted_label)

        env = os.environ.copy()
        env.update(
            {
                "FAK_DOGFOOD_MODEL": "fixture",
                "FAK_GATEWAY_ADDR": "127.0.0.1:8080",
                "FAK_GRAFANA_RUN_ID": owner,
                "FAK_GRAFANA_TEST_TOKEN": token,
                "FAK_MODEL_DIR": str(model_dir),
                "FAK_TELEMETRY_SOURCES": "fak_gateway",
                "PATH": f"{fake_bin}:/usr/bin:/bin:/usr/sbin:/sbin",
            }
        )
        subprocess.run(
            ["/bin/bash", str(grafana / "up.sh")],
            check=True,
            env=env,
            capture_output=True,
            text=True,
            start_new_session=True,
        )

        pid_file = run_dir / "fak_gateway.pid"
        metadata_text = pid_file.read_text(encoding="utf-8")
        metadata = dict(
            line.split("=", 1) for line in metadata_text.splitlines()
        )
        wrapper_pid = int(metadata["pid"])
        self.assertEqual(metadata["owner"], owner)
        self.assertEqual(metadata["supervisor"], "launchd")
        self.assertEqual(metadata["label"], owned_label)
        self.assertLaunchdJobPresent(owned_label)
        self.wait_for_http("http://127.0.0.1:8080/metrics")
        child_pid = self.wait_for_child(wrapper_pid)
        os.kill(wrapper_pid, 0)
        os.kill(child_pid, 0)

        time.sleep(0.5)
        self.assertEqual(pid_file.read_text(encoding="utf-8"), metadata_text)
        self.assertLaunchdJobPresent(owned_label)
        os.kill(wrapper_pid, 0)
        os.kill(child_pid, 0)
        self.wait_for_http("http://127.0.0.1:8080/healthz")

        subprocess.run(
            ["bash", str(grafana / "down.sh")],
            check=True,
            env=env,
            capture_output=True,
            text=True,
        )
        self.wait_for_pid_exit(wrapper_pid)
        self.wait_for_pid_exit(child_pid)
        self.assertLaunchdJobAbsent(owned_label)
        self.assertFalse(pid_file.exists())

        adopted = subprocess.Popen(
            [
                str(fake_fak),
                "serve",
                "--addr",
                "127.0.0.1:8080",
                "--engine",
                "fixture",
                "--model",
                "fixture",
            ],
            env=env,
        )
        self.addCleanup(self.stop_process, adopted)
        self.wait_for_http("http://127.0.0.1:8080/metrics")

        adopted_env = env.copy()
        adopted_env["FAK_GRAFANA_RUN_ID"] = adopted_owner
        subprocess.run(
            ["bash", str(grafana / "up.sh")],
            check=True,
            env=adopted_env,
            capture_output=True,
            text=True,
        )
        self.assertIsNone(adopted.poll(), "adopted listener must remain alive")
        self.assertFalse(pid_file.exists())
        self.assertLaunchdJobAbsent(adopted_label)

        subprocess.run(
            ["bash", str(grafana / "down.sh")],
            check=True,
            env=adopted_env,
            capture_output=True,
            text=True,
        )
        self.assertIsNone(adopted.poll(), "down must not stop an adopted listener")
        self.wait_for_http("http://127.0.0.1:8080/metrics")

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
    def stop_pid(pid: int) -> None:
        try:
            os.kill(pid, 15)
        except ProcessLookupError:
            return

    @staticmethod
    def stop_process(process: subprocess.Popen[bytes]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)

    @staticmethod
    def launchd_label(name: str, owner: str) -> str:
        return f"com.fak.grafana.{os.getuid()}.{name.replace('_', '-')}.{owner}"

    @staticmethod
    def remove_launchd_job(label: str) -> None:
        subprocess.run(
            ["launchctl", "remove", label],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )

    def assertLaunchdJobPresent(self, label: str) -> None:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertEqual(result.returncode, 0, f"launchd job {label} is absent")

    def assertLaunchdJobAbsent(self, label: str) -> None:
        result = subprocess.run(
            ["launchctl", "print", f"gui/{os.getuid()}/{label}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        self.assertNotEqual(result.returncode, 0, f"launchd job {label} remains")

    def wait_for_child(self, parent_pid: int) -> int:
        for _ in range(100):
            result = subprocess.run(
                ["pgrep", "-P", str(parent_pid)],
                check=False,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                return int(result.stdout.splitlines()[0])
            time.sleep(0.05)
        self.fail(f"no child appeared for supervisor pid {parent_pid}")

    def wait_for_http(self, url: str) -> None:
        for _ in range(100):
            try:
                with urllib.request.urlopen(url, timeout=0.2) as response:
                    if response.status == 200:
                        return
            except OSError:
                time.sleep(0.05)
        self.fail(f"listener did not become healthy at {url}")

    def wait_for_pid_exit(self, pid: int) -> None:
        for _ in range(100):
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return
            time.sleep(0.05)
        self.fail(f"pid {pid} remained alive")


if __name__ == "__main__":
    unittest.main()
