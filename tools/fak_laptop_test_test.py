#!/usr/bin/env python3
"""Hermetic tests for fak_laptop_test command selection."""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
from contextlib import redirect_stderr, redirect_stdout

sys.path.insert(0, str(Path(__file__).resolve().parent))

import fak_laptop_test as runner  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]


class LaptopRunnerTest(unittest.TestCase):
    def parse(self, *args: str):
        return runner.parse_args(list(args))

    def good_repo_report(self, head: str | None = None) -> dict:
        current = runner.repo_report(ROOT)
        return {
            "git_available": True,
            "head": head or current["head"],
            "branch": current["branch"],
            "dirty": False,
            "status_count": 0,
            "status_fingerprint": current["status_fingerprint"],
            "status_sample": current["status_sample"],
            "status_truncated": current["status_truncated"],
        }

    def good_check_report(self, proof_mode: str = "cpu+nvidia", cpu_scope: str = "smoke") -> dict:
        return {
            "schema": "fak.laptop-test.v1",
            "kind": "check",
            "host": runner.host_report(ROOT),
            "repo": self.good_repo_report(),
            "wsl": runner.wsl_report(self.parse("verify"), platform="win32"),
            "proof": {"mode": proof_mode, "stage": "postcheck", "cpu_scope": cpu_scope},
            "summary": {"ok": True, "proof_mode": proof_mode, "proof_stage": "postcheck"},
            "exit_code": 0,
            "checks": [
                {"name": "cpu", "ok": True, "detail": "go version go1.26.0 linux/amd64; GOOS=linux GOARCH=amd64"},
                {"name": "nvidia", "ok": True, "detail": "GPU 0: NVIDIA RTX"},
                {"name": "cuda-toolchain", "ok": True, "detail": "Cuda compilation tools"},
            ],
        }

    def good_run_report(self, proof_mode: str = "cpu+nvidia", cpu_scope: str = "smoke") -> dict:
        return {
            "schema": "fak.laptop-test.v1",
            "kind": "run",
            "host": runner.host_report(ROOT),
            "repo": self.good_repo_report(),
            "wsl": runner.wsl_report(self.parse("verify"), platform="win32"),
            "proof": {"mode": proof_mode, "stage": "run", "cpu_scope": cpu_scope},
            "summary": {"ok": True, "proof_mode": proof_mode, "proof_stage": "run", "cpu_scope": cpu_scope},
            "dry_run": False,
            "exit_code": 0,
            "commands": [
                {"label": "cpu", "skipped": False, "exit_code": 0},
                {"label": "nvidia-setup", "skipped": False, "exit_code": 0},
                {"label": "nvidia", "skipped": False, "exit_code": 0},
            ],
        }

    def write_json(self, path: Path, data: dict) -> None:
        path.write_text(json.dumps(data), encoding="utf-8")

    def test_host_report_uses_non_empty_fallbacks(self) -> None:
        with mock.patch.object(runner.host_platform, "node", return_value=""), \
             mock.patch.object(runner.host_platform, "system", return_value=""), \
             mock.patch.object(runner.host_platform, "release", return_value=""), \
             mock.patch.object(runner.host_platform, "machine", return_value=""):
            report = runner.host_report(ROOT)

        self.assertEqual(report["node"], "unknown")
        self.assertEqual(report["system"], "unknown")
        self.assertEqual(report["release"], "unknown")
        self.assertEqual(report["machine"], "unknown")
        self.assertEqual(report["repo_root"], str(ROOT))

    def test_repo_report_records_git_revision_and_dirty_state(self) -> None:
        calls = []

        def fake_git_output(root: Path, *args: str):
            calls.append(args)
            if args == ("rev-parse", "HEAD"):
                return True, "abc123"
            if args == ("branch", "--show-current"):
                return True, "master"
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return True, " M fak/GPU.md\n?? tools/fak_laptop_test.py"
            return False, ""

        with mock.patch.object(runner, "git_output", side_effect=fake_git_output):
            report = runner.repo_report(ROOT)

        self.assertEqual(report, {
            "git_available": True,
            "head": "abc123",
            "branch": "master",
            "dirty": True,
            "status_count": 2,
            "status_fingerprint": runner.hashlib.sha256(b" M fak/GPU.md\n?? tools/fak_laptop_test.py").hexdigest(),
            "status_sample": [" M fak/GPU.md", "?? tools/fak_laptop_test.py"],
            "status_truncated": False,
        })
        self.assertIn(("rev-parse", "HEAD"), calls)

    def test_repo_report_records_bounded_dirty_sample(self) -> None:
        status = "\n".join(f"?? generated-{idx}.txt" for idx in range(runner.STATUS_SAMPLE_LIMIT + 5))

        def fake_git_output(root: Path, *args: str):
            if args == ("rev-parse", "HEAD"):
                return True, "abc123"
            if args == ("branch", "--show-current"):
                return True, "master"
            if args == ("status", "--porcelain", "--untracked-files=all"):
                return True, status
            return False, ""

        with mock.patch.object(runner, "git_output", side_effect=fake_git_output):
            report = runner.repo_report(ROOT)

        self.assertEqual(report["status_count"], runner.STATUS_SAMPLE_LIMIT + 5)
        self.assertEqual(len(report["status_sample"]), runner.STATUS_SAMPLE_LIMIT)
        self.assertTrue(report["status_truncated"])

    def test_wsl_report_records_explicit_distro(self) -> None:
        args = self.parse("check", "--wsl-distro", "Ubuntu")

        self.assertEqual(runner.wsl_report(args, platform="win32"), {
            "platform": "win32",
            "source": "explicit",
            "distro": "Ubuntu",
        })

    def test_wsl_report_records_preferred_distro(self) -> None:
        args = self.parse("check")
        with mock.patch.object(runner, "list_wsl_distros", return_value=("Ubuntu", "Ubuntu-24.04")):
            report = runner.wsl_report(args, platform="win32")

        self.assertEqual(report, {
            "platform": "win32",
            "source": "preferred",
            "distro": "Ubuntu-24.04",
        })

    def test_wsl_report_records_default_distro(self) -> None:
        args = self.parse("check")
        with mock.patch.object(runner, "list_wsl_distros", return_value=("Ubuntu",)):
            report = runner.wsl_report(args, platform="win32")

        self.assertEqual(report, {
            "platform": "win32",
            "source": "default",
            "distro": "default",
        })

    def test_wsl_report_records_native_non_windows(self) -> None:
        args = self.parse("check")

        self.assertEqual(runner.wsl_report(args, platform="linux"), {
            "platform": "linux",
            "source": "native",
            "distro": "not-applicable",
        })

    def test_cpu_smoke_uses_existing_posix_test_runner(self) -> None:
        args = self.parse("--smoke", "cpu")
        cmd = runner.cpu_command(Path("/repo/fleet"), args, platform="linux")

        self.assertEqual(cmd.label, "cpu")
        self.assertEqual(cmd.argv[0], "bash")
        self.assertEqual(cmd.argv[1], "/repo/fleet/fak/test.sh")
        self.assertTrue(any("TestHALSessionMatchesLegacyCPUReference" in part for part in cmd.argv))
        self.assertIn("./internal/compute/", cmd.argv)
        self.assertIn("./internal/model/", cmd.argv)

    def test_cpu_passes_go_test_args_after_separator(self) -> None:
        args = self.parse("cpu", "--", "-run", "TestEvict", "./internal/model/")
        cmd = runner.cpu_command(Path("/repo/fleet"), args, platform="linux")

        self.assertEqual(cmd.argv[-3:], ("-run", "TestEvict", "./internal/model/"))

    def test_cpu_windows_delegates_to_powershell_wrapper(self) -> None:
        args = self.parse("--fast", "--wsl-distro", "Ubuntu", "cpu")
        cmd = runner.cpu_command(Path(r"C:\work\fleet"), args, platform="win32")

        self.assertEqual(cmd.argv[:5], ("powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"))
        self.assertTrue(str(cmd.argv[5]).replace("/", "\\").endswith(r"fak\test.ps1"))
        self.assertEqual(cmd.env["FAK_FAST"], "1")
        self.assertEqual(cmd.env["FAK_WSL_DISTRO"], "Ubuntu")

    def test_powershell_wrapper_is_present(self) -> None:
        wrapper = ROOT / "tools" / "fak_laptop_test.ps1"
        text = wrapper.read_text(encoding="utf-8")
        self.assertIn("fak_laptop_test.ps1 accept", text)
        self.assertIn("fak_laptop_test.ps1 accept --cpu-only", text)
        self.assertIn("fak_laptop_test.ps1 status", text)
        self.assertIn("fak_laptop_test.py", text)
        self.assertIn("ValueFromRemainingArguments", text)
        self.assertIn("function Get-WorkingPython", text)
        self.assertIn("--version", text)
        self.assertLess(text.index("Name = 'py'"), text.index("Name = 'python'"))
        self.assertIn("PrefixArgs = @('-3')", text)
        self.assertIn("exit $LASTEXITCODE", text)

    def test_cuda_setup_script_fails_when_probe_fails(self) -> None:
        setup = runner.module_dir(ROOT) / "internal" / "compute" / "setup_cuda_wsl.sh"
        text = setup.read_text(encoding="utf-8")
        self.assertIn("set -euo pipefail", text)
        self.assertIn('echo "SETUP_FAIL"', text)
        self.assertIn("exit 1", text)
        self.assertIn('echo "SETUP_OK"', text)

    def test_nvidia_windows_builds_wsl_cuda_test_command(self) -> None:
        args = self.parse("--setup", "--wsl-distro", "Ubuntu-24.04", "nvidia")
        plan = runner.nvidia_commands(Path(r"C:\work\fleet"), args, platform="win32")

        self.assertEqual([cmd.label for cmd in plan], ["nvidia-setup", "nvidia"])
        self.assertEqual(plan[0].argv[:3], ("wsl.exe", "-d", "Ubuntu-24.04"))
        self.assertIn("cd /mnt/c/work/fleet/fak", plan[0].argv[-1])
        self.assertIn("setup_cuda_wsl.sh", plan[0].argv[-1])
        self.assertIn("build_cuda.sh test", plan[1].argv[-1])

    def test_windows_prefers_ubuntu_2404_when_installed(self) -> None:
        args = self.parse("--setup", "nvidia")
        with mock.patch.object(runner, "list_wsl_distros", return_value=("Ubuntu", "Ubuntu-24.04")):
            plan = runner.nvidia_commands(Path(r"C:\work\fleet"), args, platform="win32")

        self.assertEqual(plan[0].argv[:3], ("wsl.exe", "-d", "Ubuntu-24.04"))

    def test_windows_uses_default_wsl_distro_when_ubuntu_2404_absent(self) -> None:
        args = self.parse("--setup", "nvidia")
        with mock.patch.object(runner, "list_wsl_distros", return_value=("Ubuntu",)):
            plan = runner.nvidia_commands(Path(r"C:\work\fleet"), args, platform="win32")

        self.assertEqual(plan[0].argv[0], "wsl.exe")
        self.assertNotEqual(plan[0].argv[1], "-d")

    def test_wsl_distro_list_strips_windows_nuls_and_bom(self) -> None:
        raw = "\ufeffU\x00b\x00u\x00n\x00t\x00u\x00-\x002\x004\x00.\x000\x004\x00\r\x00"

        self.assertEqual(runner.clean_wsl_distro_name(raw), "Ubuntu-24.04")

    def test_wsl_distro_list_decodes_utf16_output(self) -> None:
        raw = "Ubuntu-24.04\r\nDebian\r\n".encode("utf-16")

        self.assertEqual(runner.decode_wsl_list_output(raw).splitlines(), ["Ubuntu-24.04", "Debian"])

    def test_list_wsl_distros_handles_utf16_bytes_from_wsl_exe(self) -> None:
        completed = mock.Mock(
            stdout="Ubuntu-24.04\r\nDebian\r\n".encode("utf-16"),
            returncode=0,
        )

        runner.list_wsl_distros.cache_clear()
        try:
            with mock.patch("subprocess.run", return_value=completed):
                self.assertEqual(runner.list_wsl_distros(), ("Ubuntu-24.04", "Debian"))
        finally:
            runner.list_wsl_distros.cache_clear()

    def test_nvidia_posix_runs_from_fak_dir(self) -> None:
        args = self.parse("--nvidia-action", "build", "nvidia")
        plan = runner.nvidia_commands(Path("/repo/fleet"), args, platform="linux")

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0].cwd, Path("/repo/fleet/fak"))
        self.assertEqual(plan[0].argv, ("bash", "internal/compute/build_cuda.sh", "build"))

    def test_all_lane_orders_cpu_before_nvidia(self) -> None:
        args = self.parse("--smoke", "all")
        plan = runner.build_plan(Path("/repo/fleet"), args, platform="linux")

        self.assertEqual([cmd.label for cmd in plan], ["cpu", "nvidia"])

    def test_check_lane_reports_missing_nvidia_without_failing_by_default(self) -> None:
        args = self.parse("check")
        with mock.patch.object(runner, "run_probe", return_value=(True, "go version go1.26.0")):
            with redirect_stdout(io.StringIO()) as out:
                rc = runner.run_checks(ROOT, args, platform="darwin")
        self.assertEqual(rc, 0)
        self.assertIn("[check:cpu] OK", out.getvalue())
        self.assertIn("[check:nvidia] MISSING", out.getvalue())

    def test_check_lane_can_require_nvidia(self) -> None:
        args = self.parse("--require-nvidia", "check")
        with mock.patch.object(runner, "run_probe", return_value=(True, "go version go1.26.0")):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = runner.run_checks(ROOT, args, platform="darwin")
        self.assertEqual(rc, 1)

    def test_check_cpu_native_records_goos_goarch(self) -> None:
        args = self.parse("check")
        with mock.patch.object(runner, "run_probe", return_value=(True, "go version go1.26.0\nGOOS=linux GOARCH=amd64")) as probe:
            result = runner.check_cpu(ROOT, args, platform="linux")

        self.assertTrue(result.ok)
        self.assertIn("GOOS=linux GOARCH=amd64", result.detail)
        self.assertEqual(probe.call_args.args[0], ("bash", "-lc", runner.go_env_probe_script()))

    def test_check_cpu_windows_records_wsl_goos_goarch(self) -> None:
        args = self.parse("--wsl-distro", "Ubuntu", "check")
        with mock.patch.object(runner, "run_probe", return_value=(True, "go version go1.26.0\nGOOS=linux GOARCH=amd64")) as probe:
            result = runner.check_cpu(ROOT, args, platform="win32")

        self.assertTrue(result.ok)
        self.assertIn("WSL Go available", result.detail)
        self.assertIn("GOOS=linux GOARCH=amd64", result.detail)
        argv = probe.call_args.args[0]
        self.assertEqual(argv[:3], ("wsl.exe", "-d", "Ubuntu"))
        self.assertIn("go env GOOS", argv[-1])
        self.assertIn("go env GOARCH", argv[-1])

    def test_check_nvidia_windows_probes_wsl_gpu_passthrough(self) -> None:
        args = self.parse("--wsl-distro", "Ubuntu", "check")
        with mock.patch.object(runner, "run_probe", return_value=(True, "GPU 0: NVIDIA RTX")) as probe:
            result = runner.check_nvidia(Path(r"C:\work\fleet"), args, platform="win32")
        self.assertTrue(result.ok)
        argv = probe.call_args.args[0]
        self.assertEqual(argv[:3], ("wsl.exe", "-d", "Ubuntu"))
        self.assertIn("nvidia-smi -L", argv[-1])
        self.assertIn("libcuda.so", argv[-1])
        self.assertNotIn("nvcc --version", argv[-1])

    def test_check_cuda_toolchain_windows_probes_nvcc(self) -> None:
        args = self.parse("--wsl-distro", "Ubuntu", "check")
        with mock.patch.object(runner, "run_probe", return_value=(True, "Cuda compilation tools")) as probe:
            result = runner.check_cuda_toolchain(Path(r"C:\work\fleet"), args, platform="win32")
        self.assertTrue(result.ok)
        argv = probe.call_args.args[0]
        self.assertEqual(argv[:3], ("wsl.exe", "-d", "Ubuntu"))
        self.assertIn("nvcc --version", argv[-1])

    def test_require_nvidia_does_not_require_nvcc(self) -> None:
        args = self.parse("--require-nvidia", "check")
        checks = {
            "cpu": runner.CheckResult("cpu", True, "go ok"),
            "nvidia": runner.CheckResult("nvidia", True, "gpu ok"),
            "cuda-toolchain": runner.CheckResult("cuda-toolchain", False, "no nvcc"),
        }
        with mock.patch.object(runner, "check_cpu", return_value=checks["cpu"]), \
             mock.patch.object(runner, "check_nvidia", return_value=checks["nvidia"]), \
             mock.patch.object(runner, "check_cuda_toolchain", return_value=checks["cuda-toolchain"]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = runner.run_checks(ROOT, args, platform="win32")
        self.assertEqual(rc, 0)

    def test_require_cuda_toolchain_can_fail_independently(self) -> None:
        args = self.parse("--require-cuda-toolchain", "check")
        checks = {
            "cpu": runner.CheckResult("cpu", True, "go ok"),
            "nvidia": runner.CheckResult("nvidia", True, "gpu ok"),
            "cuda-toolchain": runner.CheckResult("cuda-toolchain", False, "no nvcc"),
        }
        with mock.patch.object(runner, "check_cpu", return_value=checks["cpu"]), \
             mock.patch.object(runner, "check_nvidia", return_value=checks["nvidia"]), \
             mock.patch.object(runner, "check_cuda_toolchain", return_value=checks["cuda-toolchain"]):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                rc = runner.run_checks(ROOT, args, platform="win32")
        self.assertEqual(rc, 1)

    def test_check_lane_writes_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "check.json"
            args = self.parse("--require-nvidia", "--out", str(out_path), "check")
            checks = {
                "cpu": runner.CheckResult("cpu", True, "go ok"),
                "nvidia": runner.CheckResult("nvidia", True, "gpu ok"),
                "cuda-toolchain": runner.CheckResult("cuda-toolchain", False, "no nvcc"),
            }
            with mock.patch.object(runner, "check_cpu", return_value=checks["cpu"]), \
                 mock.patch.object(runner, "check_nvidia", return_value=checks["nvidia"]), \
                 mock.patch.object(runner, "check_cuda_toolchain", return_value=checks["cuda-toolchain"]):
                with redirect_stdout(io.StringIO()):
                    rc = runner.run_checks(ROOT, args, platform="win32")

            self.assertEqual(rc, 0)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "fak.laptop-test.v1")
            self.assertEqual(report["kind"], "check")
            self.assertEqual(report["host"]["repo_root"], str(ROOT))
            self.assertTrue(report["host"]["python"])
            self.assertTrue(report["repo"]["git_available"])
            self.assertTrue(report["repo"]["head"])
            self.assertIsInstance(report["repo"]["dirty"], bool)
            self.assertIsInstance(report["repo"]["status_count"], int)
            self.assertIsInstance(report["repo"]["status_fingerprint"], str)
            self.assertTrue(report["repo"]["status_fingerprint"])
            self.assertIsInstance(report["repo"]["status_sample"], list)
            self.assertIsInstance(report["repo"]["status_truncated"], bool)
            self.assertEqual(report["wsl"]["platform"], "win32")
            self.assertTrue(report["wsl"]["distro"])
            self.assertEqual(report["proof"]["mode"], "cpu+nvidia")
            self.assertEqual(report["proof"]["stage"], "precheck")
            self.assertEqual(report["summary"]["ok"], True)
            self.assertEqual(report["summary"]["proof_mode"], "cpu+nvidia")
            self.assertEqual(report["summary"]["proof_stage"], "precheck")
            self.assertEqual(report["summary"]["required"], ["cpu", "nvidia"])
            self.assertEqual(report["summary"]["failed_required"], [])
            self.assertIn("cuda-toolchain", report["summary"]["missing_checks"])
            self.assertEqual(report["exit_code"], 0)
            self.assertEqual(report["required"], ["cpu", "nvidia"])
            by_name = {c["name"]: c for c in report["checks"]}
            self.assertTrue(by_name["nvidia"]["required"])
            self.assertFalse(by_name["cuda-toolchain"]["required"])

    def test_check_lane_writes_relative_report_under_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            other = Path(td) / "other"
            repo.mkdir()
            other.mkdir()
            rel_report = "fak/experiments/gpu/laptop-check.json"
            args = self.parse("--out", rel_report, "check")
            checks = {
                "cpu": runner.CheckResult("cpu", True, "go ok"),
                "nvidia": runner.CheckResult("nvidia", False, "no gpu"),
                "cuda-toolchain": runner.CheckResult("cuda-toolchain", False, "no nvcc"),
            }
            old_cwd = Path.cwd()
            try:
                os.chdir(other)
                with mock.patch.object(runner, "check_cpu", return_value=checks["cpu"]), \
                     mock.patch.object(runner, "check_nvidia", return_value=checks["nvidia"]), \
                     mock.patch.object(runner, "check_cuda_toolchain", return_value=checks["cuda-toolchain"]):
                    with redirect_stdout(io.StringIO()):
                        rc = runner.run_checks(repo, args, platform="win32")
            finally:
                os.chdir(old_cwd)

            self.assertEqual(rc, 0)
            self.assertTrue((repo / rel_report).exists())
            self.assertFalse((other / rel_report).exists())

    def test_documented_check_command_accepts_options_after_lane(self) -> None:
        args = self.parse("check", "--require-nvidia", "--out", r"fak\experiments\gpu\laptop-check.json")

        self.assertEqual(args.lane, "check")
        self.assertTrue(args.require_nvidia)
        self.assertEqual(args.out, r"fak\experiments\gpu\laptop-check.json")
        self.assertEqual(runner.proof_report(args, "check")["mode"], "cpu+nvidia")
        self.assertEqual(runner.proof_report(args, "check")["stage"], "precheck")

    def test_documented_post_setup_check_writes_verify_compatible_proof_metadata(self) -> None:
        args = self.parse(
            "check",
            "--require-nvidia",
            "--require-cuda-toolchain",
            "--out",
            r"fak\experiments\gpu\laptop-post-setup.json",
        )

        self.assertEqual(runner.proof_report(args, "check"), {
            "mode": "cpu+nvidia",
            "stage": "postcheck",
        })

    def test_documented_cpu_full_suite_forwards_args_after_separator(self) -> None:
        args = self.parse("cpu", "--", "./...")
        cmd = runner.cpu_command(Path(r"C:\work\fleet"), args, platform="win32")

        self.assertEqual(cmd.argv[:5], ("powershell.exe", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File"))
        self.assertEqual(cmd.argv[-1], "./...")

    def test_documented_all_smoke_setup_command_shape(self) -> None:
        args = self.parse("all", "--smoke", "--setup", "--out", r"fak\experiments\gpu\laptop-all.json")
        with mock.patch.object(runner, "list_wsl_distros", return_value=("Ubuntu-24.04",)):
            plan = runner.build_plan(Path(r"C:\work\fleet"), args, platform="win32")

        self.assertEqual([cmd.label for cmd in plan], ["cpu", "nvidia-setup", "nvidia"])
        self.assertTrue(any("TestHALSessionMatchesLegacyCPUReference" in part for part in plan[0].argv))
        self.assertEqual(plan[1].argv[:3], ("wsl.exe", "-d", "Ubuntu-24.04"))
        self.assertIn("setup_cuda_wsl.sh", plan[1].argv[-1])
        self.assertIn("build_cuda.sh test", plan[2].argv[-1])
        self.assertEqual(runner.proof_report(args, "run"), {
            "cpu_scope": "smoke",
            "mode": "cpu+nvidia",
            "stage": "run",
        })

    def test_documented_cpu_only_run_report_writes_verify_compatible_proof_metadata(self) -> None:
        args = self.parse("cpu", "--smoke", "--out", r"fak\experiments\gpu\laptop-cpu.json")

        self.assertEqual(runner.proof_report(args, "run"), {
            "cpu_scope": "smoke",
            "mode": "cpu-only",
            "stage": "cpu-run",
        })

    def test_documented_cpu_only_full_run_report_writes_verify_compatible_proof_metadata(self) -> None:
        args = self.parse("cpu", "--out", r"fak\experiments\gpu\laptop-cpu.json")

        self.assertEqual(runner.cpu_args(args), ("./...",))
        self.assertEqual(runner.proof_report(args, "run"), {
            "cpu_scope": "full",
            "mode": "cpu-only",
            "stage": "cpu-run",
        })

    def test_documented_cpu_only_forwarded_full_suite_counts_as_full_scope(self) -> None:
        args = self.parse("cpu", "--out", r"fak\experiments\gpu\laptop-cpu.json", "--", "./...")

        self.assertEqual(runner.cpu_args(args), ("./...",))
        self.assertEqual(runner.proof_report(args, "run"), {
            "cpu_scope": "full",
            "mode": "cpu-only",
            "stage": "cpu-run",
        })

    def test_custom_cpu_report_scope_stays_custom(self) -> None:
        args = self.parse("cpu", "--out", r"fak\experiments\gpu\laptop-cpu.json", "--", "./internal/model/")

        self.assertEqual(runner.cpu_args(args), ("./internal/model/",))
        self.assertEqual(runner.proof_report(args, "run"), {
            "cpu_scope": "custom",
            "mode": "cpu-only",
            "stage": "cpu-run",
        })

    def test_documented_verify_defaults_to_post_setup_and_all_reports(self) -> None:
        args = self.parse("verify")

        self.assertEqual(args.check_report, runner.DEFAULT_VERIFY_CHECK_REPORT)
        self.assertEqual(args.run_report, runner.DEFAULT_VERIFY_RUN_REPORT)
        self.assertIn("laptop-post-setup.json", args.check_report)
        self.assertIn("laptop-all.json", args.run_report)

    def test_documented_cpu_only_verify_defaults_to_cpu_reports(self) -> None:
        args = self.parse("verify", "--cpu-only")

        self.assertEqual(args.check_report, runner.DEFAULT_CPU_CHECK_REPORT)
        self.assertEqual(args.run_report, runner.DEFAULT_CPU_RUN_REPORT)
        self.assertIn("laptop-cpu-check.json", args.check_report)
        self.assertIn("laptop-cpu.json", args.run_report)

    def test_status_defaults_to_verify_report_paths(self) -> None:
        args = self.parse("status")

        self.assertEqual(args.check_report, runner.DEFAULT_VERIFY_CHECK_REPORT)
        self.assertEqual(args.run_report, runner.DEFAULT_VERIFY_RUN_REPORT)

    def test_status_cpu_only_defaults_to_cpu_report_paths(self) -> None:
        args = self.parse("status", "--cpu-only")

        self.assertEqual(args.check_report, runner.DEFAULT_CPU_CHECK_REPORT)
        self.assertEqual(args.run_report, runner.DEFAULT_CPU_RUN_REPORT)

    def test_accept_defaults_to_laptop_report_paths(self) -> None:
        args = self.parse("accept")

        self.assertEqual(args.precheck_report, runner.DEFAULT_PRECHECK_REPORT)
        self.assertEqual(args.check_report, runner.DEFAULT_VERIFY_CHECK_REPORT)
        self.assertEqual(args.run_report, runner.DEFAULT_VERIFY_RUN_REPORT)
        self.assertFalse(args.full_cpu)
        self.assertIn("laptop-check.json", args.precheck_report)

    def test_accept_cpu_only_defaults_to_cpu_report_paths(self) -> None:
        args = self.parse("accept", "--cpu-only")

        self.assertEqual(args.precheck_report, runner.DEFAULT_PRECHECK_REPORT)
        self.assertEqual(args.check_report, runner.DEFAULT_CPU_CHECK_REPORT)
        self.assertEqual(args.run_report, runner.DEFAULT_CPU_RUN_REPORT)
        self.assertTrue(args.cpu_only)
        self.assertFalse(args.full_cpu)

    def test_accept_lane_runs_precheck_all_postcheck_verify_in_order(self) -> None:
        args = self.parse("accept", "--wsl-distro", "Ubuntu-24.04")
        plan = [runner.Command("probe", (sys.executable, "-c", "pass"), ROOT, os.environ.copy())]
        events: list[tuple[str, object]] = []

        def fake_run_checks(root: Path, check_args, platform=None):
            events.append((
                "check",
                check_args.lane,
                check_args.require_nvidia,
                check_args.require_cuda_toolchain,
                check_args.proof_mode,
                check_args.proof_stage,
                check_args.cpu_scope,
                check_args.out,
                platform,
            ))
            return 0

        def fake_build_plan(root: Path, run_args, platform=None):
            events.append((
                "build_plan",
                run_args.lane,
                run_args.smoke,
                run_args.setup,
                run_args.proof_mode,
                run_args.proof_stage,
                run_args.cpu_scope,
                run_args.out,
                platform,
            ))
            return plan

        def fake_run_plan(run_plan, dry_run: bool, platform=None, args=None):
            events.append(("run_plan", [cmd.label for cmd in run_plan], dry_run, args.lane, args.out, platform))
            return 0

        def fake_run_verify(root: Path, verify_args):
            events.append(("verify", verify_args.check_report, verify_args.run_report))
            return 0

        with mock.patch.object(runner, "run_checks", side_effect=fake_run_checks), \
             mock.patch.object(runner, "build_plan", side_effect=fake_build_plan), \
             mock.patch.object(runner, "run_plan", side_effect=fake_run_plan), \
             mock.patch.object(runner, "run_verify", side_effect=fake_run_verify):
            with redirect_stdout(io.StringIO()) as stdout:
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 0)
        self.assertIn("[accept] OK laptop hardware proof verified", stdout.getvalue())
        self.assertIn("laptop-check.json", stdout.getvalue())
        self.assertIn("laptop-all.json", stdout.getvalue())
        self.assertIn("laptop-post-setup.json", stdout.getvalue())
        self.assertEqual(events, [
            ("check", "check", True, False, "cpu+nvidia", "precheck", "smoke", runner.DEFAULT_PRECHECK_REPORT, "win32"),
            ("build_plan", "all", True, True, "cpu+nvidia", "run", "smoke", runner.DEFAULT_VERIFY_RUN_REPORT, "win32"),
            ("run_plan", ["probe"], False, "all", runner.DEFAULT_VERIFY_RUN_REPORT, "win32"),
            ("check", "check", True, True, "cpu+nvidia", "postcheck", "smoke", runner.DEFAULT_VERIFY_CHECK_REPORT, "win32"),
            ("verify", runner.DEFAULT_VERIFY_CHECK_REPORT, runner.DEFAULT_VERIFY_RUN_REPORT),
        ])

    def test_accept_full_cpu_runs_all_cpu_packages(self) -> None:
        args = self.parse("accept", "--full-cpu")
        captured = []

        def fake_run_checks(root: Path, check_args, platform=None):
            return 0

        def fake_build_plan(root: Path, run_args, platform=None):
            captured.append((run_args.smoke, runner.cpu_args(run_args)))
            return [runner.Command("probe", (sys.executable, "-c", "pass"), ROOT, os.environ.copy())]

        with mock.patch.object(runner, "run_checks", side_effect=fake_run_checks), \
             mock.patch.object(runner, "build_plan", side_effect=fake_build_plan), \
             mock.patch.object(runner, "run_plan", return_value=0), \
             mock.patch.object(runner, "run_verify", return_value=0):
            with redirect_stdout(io.StringIO()):
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 0)
        self.assertEqual(captured, [(False, ("./...",))])

    def test_accept_cpu_only_runs_check_cpu_verify_in_order(self) -> None:
        args = self.parse("accept", "--cpu-only", "--wsl-distro", "Ubuntu-24.04")
        plan = [runner.Command("cpu", (sys.executable, "-c", "pass"), ROOT, os.environ.copy())]
        events: list[tuple[str, object]] = []

        def fake_run_checks(root: Path, check_args, platform=None):
            events.append((
                "check",
                check_args.lane,
                check_args.require_nvidia,
                check_args.require_cuda_toolchain,
                check_args.proof_mode,
                check_args.proof_stage,
                check_args.cpu_scope,
                check_args.out,
                platform,
            ))
            return 0

        def fake_build_plan(root: Path, run_args, platform=None):
            events.append((
                "build_plan",
                run_args.lane,
                run_args.smoke,
                run_args.setup,
                run_args.proof_mode,
                run_args.proof_stage,
                run_args.cpu_scope,
                run_args.out,
                platform,
            ))
            return plan

        def fake_run_plan(run_plan, dry_run: bool, platform=None, args=None):
            events.append(("run_plan", [cmd.label for cmd in run_plan], dry_run, args.lane, args.out, platform))
            return 0

        def fake_run_verify(root: Path, verify_args):
            events.append(("verify", verify_args.cpu_only, verify_args.check_report, verify_args.run_report))
            return 0

        with mock.patch.object(runner, "run_checks", side_effect=fake_run_checks), \
             mock.patch.object(runner, "build_plan", side_effect=fake_build_plan), \
             mock.patch.object(runner, "run_plan", side_effect=fake_run_plan), \
             mock.patch.object(runner, "run_verify", side_effect=fake_run_verify):
            with redirect_stdout(io.StringIO()) as stdout:
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 0)
        self.assertIn("[accept] OK laptop CPU/Intel proof verified", stdout.getvalue())
        self.assertEqual(events, [
            ("check", "check", False, False, "cpu-only", "cpu-check", "smoke", runner.DEFAULT_CPU_CHECK_REPORT, "win32"),
            ("build_plan", "cpu", True, False, "cpu-only", "cpu-run", "smoke", runner.DEFAULT_CPU_RUN_REPORT, "win32"),
            ("run_plan", ["cpu"], False, "cpu", runner.DEFAULT_CPU_RUN_REPORT, "win32"),
            ("verify", True, runner.DEFAULT_CPU_CHECK_REPORT, runner.DEFAULT_CPU_RUN_REPORT),
        ])

    def test_accept_cpu_only_full_cpu_runs_all_cpu_packages(self) -> None:
        args = self.parse("accept", "--cpu-only", "--full-cpu")
        captured = []

        def fake_build_plan(root: Path, run_args, platform=None):
            captured.append((run_args.smoke, runner.cpu_args(run_args)))
            return [runner.Command("cpu", (sys.executable, "-c", "pass"), ROOT, os.environ.copy())]

        with mock.patch.object(runner, "run_checks", return_value=0), \
             mock.patch.object(runner, "build_plan", side_effect=fake_build_plan), \
             mock.patch.object(runner, "run_plan", return_value=0), \
             mock.patch.object(runner, "run_verify", return_value=0):
            with redirect_stdout(io.StringIO()):
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 0)
        self.assertEqual(captured, [(False, ("./...",))])

    def test_accept_lane_rejects_dry_run(self) -> None:
        args = self.parse("accept", "--dry-run")

        with self.assertRaises(SystemExit):
            runner.run_accept(ROOT, args, platform="win32")

    def test_accept_lane_stops_after_precheck_failure(self) -> None:
        args = self.parse("accept")

        with mock.patch.object(runner, "run_checks", return_value=7) as checks, \
             mock.patch.object(runner, "build_plan") as build_plan, \
             mock.patch.object(runner, "run_plan") as run_plan, \
             mock.patch.object(runner, "run_verify") as verify:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 7)
        self.assertIn("phase=preflight exit_code=7", stderr.getvalue())
        checks.assert_called_once()
        build_plan.assert_not_called()
        run_plan.assert_not_called()
        verify.assert_not_called()

    def test_accept_lane_stops_after_run_failure(self) -> None:
        args = self.parse("accept")
        plan = [runner.Command("probe", (sys.executable, "-c", "pass"), ROOT, os.environ.copy())]

        with mock.patch.object(runner, "run_checks", return_value=0) as checks, \
             mock.patch.object(runner, "build_plan", return_value=plan), \
             mock.patch.object(runner, "run_plan", return_value=9), \
             mock.patch.object(runner, "run_verify") as verify:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 9)
        self.assertIn("phase=run exit_code=9", stderr.getvalue())
        checks.assert_called_once()
        verify.assert_not_called()

    def test_accept_lane_stops_after_postcheck_failure(self) -> None:
        args = self.parse("accept")
        plan = [runner.Command("probe", (sys.executable, "-c", "pass"), ROOT, os.environ.copy())]
        check_results = [0, 11]

        def fake_run_checks(root: Path, check_args, platform=None):
            return check_results.pop(0)

        with mock.patch.object(runner, "run_checks", side_effect=fake_run_checks) as checks, \
             mock.patch.object(runner, "build_plan", return_value=plan), \
             mock.patch.object(runner, "run_plan", return_value=0), \
             mock.patch.object(runner, "run_verify") as verify:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 11)
        self.assertIn("phase=post-setup exit_code=11", stderr.getvalue())
        self.assertEqual(checks.call_count, 2)
        verify.assert_not_called()

    def test_accept_lane_stops_after_verify_failure(self) -> None:
        args = self.parse("accept")
        plan = [runner.Command("probe", (sys.executable, "-c", "pass"), ROOT, os.environ.copy())]

        with mock.patch.object(runner, "run_checks", return_value=0), \
             mock.patch.object(runner, "build_plan", return_value=plan), \
             mock.patch.object(runner, "run_plan", return_value=0), \
             mock.patch.object(runner, "run_verify", return_value=13):
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_accept(ROOT, args, platform="win32")

        self.assertEqual(rc, 13)
        self.assertIn("phase=verify exit_code=13", stderr.getvalue())

    def test_run_lane_writes_dry_run_json_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "run.json"
            args = self.parse("--smoke", "--out", str(out_path), "all", "--setup", "--dry-run")
            plan = runner.build_plan(Path("/repo/fleet"), args, platform="linux")
            with redirect_stdout(io.StringIO()):
                rc = runner.run_plan(plan, dry_run=True, platform="linux", args=args)

            self.assertEqual(rc, 0)
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["schema"], "fak.laptop-test.v1")
            self.assertEqual(report["kind"], "run")
            self.assertEqual(report["host"]["repo_root"], str(ROOT))
            self.assertTrue(report["host"]["machine"])
            self.assertTrue(report["repo"]["git_available"])
            self.assertTrue(report["repo"]["head"])
            self.assertIsInstance(report["repo"]["dirty"], bool)
            self.assertIsInstance(report["repo"]["status_count"], int)
            self.assertIsInstance(report["repo"]["status_fingerprint"], str)
            self.assertTrue(report["repo"]["status_fingerprint"])
            self.assertIsInstance(report["repo"]["status_sample"], list)
            self.assertIsInstance(report["repo"]["status_truncated"], bool)
            self.assertEqual(report["wsl"]["platform"], "linux")
            self.assertEqual(report["wsl"]["source"], "native")
            self.assertEqual(report["proof"], {"cpu_scope": "smoke", "mode": "cpu+nvidia", "stage": "run"})
            self.assertEqual(report["summary"]["ok"], True)
            self.assertEqual(report["summary"]["proof_mode"], "cpu+nvidia")
            self.assertEqual(report["summary"]["proof_stage"], "run")
            self.assertEqual(report["summary"]["cpu_scope"], "smoke")
            self.assertEqual(report["summary"]["command_count"], 3)
            self.assertEqual(report["summary"]["failed_commands"], [])
            self.assertEqual(report["summary"]["skipped_commands"], ["cpu", "nvidia", "nvidia-setup"])
            self.assertEqual(report["lane"], "all")
            self.assertTrue(report["dry_run"])
            self.assertEqual(report["exit_code"], 0)
            self.assertEqual([c["label"] for c in report["commands"]], ["cpu", "nvidia-setup", "nvidia"])
            self.assertTrue(all(c["skipped"] for c in report["commands"]))

    def test_run_lane_writes_relative_report_under_repo_root(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            repo = Path(td) / "repo"
            other = Path(td) / "other"
            repo.mkdir()
            other.mkdir()
            rel_report = "fak/experiments/gpu/laptop-all.json"
            args = self.parse("--out", rel_report, "cpu", "--dry-run")
            plan = [runner.Command("cpu", ("go", "test", "./..."), repo, os.environ.copy())]
            old_cwd = Path.cwd()
            try:
                os.chdir(other)
                with mock.patch.object(runner, "repo_root", return_value=repo):
                    with redirect_stdout(io.StringIO()):
                        rc = runner.run_plan(plan, dry_run=True, platform="linux", args=args)
            finally:
                os.chdir(old_cwd)

            self.assertEqual(rc, 0)
            self.assertTrue((repo / rel_report).exists())
            self.assertFalse((other / rel_report).exists())

    def test_report_mode_streams_and_captures_command_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "run.json"
            args = self.parse("--out", str(out_path), "cpu")
            cmd = runner.Command(
                "probe",
                (sys.executable, "-c", "print('streamed-line')"),
                ROOT,
                os.environ.copy(),
            )

            with redirect_stdout(io.StringIO()) as stdout:
                rc = runner.run_plan([cmd], dry_run=False, platform="linux", args=args)

            self.assertEqual(rc, 0)
            self.assertIn("streamed-line", stdout.getvalue())
            report = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertEqual(report["commands"][0]["stdout_tail"].strip(), "streamed-line")
            self.assertEqual(report["commands"][0]["stderr_tail"], "")
            self.assertEqual(report["proof"], {"cpu_scope": "full", "mode": "manual", "stage": "cpu"})

    def test_verify_lane_accepts_good_laptop_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, self.good_run_report())
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stdout(io.StringIO()) as stdout:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 0)
            self.assertIn("[verify] OK", stdout.getvalue())

    def test_status_lane_prints_compact_summary_for_good_reports(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, self.good_run_report())
            args = self.parse("status", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stdout(io.StringIO()) as stdout:
                rc = runner.run_status(ROOT, args)

            text = stdout.getvalue()
            expected_head = runner.repo_report(ROOT)["head"][:12]
            self.assertEqual(rc, 0)
            self.assertIn("[status] mode=cpu+nvidia", text)
            self.assertIn("[status:check] OK", text)
            self.assertIn("proof=cpu+nvidia/postcheck", text)
            self.assertIn(f"repo=head={expected_head}", text)
            self.assertIn("checks=cpu:OK,nvidia:OK,cuda-toolchain:OK", text)
            self.assertIn("[status:run] OK", text)
            self.assertIn("commands=cpu:OK,nvidia-setup:OK,nvidia:OK", text)

    def test_status_cpu_only_accepts_cpu_artifacts_without_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report(proof_mode="cpu-only")
            check_report["checks"][1]["ok"] = False
            check_report["checks"][1]["detail"] = "no gpu"
            check_report["checks"][2]["ok"] = False
            check_report["checks"][2]["detail"] = "no nvcc"
            run_report = self.good_run_report(proof_mode="cpu-only")
            run_report["commands"] = [{"label": "cpu", "skipped": False, "exit_code": 0}]
            self.write_json(check_path, check_report)
            self.write_json(run_path, run_report)
            args = self.parse("status", "--cpu-only", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stdout(io.StringIO()) as stdout:
                rc = runner.run_status(ROOT, args)

            self.assertEqual(rc, 0)
            self.assertIn("mode=cpu-only", stdout.getvalue())
            self.assertIn("checks=cpu:OK,nvidia:MISSING,cuda-toolchain:MISSING", stdout.getvalue())

    def test_status_lane_returns_nonzero_for_failed_summary(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            run_report = self.good_run_report()
            run_report["summary"]["ok"] = False
            run_report["exit_code"] = 1
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, run_report)
            args = self.parse("status", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_status(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("[status:run] FAIL", stdout.getvalue())
            self.assertIn("summary.ok", stderr.getvalue())

    def test_status_lane_returns_nonzero_for_dry_run_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            run_report = self.good_run_report()
            run_report["dry_run"] = True
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, run_report)
            args = self.parse("status", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_status(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("--dry-run", stderr.getvalue())

    def test_verify_lane_rejects_stale_repo_head(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            run_report = self.good_run_report()
            check_report["repo"]["head"] = "old-head"
            run_report["repo"]["head"] = "old-head"
            self.write_json(check_path, check_report)
            self.write_json(run_path, run_report)
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))
            current_repo = self.good_repo_report(head="new-head")

            with mock.patch.object(runner, "repo_report", return_value=current_repo):
                with redirect_stderr(io.StringIO()) as stderr:
                    rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("current HEAD", stderr.getvalue())
            self.assertIn("--allow-stale-repo", stderr.getvalue())

    def test_verify_lane_can_allow_stale_repo_head(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            run_report = self.good_run_report()
            check_report["repo"]["head"] = "old-head"
            run_report["repo"]["head"] = "old-head"
            self.write_json(check_path, check_report)
            self.write_json(run_path, run_report)
            args = self.parse(
                "verify",
                "--allow-stale-repo",
                "--check-report",
                str(check_path),
                "--run-report",
                str(run_path),
            )
            current_repo = self.good_repo_report(head="new-head")

            with mock.patch.object(runner, "repo_report", return_value=current_repo):
                with redirect_stdout(io.StringIO()) as stdout:
                    rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 0)
            self.assertIn("[verify] OK", stdout.getvalue())

    def test_status_lane_rejects_stale_repo_head(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            run_report = self.good_run_report()
            check_report["repo"]["head"] = "old-head"
            run_report["repo"]["head"] = "old-head"
            self.write_json(check_path, check_report)
            self.write_json(run_path, run_report)
            args = self.parse("status", "--check-report", str(check_path), "--run-report", str(run_path))
            current_repo = self.good_repo_report(head="new-head")

            with mock.patch.object(runner, "repo_report", return_value=current_repo):
                with redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()) as stderr:
                    rc = runner.run_status(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("[status:check] FAIL", stdout.getvalue())
            self.assertIn("current HEAD", stderr.getvalue())

    def test_verify_lane_rejects_stale_dirty_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            run_report = self.good_run_report()
            check_report["repo"]["status_fingerprint"] = "old-fingerprint"
            run_report["repo"]["status_fingerprint"] = "old-fingerprint"
            self.write_json(check_path, check_report)
            self.write_json(run_path, run_report)
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))
            current_repo = self.good_repo_report()
            current_repo["status_fingerprint"] = "new-fingerprint"

            with mock.patch.object(runner, "repo_report", return_value=current_repo):
                with redirect_stderr(io.StringIO()) as stderr:
                    rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("status_fingerprint", stderr.getvalue())
            self.assertIn("--allow-stale-repo", stderr.getvalue())

    def test_verify_cpu_only_accepts_cpu_artifacts_without_nvidia(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report(proof_mode="cpu-only")
            check_report["checks"][1]["ok"] = False
            check_report["checks"][1]["detail"] = "no gpu"
            check_report["checks"][2]["ok"] = False
            check_report["checks"][2]["detail"] = "no nvcc"
            run_report = self.good_run_report(proof_mode="cpu-only")
            run_report["commands"] = [{"label": "cpu", "skipped": False, "exit_code": 0}]
            self.write_json(check_path, check_report)
            self.write_json(run_path, run_report)
            args = self.parse("verify", "--cpu-only", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stdout(io.StringIO()) as stdout:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 0)
            self.assertIn("[verify] OK", stdout.getvalue())

    def test_verify_cpu_only_rejects_missing_cpu_run(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            run_report = self.good_run_report()
            run_report["commands"] = [{"label": "nvidia", "skipped": False, "exit_code": 0}]
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, run_report)
            args = self.parse("verify", "--cpu-only", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("missing command label 'cpu'", stderr.getvalue())

    def test_verify_lane_rejects_wrong_proof_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            self.write_json(check_path, self.good_check_report(proof_mode="cpu-only"))
            self.write_json(run_path, self.good_run_report(proof_mode="cpu-only"))
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("proof.mode", stderr.getvalue())

    def test_verify_full_cpu_rejects_smoke_run_report(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            self.write_json(check_path, self.good_check_report(cpu_scope="full"))
            self.write_json(run_path, self.good_run_report(cpu_scope="smoke"))
            args = self.parse("verify", "--full-cpu", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("proof.cpu_scope", stderr.getvalue())

    def test_verify_lane_rejects_missing_nvidia_success(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            check_report["checks"][1]["ok"] = False
            check_report["checks"][1]["detail"] = "no gpu"
            self.write_json(check_path, check_report)
            self.write_json(run_path, self.good_run_report())
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("nvidia", stderr.getvalue())

    def test_verify_lane_rejects_missing_cpu_target_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            check_report["checks"][0]["detail"] = "go version go1.26.0"
            self.write_json(check_path, check_report)
            self.write_json(run_path, self.good_run_report())
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("GOOS/GOARCH", stderr.getvalue())

    def test_verify_lane_rejects_dry_run_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            run_report = self.good_run_report()
            run_report["dry_run"] = True
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, run_report)
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("--dry-run", stderr.getvalue())

    def test_verify_lane_rejects_missing_host_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            del check_report["host"]
            self.write_json(check_path, check_report)
            self.write_json(run_path, self.good_run_report())
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("missing host metadata", stderr.getvalue())

    def test_verify_lane_rejects_missing_repo_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            run_report = self.good_run_report()
            del run_report["repo"]
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, run_report)
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("missing repo metadata", stderr.getvalue())

    def test_verify_lane_rejects_missing_summary_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            check_report = self.good_check_report()
            del check_report["summary"]
            self.write_json(check_path, check_report)
            self.write_json(run_path, self.good_run_report())
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("missing summary metadata", stderr.getvalue())

    def test_verify_lane_rejects_missing_wsl_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            check_path = Path(td) / "check.json"
            run_path = Path(td) / "run.json"
            run_report = self.good_run_report()
            del run_report["wsl"]
            self.write_json(check_path, self.good_check_report())
            self.write_json(run_path, run_report)
            args = self.parse("verify", "--check-report", str(check_path), "--run-report", str(run_path))

            with redirect_stderr(io.StringIO()) as stderr:
                rc = runner.run_verify(ROOT, args)

            self.assertEqual(rc, 1)
            self.assertIn("missing WSL metadata", stderr.getvalue())

    def test_darwin_rejects_nvidia_lane_before_running(self) -> None:
        args = self.parse("nvidia")
        with self.assertRaises(SystemExit):
            runner.nvidia_commands(Path("/repo/fleet"), args, platform="darwin")

    def test_dry_run_does_not_execute_subprocess(self) -> None:
        args = self.parse("--smoke", "cpu")
        plan = runner.build_plan(Path("/repo/fleet"), args, platform="linux")
        with mock.patch("subprocess.run") as run:
            with redirect_stdout(io.StringIO()):
                rc = runner.run_plan(plan, dry_run=True, platform="linux")
        self.assertEqual(rc, 0)
        run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
