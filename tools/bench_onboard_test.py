#!/usr/bin/env python3
"""Tests for bench_onboard.py — registering a new benchmark machine.

The highest-value thing to pin here is a path, not a computation. The header
comment records a real outage: ``BENCHMARK_DIR`` once carried a stale ``fak/``
prefix left over from when the Go module sat in a subdirectory, so onboarding
wrote ``specs.json`` to a directory ``bench_catalog.py build`` never scans — the
machine registered "successfully" and then silently never appeared in the
catalog. That is a SILENT failure with a loud-looking success message, so the
root is asserted directly.

Around it: hardware detection must degrade rather than raise when a probe binary
is absent (an onboarding run on a GPU-less laptop is normal), and ``save_specs``
must refuse to clobber an existing registration unless ``--replace`` was asked
for, because overwriting specs.json silently rewrites the provenance of every
benchmark already published under that machine id.

Every probe is stubbed: this test never spawns a subprocess, never reads real
hardware, and never writes outside a temp directory.
"""
import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench_onboard as M  # noqa: E402


class Stub:
    """Swap module globals for the duration of a test."""

    def __init__(self, case):
        self.case = case

    def set(self, name, value):
        original = getattr(M, name)
        setattr(M, name, value)
        self.case.addCleanup(lambda: setattr(M, name, original))

    def quiet(self):
        """Swallow the tool's operator-facing chatter so the runner stays readable."""
        stack = contextlib.ExitStack()
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
        self.case.addCleanup(stack.close)

    def platform(self, system="Linux", machine="x86_64", node="host-1",
                 processor="Some CPU"):
        self.set("platform", types.SimpleNamespace(
            system=lambda: system, machine=lambda: machine, node=lambda: node,
            processor=lambda: processor, version=lambda: "#1 SMP",
            release=lambda: "6.1.0", python_version=lambda: "3.13.0"))

    def probes(self, which=(), outputs=None):
        """`which` = binaries that exist; `outputs` = {cmd[0]: stdout}."""
        outputs = outputs or {}
        self.set("shutil_which", lambda name: name in which)
        self.set("run_command", lambda cmd: outputs.get(cmd[0]))


class BenchmarkRoot(unittest.TestCase):
    def test_specs_are_written_where_the_catalog_builder_actually_looks(self):
        # Regression guard for the stale `fak/` prefix fossil: this root MUST
        # match bench_catalog.py's, or a registration is silently invisible.
        self.assertEqual(M.BENCHMARK_DIR,
                         Path(M.__file__).resolve().parents[1] / "experiments" / "benchmark")
        self.assertEqual(M.MACHINES_DIR, M.BENCHMARK_DIR / "machines")
        self.assertNotIn("fak", M.BENCHMARK_DIR.relative_to(
            Path(M.__file__).resolve().parents[1]).parts)


class RunCommand(unittest.TestCase):
    def fake_subprocess(self, **kwargs):
        stub = Stub(self)
        stub.set("subprocess", types.SimpleNamespace(
            TimeoutExpired=TimeoutError, FileNotFoundError=FileNotFoundError, **kwargs))

    def test_stdout_is_returned_stripped_on_success(self):
        self.fake_subprocess(run=lambda *a, **k: types.SimpleNamespace(
            returncode=0, stdout="  hello\n"))
        self.assertEqual(M.run_command(["anything"]), "hello")

    def test_a_non_zero_exit_yields_none_rather_than_partial_output(self):
        self.fake_subprocess(run=lambda *a, **k: types.SimpleNamespace(
            returncode=2, stdout="garbage"))
        self.assertIsNone(M.run_command(["anything"]))

    def test_a_missing_binary_yields_none_instead_of_raising(self):
        def boom(*a, **k):
            raise FileNotFoundError("nvidia-smi")
        self.fake_subprocess(run=boom)
        self.assertIsNone(M.run_command(["nvidia-smi"]))


class DetectCpu(unittest.TestCase):
    def test_lscpu_core_counts_override_the_logical_cpu_count(self):
        stub = Stub(self)
        stub.platform(machine="x86_64", processor="AMD EPYC")
        stub.probes(which={"lscpu"}, outputs={"lscpu": (
            "Architecture:            x86_64\n"
            "CPU(s):                  128\n"
            "Core(s) per socket:      64\n"
            "NUMA node0 CPU(s):       0-63\n")})
        cpu = M.detect_cpu()
        self.assertEqual(cpu["cores_physical"], 64)
        self.assertEqual(cpu["cores_logical"], 128)
        self.assertEqual(cpu["model"], "AMD EPYC")
        self.assertEqual(cpu["architecture"], "x86_64")

    def test_no_lscpu_falls_back_to_the_interpreter_cpu_count(self):
        import os
        stub = Stub(self)
        stub.platform(machine="x86_64")
        stub.probes(which=set())
        cpu = M.detect_cpu()
        self.assertEqual(cpu["cores_logical"], os.cpu_count() or 1)
        self.assertGreaterEqual(cpu["cores_logical"], 1)

    def test_apple_silicon_uses_sysctl_for_both_core_counts(self):
        stub = Stub(self)
        stub.platform(system="Darwin", machine="arm64")
        stub.set("run_command", lambda cmd: {"hw.physicalcpu": "10",
                                             "hw.logicalcpu": "10"}.get(cmd[-1]))
        stub.set("shutil_which", lambda name: False)
        cpu = M.detect_cpu()
        self.assertEqual((cpu["cores_physical"], cpu["cores_logical"]), (10, 10))

    def test_the_architecture_is_normalised_to_lower_case(self):
        stub = Stub(self)
        stub.platform(machine="AMD64")
        stub.probes(which=set())
        self.assertEqual(M.detect_cpu()["architecture"], "amd64")


class DetectGpu(unittest.TestCase):
    def test_every_nvidia_gpu_row_is_parsed_with_memory_converted_to_gb(self):
        stub = Stub(self)
        stub.platform()
        stub.probes(which={"nvidia-smi"}, outputs={"nvidia-smi": None})
        stub.set("run_command", lambda cmd: (
            "NVIDIA A100-SXM4-80GB, 81920 MiB\nNVIDIA A100-SXM4-80GB, 81920 MiB"
            if "--query-gpu=name,memory.total" in cmd else "8.0"))
        gpus = M.detect_gpu()
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["model"], "NVIDIA A100-SXM4-80GB")
        self.assertEqual(gpus[0]["memory_gb"], 80.0)
        self.assertEqual(gpus[0]["compute_capability"], "8.0")

    def test_no_probe_binary_at_all_reports_no_gpu_rather_than_failing(self):
        stub = Stub(self)
        stub.platform()
        stub.probes(which=set())
        self.assertEqual(M.detect_gpu(), [])

    def test_an_amd_card_is_recognised_only_when_no_nvidia_card_was_found(self):
        stub = Stub(self)
        stub.platform()
        stub.probes(which={"rocm-smi"},
                    outputs={"rocm-smi": "Card series:  Radeon Instinct MI250X"})
        self.assertEqual(M.detect_gpu(), [{"model": "Radeon Instinct MI250X"}])

    def test_apple_unified_memory_is_labelled_rather_than_guessed_in_gb(self):
        stub = Stub(self)
        stub.platform(system="Darwin", machine="arm64")
        stub.probes(which=set(),
                    outputs={"system_profiler": "  Chip:  Apple M3 Max\n"})
        self.assertEqual(M.detect_gpu(), [{"model": "Apple M3 Max",
                                           "memory_gb": "unified"}])


class DetectRam(unittest.TestCase):
    def test_macos_memsize_bytes_are_converted_to_gb(self):
        stub = Stub(self)
        stub.platform(system="Darwin")
        stub.set("run_command", lambda cmd: str(64 * 1024**3))
        self.assertEqual(M.detect_ram(), 64.0)

    def test_windows_reads_total_physical_memory_from_the_cim_api(self):
        # wmic.exe is gone on Win11 24H2+; the CIM fallback is the supported path.
        stub = Stub(self)
        stub.platform(system="Windows")
        stub.set("run_command", lambda cmd: f"\r\n{32 * 1024**3}\r\n")
        self.assertEqual(M.detect_ram(), 32.0)

    def test_an_unreadable_probe_reports_zero_instead_of_raising(self):
        stub = Stub(self)
        stub.platform(system="Darwin")
        stub.set("run_command", lambda cmd: None)
        self.assertEqual(M.detect_ram(), 0)


class DetectRuntime(unittest.TestCase):
    def test_the_go_version_token_is_pulled_out_of_the_banner(self):
        stub = Stub(self)
        stub.platform()
        stub.probes(which=set(),
                    outputs={"go": "go version go1.26.0 windows/amd64"})
        runtime = M.detect_runtime()
        self.assertEqual(runtime["go_version"], "go1.26.0")
        self.assertEqual(runtime["python_version"], "3.13.0")

    def test_no_go_toolchain_simply_omits_the_key(self):
        stub = Stub(self)
        stub.platform()
        stub.probes(which=set())
        self.assertNotIn("go_version", M.detect_runtime())

    def test_the_cuda_driver_version_is_scraped_from_the_smi_header(self):
        stub = Stub(self)
        stub.platform()
        stub.probes(which={"nvidia-smi"}, outputs={
            "nvidia-smi": "| NVIDIA-SMI 550.54.15   Driver Version: 550.54.15   "
                          "CUDA Version: 12.4     |"})
        self.assertEqual(M.detect_runtime()["cuda_driver_version"], "12.4")


class GenerateSpecs(unittest.TestCase):
    def stub_detectors(self):
        stub = Stub(self)
        stub.platform(node="bench-box")
        stub.set("detect_cpu", lambda: {"model": "cpu"})
        stub.set("detect_gpu", lambda: [{"model": "gpu"}])
        stub.set("detect_ram", lambda: 64.0)
        stub.set("detect_runtime", lambda: {"python_version": "3.13.0"})
        return stub

    def test_the_record_carries_its_schema_id_and_the_declared_tags(self):
        self.stub_detectors()
        specs = M.generate_specs("bench-box", ["gpu", "linux"])
        self.assertEqual(specs["$schema"], "benchmark/machine-specs.v1")
        self.assertEqual(specs["machine_id"], "bench-box")
        self.assertEqual(specs["hostname"], "bench-box")
        self.assertEqual(specs["tags"], ["gpu", "linux"])

    def test_hardware_os_and_runtime_are_all_nested_under_the_record(self):
        self.stub_detectors()
        specs = M.generate_specs("bench-box", [])
        self.assertEqual(set(specs["hardware"]), {"cpu", "gpu", "ram_gb"})
        self.assertIn("os", specs)
        self.assertIn("runtime", specs)

    def test_the_registration_timestamp_is_utc_and_iso_8601(self):
        from datetime import datetime, timezone
        self.stub_detectors()
        stamp = M.generate_specs("bench-box", [])["registered_at"]
        parsed = datetime.fromisoformat(stamp)
        self.assertEqual(parsed.utcoffset(), timezone.utc.utcoffset(None))


class SaveSpecs(unittest.TestCase):
    def setUp(self):
        self.machines = Path(tempfile.mkdtemp()) / "machines"
        stub = Stub(self)
        stub.set("MACHINES_DIR", self.machines)
        stub.quiet()
        self.specs = {"machine_id": "node-a", "tags": ["gpu"]}

    def path(self):
        return self.machines / "node-a" / "specs.json"

    def test_a_first_registration_creates_the_directory_and_the_file(self):
        self.assertTrue(M.save_specs(self.specs))
        self.assertEqual(json.loads(self.path().read_text(encoding="utf-8")), self.specs)

    def test_an_existing_registration_is_never_silently_overwritten(self):
        self.assertTrue(M.save_specs(self.specs))
        self.path().write_text('{"machine_id": "node-a", "tags": ["kept"]}',
                               encoding="utf-8")
        self.assertFalse(M.save_specs(self.specs))
        self.assertEqual(json.loads(self.path().read_text(encoding="utf-8"))["tags"],
                         ["kept"])

    def test_replace_overwrites_deliberately(self):
        self.assertTrue(M.save_specs(self.specs))
        self.assertTrue(M.save_specs({"machine_id": "node-a", "tags": ["new"]},
                                     overwrite=True))
        self.assertEqual(json.loads(self.path().read_text(encoding="utf-8"))["tags"],
                         ["new"])


class Main(unittest.TestCase):
    def setUp(self):
        self.machines = Path(tempfile.mkdtemp()) / "machines"
        stub = Stub(self)
        stub.set("MACHINES_DIR", self.machines)
        stub.set("generate_specs",
                 lambda machine_id, tags: {"machine_id": machine_id, "tags": tags})
        stub.quiet()

    def test_non_interactive_mode_requires_a_machine_id(self):
        self.assertEqual(M.main([]), 1)
        self.assertFalse(self.machines.exists())

    def test_a_named_machine_is_registered_with_its_tags_split_and_trimmed(self):
        self.assertEqual(M.main(["--machine-id", "node-a", "--tags", "gpu, a100 ,linux"]), 0)
        saved = json.loads((self.machines / "node-a" / "specs.json")
                           .read_text(encoding="utf-8"))
        self.assertEqual(saved["tags"], ["gpu", "a100", "linux"])

    def test_registering_the_same_machine_twice_fails_loudly(self):
        self.assertEqual(M.main(["--machine-id", "node-a"]), 0)
        self.assertEqual(M.main(["--machine-id", "node-a"]), 1)
        self.assertEqual(M.main(["--machine-id", "node-a", "--replace"]), 0)


if __name__ == "__main__":
    unittest.main()
