#!/usr/bin/env python3
"""Hermetic, offline tests for the GCP benchmark path.

No gcloud, no network, no GCP auth -- these lock the pure-data registry contract
(gcp_accel) and the provisioner's offline logic (tier resolution, command
rendering, the always-teardown finally, idempotent delete). The live gcloud calls
are stubbed via a fake Runner.
"""
from __future__ import annotations

import sys
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gcp_accel  # noqa: E402
import gcp_bench  # noqa: E402


class RegistryContractTest(unittest.TestCase):
    """gcp_accel is the shared contract the probe + provisioner both bind to."""

    def test_blackwell_tiers_present_with_exact_machine_types(self):
        b200 = gcp_accel.by_slug("a4-b200")
        gb200 = gcp_accel.by_slug("a4x-gb200")
        self.assertIsNotNone(b200)
        self.assertIsNotNone(gb200)
        # Exact gcloud strings -- a typo here silently fails provisioning.
        self.assertEqual(b200.machine_type, "a4-highgpu-8g")
        self.assertEqual(b200.accelerator_type, "nvidia-b200")
        self.assertEqual(b200.gpu_count, 8)
        self.assertEqual(gb200.machine_type, "a4x-highgpu-4g")
        self.assertEqual(gb200.accelerator_type, "nvidia-gb200")
        self.assertTrue(b200.blackwell and gb200.blackwell)

    def test_blackwell_first_ladder_ordering(self):
        # Default ladder is newest-silicon-first and the first two are Blackwell.
        ladder = gcp_accel.fallback_ladder()
        self.assertTrue(ladder[0].blackwell)
        ranks = [t.gen_rank for t in ladder]
        self.assertEqual(ranks, sorted(ranks, reverse=True))

    def test_blackwell_only_excludes_hopper_and_ada(self):
        strict = gcp_accel.fallback_ladder(blackwell_only=True)
        self.assertTrue(all(t.blackwell for t in strict))
        self.assertTrue(all(t.arch == "blackwell" for t in strict))

    def test_proof_tier_is_cheapest(self):
        proof = gcp_accel.proof_tier()
        self.assertLessEqual(
            proof.approx_usd_per_hour,
            min(t.approx_usd_per_hour for t in gcp_accel.TIERS),
        )

    def test_accelerator_flag_shape(self):
        b200 = gcp_accel.by_slug("a4-b200")
        self.assertEqual(gcp_accel.accelerator_flag(b200), "type=nvidia-b200,count=8")

    def test_boot_image_is_a_cuda_dlvm(self):
        fam, proj = gcp_accel.boot_image()
        self.assertIn("cu1", fam)  # a CUDA image family
        self.assertTrue(proj)


class FakeRunner(gcp_bench.Runner):
    """Records gcloud arg-lists instead of executing them."""

    def __init__(self):
        super().__init__(dry_run=True, project="p", account="a")
        self.calls: list[list[str]] = []

    def run(self, args, *, capture=False, timeout=600, check=True):
        self.calls.append(args)
        return SimpleNamespace(returncode=0, stdout="", stderr="", args=args)


class ProvisionerLogicTest(unittest.TestCase):
    def _args(self, **kw):
        base = dict(tier=None, blackwell=False, proof=False, zone=None,
                    project="p", account="a", spot=False, keep=False, dry_run=True)
        base.update(kw)
        return SimpleNamespace(**base)

    def test_resolve_tier_pinned(self):
        r = FakeRunner()
        t = gcp_bench.resolve_tier(self._args(tier="a3-ultra-h200"), r)
        self.assertEqual(t.slug, "a3-ultra-h200")

    def test_resolve_tier_proof_is_cheapest(self):
        r = FakeRunner()
        t = gcp_bench.resolve_tier(self._args(proof=True), r)
        self.assertEqual(t.slug, gcp_accel.proof_tier().slug)

    def test_resolve_tier_unknown_returns_none(self):
        r = FakeRunner()
        self.assertIsNone(gcp_bench.resolve_tier(self._args(tier="nope"), r))

    def test_provision_self_accelerated_omits_accelerator_flag(self):
        # a4/a4x/a3/g2 carry their GPUs implicitly in the machine type; gcloud
        # REJECTS --accelerator on them. The provisioner must NOT emit it.
        r = FakeRunner()
        tier = gcp_accel.by_slug("a4-b200")
        gcp_bench.provision(r, tier, "fak-bench-x", "us-central1-b",
                            Path("/tmp/startup.sh"), spot=False)
        self.assertEqual(len(r.calls), 1)
        flat = " ".join(r.calls[0])
        self.assertIn("instances create fak-bench-x", flat)
        self.assertIn("--machine-type=a4-highgpu-8g", flat)
        self.assertNotIn("--accelerator", flat)
        self.assertIn("--zone=us-central1-b", flat)
        # B200 mandates local SSD -- the provisioner must attach it.
        self.assertIn("--local-ssd=interface=NVME", flat)

    def test_provision_n1_tier_emits_explicit_accelerator(self):
        # The older N1 family (T4) DOES need an explicit --accelerator.
        r = FakeRunner()
        tier = gcp_accel.by_slug("n1-t4")
        gcp_bench.provision(r, tier, "fak-bench-t4", "us-central1-a",
                            Path("/tmp/startup.sh"), spot=False)
        flat = " ".join(r.calls[0])
        self.assertIn("--machine-type=n1-standard-8", flat)
        # GCP's real T4 accelerator string is nvidia-tesla-t4 (not nvidia-t4);
        # the bare form makes the SKU read as NOT_OFFERED. Verified live.
        self.assertIn("--accelerator=type=nvidia-tesla-t4,count=1", flat)

    def test_provision_spot_adds_termination_action(self):
        r = FakeRunner()
        tier = gcp_accel.by_slug("g2-l4")
        gcp_bench.provision(r, tier, "n", "us-central1-a",
                            Path("/tmp/s.sh"), spot=True)
        flat = " ".join(r.calls[0])
        self.assertIn("--provisioning-model=SPOT", flat)
        self.assertIn("--instance-termination-action=DELETE", flat)

    def test_provision_sets_max_run_duration_ttl_by_default(self):
        # The dominant leak mode is a killed launcher whose `finally` teardown never
        # runs. A GCP-side --max-run-duration deletes the VM no matter what the client
        # does -- the belt-and-suspenders. Default is 2h => 7200s, DELETE on expiry.
        r = FakeRunner()
        tier = gcp_accel.by_slug("g2-l4")
        gcp_bench.provision(r, tier, "n", "us-central1-a", Path("/tmp/s.sh"), spot=False)
        flat = " ".join(r.calls[0])
        self.assertIn("--max-run-duration=7200s", flat)
        self.assertIn("--instance-termination-action=DELETE", flat)

    def test_provision_termination_action_emitted_once_under_spot_and_ttl(self):
        # --instance-termination-action must appear EXACTLY once even when both --spot
        # and the TTL ask for it (gcloud rejects a duplicated flag).
        r = FakeRunner()
        tier = gcp_accel.by_slug("g2-l4")
        gcp_bench.provision(r, tier, "n", "us-central1-a", Path("/tmp/s.sh"),
                            spot=True, max_run_hours=2.0)
        self.assertEqual(
            r.calls[0].count("--instance-termination-action=DELETE"), 1,
            "termination-action must be emitted once, not duplicated")

    def test_provision_ttl_zero_disables_max_run_duration(self):
        # --max-run-hours 0 opts out (e.g. a deliberately long debug run); no TTL flag,
        # and with no --spot, no termination-action either.
        r = FakeRunner()
        tier = gcp_accel.by_slug("g2-l4")
        gcp_bench.provision(r, tier, "n", "us-central1-a", Path("/tmp/s.sh"),
                            spot=False, max_run_hours=0)
        flat = " ".join(r.calls[0])
        self.assertNotIn("--max-run-duration", flat)
        self.assertNotIn("--instance-termination-action", flat)

    def test_startup_warns_about_preexisting_bench_instances(self):
        class LeakedRunner(FakeRunner):
            def __init__(self):
                super().__init__()
                self.dry_run = False

            def run(self, args, *, capture=False, timeout=600, check=True):
                self.calls.append(args)
                rows = [{
                    "name": "fak-bench-g2-l4-20260620t000000z",
                    "zone": "https://www.googleapis.com/compute/v1/projects/p/zones/us-central1-a",
                    "status": "RUNNING",
                    "creationTimestamp": "2026-06-20T00:00:00.000-07:00",
                }]
                return SimpleNamespace(returncode=0, stdout=json.dumps(rows), stderr="")

        import json
        r = LeakedRunner()
        r.exe = "gcloud"
        out = StringIO()
        with redirect_stdout(out):
            gcp_bench.warn_preexisting_bench_instances(r)
        flat = " ".join(r.calls[0])
        self.assertIn("instances list", flat)
        self.assertIn("--filter=name~^fak-bench-", flat)
        text = out.getvalue()
        self.assertIn("fak-bench-g2-l4-20260620t000000z", text)
        self.assertIn(
            "gcloud --project p --account a compute instances delete "
            "fak-bench-g2-l4-20260620t000000z --zone=us-central1-a --quiet",
            text,
        )

    def test_resolve_instance_ip_describes_current_nat_ip(self):
        class IPRunner(FakeRunner):
            def __init__(self):
                super().__init__()
                self.dry_run = False

            def run(self, args, *, capture=False, timeout=600, check=True):
                self.calls.append(args)
                return SimpleNamespace(returncode=0, stdout="203.0.113.44\n", stderr="")

        r = IPRunner()
        ip = gcp_bench.resolve_instance_ip(r, "fak-bench-g2-l4-20260629t000000z",
                                           "us-central1-a")
        self.assertEqual(ip, "203.0.113.44")
        self.assertEqual(
            r.calls[0],
            [
                "compute", "instances", "describe", "fak-bench-g2-l4-20260629t000000z",
                "--zone=us-central1-a",
                "--format=get(networkInterfaces[0].accessConfigs[0].natIP)",
            ],
        )

    def test_main_runs_preexisting_bench_sweep_at_startup(self):
        class LeakedRunner(FakeRunner):
            def __init__(self):
                super().__init__()
                self.dry_run = False

            def run(self, args, *, capture=False, timeout=600, check=True):
                self.calls.append(args)
                if "instances" in args and "list" in args:
                    return SimpleNamespace(
                        returncode=0,
                        stdout='[{"name":"fak-bench-old","zone":"zones/us-central1-a"}]',
                        stderr="",
                    )
                return SimpleNamespace(returncode=0, stdout="", stderr="")

        r = LeakedRunner()
        out = StringIO()
        with mock.patch.object(gcp_bench, "Runner", lambda dry_run, project, account: r), \
                mock.patch.object(gcp_bench, "resolve_tier", lambda args, runner: None), \
                redirect_stdout(out):
            rc = gcp_bench.main(["--tier", "g2-l4"])
        self.assertEqual(rc, 2)
        self.assertEqual(r.calls[0][0:3], ["compute", "instances", "list"])
        self.assertIn("fak-bench-old", out.getvalue())

    def test_readback_retries_then_succeeds_on_transient_flake(self):
        # The benchmark already burned GPU minutes and teardown follows the read, so a
        # transient SSH/IAP stall on `sudo cat result.json` must NOT discard a real
        # result -- the read is retried. First cat flakes (rc=1), second returns the JSON.
        good = '{"schema":"fak.gcp-vm-bench.v2","ok":true,"decode_tok_per_sec":70.7}'

        class FlakyReadback(FakeRunner):
            def __init__(self):
                super().__init__()
                self.cat_calls = 0

            def scp(self, *a, **k):
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def run(self, args, *, capture=False, timeout=600, check=True):
                self.calls.append(args)
                joined = " ".join(args)
                if "cat /opt/gcp-bench/result.json" in joined:
                    self.cat_calls += 1
                    if self.cat_calls == 1:  # first read flakes
                        return SimpleNamespace(returncode=1, stdout="", stderr="stall")
                    return SimpleNamespace(returncode=0, stdout=good, stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

        r = FlakyReadback()
        r.dry_run = False
        with mock.patch.object(gcp_bench.time, "sleep", lambda *_: None):
            out = gcp_bench.run_driver_over_ssh(
                r, "n", "us-central1-a", Path("/tmp/src.tgz"), Path("/tmp/drv.sh"))
        self.assertIsNotNone(out)
        self.assertEqual(out["decode_tok_per_sec"], 70.7)
        self.assertEqual(r.cat_calls, 2, "should have retried the flaked readback once")

    def test_readback_rejects_partial_json_without_schema(self):
        # A truncated/partial write (no schema key) must NOT be accepted as a result:
        # that would record a phantom row. After retries exhaust it returns None.
        class PartialReadback(FakeRunner):
            def scp(self, *a, **k):
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def run(self, args, *, capture=False, timeout=600, check=True):
                self.calls.append(args)
                if "cat /opt/gcp-bench/result.json" in " ".join(args):
                    return SimpleNamespace(returncode=0, stdout='{"ok":true}', stderr="")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

        r = PartialReadback()
        r.dry_run = False
        with mock.patch.object(gcp_bench.time, "sleep", lambda *_: None):
            out = gcp_bench.run_driver_over_ssh(
                r, "n", "us-central1-a", Path("/tmp/src.tgz"), Path("/tmp/drv.sh"))
        self.assertIsNone(out)

    def test_teardown_is_idempotent_on_missing(self):
        class GoneRunner(FakeRunner):
            def run(self, args, *, capture=False, timeout=600, check=True):
                self.calls.append(args)
                raise RuntimeError("The resource ... was not found")
        r = GoneRunner()
        # Must NOT raise: a missing instance is a successful teardown.
        gcp_bench.teardown(r, "ghost", "us-central1-a")

    def test_setup_script_is_a_trivial_boot_marker(self):
        # The boot startup-script is intentionally trivial -- the real work is the
        # SCP'd driver. It must just mark readiness, never block the boot.
        body = gcp_bench.render_setup_script()
        self.assertIn("boot.marker", body)
        self.assertIn("mkdir -p /opt/gcp-bench", body)
        # No heavy build in the boot path.
        self.assertNotIn("-DGGML_CUDA=ON", body)

    def test_machine_id_and_instance_name_are_slugged(self):
        tier = gcp_accel.by_slug("a4-b200")
        self.assertEqual(gcp_bench.machine_id_for(tier), "gcp-a4-b200")
        self.assertTrue(gcp_bench.instance_name(tier).startswith("fak-bench-a4-b200-"))


class EngineSelectionTest(unittest.TestCase):
    """The modular engine registry + --engine resolution."""

    def test_all_is_the_curated_default_and_excludes_vllm(self):
        # `all` stays the fak-vs-llama.cpp default; vLLM is opt-in (a ~5 GB install), so
        # it must NOT be pulled into every default run.
        self.assertEqual(gcp_bench.resolve_engines("all"), ["llama", "fak-cpu", "fak-cuda"])
        self.assertNotIn("vllm", gcp_bench.resolve_engines("all"))

    def test_fak_alias_is_both_fak_engines(self):
        self.assertEqual(gcp_bench.resolve_engines("fak"), ["fak-cpu", "fak-cuda"])

    def test_single_engine(self):
        self.assertEqual(gcp_bench.resolve_engines("fak-cuda"), ["fak-cuda"])

    def test_comma_list_is_normalized_to_run_order(self):
        # Order follows ENGINE_ORDER regardless of how the user listed them.
        self.assertEqual(gcp_bench.resolve_engines("fak-cuda,llama"), ["llama", "fak-cuda"])

    def test_vllm_is_a_first_class_selectable_engine(self):
        # vLLM is a first-class peer to llama.cpp: registered, CUDA-flagged, selectable
        # alone or in a comma list, and ordered right after llama (the headline baseline).
        self.assertIn("vllm", gcp_bench.ENGINES)
        self.assertTrue(gcp_bench.ENGINES["vllm"].needs_cuda)
        self.assertIn("vllm", gcp_bench.ENGINE_ORDER)
        self.assertEqual(gcp_bench.resolve_engines("vllm"), ["vllm"])
        # The canonical engine head-to-head, in stable order regardless of input order.
        self.assertEqual(gcp_bench.resolve_engines("vllm,llama"), ["llama", "vllm"])
        self.assertEqual(gcp_bench.resolve_engines("fak-cuda,vllm,llama"),
                         ["llama", "vllm", "fak-cuda"])

    def test_fak_cuda_q8_is_an_optin_apples_to_apples_engine(self):
        # fak-cuda-q8 is the Q8 device GEMV row (apples-to-apples vs llama.cpp Q8_0).
        # Like vLLM it is opt-in: registered + CUDA-flagged + selectable, ordered right
        # after fak-cuda, but NOT in the curated `all` default until a green Hopper run
        # witnesses the device Q8 GEMV (then it is promoted into DEFAULT_ALL).
        self.assertIn("fak-cuda-q8", gcp_bench.ENGINES)
        self.assertTrue(gcp_bench.ENGINES["fak-cuda-q8"].needs_cuda)
        self.assertIn("fak-cuda-q8", gcp_bench.ENGINE_ORDER)
        self.assertNotIn("fak-cuda-q8", gcp_bench.resolve_engines("all"))
        self.assertEqual(gcp_bench.resolve_engines("fak-cuda-q8"), ["fak-cuda-q8"])
        # The apples-to-apples head-to-head, in stable ENGINE_ORDER regardless of input order.
        self.assertEqual(gcp_bench.resolve_engines("fak-cuda-q8,fak-cuda,llama"),
                         ["llama", "fak-cuda", "fak-cuda-q8"])

    def test_fak_cuda_tf32_is_an_optin_prefill_tensorcore_engine(self):
        # fak-cuda-tf32 is the TF32 tensor-core SGEMM row (Lever 4: the f32 device path with
        # FAK_CUDA_TF32=1, the compute-bound prefill lever). Like fak-cuda-q8 it is opt-in:
        # registered + CUDA-flagged + selectable, ordered last (after fak-cuda-q8), but NOT in the
        # curated `all` default until a green Hopper run witnesses its prefill gain.
        self.assertIn("fak-cuda-tf32", gcp_bench.ENGINES)
        self.assertTrue(gcp_bench.ENGINES["fak-cuda-tf32"].needs_cuda)
        self.assertIn("fak-cuda-tf32", gcp_bench.ENGINE_ORDER)
        self.assertNotIn("fak-cuda-tf32", gcp_bench.resolve_engines("all"))
        # It is also not pulled in by the `fak` alias (that stays the cpu+f32-cuda pair).
        self.assertNotIn("fak-cuda-tf32", gcp_bench.resolve_engines("fak"))
        self.assertEqual(gcp_bench.resolve_engines("fak-cuda-tf32"), ["fak-cuda-tf32"])
        # The TF32 prefill head-to-head, in stable ENGINE_ORDER regardless of input order.
        self.assertEqual(gcp_bench.resolve_engines("fak-cuda-tf32,fak-cuda,llama"),
                         ["llama", "fak-cuda", "fak-cuda-tf32"])

    def test_unknown_engine_raises(self):
        with self.assertRaises(ValueError):
            gcp_bench.resolve_engines("nope")

    def test_engine_registry_flags_cuda_correctly(self):
        self.assertTrue(gcp_bench.ENGINES["llama"].needs_cuda)
        self.assertTrue(gcp_bench.ENGINES["vllm"].needs_cuda)
        self.assertTrue(gcp_bench.ENGINES["fak-cuda"].needs_cuda)
        self.assertFalse(gcp_bench.ENGINES["fak-cpu"].needs_cuda)


class DriverScriptTest(unittest.TestCase):
    """The rendered on-VM driver: shared setup + selected engines + combiner."""

    def _driver(self, engines, cc="89"):
        return gcp_bench.render_driver_script(
            engines, gcp_bench.DEFAULT_HF_REPO, gcp_bench.DEFAULT_HF_FILE, cc)

    def test_no_unresolved_placeholders(self):
        body = self._driver(["llama", "fak-cpu", "fak-cuda"])
        self.assertNotIn("@@", body)

    def test_shared_preamble_present_once(self):
        body = self._driver(["llama", "fak-cpu", "fak-cuda"])
        # the model is fetched ONCE and shared across engines
        self.assertEqual(body.count("hf_hub_download(repo_id="), 1)
        self.assertIn("nvidia-smi -L", body)
        self.assertIn("GOTOOLCHAIN=auto", body)
        self.assertIn("tar -C \"$WORK\" -xzf /tmp/fak-src.tgz", body)

    def test_only_selected_engine_fragments_emitted(self):
        body = self._driver(["fak-cuda"])
        self.assertIn("engine_fak_cuda()", body)
        self.assertNotIn("engine_llama()", body)
        self.assertNotIn("engine_fak_cpu()", body)
        self.assertIn("run_one fak-cuda engine_fak_cuda", body)

    def test_llama_engine_targets_gpu(self):
        body = self._driver(["llama"])
        self.assertIn("-DGGML_CUDA=ON", body)
        self.assertIn("-ngl 99", body)
        self.assertIn("-DCMAKE_CUDA_ARCHITECTURES=\"$CUDA_CC\"", body)

    def test_fak_cuda_is_f32_device_path_with_honesty_gate(self):
        # fak-cuda is the un-narrowed f32 device baseline (no -lean/-quant on this row);
        # the Q8 device path is the separate fak-cuda-q8 engine. -require-non-reference
        # makes a silent CPU fallback fail loudly instead of mislabeling the number.
        body = self._driver(["fak-cuda"])
        self.assertIn("-backend cuda", body)
        self.assertIn("-require-non-reference", body)
        self.assertNotIn("modelbench-cuda\" -gguf \"$MODEL\" -lean", body)
        self.assertIn("build_cuda.sh build", body)

    def test_fak_cuda_q8_renders_lean_device_q8_with_honesty_gate(self):
        # fak-cuda-q8 is the apples-to-apples Q8 device GEMV row: -lean (resident Q8
        # codes+scales, native device GEMV) + -backend cuda + the non-reference gate.
        body = self._driver(["fak-cuda-q8"])
        self.assertIn("engine_fak_cuda_q8()", body)
        self.assertIn("run_one fak-cuda-q8 engine_fak_cuda_q8", body)
        self.assertIn("modelbench-cuda\" -gguf \"$MODEL\" -lean -backend cuda", body)
        self.assertIn("-require-non-reference", body)
        # It reuses the binary fak-cuda built rather than unconditionally recompiling.
        self.assertIn("[ ! -x \"$WORK/bin/modelbench-cuda\" ]", body)

    def test_fak_cuda_tf32_renders_f32_path_with_tf32_env(self):
        # fak-cuda-tf32 is the TF32 tensor-core PREFILL row (Lever 4): the f32 device path
        # (NOT -lean, so f32 weights stay resident) with FAK_CUDA_TF32=1 exported so the f32
        # SGEMM runs on the tensor cores. It reuses the modelbench-cuda binary + keeps the
        # non-reference honesty gate.
        body = self._driver(["fak-cuda-tf32"])
        self.assertIn("engine_fak_cuda_tf32()", body)
        self.assertIn("run_one fak-cuda-tf32 engine_fak_cuda_tf32", body)
        # The TF32 env knob is what flips the cuBLAS math mode on the f32 GEMM.
        self.assertIn("FAK_CUDA_TF32=1", body)
        self.assertIn("-require-non-reference", body)
        # It reuses the binary fak-cuda built rather than unconditionally recompiling.
        self.assertIn("[ ! -x \"$WORK/bin/modelbench-cuda\" ]", body)
        # This is the f32 device path: the TF32 row must NOT pass -lean (that would be the Q8
        # device GEMV row, fak-cuda-q8). The tensor-core lever applies to the f32 SGEMM.
        self.assertNotIn("modelbench-cuda\" -gguf \"$MODEL\" -lean", body)

    def test_fak_cpu_uses_lean_q8(self):
        body = self._driver(["fak-cpu"])
        self.assertIn("-gguf \"$MODEL\" -lean", body)

    def test_vllm_engine_fragment_renders_and_discloses_precision(self):
        body = self._driver(["llama", "vllm"])
        self.assertNotIn("@@", body)
        self.assertIn("engine_vllm()", body)
        self.assertIn("run_one vllm engine_vllm", body)
        self.assertIn("bench throughput", body)
        self.assertIn("pip install -q -U vllm", body)
        self.assertIn("norm_vllm.py", body)
        # vLLM runs the HF (bf16) form of the SAME model, derived by stripping -GGUF.
        self.assertIn("${MODEL_REPO%-GGUF}", body)
        self.assertIn("--dtype bfloat16", body)
        # The norm helper is written to the VM and discloses the precision/aggregate
        # difference rather than silently passing it off as the llama/fak Q8 number.
        self.assertIn('cat > "$WORK/norm_vllm.py"', body)
        self.assertIn("precision", gcp_bench._PY_NORM_VLLM)

    def test_vllm_normalizer_picks_output_throughput_and_never_fabricates(self):
        # Exercise the real _PY_NORM_VLLM helper hermetically: a sample vLLM throughput
        # JSON -> a normalized row with the output tok/s as decode, bf16 precision, and
        # an honest aggregate/precision note. A JSON with no known key must mark ok=False
        # (a diagnosed miss), NEVER invent a number.
        import json
        import os
        import subprocess
        import sys
        import tempfile
        d = tempfile.mkdtemp()
        helper = Path(d) / "norm_vllm.py"
        gcp_bench.write_lf(helper, gcp_bench._PY_NORM_VLLM.strip("\n") + "\n")

        good = Path(d) / "vllm-throughput.json"
        good.write_text(json.dumps({
            "requests_per_second": 7.1, "output_throughput": 1234.5,
            "total_token_throughput": 4567.8, "elapsed_time": 9.0,
        }), encoding="utf-8")
        out = Path(d) / "engine-vllm.json"
        subprocess.run([sys.executable, str(helper), str(good), "Qwen/Qwen2.5-3B-Instruct", str(out)],
                       check=True)
        row = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(row["engine"], "vllm")
        self.assertTrue(row["ok"])
        self.assertEqual(row["decode_tok_per_sec"], 1234.5)
        self.assertEqual(row["precision"], "bf16")
        self.assertIsNone(row["prefill_tok_per_sec"])
        self.assertIn("AGGREGATE", row["note"])

        # No recognizable throughput key + no total/elapsed pair -> ok=False, not a fake.
        empty = Path(d) / "empty.json"
        empty.write_text(json.dumps({"requests_per_second": 7.1}), encoding="utf-8")
        out2 = Path(d) / "engine-vllm-2.json"
        subprocess.run([sys.executable, str(helper), str(empty), "m", str(out2)], check=True)
        row2 = json.loads(out2.read_text(encoding="utf-8"))
        self.assertFalse(row2["ok"])
        self.assertIsNone(row2["decode_tok_per_sec"])
        for p in (helper, good, out, empty, out2):
            os.remove(p)

    def test_arch_threaded_from_compute_capability(self):
        body = self._driver(["fak-cuda"], cc="100")  # Blackwell sm_100
        self.assertIn('CUDA_CC="100"', body)
        self.assertIn('FAK_CUDA_ARCH="$CUDA_CC"', body)

    def test_combiner_reports_failure_when_all_engines_fail(self):
        # The combiner must NOT emit a false-green result.json: an all-failed run
        # carries ok=false + an error so the catalog never records a phantom success.
        self.assertIn('out["ok"]=n_ok>0', gcp_bench._PY_COMBINE)
        self.assertIn("all %d engine(s) failed", gcp_bench._PY_COMBINE)

    def test_fatal_builds_json_via_python_not_raw_interpolation(self):
        body = self._driver(["llama"])
        # robust JSON (python3, present pre-apt) rather than echo-with-raw-$1
        self.assertIn("import json,sys;open(sys.argv[2]", body)

    def test_write_lf_has_no_carriage_returns(self):
        # The real failure mode: write_text translates \n->\r\n on Windows and the
        # VM's bash rejects CRLF. write_lf must emit pure LF regardless of host OS.
        import tempfile
        import os
        body = self._driver(["llama", "fak-cpu", "fak-cuda"])
        self.assertIn("\n", body)
        d = tempfile.mkdtemp()
        p = Path(d) / "drv.sh"
        gcp_bench.write_lf(p, body)
        raw = p.read_bytes()
        self.assertNotIn(b"\r", raw, "rendered driver must be LF-only (CRLF breaks bash on the VM)")
        os.remove(p)


class SourceTarballLayoutTest(unittest.TestCase):
    """The on-VM driver guards on `$SRC/go.mod` where $SRC=$WORK/fak, so the
    tarball MUST carry a fak/go.mod member regardless of repo layout. A
    fak/-prefix regression here silently spends a VM then fatals the driver."""

    def test_fak_src_is_a_real_module_root(self):
        # Auto-detect must land on a dir that actually has a go.mod.
        self.assertTrue((gcp_bench.FAK_SRC / "go.mod").is_file(),
                        f"FAK_SRC {gcp_bench.FAK_SRC} has no go.mod")

    def test_tarball_contains_fak_go_mod_and_excludes_root_binary(self):
        import tarfile
        import tempfile
        import os
        d = tempfile.mkdtemp()
        dest = Path(d) / "src.tgz"
        gcp_bench.make_source_tarball(dest, dry_run=False)
        with tarfile.open(dest, "r:gz") as tf:
            names = tf.getnames()
        self.assertIn("fak/go.mod", names,
                      "driver's $SRC/go.mod guard needs a fak/go.mod member")
        # When the source is the repo root, the compiled root `fak` binary must
        # not ride along as fak/fak.
        if gcp_bench.FAK_SRC == gcp_bench.ROOT:
            self.assertNotIn("fak/fak", names, "root fak binary must be excluded")
        os.remove(dest)


if __name__ == "__main__":
    unittest.main()
