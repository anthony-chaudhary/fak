#!/usr/bin/env python3
"""Tests for gcp_accel.py + the pure-data paths of gcp_gpu_probe.py.

No GCP calls: everything here is the registry + the quota-verdict logic fed
synthetic quota maps, so it runs in the offline suite.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gcp_accel
import gcp_gpu_probe as probe


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def test_blackwell_tiers_exist_with_exact_machine_types():
    a4 = gcp_accel.by_slug("a4-b200")
    assert a4 is not None
    assert a4.machine_type == "a4-highgpu-8g"
    assert a4.accelerator_type == "nvidia-b200"
    assert a4.gpu_count == 8
    assert a4.blackwell is True
    a4x = gcp_accel.by_slug("a4x-gb200")
    assert a4x.machine_type == "a4x-highgpu-4g"
    assert a4x.accelerator_type == "nvidia-gb200"
    assert a4x.blackwell is True


def test_a100_tiers_exist_for_the_available_path():
    # "whatever is available, a100 ok": the A100 serving tiers the GCP bring-up branches to
    # the pure-fak-kernel / llama.cpp paths for (both Ampere sm_80, below the DSA floor).
    ultra = gcp_accel.by_slug("a2-ultra-a100-80gb")
    assert ultra is not None
    assert ultra.machine_type == "a2-ultragpu-8g"
    # 80GB A100 dropped the Tesla prefix; 40GB keeps it (verified vs GCP GPU docs 2026-06-26).
    assert ultra.accelerator_type == "nvidia-a100-80gb"
    assert ultra.gpu_count == 8
    assert ultra.gpu_mem_gb_each == 80
    assert ultra.arch == "ampere"
    assert ultra.compute_capability == "80"  # sm_80, below the sm_90 DSA floor
    assert ultra.blackwell is False

    high = gcp_accel.by_slug("a2-high-a100-40gb")
    assert high is not None
    assert high.machine_type == "a2-highgpu-8g"
    assert high.accelerator_type == "nvidia-tesla-a100"
    assert high.gpu_count == 8
    assert high.gpu_mem_gb_each == 40
    assert high.compute_capability == "80"


def test_a100_emit_shell_reports_sub_dsa_compute_cap():
    # gcp-glm-serve.sh branches to the llama.cpp / pure-fak-kernel path when GLM_COMPUTE_CAP < 90.
    out = gcp_accel.emit_shell("a2-ultra-a100-80gb")
    lines = dict(line.split("=", 1) for line in out.splitlines())
    import shlex
    got = {k: shlex.split(v)[0] if v else "" for k, v in lines.items()}
    assert got["GLM_MACHINE_TYPE"] == "a2-ultragpu-8g"
    assert got["GLM_ACCEL_FLAG"] == "type=nvidia-a100-80gb,count=8"
    assert got["GLM_COMPUTE_CAP"] == "80"
    assert int(got["GLM_COMPUTE_CAP"]) < 90


def test_a100_tiers_rank_below_hopper_above_l4():
    pos = {t.slug: i for i, t in enumerate(gcp_accel.fallback_ladder())}
    # Datacenter A100 outranks the cheap Ada/Turing proof tiers (better serving choice) but
    # sits below the Hopper baseline in the preference ladder.
    assert pos["a3-high-h100"] < pos["a2-ultra-a100-80gb"] < pos["g2-l4"]
    assert pos["a2-ultra-a100-80gb"] < pos["a2-high-a100-40gb"]


def test_ladder_is_newest_silicon_first():
    ladder = gcp_accel.fallback_ladder()
    ranks = [t.gen_rank for t in ladder]
    assert ranks == sorted(ranks, reverse=True), "ladder must be newest-first"
    assert ladder[0].blackwell, "newest tier in the full ladder is Blackwell"


def test_blackwell_only_ladder_excludes_hopper_and_older():
    bw = gcp_accel.fallback_ladder(blackwell_only=True)
    assert bw, "there is at least one Blackwell tier"
    assert all(t.blackwell for t in bw)
    slugs = {t.slug for t in bw}
    assert "a3-ultra-h200" not in slugs
    assert "n1-t4" not in slugs


def test_proof_tier_is_cheapest():
    t = gcp_accel.proof_tier()
    assert t.approx_usd_per_hour == min(x.approx_usd_per_hour for x in gcp_accel.TIERS)
    assert t.gpu_count == 1, "the proof tier is a single-GPU VM"


def test_accelerator_flag_shape():
    a4 = gcp_accel.by_slug("a4-b200")
    assert gcp_accel.accelerator_flag(a4) == "type=nvidia-b200,count=8"


def test_boot_image_is_a_cuda_image():
    fam, proj = gcp_accel.boot_image()
    assert "cu" in fam  # a CUDA image family
    assert proj == "deeplearning-platform-release"


def test_emit_shell_is_evalable_and_matches_the_registry():
    # The bring-up script (scripts/gcp-glm-serve.sh) sources these, so they must be the
    # exact registry strings, eval-able, and prefixed.
    out = gcp_accel.emit_shell("a3-ultra-h200")
    t = gcp_accel.by_slug("a3-ultra-h200")
    fam, proj = gcp_accel.boot_image()
    lines = dict(line.split("=", 1) for line in out.splitlines())
    # shlex.quote wraps nothing for bare tokens; strip any quoting for the compare.
    import shlex
    got = {k: shlex.split(v)[0] if v else "" for k, v in lines.items()}
    assert got["GLM_MACHINE_TYPE"] == t.machine_type == "a3-ultragpu-8g"
    assert got["GLM_ACCEL_FLAG"] == "type=nvidia-h200-141gb,count=8"
    assert got["GLM_GPU_COUNT"] == "8"
    assert got["GLM_COMPUTE_CAP"] == "90"  # sm_90+, the DSA floor the preflight enforces
    assert got["GLM_IMAGE_FAMILY"] == fam
    assert got["GLM_IMAGE_PROJECT"] == proj
    assert got["GLM_DEFAULT_ZONE"] == t.common_zones[0]


def test_emit_shell_custom_prefix_and_unknown_slug():
    out = gcp_accel.emit_shell("a4-b200", prefix="X")
    assert "X_MACHINE_TYPE=" in out and "X_BLACKWELL=1" in out
    import pytest
    with pytest.raises(KeyError):
        gcp_accel.emit_shell("no-such-tier")


# --------------------------------------------------------------------------- #
# Probe verdict logic (synthetic quota maps -- no network)
# --------------------------------------------------------------------------- #

def _quota(global_cap, grants=None):
    return {"global": global_cap, "by_family_region": dict(grants or {})}


def test_global_cap_blocks_multi_gpu_tier_even_with_family_grant():
    # Family grant is generous, but the project-global cap is 1 -> NO_QUOTA.
    a4 = gcp_accel.by_slug("a4-b200")
    q = _quota(1, {("NVIDIA_B200", "us-central1"): 8})
    v = probe.probe_tier(a4, project="p", account="a", quota=q,
                         reservation_list=[], zone_override="us-central1-b")
    assert v["verdict"] == "NO_QUOTA"
    assert "global GPU cap" in v["reason"]


def test_family_grant_plus_global_headroom_is_provisionable(monkeypatch):
    # offered=True is the only piece that needs gcloud; stub it.
    monkeypatch.setattr(probe, "accelerator_offered", lambda *a, **k: True)
    a4 = gcp_accel.by_slug("a4-b200")
    q = _quota(8, {("NVIDIA_B200", "us-central1"): 8})
    v = probe.probe_tier(a4, project="p", account="a", quota=q,
                         reservation_list=[], zone_override="us-central1-b")
    assert v["verdict"] == "PROVISIONABLE", v
    assert v["zone"] == "us-central1-b"


def test_reservation_rescues_zero_family_grant(monkeypatch):
    monkeypatch.setattr(probe, "accelerator_offered", lambda *a, **k: True)
    a4 = gcp_accel.by_slug("a4-b200")
    q = _quota(8, {})  # no family grant at all
    reservations = [{
        "name": "my-b200-res",
        "specificReservation": {"instanceProperties": {"machineType": "a4-highgpu-8g"}},
    }]
    v = probe.probe_tier(a4, project="p", account="a", quota=q,
                         reservation_list=reservations, zone_override="us-central1-b")
    assert v["verdict"] == "PROVISIONABLE", v
    assert v["reservation"] == "my-b200-res"


def test_not_offered_when_zone_lacks_the_accelerator(monkeypatch):
    monkeypatch.setattr(probe, "accelerator_offered", lambda *a, **k: False)
    a4 = gcp_accel.by_slug("a4-b200")
    q = _quota(8, {("NVIDIA_B200", "us-central1"): 8})
    v = probe.probe_tier(a4, project="p", account="a", quota=q,
                         reservation_list=[], zone_override="us-central1-b")
    assert v["verdict"] == "NOT_OFFERED"


def test_limit_val_parses_string_value():
    assert probe._limit_val({"details": {"value": "1"}}) == 1.0
    assert probe._limit_val({"details": {"value": 8}}) == 8.0
    assert probe._limit_val({"details": {}}) is None
    assert probe._limit_val({}) is None


def test_zone_region_strips_zone_suffix():
    assert probe.zone_region("us-central1-b") == "us-central1"
    assert probe.zone_region("europe-west4-a") == "europe-west4"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
