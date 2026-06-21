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
