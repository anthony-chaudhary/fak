#!/usr/bin/env python3
"""Offline parsing and quota-verdict tests for the GCP GPU probe."""

import gcp_accel
import gcp_gpu_probe as probe


def test_zone_and_quota_value_parsing_are_fail_closed():
    assert probe.zone_region("us-central1-b") == "us-central1"
    assert probe.zone_region("europe-west4-a") == "europe-west4"
    assert probe._limit_val({"details": {"value": "8"}}) == 8.0
    assert probe._limit_val({"details": {}}) is None
    assert probe._limit_val({}) is None


def test_global_gpu_cap_blocks_tier_before_offering_probe():
    tier = gcp_accel.by_slug("a4-b200")
    verdict = probe.probe_tier(
        tier,
        project="p",
        account="a",
        quota={
            "global": 1,
            "by_family_region": {("NVIDIA_B200", "us-central1"): 8},
        },
        reservation_list=[],
        zone_override="us-central1-b",
    )
    assert verdict["verdict"] == "NO_QUOTA"
    assert "global GPU cap" in verdict["reason"]


def test_stale_detection_recognizes_auth_and_project_drift():
    assert probe._looks_stale("Reauthentication failed; run gcloud auth login")
    assert probe._looks_stale("invalid_grant while refreshing the access token")
    assert not probe._looks_stale("quota unavailable in requested region")
