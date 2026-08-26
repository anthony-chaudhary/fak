#!/usr/bin/env python3
"""Request-shape tests for GCP quota preference planning."""

import gcp_quota_request as quota


def test_build_requests_covers_global_and_family_limits():
    specs = quota.build_requests(
        project="p",
        gpus=8,
        region="us-central1",
        family="NVIDIA_B200",
        contact="ops@example.com",
        just="benchmark capacity",
    )

    assert len(specs) == 2
    global_spec, family_spec = specs
    assert global_spec["body"]["quotaId"] == "GPUS-ALL-REGIONS-per-project"
    assert family_spec["body"]["quotaId"] == "GPUS-PER-GPU-FAMILY-per-project-region"
    assert family_spec["body"]["dimensions"] == {
        "region": "us-central1",
        "gpu_family": "NVIDIA_B200",
    }
    assert all(spec["body"]["quotaConfig"]["preferredValue"] == 8 for spec in specs)
    assert all(spec["body"]["contactEmail"] == "ops@example.com" for spec in specs)


def test_preference_url_distinguishes_create_from_patch():
    create = quota.preference_url("my-project", "pref-one", create=True)
    patch = quota.preference_url("my-project", "pref-one", create=False)
    assert create.endswith("/quotaPreferences?quotaPreferenceId=pref-one")
    assert patch.endswith("/quotaPreferences/pref-one")

