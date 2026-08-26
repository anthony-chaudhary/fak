#!/usr/bin/env python3
"""Behavior tests for GLM docs-lane guard routing and quota backoff."""

import datetime as dt
import os
import tempfile
from pathlib import Path

import dispatch_glm_docs as dispatch


def test_guard_gateway_default_override_and_operator_precedence():
    saved = os.environ.get("FAK_GLM_GUARD_GATEWAY")
    try:
        os.environ.pop("FAK_GLM_GUARD_GATEWAY", None)
        env = {}
        assert dispatch.apply_glm_guard_gateway(env) == dispatch.DEFAULT_GLM_GUARD_GATEWAY
        assert env[dispatch.GLM_GUARD_BASEURL_ENV] == dispatch.DEFAULT_GLM_GUARD_GATEWAY

        env = {dispatch.GLM_GUARD_BASEURL_ENV: "http://operator.example/v1"}
        os.environ["FAK_GLM_GUARD_GATEWAY"] = "http://ignored.example/v1"
        assert dispatch.apply_glm_guard_gateway(env) == "http://operator.example/v1"
    finally:
        if saved is None:
            os.environ.pop("FAK_GLM_GUARD_GATEWAY", None)
        else:
            os.environ["FAK_GLM_GUARD_GATEWAY"] = saved


def test_exhaustion_uses_freshest_matching_reset_and_ignores_expired():
    with tempfile.TemporaryDirectory() as td:
        runs = Path(td)
        expired = (dt.date.today() - dt.timedelta(days=1)).isoformat()
        future = (dt.date.today() + dt.timedelta(days=2)).isoformat()
        old = runs / "resolve-old.log"
        old.write_text(
            f"zai-coding-plan Limit Exhausted reset at {expired}", encoding="utf-8"
        )
        fresh = runs / "resolve-fresh.log"
        fresh.write_text(
            f"zai-coding-plan Limit Exhausted reset at {future}", encoding="utf-8"
        )
        old.touch()
        fresh.touch()
        assert dispatch.glm_provider_exhausted(runs) == future


def test_gateway_preflight_turns_probe_exception_into_typed_api_error():
    def broken_probe(_row, **_kwargs):
        raise RuntimeError("boom")

    verdict = dispatch.gateway_preflight({"tag": "seat"}, probe=broken_probe)
    assert verdict["status"] == "APIERR"
    assert verdict["block_kind"] == "apierr"
    assert "boom" in verdict["block_reason"]

