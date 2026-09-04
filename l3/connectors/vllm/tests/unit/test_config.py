"""Tests for l3_vllm.config.CamaConnectorConfig."""

from l3_vllm.config import CamaConnectorConfig


def test_defaults():
    c = CamaConnectorConfig.from_extra_config({})
    assert c.remote_addr == "localhost"
    assert c.remote_port == 18001
    assert c.engine_namespace == "vllm_l3"


def test_overrides_and_unknown_keys_ignored(caplog):
    import logging
    caplog.set_level(logging.WARNING)
    c = CamaConnectorConfig.from_extra_config({
        "remote_addr": "10.0.0.5",
        "remote_port": 9999,
        "engine_namespace": "vllm_prefill",
        "totally_made_up_key": True,
    })
    assert c.remote_addr == "10.0.0.5"
    assert c.remote_port == 9999
    assert c.engine_namespace == "vllm_prefill"
    assert any("totally_made_up_key" in rec.message for rec in caplog.records)


def test_none_extra():
    c = CamaConnectorConfig.from_extra_config(None)
    assert c.remote_addr == "localhost"
