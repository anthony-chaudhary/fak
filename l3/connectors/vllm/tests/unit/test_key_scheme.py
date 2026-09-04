"""Tests for l3_vllm.key_scheme."""

from l3_vllm.key_scheme import KeyScheme, KeySchemeConfig


def test_mha_keys_no_pp():
    ks = KeyScheme(KeySchemeConfig(tp_rank=0, pp_rank=0, pp_size=1, is_mla=False))
    keys = ks.keys_for_block("abc123")
    assert keys == ["vllm_l3_abc123_0_k", "vllm_l3_abc123_0_v"]


def test_mha_keys_with_pp():
    ks = KeyScheme(KeySchemeConfig(tp_rank=2, pp_rank=1, pp_size=4, is_mla=False))
    keys = ks.keys_for_block("abc")
    assert keys == ["vllm_l3_abc_2_1_k", "vllm_l3_abc_2_1_v"]


def test_mla_keys_no_pp():
    ks = KeyScheme(KeySchemeConfig(tp_rank=0, pp_rank=0, pp_size=1, is_mla=True))
    keys = ks.keys_for_block("xyz")
    assert keys == ["vllm_l3_xyz"]


def test_mla_keys_with_pp():
    ks = KeyScheme(KeySchemeConfig(tp_rank=0, pp_rank=2, pp_size=4, is_mla=True))
    keys = ks.keys_for_block("xyz")
    assert keys == ["vllm_l3_xyz_2"]


def test_block_hash_bytes_to_hex():
    ks = KeyScheme(KeySchemeConfig())
    keys = ks.keys_for_block(b"\xde\xad\xbe\xef")
    assert keys[0].startswith("vllm_l3_deadbeef")


def test_block_hash_int_to_hex():
    ks = KeyScheme(KeySchemeConfig())
    keys = ks.keys_for_block(0xCAFE)
    assert keys[0].startswith("vllm_l3_cafe")


def test_engine_namespace_override():
    ks = KeyScheme(KeySchemeConfig(engine_namespace="vllm_prefill"))
    keys = ks.keys_for_block("a1")
    assert all(k.startswith("vllm_prefill_a1") for k in keys)


def test_keys_per_block():
    assert KeyScheme(KeySchemeConfig(is_mla=False)).keys_per_block == 2
    assert KeyScheme(KeySchemeConfig(is_mla=True)).keys_per_block == 1


def test_keys_for_blocks_flattens():
    ks = KeyScheme(KeySchemeConfig(is_mla=False))
    out = ks.keys_for_blocks(["a", "b"])
    assert out == [
        "vllm_l3_a_0_k", "vllm_l3_a_0_v",
        "vllm_l3_b_0_k", "vllm_l3_b_0_v",
    ]
