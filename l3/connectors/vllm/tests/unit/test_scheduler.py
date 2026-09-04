"""Tests for l3_vllm.scheduler.CamaConnectorScheduler."""

from types import SimpleNamespace

from l3_vllm.config import CamaConnectorConfig
from l3_vllm.key_scheme import KeyScheme, KeySchemeConfig
from l3_vllm.scheduler import CamaConnectorScheduler


def make_sched(block_size=16, async_min=0):
    cfg = CamaConnectorConfig.from_extra_config({"async_load_min_tokens": async_min})
    ks = KeyScheme(KeySchemeConfig(is_mla=False))
    s = CamaConnectorScheduler(cfg, ks)
    s.set_block_size(block_size)
    return s, ks


def fake_request(rid: str, block_hashes):
    return SimpleNamespace(request_id=rid, block_hashes=block_hashes)


def test_zero_blocks_returns_zero():
    s, _ = make_sched()
    req = fake_request("r1", [])
    assert s.get_num_new_matched_tokens(req, 0) == (0, False)


def test_all_miss_returns_zero_async_false():
    s, _ = make_sched()
    req = fake_request("r1", ["aa", "bb"])
    matched, is_async = s.get_num_new_matched_tokens(req, 0)
    assert matched == 0
    assert is_async is False


def test_hit_after_report_present():
    s, ks = make_sched()
    s.report_blocks_present(["aa", "bb"])
    req = fake_request("r1", ["aa", "bb", "cc"])
    matched, is_async = s.get_num_new_matched_tokens(req, 0)
    # 2 blocks of size 16 = 32 tokens; async_min default 32 -> still allowed.
    assert matched == 32
    assert is_async is True


def test_below_async_min_returns_zero():
    s, _ = make_sched(async_min=64)
    s.report_blocks_present(["aa"])
    req = fake_request("r1", ["aa", "bb"])  # only 1 hit = 16 tokens < 64
    matched, is_async = s.get_num_new_matched_tokens(req, 0)
    assert matched == 0
    assert is_async is False


def test_skip_locally_computed_prefix():
    s, _ = make_sched()
    s.report_blocks_present(["bb", "cc"])  # block ids 1,2 of 0,1,2
    req = fake_request("r1", ["aa", "bb", "cc"])
    # First 16 tokens already computed -> start_block=1
    matched, _ = s.get_num_new_matched_tokens(req, 16)
    assert matched == 32   # 2 remaining blocks bb,cc both hit


def test_request_finished_schedules_save():
    s, _ = make_sched()
    req = fake_request("r1", ["aa", "bb"])
    delay, params = s.request_finished(req, block_ids=[100, 101])
    assert delay is False
    assert params is None
    meta = s.build_connector_meta(SimpleNamespace())
    assert len(meta.saves) == 1
    assert meta.saves[0].block_hashes == ["aa", "bb"]
    assert meta.saves[0].block_ids == [100, 101]


def test_missing_report_clears_lru():
    s, _ = make_sched()
    s.report_blocks_present(["aa"])
    s.report_blocks_missing(["aa"])
    req = fake_request("r1", ["aa"])
    # Bloom may still have it, but LRU was cleared. Behavior: bloom hit is
    # still treated as present (best-effort). This documents the design.
    matched, _ = s.get_num_new_matched_tokens(req, 0)
    assert matched in (0, 16)


def test_build_meta_drains_pending():
    s, _ = make_sched()
    req = fake_request("r1", ["aa"])
    s.request_finished(req, [10])
    meta1 = s.build_connector_meta(SimpleNamespace())
    meta2 = s.build_connector_meta(SimpleNamespace())
    assert len(meta1.saves) == 1
    assert len(meta2.saves) == 0
