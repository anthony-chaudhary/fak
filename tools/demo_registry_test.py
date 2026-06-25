#!/usr/bin/env python3
"""Tests for the shared browser-demo registry."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_registry as dr  # noqa: E402


def test_registry_has_unique_names_base_paths_and_api_paths() -> None:
    names = [d.name for d in dr.DEMOS]
    bases = [d.base_path for d in dr.DEMOS]
    assert len(names) == len(set(names)), names
    assert len(bases) == len(set(bases)), bases
    for demo in dr.DEMOS:
        assert demo.base_path.startswith("/"), demo
        assert demo.api_path.startswith("api/"), demo
        assert demo.page_marker, demo
        assert demo.default_port > 0, demo
        if demo.hosted_path:
            assert demo.hosted_path.startswith("/"), demo
            assert demo.hosted_api_keys, demo


def test_demo_url_normalizes_base_path_and_relative_api() -> None:
    assert dr.demo_url(1234, "guarddemo") == "http://127.0.0.1:1234/guarddemo/"
    assert dr.demo_url(1234, "/guarddemo/", "/api/scenarios") == "http://127.0.0.1:1234/guarddemo/api/scenarios"


def test_hosted_demo_urls_come_from_registry_metadata() -> None:
    urls = dr.hosted_demo_urls("136.111.250.205")
    assert urls == {
        "turntaxdemo": "http://136.111.250.205:8150/",
        "ctxdemo": "http://136.111.250.205:8153/",
        "demorace": "http://136.111.250.205/demorace/",
    }, urls


def test_hosted_url_omits_local_only_demo() -> None:
    guard = dr.demo_map()["guarddemo"]
    assert dr.hosted_url("136.111.250.205", guard) == ""


def test_select_demos_reports_unknown_without_dropping_known() -> None:
    demos, unknown = dr.select_demos(["guarddemo", "ghost"])
    assert [d.name for d in demos] == ["guarddemo"]
    assert unknown == ["ghost"]


def test_registry_defects_flags_unregistered_page_demo() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cmd" / "guarddemo").mkdir(parents=True)
        (root / "cmd" / "newdemo").mkdir(parents=True)
        (root / "cmd" / "guarddemo" / "page.html").write_text("<html></html>", encoding="utf-8")
        (root / "cmd" / "newdemo" / "page.html").write_text("<html></html>", encoding="utf-8")
        demos = (dr.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety"),)
        defects = dr.registry_defects(root, demos=demos)
        assert defects == ["browser demo has page.html but is not in DEMOS registry: cmd/newdemo"], defects


def test_registry_defects_flags_stale_registered_demo() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cmd" / "guarddemo").mkdir(parents=True)
        (root / "cmd" / "guarddemo" / "page.html").write_text("<html></html>", encoding="utf-8")
        demos = (
            dr.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety"),
            dr.Demo("ghostdemo", "/ghost", "api/ping", "ghost"),
        )
        defects = dr.registry_defects(root, demos=demos)
        assert defects == ["DEMOS registry names a browser demo without page.html: cmd/ghostdemo"], defects


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("demo_registry_test: OK")
