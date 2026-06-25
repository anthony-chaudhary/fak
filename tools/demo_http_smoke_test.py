#!/usr/bin/env python3
"""Tests for demo_http_smoke.py.

These unit tests do not start Go servers; the script itself is the dynamic smoke.
Run: `python tools/demo_http_smoke_test.py`, or
`python -m pytest tools/demo_http_smoke_test.py -q`.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_http_smoke as dhs  # noqa: E402


def test_demo_url_normalizes_base_path_and_relative_api() -> None:
    assert dhs.demo_url(1234, "guarddemo") == "http://127.0.0.1:1234/guarddemo/"
    assert dhs.demo_url(1234, "/guarddemo/", "/api/scenarios") == "http://127.0.0.1:1234/guarddemo/api/scenarios"


def test_select_demos_reports_unknown_without_dropping_known() -> None:
    demos, unknown = dhs.select_demos(["guarddemo", "ghost"])
    assert [d.name for d in demos] == ["guarddemo"]
    assert unknown == ["ghost"]


def test_demo_registry_has_unique_base_paths_and_api_paths() -> None:
    names = [d.name for d in dhs.DEMOS]
    bases = [d.base_path for d in dhs.DEMOS]
    assert len(names) == len(set(names)), names
    assert len(bases) == len(set(bases)), bases
    for demo in dhs.DEMOS:
        assert demo.base_path.startswith("/"), demo
        assert demo.api_path.startswith("api/"), demo
        assert demo.page_marker, demo


def test_smoke_demo_reports_both_base_path_sources(monkeypatch=None) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def fake_smoke_server(workspace, exe, demo, timeout_s, base_source):  # type: ignore[no-untyped-def]
        calls.append(base_source)
        return {"demo": demo.name, "ok": True, "base_source": base_source, "defects": []}

    old_build = dhs.build_demo
    old_smoke = dhs.smoke_server
    try:
        dhs.build_demo = lambda workspace, tmp, demo, timeout_s: (tmp / "demo.exe", "")  # type: ignore[assignment]
        dhs.smoke_server = fake_smoke_server  # type: ignore[assignment]
        row = dhs.smoke_demo(Path("."), Path("."), dhs.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety"), 1.0)
    finally:
        dhs.build_demo = old_build  # type: ignore[assignment]
        dhs.smoke_server = old_smoke  # type: ignore[assignment]

    assert row["ok"], row
    assert calls == ["flag", "env"], calls


def test_registry_defects_flags_unregistered_page_demo() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cmd" / "guarddemo").mkdir(parents=True)
        (root / "cmd" / "newdemo").mkdir(parents=True)
        (root / "cmd" / "guarddemo" / "page.html").write_text("<html></html>", encoding="utf-8")
        (root / "cmd" / "newdemo" / "page.html").write_text("<html></html>", encoding="utf-8")
        demos = (dhs.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety"),)
        defects = dhs.registry_defects(root, demos=demos)
        assert defects == ["browser demo has page.html but is not in DEMOS registry: cmd/newdemo"], defects


def test_registry_defects_flags_stale_registered_demo() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "cmd" / "guarddemo").mkdir(parents=True)
        (root / "cmd" / "guarddemo" / "page.html").write_text("<html></html>", encoding="utf-8")
        demos = (
            dhs.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety"),
            dhs.Demo("ghostdemo", "/ghost", "api/ping", "ghost"),
        )
        defects = dhs.registry_defects(root, demos=demos)
        assert defects == ["DEMOS registry names a browser demo without page.html: cmd/ghostdemo"], defects


def test_page_static_defects_accepts_relative_api_references() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        page = root / "cmd" / "guarddemo" / "page.html"
        page.parent.mkdir(parents=True)
        page.write_text("fetch('api/scenarios'); new EventSource(`api/run`);", encoding="utf-8")
        demo = dhs.Demo("guarddemo", "/guarddemo", "api/scenarios", "safety")
        assert dhs.page_static_defects(root, demo) == []


def test_page_static_defects_rejects_root_relative_browser_refs_and_missing_api() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        page = root / "cmd" / "guarddemo" / "page.html"
        page.parent.mkdir(parents=True)
        page.write_text("fetch('/api/scenarios'); const css='/static/app.css';", encoding="utf-8")
        demo = dhs.Demo("guarddemo", "/guarddemo", "api/run", "safety")
        defects = dhs.page_static_defects(root, demo)
        assert "guarddemo: page.html does not reference registered API path 'api/run'" in defects
        assert "guarddemo: page.html has root-relative browser reference '/api/scenarios'; use a relative path" in defects
        assert "guarddemo: page.html has root-relative browser reference '/static/app.css'; use a relative path" in defects


def test_validate_api_payload_checks_expected_shapes() -> None:
    assert dhs.validate_api_payload(dhs.Demo("guarddemo", "/g", "api/scenarios", "x"), {"scenarios": [{"id": "x"}]}) == []
    assert dhs.validate_api_payload(dhs.Demo("turntaxdemo", "/t", "api/suites", "x"), {"suites": [{"id": "x"}]}) == []
    assert dhs.validate_api_payload(
        dhs.Demo("demorace", "/r", "api/ladder", "x"),
        {"models": [{"name": "tiny"}], "prefill_tok_ratio": {"b_over_c": 2.0}},
    ) == []
    assert dhs.validate_api_payload(
        dhs.Demo("unseedemo", "/u", "api/events", "x"),
        {"witness": {}, "frames": [{"act": 1}], "fences": [{"label": "ok"}]},
    ) == []

    assert "guarddemo API missing non-empty scenarios list" in dhs.validate_api_payload(
        dhs.Demo("guarddemo", "/g", "api/scenarios", "x"), {"scenarios": []}
    )
    assert "demorace API missing prefill_tok_ratio.b_over_c > 1" in dhs.validate_api_payload(
        dhs.Demo("demorace", "/r", "api/ladder", "x"), {"models": [{"name": "tiny"}]}
    )


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("demo_http_smoke_test: OK")
