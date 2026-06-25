#!/usr/bin/env python3
"""Tests for demo_live_links.py.

Run: `python tools/demo_live_links_test.py`, or
`python -m pytest tools/demo_live_links_test.py -q`.
"""
from __future__ import annotations

import contextlib
import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import demo_live_links as dl  # noqa: E402


GOOD = """
<!-- The demo host is plain HTTP. -->
<link rel="canonical" href="https://anthony-chaudhary.github.io/fak/demos.html">
<meta property="og:url" content="https://anthony-chaudhary.github.io/fak/demos.html">
<meta property="og:image" content="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/social-preview.png">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:image" content="https://raw.githubusercontent.com/anthony-chaudhary/fak/main/visuals/social-preview.png">
<a class="card" href="run-the-demos.html">guard</a>
<a class="card" href="http://136.111.250.205:8150/">turntax</a>
<a class="card" href="http://136.111.250.205:8153/">ctx</a>
<a class="card" href="http://136.111.250.205/demorace/">race</a>
<a href="http://136.111.250.205/">hub</a>
"""

PNG = b"\x89PNG\r\n\x1a\n" + (b"x" * 2048)


def write_social_preview(root: Path, data: bytes = PNG) -> None:
    path = root / "visuals" / "social-preview.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_demo_doc(root: Path, html: str = GOOD) -> None:
    path = root / "docs" / "demos.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(html, encoding="utf-8")


def test_hosted_witnesses_reuse_browser_demo_registry() -> None:
    assert dl.HOSTED_DEMO_URLS["turntaxdemo"] == "http://136.111.250.205:8150/"
    assert dl.HOSTED_DEMO_URLS["ctxdemo"] == "http://136.111.250.205:8153/"
    assert dl.HOSTED_DEMO_URLS["demorace"] == "http://136.111.250.205/demorace/"
    turntax = dl.HOSTED_WITNESSES[dl.HOSTED_DEMO_URLS["turntaxdemo"]]
    assert turntax["api"] == "http://136.111.250.205:8150/api/suites"
    assert turntax["page_markers"] == ["<title>fak · turn-tax demo</title>"]


def test_static_audit_accepts_current_shape() -> None:
    a = dl.static_audit(GOOD)
    assert a["hosted_card_count"] == 3, a
    assert a["defects"] == [], a
    assert len(a["links"]) == 4, a


def test_static_audit_rejects_stale_guard_hosted_link() -> None:
    bad = GOOD.replace('href="run-the-demos.html"', 'href="http://136.111.250.205/guarddemo/"')
    a = dl.static_audit(bad)
    assert any("stale hosted demo link" in d for d in a["defects"]), a
    assert any("for guarddemo" in d for d in a["defects"]), a


def test_known_stale_match_identifies_demo() -> None:
    assert dl.known_stale_match("http://136.111.250.205/guarddemo/") == {
        "prefix": "http://136.111.250.205/guarddemo/",
        "demo": "guarddemo",
    }
    assert dl.known_stale_match("http://136.111.250.205/unsee/api/events") == {
        "prefix": "http://136.111.250.205/unsee/",
        "demo": "unseedemo",
    }
    assert dl.known_stale_match("http://136.111.250.205:8150/") is None


def test_static_audit_rejects_unexpected_hosted_link_even_with_same_count() -> None:
    bad = GOOD.replace("http://136.111.250.205:8153/", "http://136.111.250.205:9999/")
    a = dl.static_audit(bad)
    assert any("expected hosted demo link missing: http://136.111.250.205:8153/" in d for d in a["defects"]), a
    assert any("unexpected hosted demo link: http://136.111.250.205:9999/" in d for d in a["defects"]), a


def test_static_audit_rejects_wrong_hosted_link_role() -> None:
    bad = GOOD.replace(
        '<a href="http://136.111.250.205/">hub</a>',
        '<a class="card" href="http://136.111.250.205/">hub</a>',
    )
    a = dl.static_audit(bad)
    assert any("hosted demo link role changed: http://136.111.250.205/ should be a non-card link" in d for d in a["defects"]), a


def test_static_audit_rejects_stale_path_prefixes() -> None:
    for stale in ("turntax", "ctxdemo", "unsee"):
        bad = GOOD.replace("http://136.111.250.205:8150/", f"http://136.111.250.205/{stale}/")
        a = dl.static_audit(bad)
        assert any("stale hosted demo link" in d for d in a["defects"]), (stale, a)


def test_static_audit_requires_plain_http_disclosure() -> None:
    a = dl.static_audit(GOOD.replace("plain HTTP", "public demo host"))
    assert any("does not disclose plain HTTP" in d for d in a["defects"]), a


def test_static_audit_rejects_https_ip_host_link() -> None:
    bad = GOOD.replace("http://136.111.250.205:8150/", "https://136.111.250.205:8150/")
    a = dl.static_audit(bad)
    assert any("uses https://" in d for d in a["defects"]), a


def test_https_alternative_only_rewrites_http_links() -> None:
    assert dl.https_alternative("http://136.111.250.205:8150/path?q=1") == "https://136.111.250.205:8150/path?q=1"
    assert dl.https_alternative("https://136.111.250.205:8150/path") == ""
    assert dl.https_alternative("run-the-demos.html") == ""


def test_require_https_defects_report_unavailable_alternatives() -> None:
    probes = [
        {
            "from": "http://136.111.250.205:8150/",
            "href": "https://136.111.250.205:8150/",
            "ok": False,
            "status": 0,
            "error": "timed out",
        },
        {
            "from": "http://136.111.250.205:8153/",
            "href": "https://136.111.250.205:8153/",
            "ok": True,
            "status": 200,
            "error": "",
        },
    ]
    defects = dl.https_transport_defects(probes)
    assert defects == [
        "HTTPS alternative unavailable: http://136.111.250.205:8150/ -> "
        "https://136.111.250.205:8150/ (0 timed out)"
    ]


def test_require_https_payload_uses_transport_finding() -> None:
    payload = dl.build_payload(
        dl.repo_root(),
        "docs/demos.html",
        {
            "links": [{"href": "http://136.111.250.205:8150/"}],
            "defects": [
                "HTTPS alternative unavailable: http://136.111.250.205:8150/ -> "
                "https://136.111.250.205:8150/ (0 timed out)"
            ],
            "status_matrix": [],
            "status_summary": {"action": 0},
        },
        live=True,
        require_https=True,
    )
    assert payload["verdict"] == "ACTION", payload
    assert payload["finding"] == "https_transport_debt", payload
    assert payload["require_https"] is True, payload
    rendered = dl.render_status(payload)
    assert "demo-https-status: ACTION (https_transport_debt)" in rendered, rendered
    assert "policy: ACTION https_transport_debt (aggregate defect; rows may still be HTTP-healthy)" in rendered, rendered


def test_readiness_renderer_summarizes_failed_actions() -> None:
    ok_payload = dl.build_payload(
        dl.repo_root(),
        "docs/demos.html",
        {"links": [], "defects": []},
        live=False,
    )
    https_payload = dl.build_payload(
        dl.repo_root(),
        "docs/demos.html",
        {
            "links": [],
            "defects": [
                "HTTPS alternative unavailable: http://136.111.250.205/ -> "
                "https://136.111.250.205/ (0 timed out)"
            ],
            "status_matrix": [],
            "status_summary": {"action": 0},
        },
        live=True,
        require_https=True,
    )
    payload = dl.build_readiness_payload([
        {"name": "static", "surface": "local", "command": "python tools/demo_live_links.py", "payload": ok_payload},
        {"name": "https", "surface": "launch", "command": "make demo-https-status", "payload": https_payload},
    ])
    rendered = dl.render_readiness(payload)
    assert payload["verdict"] == "ACTION", payload
    assert payload["surface_summary"] == {"local": {"ok": 1, "total": 1}, "launch": {"ok": 0, "total": 1}}, payload
    assert "demo-readiness-status: ACTION (demo_readiness_debt)" in rendered, rendered
    assert "surfaces: local=1/1 launch=0/1" in rendered, rendered
    assert "scope: external launch/pages deployment has debt" in rendered, rendered
    assert "static     local     OK" in rendered, rendered
    assert "https      launch    ACTION" in rendered, rendered
    assert "ACTION  https_transport_debt" in rendered, rendered
    assert "details:" in rendered, rendered
    assert "static: 0 hosted demo link(s) audited; no stale hosted paths" in rendered, rendered
    assert "https: defects=1" in rendered, rendered
    assert "actions:" in rendered, rendered
    assert "https: terminate TLS for hosted demos" in rendered, rendered


def test_readiness_scope_note_splits_local_hosted_from_external_debt() -> None:
    summary = {
        "local": {"ok": 1, "total": 1},
        "hosted": {"ok": 1, "total": 1},
        "launch": {"ok": 0, "total": 1},
        "pages": {"ok": 0, "total": 1},
    }
    assert dl.readiness_scope_note(summary) == (
        "local demo source and live HTTP are healthy; remaining debt is external launch/pages deployment"
    )
    assert dl.readiness_scope_note({"local": {"ok": 0, "total": 1}}) == (
        "local demo source has debt; fix checked-in docs or demo contracts before deployment"
    )


def test_require_https_payload_from_live_reuses_probe_results() -> None:
    live_payload = dl.build_payload(
        dl.repo_root(),
        "docs/demos.html",
        {
            "links": [],
            "defects": [],
            "https_probes": [{
                "from": "http://136.111.250.205/",
                "href": "https://136.111.250.205/",
                "ok": False,
                "status": 0,
                "error": "timed out",
            }],
            "status_matrix": [],
            "status_summary": {"action": 0},
        },
        live=True,
    )
    strict = dl.require_https_payload_from_live(live_payload)
    assert strict["verdict"] == "ACTION", strict
    assert strict["finding"] == "https_transport_debt", strict
    assert strict["require_https"] is True, strict
    assert any("HTTPS alternative unavailable" in d for d in strict["audit"]["defects"]), strict
    assert live_payload["audit"]["defects"] == [], live_payload


def test_collect_readiness_reuses_live_payload_for_https() -> None:
    calls: list[dict[str, object]] = []
    real_collect = dl.collect

    def fake_collect(workspace: Path, **kwargs: object) -> dict[str, object]:
        calls.append(kwargs)
        audit = {"links": [], "defects": [], "status_matrix": [], "status_summary": {"action": 0}}
        if kwargs.get("live"):
            audit["https_probes"] = [{
                "from": "http://136.111.250.205/",
                "href": "https://136.111.250.205/",
                "ok": False,
                "status": 0,
                "error": "timed out",
            }]
        return dl.build_payload(
            workspace,
            "docs/demos.html",
            audit,
            live=bool(kwargs.get("live")),
            published=bool(kwargs.get("published")),
            require_https=bool(kwargs.get("require_https")),
        )

    dl.collect = fake_collect  # type: ignore[assignment]
    try:
        payload = dl.collect_readiness(dl.repo_root())
    finally:
        dl.collect = real_collect  # type: ignore[assignment]

    assert len(calls) == 3, calls
    assert not any(call.get("require_https") for call in calls), calls
    checks = {check["name"]: check["payload"] for check in payload["checks"]}
    surfaces = {check["name"]: check["surface"] for check in payload["checks"]}
    assert checks["live"]["ok"], checks
    assert checks["https"]["finding"] == "https_transport_debt", checks
    assert surfaces == {"static": "local", "live": "hosted", "https": "launch", "published": "pages"}, surfaces


def test_readiness_cli_rejects_mode_combinations() -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            dl.main(["--readiness", "--live"])
        except SystemExit as exc:
            assert exc.code == 2, exc
        else:
            raise AssertionError("--readiness with --live should exit through argparse")
    assert "--readiness runs all modes" in stderr.getvalue()


def test_require_https_cli_requires_live_probe() -> None:
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr):
        try:
            dl.main(["--require-https"])
        except SystemExit as exc:
            assert exc.code == 2, exc
        else:
            raise AssertionError("--require-https without --live should exit through argparse")
    assert "--require-https needs --live" in stderr.getvalue()


def test_witness_helpers_flag_missing_page_and_api_shape() -> None:
    assert dl.missing_markers("abc", ["a", "z"]) == ["z"]
    assert dl.missing_json_keys('{"models":[]}', ["models", "scenarios"]) == ["scenarios"]
    assert dl.missing_json_keys("not-json", ["models"]) == ["<invalid-json>"]
    assert dl.missing_json_keys("[]", ["models"]) == ["<non-object-json>"]


def test_probe_live_witness_accepts_expected_page_and_api() -> None:
    pages = {
        "http://136.111.250.205:8150/": {
            "ok": True,
            "status": 200,
            "text": "<title>fak · turn-tax demo</title>",
            "error": "",
        },
        "http://136.111.250.205:8150/api/suites": {
            "ok": True,
            "status": 200,
            "text": '{"suites":[]}',
            "error": "",
        },
    }

    def fake_fetch(url: str, timeout_s: float) -> dict[str, object]:
        return pages[url]

    witness = dl.probe_live_witness("http://136.111.250.205:8150/", 1.0, fetcher=fake_fetch)
    assert witness["ok"], witness
    assert witness["page_status"] == 200, witness
    assert witness["api_status"] == 200, witness


def test_probe_live_witness_rejects_wrong_page_and_api() -> None:
    pages = {
        "http://136.111.250.205:8150/": {
            "ok": True,
            "status": 200,
            "text": "<title>wrong</title>",
            "error": "",
        },
        "http://136.111.250.205:8150/api/suites": {
            "ok": True,
            "status": 200,
            "text": '{"other":[]}',
            "error": "",
        },
    }

    def fake_fetch(url: str, timeout_s: float) -> dict[str, object]:
        return pages[url]

    witness = dl.probe_live_witness("http://136.111.250.205:8150/", 1.0, fetcher=fake_fetch)
    assert not witness["ok"], witness
    assert any("page witness missing marker" in d for d in witness["defects"]), witness
    assert any("api witness" in d and "missing key" in d for d in witness["defects"]), witness


def test_probe_live_witness_rejects_missing_witness_spec() -> None:
    witness = dl.probe_live_witness("http://136.111.250.205:9999/", 1.0)
    assert not witness["ok"], witness
    assert witness["skipped"], witness
    assert witness["defects"] == ["no hosted witness registered"], witness


def test_hosted_status_matrix_folds_http_https_and_witness_state() -> None:
    href = "http://136.111.250.205:8150/"
    audit = {
        "links": [{"href": href, "text": "turntax", "card": True}],
        "probes": [{"href": href, "ok": True, "status": 200, "error": ""}],
        "https_probes": [{
            "from": href,
            "href": "https://136.111.250.205:8150/",
            "ok": False,
            "status": 0,
            "error": "wrong version number",
        }],
        "witnesses": [{
            "href": href,
            "ok": True,
            "skipped": False,
            "page_status": 200,
            "api": "http://136.111.250.205:8150/api/suites",
            "api_status": 200,
            "defects": [],
        }],
    }
    rows = dl.hosted_status_matrix(audit)
    hosted_rows = [row for row in rows if row["href"] == href]
    assert len(hosted_rows) == 1, rows
    row = hosted_rows[0]
    assert row["demo"] == "turntaxdemo", row
    assert row["role"] == "card", row
    assert row["transport"] == "plain_http", row
    assert row["http"]["checked"] and row["http"]["ok"], row
    assert row["page"] == {"checked": True, "ok": True, "status": 200}, row
    assert row["api"]["status"] == 200, row
    assert row["https"]["state"] == "unavailable", row
    assert {row["demo"] for row in rows if row["status"] == "local_only"} == {"guarddemo", "unseedemo"}, rows
    summary = dl.status_summary(rows)
    assert summary["hosted"] == 1, summary
    assert summary["hosted_links"] == 1, summary
    assert summary["hosted_demos"] == 1, summary
    assert summary["hub"] == 0, summary
    assert summary["ok"] == 1, summary
    assert summary["local_only"] == 2, summary
    assert summary["https"]["unavailable"] == 1, summary
    assert summary["https"]["not_applicable"] == 2, summary


def test_collect_real_doc_includes_static_status_matrix() -> None:
    payload = dl.collect(dl.repo_root(), live=False)
    rows = payload["audit"]["status_matrix"]
    assert rows, payload
    assert {row["demo"] for row in rows} == {
        "guarddemo", "turntaxdemo", "ctxdemo", "demorace", "unseedemo", "hub",
    }, rows
    local_only = [row for row in rows if row["status"] == "local_only"]
    assert {row["demo"] for row in local_only} == {"guarddemo", "unseedemo"}, rows
    assert all(row["transport"] == "not_hosted" for row in local_only), rows
    hosted = [row for row in rows if row["status"] == "ok"]
    assert {row["demo"] for row in hosted} == {"turntaxdemo", "ctxdemo", "demorace", "hub"}, rows
    assert all(row["http"]["checked"] is False for row in hosted), rows
    assert all(row["https"]["state"] == "not_checked" for row in hosted), rows
    summary = payload["audit"]["status_summary"]
    assert summary["hosted"] == 4, summary
    assert summary["hosted_links"] == 4, summary
    assert summary["hosted_demos"] == 3, summary
    assert summary["hub"] == 1, summary
    assert summary["check"] == 4, summary
    assert summary["action"] == 0, summary
    assert summary["local_only"] == 2, summary
    rendered = dl.render(payload)
    assert "demo status:" in rendered, rendered
    assert "LOCAL guarddemo local-only" in rendered, rendered
    assert "LOCAL unseedemo local-only" in rendered, rendered


def test_status_title_names_static_live_and_published_modes() -> None:
    assert dl.status_title({"live": False, "published": False}) == "demo-static-status"
    assert dl.status_title({"live": True, "published": False}) == "demo-live-status"
    assert dl.status_title({"live": True, "require_https": True, "published": False}) == "demo-https-status"
    assert dl.status_title({"live": False, "published": True}) == "demo-published-status"


def test_format_count_map_is_stable() -> None:
    assert dl.format_count_map({}) == "-"
    assert dl.format_count_map({"z": 2, "a": 1}) == "a:1,z:2"


def test_render_summary_line_uses_status_summary() -> None:
    payload = dl.collect(dl.repo_root(), live=False)
    assert dl.render_summary_line(payload["audit"]).startswith(
        "summary: hosted-links=4 hosted-demos=3 hub=1 ok=0 check=4 action=0 local-only=2"
    )


def test_render_status_includes_hosted_and_local_only_rows() -> None:
    payload = dl.collect(dl.repo_root(), live=False)
    rendered = dl.render_status(payload)
    assert rendered.startswith("demo-static-status: OK"), rendered
    assert "summary: hosted-links=4 hosted-demos=3 hub=1 ok=0 check=4 action=0 local-only=2" in rendered, rendered
    assert "turntaxdemo card        CHECK" in rendered, rendered
    assert "guarddemo" in rendered and "local-only" in rendered and "LOCAL" in rendered, rendered
    assert "unseedemo" in rendered and "local-only" in rendered and "LOCAL" in rendered, rendered
    assert "(not hosted)" in rendered, rendered


def test_collect_published_matrix_names_stale_demo_instead_of_unknown() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_demo_doc(root)
        write_social_preview(root)
        stale_published = GOOD.replace('href="run-the-demos.html"', 'href="http://136.111.250.205/guarddemo/"')

        def fake_fetch(url: str, timeout_s: float) -> dict[str, object]:
            return {"ok": True, "status": 200, "text": stale_published, "error": ""}

        def fake_bytes_fetch(url: str, timeout_s: float) -> dict[str, object]:
            return {"ok": True, "status": 200, "content_type": "image/png", "body": PNG, "error": ""}

        payload = dl.collect(root, published=True, fetcher=fake_fetch, bytes_fetcher=fake_bytes_fetch)
        rows = payload["audit"]["status_matrix"]
        guard = next(row for row in rows if row["href"] == "http://136.111.250.205/guarddemo/")
        assert guard["demo"] == "guarddemo", rows
        assert guard["status"] == "action", rows
        assert any("stale hosted demo link for guarddemo" in d for d in guard["defects"]), rows
        assert [row["demo"] for row in rows].count("guarddemo") == 1, rows
        summary = payload["audit"]["status_summary"]
        assert summary["hosted"] == 5, summary
        assert summary["hosted_links"] == 5, summary
        assert summary["hosted_demos"] == 4, summary
        assert summary["hub"] == 1, summary
        assert summary["action"] == 1, summary
        assert summary["check"] == 4, summary
        assert summary["local_only"] == 1, summary
        rendered = dl.render(payload)
        assert "ACTION guarddemo card" in rendered, rendered
        status = dl.render_status(payload)
        assert "demo-published-status: ACTION (published_deployment_drift)" in status, status
        assert "defects=3" in status, status
        assert "guarddemo" in status and "card" in status and "ACTION" in status, status
        assert "unexpected hosted demo link: http://136.111.250.205/guarddemo/" in status, status
        assert "hosted card count changed: found 4, want 3" in status, status
        assert "local-source: OK docs/demos.html" in status, status


def test_collect_real_doc_is_clean_static() -> None:
    payload = dl.collect(dl.repo_root(), live=False)
    assert payload["ok"], payload


def test_render_includes_social_metadata() -> None:
    payload = dl.collect(dl.repo_root(), live=False)
    rendered = dl.render(payload)
    assert "social metadata:" in rendered, rendered
    assert "local ok visuals/social-preview.png" in rendered, rendered
    assert "summary: hosted-links=4 hosted-demos=3 hub=1" in rendered, rendered


def test_page_metadata_audit_accepts_expected_tags_and_asset() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_social_preview(root)
        audit = dl.page_metadata_audit(root, GOOD)
        assert audit["defects"] == [], audit
        assert audit["social_preview_asset"] == "ok", audit


def test_page_metadata_audit_rejects_bad_tags_and_asset() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_social_preview(root, b"not-png")
        bad = GOOD.replace(dl.PUBLISHED_DOC_URL, "https://example.invalid/fak/demos.html", 1)
        bad = bad.replace(dl.SOCIAL_PREVIEW_URL, "https://example.invalid/card.png", 1)
        audit = dl.page_metadata_audit(root, bad)
        assert any("metadata canonical" in d for d in audit["defects"]), audit
        assert any("metadata og:image" in d for d in audit["defects"]), audit
        assert any("social image asset is not a PNG" in d for d in audit["defects"]), audit


def test_probe_remote_social_image_accepts_png() -> None:
    def fake_fetch(url: str, timeout_s: float) -> dict[str, object]:
        return {"ok": True, "status": 200, "content_type": "image/png", "body": PNG, "error": ""}

    probe = dl.probe_remote_social_image(dl.SOCIAL_PREVIEW_URL, 1.0, fetcher=fake_fetch)
    assert probe["ok"], probe
    assert probe["defects"] == [], probe


def test_probe_remote_social_image_rejects_unreachable_or_non_png() -> None:
    def missing_fetch(url: str, timeout_s: float) -> dict[str, object]:
        return {"ok": False, "status": 404, "content_type": "", "body": b"", "error": "not found"}

    missing = dl.probe_remote_social_image(dl.SOCIAL_PREVIEW_URL, 1.0, fetcher=missing_fetch)
    assert not missing["ok"], missing
    assert any("remote social image unreachable" in d for d in missing["defects"]), missing

    def bad_fetch(url: str, timeout_s: float) -> dict[str, object]:
        return {"ok": True, "status": 200, "content_type": "text/plain", "body": b"not-png", "error": ""}

    bad = dl.probe_remote_social_image(dl.SOCIAL_PREVIEW_URL, 1.0, fetcher=bad_fetch)
    assert not bad["ok"], bad
    assert any("remote social image is not a PNG" in d for d in bad["defects"]), bad
    assert any("content-type is not image/png" in d for d in bad["defects"]), bad


def test_collect_published_uses_fetched_html() -> None:
    def fake_fetch(url: str, timeout_s: float) -> dict[str, object]:
        assert url == dl.PUBLISHED_DOC_URL
        return {"ok": True, "status": 200, "text": GOOD, "error": ""}

    def fake_bytes_fetch(url: str, timeout_s: float) -> dict[str, object]:
        assert url == dl.SOCIAL_PREVIEW_URL
        return {"ok": True, "status": 200, "content_type": "image/png", "body": PNG, "error": ""}

    payload = dl.collect(dl.repo_root(), published=True, fetcher=fake_fetch, bytes_fetcher=fake_bytes_fetch)
    assert payload["ok"], payload
    assert payload["doc"] == dl.PUBLISHED_DOC_URL, payload
    assert payload["published"], payload
    assert payload["audit"]["metadata"]["remote_social_preview"]["ok"], payload


def test_collect_published_marks_clean_local_source_as_deployment_drift() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        write_demo_doc(root)
        write_social_preview(root)
        stale_published = GOOD.replace('href="run-the-demos.html"', 'href="http://136.111.250.205/guarddemo/"')

        def fake_fetch(url: str, timeout_s: float) -> dict[str, object]:
            assert url == dl.PUBLISHED_DOC_URL
            return {"ok": True, "status": 200, "text": stale_published, "error": ""}

        def fake_bytes_fetch(url: str, timeout_s: float) -> dict[str, object]:
            assert url == dl.SOCIAL_PREVIEW_URL
            return {"ok": True, "status": 200, "content_type": "image/png", "body": PNG, "error": ""}

        payload = dl.collect(root, published=True, fetcher=fake_fetch, bytes_fetcher=fake_bytes_fetch)
        assert not payload["ok"], payload
        assert payload["finding"] == "published_deployment_drift", payload
        assert "local docs/demos.html is clean" in payload["reason"], payload
        assert payload["local_source"]["ok"], payload
        rendered = dl.render(payload)
        assert "local source:" in rendered, rendered
        assert "OK   docs/demos.html" in rendered, rendered


def test_collect_published_reports_fetch_failure() -> None:
    def fake_fetch(url: str, timeout_s: float) -> dict[str, object]:
        return {"ok": False, "status": 404, "text": "", "error": "not found"}

    payload = dl.collect(dl.repo_root(), published=True, fetcher=fake_fetch)
    assert not payload["ok"], payload
    assert payload["verdict"] == "AUDIT_ERROR", payload
    assert "fetch published demos page: 404 not found" in payload["reason"], payload


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("demo_live_links_test: OK")
