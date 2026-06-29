#!/usr/bin/env python3
"""Audit the hosted links on docs/demos.html.

The demo scorecards prove local demos are runnable and documented. This tool watches a
different failure mode: the public demos page can link to a hosted VM path that no longer
exists. By default the audit is static and network-free: it parses docs/demos.html,
checks which cards point at the hosted demo host, rejects known stale paths such as
/guarddemo/, and requires the page to say that hosted demos are plain HTTP.

Use --live when intentionally checking the external VM. That mode probes every hosted
link with a short timeout, checks hosted page/API witnesses, and fails if any hosted link
is no longer reachable or serves the wrong demo. It also probes HTTPS alternatives for
HTTP hosted links with a shorter timeout; those probes are evidence for the plain-HTTP
deployment note, and become an action item if HTTPS starts working.

Run from the repo root:

    python tools/demo_live_links.py
    python tools/demo_live_links.py --live
    python tools/demo_live_links.py --live --require-https
    python tools/demo_live_links.py --published
    python tools/demo_live_links.py --json --live

In --published mode, ACTION (published_deployment_drift) means the repository's
docs/demos.html is already clean but GitHub Pages is still serving stale HTML.
Use --live --require-https when a launch gate needs hosted demos to be embeddable
from HTTPS pages instead of merely reachable through top-level HTTP navigation.
"""
from __future__ import annotations

import argparse
import copy
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import demo_registry as dr  # noqa: E402

SCHEMA = "fak-demo-live-links/1"
DEFAULT_DOC = "docs/demos.html"
PUBLISHED_DOC_URL = "https://anthony-chaudhary.github.io/fak/demos.html"
SOCIAL_PREVIEW_PATH = "visuals/social-preview.png"
SOCIAL_PREVIEW_URL = f"https://raw.githubusercontent.com/anthony-chaudhary/fak/main/{SOCIAL_PREVIEW_PATH}"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
HOSTED_HOST = "136.111.250.205"
HOSTED_HUB_URL = f"http://{HOSTED_HOST}/"
HOSTED_DEMO_URLS = dr.hosted_demo_urls(HOSTED_HOST)
EXPECTED_HOSTED_LINKS: dict[str, bool] = {
    **{href: True for href in HOSTED_DEMO_URLS.values()},
    HOSTED_HUB_URL: False,
}
EXPECTED_HOSTED_CARD_COUNT = sum(1 for is_card in EXPECTED_HOSTED_LINKS.values() if is_card)

# Paths/ports proven stale during the live audit. Keep this list narrow: it prevents
# accidentally reintroducing a link that currently 404s or times out, while still letting
# the hosted set be updated deliberately when the deployment changes.
KNOWN_STALE_PREFIXES: tuple[tuple[str, str], ...] = (
    (f"http://{HOSTED_HOST}/guarddemo/", "guarddemo"),
    (f"http://{HOSTED_HOST}:8151/", "guarddemo"),
    (f"http://{HOSTED_HOST}:8151/api/", "guarddemo"),
    (f"http://{HOSTED_HOST}:8156/", "unseedemo"),
    (f"http://{HOSTED_HOST}:8156/api/", "unseedemo"),
    (f"http://{HOSTED_HOST}/turntax/", "turntaxdemo"),
    (f"http://{HOSTED_HOST}/ctxdemo/", "ctxdemo"),
    (f"http://{HOSTED_HOST}/unsee/", "unseedemo"),
)


def known_stale_match(href: str) -> dict[str, str] | None:
    for prefix, demo in KNOWN_STALE_PREFIXES:
        if href.startswith(prefix):
            return {"prefix": prefix, "demo": demo}
    return None


def hosted_witnesses() -> dict[str, dict[str, Any]]:
    by_name = dr.demo_map()
    witnesses: dict[str, dict[str, Any]] = {}
    for name, base_url in HOSTED_DEMO_URLS.items():
        demo = by_name[name]
        witnesses[base_url] = {
            "demo": name,
            "page_markers": [f"<title>fak · {demo.page_marker}</title>"],
            "api": urllib.parse.urljoin(base_url, demo.api_path),
            "api_keys": list(demo.hosted_api_keys),
        }
    witnesses[HOSTED_HUB_URL] = {
        "page_markers": ["<title>fak — the agent kernel · live demos</title>"],
    }
    return witnesses


HOSTED_WITNESSES: dict[str, dict[str, Any]] = hosted_witnesses()


@dataclass(frozen=True)
class Link:
    href: str
    text: str
    classes: tuple[str, ...]

    @property
    def is_card(self) -> bool:
        return "card" in self.classes


class AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.links: list[Link] = []
        self._stack: list[dict[str, Any]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        d = {k.lower(): v or "" for k, v in attrs}
        classes = tuple(c for c in d.get("class", "").split() if c)
        self._stack.append({"href": d.get("href", ""), "classes": classes, "parts": []})

    def handle_data(self, data: str) -> None:
        if self._stack:
            self._stack[-1]["parts"].append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() != "a" or not self._stack:
            return
        cur = self._stack.pop()
        text = " ".join("".join(cur["parts"]).split())
        self.links.append(Link(cur["href"], text, cur["classes"]))


class MetadataParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.canonical: list[str] = []
        self.meta: dict[str, list[str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        d = {k.lower(): v or "" for k, v in attrs}
        if tag.lower() == "link":
            rels = {part.lower() for part in d.get("rel", "").split()}
            if "canonical" in rels and d.get("href"):
                self.canonical.append(d["href"])
            return
        if tag.lower() != "meta":
            return
        key = d.get("property") or d.get("name")
        if key and "content" in d:
            self.meta.setdefault(key, []).append(d["content"])


def extract_links(html: str) -> list[Link]:
    p = AnchorParser()
    p.feed(html)
    return p.links


def extract_metadata(html: str) -> dict[str, Any]:
    p = MetadataParser()
    p.feed(html)
    return {"canonical": p.canonical, "meta": p.meta}


def _expect_metadata_value(defects: list[str], values: list[str], label: str, expected: str) -> None:
    unique = sorted(set(values))
    if not unique:
        defects.append(f"demos page metadata missing {label}: {expected}")
        return
    if unique != [expected]:
        defects.append(f"demos page metadata {label}={unique!r}, want {expected!r}")


def page_metadata_audit(workspace: Path, html: str) -> dict[str, Any]:
    metadata = extract_metadata(html)
    meta = metadata["meta"]
    defects: list[str] = []

    _expect_metadata_value(defects, metadata["canonical"], "canonical", PUBLISHED_DOC_URL)
    _expect_metadata_value(defects, meta.get("og:url", []), "og:url", PUBLISHED_DOC_URL)
    _expect_metadata_value(defects, meta.get("og:image", []), "og:image", SOCIAL_PREVIEW_URL)
    _expect_metadata_value(defects, meta.get("twitter:image", []), "twitter:image", SOCIAL_PREVIEW_URL)
    _expect_metadata_value(defects, meta.get("twitter:card", []), "twitter:card", "summary_large_image")

    asset = workspace / SOCIAL_PREVIEW_PATH
    asset_status = "ok"
    try:
        head = asset.read_bytes()[:len(PNG_MAGIC)]
    except OSError as exc:
        asset_status = f"missing: {exc}"
        defects.append(f"demos page social image asset missing: {SOCIAL_PREVIEW_PATH}")
    else:
        if head != PNG_MAGIC:
            asset_status = "not-png"
            defects.append(f"demos page social image asset is not a PNG: {SOCIAL_PREVIEW_PATH}")
        elif asset.stat().st_size < 1024:
            asset_status = "too-small"
            defects.append(f"demos page social image asset is unexpectedly small: {SOCIAL_PREVIEW_PATH}")

    return {
        "canonical": metadata["canonical"],
        "meta": {key: meta.get(key, []) for key in ("og:url", "og:image", "twitter:card", "twitter:image")},
        "social_preview_path": SOCIAL_PREVIEW_PATH,
        "social_preview_url": SOCIAL_PREVIEW_URL,
        "social_preview_asset": asset_status,
        "defects": defects,
    }


def is_hosted_link(href: str, host: str = HOSTED_HOST) -> bool:
    u = urllib.parse.urlparse(href)
    return u.scheme in {"http", "https"} and u.hostname == host


def https_alternative(href: str) -> str:
    u = urllib.parse.urlparse(href)
    if u.scheme != "http":
        return ""
    return urllib.parse.urlunparse(u._replace(scheme="https"))


def static_audit(html: str, *, host: str = HOSTED_HOST) -> dict[str, Any]:
    links = extract_links(html)
    hosted = [link for link in links if is_hosted_link(link.href, host)]
    hosted_cards = [link for link in hosted if link.is_card]
    defects: list[str] = []
    soft: list[str] = []

    for link in hosted:
        stale = known_stale_match(link.href)
        if stale:
            defects.append(f"stale hosted demo link for {stale['demo']}: {link.href}")

    if host == HOSTED_HOST:
        actual_hrefs = [link.href for link in hosted]
        actual_set = set(actual_hrefs)
        expected_set = set(EXPECTED_HOSTED_LINKS)
        if len(actual_hrefs) != len(actual_set):
            defects.append("duplicate hosted demo link found; keep each hosted URL unique")
        for href in sorted(expected_set - actual_set):
            defects.append(f"expected hosted demo link missing: {href}")
        for href in sorted(actual_set - expected_set):
            defects.append(f"unexpected hosted demo link: {href}")
        for href, want_card in EXPECTED_HOSTED_LINKS.items():
            roles = [link.is_card for link in hosted if link.href == href]
            if roles and want_card not in roles:
                want = "card" if want_card else "non-card link"
                defects.append(f"hosted demo link role changed: {href} should be a {want}")
            if href not in HOSTED_WITNESSES:
                defects.append(f"hosted demo link lacks live witness spec: {href}")

    if len(hosted_cards) != EXPECTED_HOSTED_CARD_COUNT:
        defects.append(
            f"hosted card count changed: found {len(hosted_cards)}, "
            f"want {EXPECTED_HOSTED_CARD_COUNT}; update docs and this audit together"
        )

    if any(urllib.parse.urlparse(link.href).scheme == "http" for link in hosted):
        if "plain HTTP" not in html:
            defects.append("hosted links use http:// but docs/demos.html does not disclose plain HTTP")
        soft.append("hosted demo links are plain HTTP; top-level navigation works, embedding from HTTPS does not")

    https_links = [link.href for link in hosted if urllib.parse.urlparse(link.href).scheme == "https"]
    if https_links:
        defects.append(f"hosted demo link uses https:// for the IP host; verify TLS first: {https_links[0]}")

    return {
        "host": host,
        "links": [link_row(link) for link in hosted],
        "hosted_card_count": len(hosted_cards),
        "defects": defects,
        "soft": soft,
    }


def link_row(link: Link) -> dict[str, Any]:
    return {"href": link.href, "text": link.text, "card": link.is_card}


def probe_url(url: str, timeout_s: float) -> dict[str, Any]:
    headers = {"User-Agent": "fak-demo-live-links/1"}
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                return {"ok": 200 <= resp.status < 400, "status": resp.status, "method": method, "error": ""}
        except urllib.error.HTTPError as exc:
            if method == "HEAD" and exc.code in {405, 501}:
                continue
            return {"ok": False, "status": exc.code, "method": method, "error": exc.reason}
        except (OSError, TimeoutError) as exc:
            return {"ok": False, "status": 0, "method": method, "error": str(exc)}
    return {"ok": False, "status": 0, "method": "GET", "error": "unreachable"}


def fetch_url_text(url: str, timeout_s: float, *, limit: int = 262_144) -> dict[str, Any]:
    headers = {"User-Agent": "fak-demo-live-links/1"}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read(limit + 1)
            truncated = len(body) > limit
            if truncated:
                body = body[:limit]
            text = body.decode("utf-8", errors="replace")
            return {
                "ok": 200 <= resp.status < 400,
                "status": resp.status,
                "content_type": resp.headers.get("content-type", ""),
                "text": text,
                "truncated": truncated,
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "content_type": "", "text": "", "truncated": False, "error": str(exc.reason)}
    except (OSError, TimeoutError) as exc:
        return {"ok": False, "status": 0, "content_type": "", "text": "", "truncated": False, "error": str(exc)}


def fetch_url_bytes(url: str, timeout_s: float, *, limit: int = 1_048_576) -> dict[str, Any]:
    headers = {"User-Agent": "fak-demo-live-links/1"}
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read(limit + 1)
            truncated = len(body) > limit
            if truncated:
                body = body[:limit]
            return {
                "ok": 200 <= resp.status < 400,
                "status": resp.status,
                "content_type": resp.headers.get("content-type", ""),
                "body": body,
                "truncated": truncated,
                "error": "",
            }
    except urllib.error.HTTPError as exc:
        return {"ok": False, "status": exc.code, "content_type": "", "body": b"", "truncated": False, "error": str(exc.reason)}
    except (OSError, TimeoutError) as exc:
        return {"ok": False, "status": 0, "content_type": "", "body": b"", "truncated": False, "error": str(exc)}


def probe_remote_social_image(url: str, timeout_s: float, *,
                              fetcher: Any = fetch_url_bytes) -> dict[str, Any]:
    fetched = fetcher(url, timeout_s)
    defects: list[str] = []
    if not fetched.get("ok"):
        defects.append(f"remote social image unreachable: {url} ({fetched.get('status', 0)} {fetched.get('error', '')})")
    else:
        body = fetched.get("body", b"")
        if not isinstance(body, (bytes, bytearray)) or not bytes(body).startswith(PNG_MAGIC):
            defects.append(f"remote social image is not a PNG: {url}")
        if "image/png" not in str(fetched.get("content_type", "")).lower():
            defects.append(f"remote social image content-type is not image/png: {fetched.get('content_type', '')}")
    return {
        "url": url,
        "ok": not defects,
        "status": fetched.get("status", 0),
        "content_type": fetched.get("content_type", ""),
        "defects": defects,
    }


def missing_markers(text: str, markers: list[str]) -> list[str]:
    return [marker for marker in markers if marker not in text]


def missing_json_keys(text: str, keys: list[str]) -> list[str]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return ["<invalid-json>"]
    if not isinstance(data, dict):
        return ["<non-object-json>"]
    return [key for key in keys if key not in data]


def probe_live_witness(url: str, timeout_s: float, *,
                       fetcher: Any = fetch_url_text) -> dict[str, Any]:
    spec = HOSTED_WITNESSES.get(url)
    if not spec:
        return {"href": url, "ok": False, "skipped": True, "defects": ["no hosted witness registered"]}

    defects: list[str] = []
    page = fetcher(url, timeout_s)
    if not page.get("ok"):
        defects.append(f"page witness unreachable ({page.get('status', 0)} {page.get('error', '')})")
    else:
        missing = missing_markers(str(page.get("text", "")), list(spec.get("page_markers", [])))
        for marker in missing:
            defects.append(f"page witness missing marker: {marker}")

    api_url = spec.get("api", "")
    api: dict[str, Any] | None = None
    if api_url:
        api = fetcher(api_url, timeout_s)
        if not api.get("ok"):
            defects.append(f"api witness unreachable {api_url} ({api.get('status', 0)} {api.get('error', '')})")
        else:
            missing = missing_json_keys(str(api.get("text", "")), list(spec.get("api_keys", [])))
            for key in missing:
                defects.append(f"api witness {api_url} missing key: {key}")

    return {
        "href": url,
        "ok": not defects,
        "skipped": False,
        "page_status": page.get("status", 0),
        "api": api_url,
        "api_status": api.get("status", 0) if api else 0,
        "defects": defects,
    }


def hosted_status_matrix(audit: dict[str, Any]) -> list[dict[str, Any]]:
    """Fold link/probe/witness rows into one per-hosted-endpoint status table."""
    probes = {row.get("href"): row for row in audit.get("probes", [])}
    https_probes = {row.get("from"): row for row in audit.get("https_probes", [])}
    witnesses = {row.get("href"): row for row in audit.get("witnesses", [])}
    rows: list[dict[str, Any]] = []
    seen_demos: set[str] = set()

    for link in audit.get("links", []):
        href = link.get("href", "")
        spec = HOSTED_WITNESSES.get(href, {})
        stale = known_stale_match(href)
        probe = probes.get(href)
        witness = witnesses.get(href)
        https_url = https_alternative(href)
        https_probe = https_probes.get(href)
        row_defects: list[str] = []

        http_checked = probe is not None
        http_ok = bool(probe.get("ok")) if probe else None
        https_checked = https_probe is not None
        if not https_url:
            https_state = "not_applicable"
        elif https_probe is None:
            https_state = "not_checked"
        elif https_probe.get("ok"):
            https_state = "available"
            row_defects.append("HTTPS alternative is reachable; update hosted link to https://")
        else:
            https_state = "unavailable"

        if stale:
            row_defects.append(f"stale hosted demo link for {stale['demo']}")
        elif href and href not in EXPECTED_HOSTED_LINKS:
            row_defects.append("unexpected hosted demo link")
        if probe and not probe.get("ok"):
            row_defects.append("hosted link unreachable")
        if witness:
            row_defects.extend(str(d) for d in witness.get("defects", []))
        elif not stale and href not in HOSTED_WITNESSES:
            row_defects.append("no hosted witness registered")

        demo_name = spec.get("demo") or (stale["demo"] if stale else "hub" if href == HOSTED_HUB_URL else "unknown")
        if demo_name not in {"hub", "unknown"}:
            seen_demos.add(demo_name)

        rows.append({
            "demo": demo_name,
            "href": href,
            "role": "card" if link.get("card") else "link",
            "status": "action" if row_defects else "ok",
            "transport": "plain_http" if urllib.parse.urlparse(href).scheme == "http" else "https",
            "defects": row_defects,
            "http": {
                "checked": http_checked,
                "ok": http_ok,
                "status": int(probe.get("status", 0)) if probe else 0,
                "error": probe.get("error", "") if probe else "",
            },
            "page": {
                "checked": witness is not None and not witness.get("skipped", False),
                "ok": bool(witness.get("ok")) if witness else None,
                "status": int(witness.get("page_status", 0)) if witness else 0,
            },
            "api": {
                "href": witness.get("api") if witness else spec.get("api", ""),
                "checked": witness is not None and bool(witness.get("api")),
                "status": int(witness.get("api_status", 0)) if witness else 0,
            },
            "https": {
                "href": https_url,
                "checked": https_checked,
                "ok": bool(https_probe.get("ok")) if https_probe else None,
                "status": int(https_probe.get("status", 0)) if https_probe else 0,
                "state": https_state,
                "error": https_probe.get("error", "") if https_probe else "",
            },
        })
    for demo in dr.DEMOS:
        if demo.name in seen_demos or demo.hosted_path:
            continue
        rows.append({
            "demo": demo.name,
            "href": "",
            "role": "local-only",
            "status": "local_only",
            "transport": "not_hosted",
            "defects": [],
            "http": {"checked": False, "ok": None, "status": 0, "error": ""},
            "page": {"checked": False, "ok": None, "status": 0},
            "api": {"href": demo.api_path, "checked": False, "status": 0},
            "https": {"href": "", "checked": False, "ok": None, "status": 0, "state": "not_applicable", "error": ""},
        })
    return rows


def status_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(rows),
        "hosted": 0,
        "hosted_links": 0,
        "hosted_demos": 0,
        "hub": 0,
        "local_only": 0,
        "ok": 0,
        "action": 0,
        "check": 0,
        "https": {},
    }
    for row in rows:
        status = row.get("status", "")
        if status == "local_only":
            summary["local_only"] += 1
        else:
            summary["hosted"] += 1
            summary["hosted_links"] += 1
            if row.get("demo") == "hub":
                summary["hub"] += 1
            else:
                summary["hosted_demos"] += 1
        if status == "ok":
            if row.get("http", {}).get("checked"):
                summary["ok"] += 1
            else:
                summary["check"] += 1
        elif status == "action":
            summary["action"] += 1
        elif status == "local_only":
            pass
        else:
            summary["check"] += 1
        https_state = str(row.get("https", {}).get("state", "unknown"))
        https = summary["https"]
        https[https_state] = https.get(https_state, 0) + 1
    return summary


def https_transport_defects(probes: list[dict[str, Any]]) -> list[str]:
    defects = []
    for probe in probes:
        if probe.get("ok"):
            continue
        status = int(probe.get("status", 0))
        error = str(probe.get("error", "")).strip()
        detail = f"{status} {error}".strip()
        defects.append(
            "HTTPS alternative unavailable: "
            f"{probe.get('from', '<unknown>')} -> {probe.get('href', '<unknown>')} ({detail})"
        )
    return defects


def collect(workspace: Path, *, doc: str = DEFAULT_DOC, live: bool = False,
            timeout_s: float = 8.0, published: bool = False, require_https: bool = False,
            fetcher: Any = fetch_url_text, bytes_fetcher: Any = fetch_url_bytes) -> dict[str, Any]:
    if published:
        fetched = fetcher(PUBLISHED_DOC_URL, timeout_s)
        if not fetched.get("ok"):
            return build_payload(
                workspace,
                PUBLISHED_DOC_URL,
                {},
                live=live,
                published=published,
                require_https=require_https,
                error=f"fetch published demos page: {fetched.get('status', 0)} {fetched.get('error', '')}",
            )
        html = str(fetched.get("text", ""))
        payload = collect_html(
            workspace,
            PUBLISHED_DOC_URL,
            html,
            live=live,
            timeout_s=timeout_s,
            published=published,
            require_https=require_https,
            bytes_fetcher=bytes_fetcher,
        )
        if not payload.get("ok"):
            local = local_source_status(workspace, doc)
            payload["local_source"] = local
            if local.get("ok"):
                defects = payload.get("audit", {}).get("defects", [])
                payload["finding"] = "published_deployment_drift"
                payload["reason"] = f"{len(defects)} published-page defect(s); local {doc} is clean"
                payload["next_action"] = "republish GitHub Pages or wait for Pages to catch up, then rerun"
        return payload

    path = workspace / doc
    try:
        html = path.read_text(encoding="utf-8")
    except OSError as exc:
        return build_payload(
            workspace,
            doc,
            {},
            live=live,
            published=published,
            require_https=require_https,
            error=f"read {doc}: {exc}",
        )
    return collect_html(
        workspace,
        doc,
        html,
        live=live,
        timeout_s=timeout_s,
        published=published,
        require_https=require_https,
    )


def local_source_status(workspace: Path, doc: str = DEFAULT_DOC) -> dict[str, Any]:
    path = workspace / doc
    try:
        html = path.read_text(encoding="utf-8")
    except OSError as exc:
        return {"doc": doc, "ok": False, "defects": [f"read local {doc}: {exc}"]}
    audit = static_audit(html)
    metadata = page_metadata_audit(workspace, html)
    defects = audit["defects"] + metadata["defects"]
    return {"doc": doc, "ok": not defects, "defects": defects}


def collect_html(workspace: Path, doc: str, html: str, *,
                 live: bool = False, timeout_s: float = 8.0,
                 published: bool = False, require_https: bool = False,
                 bytes_fetcher: Any = fetch_url_bytes) -> dict[str, Any]:
    audit = static_audit(html)
    metadata = page_metadata_audit(workspace, html)
    if published:
        remote = probe_remote_social_image(SOCIAL_PREVIEW_URL, min(timeout_s, 8.0), fetcher=bytes_fetcher)
        metadata["remote_social_preview"] = remote
        metadata["defects"] = metadata["defects"] + remote["defects"]
    audit["metadata"] = metadata
    audit["defects"] = audit["defects"] + metadata["defects"]
    if live:
        probes = []
        https_probes = []
        witnesses = []
        live_defects = []
        for row in audit["links"]:
            p = probe_url(row["href"], timeout_s)
            p["href"] = row["href"]
            probes.append(p)
            if not p["ok"]:
                live_defects.append(f"hosted link unreachable: {row['href']} ({p['status']} {p['error']})")
            w = probe_live_witness(row["href"], min(timeout_s, 8.0))
            witnesses.append(w)
            for defect in w.get("defects", []):
                live_defects.append(f"hosted witness failed: {row['href']} ({defect})")
            alt = https_alternative(row["href"])
            if alt:
                hp = probe_url(alt, min(timeout_s, 3.0))
                hp["href"] = alt
                hp["from"] = row["href"]
                https_probes.append(hp)
                if hp["ok"]:
                    live_defects.append(f"HTTPS alternative is reachable; update hosted demo link to https:// {alt}")
        if require_https:
            live_defects.extend(https_transport_defects(https_probes))
        audit["probes"] = probes
        audit["https_probes"] = https_probes
        audit["witnesses"] = witnesses
        audit["defects"] = audit["defects"] + live_defects
    audit["status_matrix"] = hosted_status_matrix(audit)
    audit["status_summary"] = status_summary(audit["status_matrix"])
    return build_payload(workspace, doc, audit, live=live, published=published, require_https=require_https)


def build_payload(workspace: Path, doc: str, audit: dict[str, Any], *,
                  live: bool, published: bool = False, require_https: bool = False,
                  error: str | None = None) -> dict[str, Any]:
    if error:
        return {
            "schema": SCHEMA,
            "ok": False,
            "verdict": "AUDIT_ERROR",
            "finding": "tooling_error",
            "reason": error,
            "next_action": "fix the docs path, then rerun",
            "workspace": str(workspace),
            "doc": doc,
            "live": live,
            "published": published,
            "require_https": require_https,
        }
    defects = audit.get("defects", [])
    ok = not defects
    n_links = len(audit.get("links", []))
    if ok:
        verdict, finding = "OK", "hosted_demo_links_clean"
        reason = f"{n_links} hosted demo link(s) audited; no stale hosted paths"
        if live:
            reason += "; all live probes and witnesses pass"
        next_action = "rerun after hosted demo URL or deployment changes"
    elif require_https and any(str(d).startswith("HTTPS alternative unavailable:") for d in defects):
        verdict, finding = "ACTION", "https_transport_debt"
        count = sum(1 for d in defects if str(d).startswith("HTTPS alternative unavailable:"))
        reason = f"{count} hosted HTTPS transport defect(s) in {doc}"
        next_action = "terminate TLS for hosted demos or run without --require-https when top-level HTTP links are acceptable"
    else:
        verdict, finding = "ACTION", "hosted_demo_link_debt"
        reason = f"{len(defects)} hosted-demo link defect(s) in {doc}"
        next_action = "remove stale hosted links, point local-only demos at run-the-demos, or fix the deployment"
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": str(workspace),
        "doc": doc,
        "live": live,
        "published": published,
        "require_https": require_https,
        "audit": audit,
    }


def build_readiness_payload(checks: list[dict[str, Any]]) -> dict[str, Any]:
    ok_count = sum(1 for check in checks if check["payload"].get("ok"))
    ok = ok_count == len(checks)
    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": "OK" if ok else "ACTION",
        "finding": "demo_readiness_clean" if ok else "demo_readiness_debt",
        "reason": f"{ok_count}/{len(checks)} demo readiness check(s) pass",
        "surface_summary": readiness_surface_summary(checks),
        "checks": checks,
    }


def readiness_surface_summary(checks: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for check in checks:
        surface = str(check.get("surface") or "unspecified")
        bucket = summary.setdefault(surface, {"ok": 0, "total": 0})
        bucket["total"] += 1
        if check["payload"].get("ok"):
            bucket["ok"] += 1
    return summary


def require_https_payload_from_live(payload: dict[str, Any]) -> dict[str, Any]:
    workspace = Path(str(payload.get("workspace", repo_root())))
    doc = str(payload.get("doc", DEFAULT_DOC))
    audit = copy.deepcopy(payload.get("audit") or {})
    if not audit:
        return build_payload(
            workspace,
            doc,
            {},
            live=True,
            require_https=True,
            error=f"derive HTTPS readiness from live check: {payload.get('reason', 'missing audit')}",
        )
    defects = list(audit.get("defects") or [])
    for defect in https_transport_defects(audit.get("https_probes") or []):
        if defect not in defects:
            defects.append(defect)
    audit["defects"] = defects
    audit["status_matrix"] = hosted_status_matrix(audit)
    audit["status_summary"] = status_summary(audit["status_matrix"])
    return build_payload(workspace, doc, audit, live=True, require_https=True)


def collect_readiness(workspace: Path, *, doc: str = DEFAULT_DOC, timeout_s: float = 8.0,
                      published_timeout_s: float = 12.0,
                      fetcher: Any = fetch_url_text,
                      bytes_fetcher: Any = fetch_url_bytes) -> dict[str, Any]:
    static_payload = collect(workspace, doc=doc, live=False, timeout_s=timeout_s)
    live_payload = collect(workspace, doc=doc, live=True, timeout_s=timeout_s)
    https_payload = require_https_payload_from_live(live_payload)
    published_payload = collect(
        workspace,
        doc=doc,
        published=True,
        timeout_s=published_timeout_s,
        fetcher=fetcher,
        bytes_fetcher=bytes_fetcher,
    )
    checks = [
        {
            "name": "static",
            "surface": "local",
            "command": "python tools/demo_live_links.py",
            "payload": static_payload,
        },
        {
            "name": "live",
            "surface": "hosted",
            "command": "make demo-live-status",
            "payload": live_payload,
        },
        {
            "name": "https",
            "surface": "launch",
            "command": "make demo-https-status",
            "payload": https_payload,
        },
        {
            "name": "published",
            "surface": "pages",
            "command": "make demo-published-status",
            "payload": published_payload,
        },
    ]
    return build_readiness_payload(checks)


def render(payload: dict[str, Any]) -> str:
    lines = [
        f"demo-live-links: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        f"  next: {payload['next_action']}",
    ]
    audit = payload.get("audit") or {}
    lines.append("")
    lines.append("hosted links:")
    for row in audit.get("links", []):
        kind = "card" if row["card"] else "link"
        lines.append(f"  {kind:4} {row['href']}")
    if audit.get("status_matrix"):
        lines.append("")
        lines.append(render_summary_line(audit))
        lines.append("")
        lines.append("demo status:")
        for row in audit["status_matrix"]:
            http = row["http"]
            https = row["https"]
            if row.get("status") == "local_only":
                state = "LOCAL"
                http_part = "http=local-only"
            elif row.get("status") == "action":
                state = "ACTION"
                http_part = f"http={http['status']}" if http["checked"] else "http=not-checked"
            elif http["checked"]:
                state = "OK" if http["ok"] else "FAIL"
                http_part = f"http={http['status']}"
            else:
                state = "CHECK"
                http_part = "http=not-checked"
            page_part = ""
            if row["page"]["checked"]:
                page_part = f" page={row['page']['status']}"
            api_part = ""
            if row["api"]["href"]:
                api_part = f" api={row['api']['status'] if row['api']['checked'] else 'not-checked'}"
            href_part = row["href"] or "(not hosted)"
            lines.append(
                f"  {state:5} {row['demo']} {row['role']} {http_part}{page_part}{api_part} "
                f"https={https['state']} {href_part}"
            )
            for defect in row.get("defects", [])[:3]:
                lines.append(f"        - {defect}")
    if audit.get("metadata"):
        metadata = audit["metadata"]
        lines.append("")
        lines.append("social metadata:")
        lines.append(f"  local {metadata.get('social_preview_asset', '<unknown>')} {metadata.get('social_preview_path', '')}")
        remote = metadata.get("remote_social_preview")
        if remote:
            status = "OK" if remote.get("ok") else "FAIL"
            lines.append(
                f"  {status:4} remote {remote.get('status', 0)} "
                f"{remote.get('content_type', '')} {remote.get('url', '')}"
            )
    if payload.get("local_source"):
        local = payload["local_source"]
        status = "OK" if local.get("ok") else "FAIL"
        lines.append("")
        lines.append("local source:")
        lines.append(f"  {status:4} {local.get('doc', DEFAULT_DOC)}")
        for defect in local.get("defects", [])[:5]:
            lines.append(f"       {defect}")
    if audit.get("probes"):
        lines.append("")
        lines.append("live probes:")
        for p in audit["probes"]:
            status = "OK" if p["ok"] else "FAIL"
            lines.append(f"  {status:4} {p['status']:>3} {p['href']}")
    if audit.get("https_probes"):
        lines.append("")
        lines.append("https alternatives:")
        for p in audit["https_probes"]:
            status = "OK" if p["ok"] else "NO"
            err = "" if p["ok"] else f" ({p['error']})"
            lines.append(f"  {status:4} {p['status']:>3} {p['href']}{err}")
    if audit.get("witnesses"):
        lines.append("")
        lines.append("live witnesses:")
        for w in audit["witnesses"]:
            if w.get("skipped"):
                lines.append(f"  FAIL no-witness {w['href']}")
                continue
            status = "OK" if w["ok"] else "FAIL"
            api = f" api={w['api_status']}" if w.get("api") else ""
            lines.append(f"  {status:4} page={w['page_status']}{api} {w['href']}")
    if audit.get("defects"):
        lines.append("")
        lines.append("defects:")
        for d in audit["defects"]:
            lines.append(f"  - {d}")
    if audit.get("soft"):
        lines.append("")
        lines.append("soft:")
        for s in audit["soft"]:
            lines.append(f"  - {s}")
    return "\n".join(lines)


def readiness_check_summary(payload: dict[str, Any]) -> str:
    audit = payload.get("audit") or {}
    if audit.get("status_matrix"):
        return render_summary_line(audit).removeprefix("summary: ")
    if audit.get("defects"):
        return f"defects={len(audit['defects'])}"
    return str(payload.get("reason", "no status summary"))


def format_readiness_surface_summary(summary: dict[str, dict[str, int]]) -> str:
    if not summary:
        return "-"
    return " ".join(f"{surface}={counts.get('ok', 0)}/{counts.get('total', 0)}" for surface, counts in summary.items())


def readiness_scope_note(summary: dict[str, dict[str, int]]) -> str:
    def clean(surface: str) -> bool:
        counts = summary.get(surface, {})
        return counts.get("total", 0) > 0 and counts.get("ok", 0) == counts.get("total", 0)

    def failing(surface: str) -> bool:
        counts = summary.get(surface, {})
        return counts.get("total", 0) > 0 and counts.get("ok", 0) < counts.get("total", 0)

    if clean("local") and clean("hosted") and (failing("launch") or failing("pages")):
        return "local demo source and live HTTP are healthy; remaining debt is external launch/pages deployment"
    if failing("local"):
        return "local demo source has debt; fix checked-in docs or demo contracts before deployment"
    if failing("hosted"):
        return "live hosted demo has debt; fix the VM/page/API witness before launch"
    if failing("launch") or failing("pages"):
        return "external launch/pages deployment has debt"
    return "all readiness surfaces are healthy"


def render_readiness(payload: dict[str, Any]) -> str:
    checks = payload.get("checks") or []
    surface_summary = payload.get("surface_summary") or readiness_surface_summary(checks)
    lines = [
        f"demo-readiness-status: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        f"surfaces: {format_readiness_surface_summary(surface_summary)}",
        f"scope: {readiness_scope_note(surface_summary)}",
        "check      surface   verdict finding                      command",
        "---------- --------- ------- ---------------------------- -------------------------------",
    ]
    for check in checks:
        p = check["payload"]
        lines.append(
            f"{check['name']:<10} {str(check.get('surface', '')):<9} "
            f"{p['verdict']:<7} {p['finding']:<28} {check['command']}"
        )
    if checks:
        lines.append("details:")
        for check in checks:
            lines.append(f"  - {check['name']}: {readiness_check_summary(check['payload'])}")
    failed = [check for check in checks if not check["payload"].get("ok")]
    if failed:
        lines.append("actions:")
        for check in failed:
            p = check["payload"]
            lines.append(f"  - {check['name']}: {p.get('next_action', p.get('reason', 'rerun the check'))}")
    return "\n".join(lines)


def status_title(payload: dict[str, Any]) -> str:
    if payload.get("published"):
        return "demo-published-status"
    if payload.get("require_https"):
        return "demo-https-status"
    if payload.get("live"):
        return "demo-live-status"
    return "demo-static-status"


def format_count_map(counts: dict[str, int]) -> str:
    if not counts:
        return "-"
    return ",".join(f"{key}:{counts[key]}" for key in sorted(counts))


def render_summary_line(audit: dict[str, Any]) -> str:
    rows = audit.get("status_matrix") or []
    summary = audit.get("status_summary") or status_summary(rows)
    defect_count = len(audit.get("defects") or [])
    defects_part = f"defects={defect_count} " if defect_count else ""
    return (
        "summary: "
        f"hosted-links={summary.get('hosted_links', summary.get('hosted', 0))} "
        f"hosted-demos={summary.get('hosted_demos', 0)} "
        f"hub={summary.get('hub', 0)} "
        f"ok={summary.get('ok', 0)} "
        f"check={summary.get('check', 0)} "
        f"action={summary.get('action', 0)} "
        f"local-only={summary.get('local_only', 0)} "
        f"{defects_part}"
        f"https={format_count_map(summary.get('https', {}))}"
    )


def render_status(payload: dict[str, Any]) -> str:
    """Compact deployment matrix: one row per hosted/local-only browser demo."""
    audit = payload.get("audit") or {}
    rows = audit.get("status_matrix") or []
    summary = audit.get("status_summary") or status_summary(rows)
    lines = [
        f"{status_title(payload)}: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        render_summary_line(audit),
    ]
    if payload.get("verdict") != "OK" and not summary.get("action") and audit.get("defects"):
        lines.append(f"policy: {payload['verdict']} {payload['finding']} (aggregate defect; rows may still be HTTP-healthy)")
    lines.extend([
        "demo        role        status   http             api          https          url",
        "----------- ----------- --------  ---------------- ------------ -------------- -------------------------------",
    ])
    for row in rows:
        http = row["http"]
        api = row["api"]
        https = row["https"]
        if row["status"] == "local_only":
            status = "LOCAL"
            http_cell = "local-only"
        elif row["status"] == "action":
            status = "ACTION"
            http_cell = str(http["status"]) if http["checked"] else "not-checked"
        elif http["checked"]:
            status = "OK" if http["ok"] else "FAIL"
            http_cell = str(http["status"])
        else:
            status = "CHECK"
            http_cell = "not-checked"
        api_cell = str(api["status"]) if api.get("checked") else ("not-checked" if api.get("href") else "-")
        href = row.get("href") or "(not hosted)"
        lines.append(
            f"{row['demo']:<11} {row['role']:<11} {status:<8} "
            f"{http_cell:<16} {api_cell:<12} {https['state']:<14} {href}"
        )
        for defect in row.get("defects", [])[:2]:
            lines.append(f"  - {row['demo']}: {defect}")
    if audit.get("defects"):
        lines.append("defects:")
        for defect in audit["defects"]:
            lines.append(f"  - {defect}")
    if payload.get("local_source"):
        local = payload["local_source"]
        state = "OK" if local.get("ok") else "FAIL"
        lines.append(f"local-source: {state} {local.get('doc', DEFAULT_DOC)}")
    if audit.get("soft"):
        for item in audit["soft"]:
            lines.append(f"soft: {item}")
    return "\n".join(lines)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit hosted links on docs/demos.html.")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--doc", default=DEFAULT_DOC, help=f"demo page to audit (default: {DEFAULT_DOC})")
    ap.add_argument("--live", action="store_true", help="probe hosted HTTP links")
    ap.add_argument("--readiness", action="store_true", help="run static, live, HTTPS, and published status checks")
    ap.add_argument(
        "--published",
        action="store_true",
        help=f"fetch and audit the published GitHub Pages demos page ({PUBLISHED_DOC_URL})",
    )
    ap.add_argument("--timeout", type=float, default=8.0, help="per-link timeout in seconds for --live")
    ap.add_argument(
        "--require-https",
        action="store_true",
        help="with --live, fail if any hosted HTTP demo lacks a reachable HTTPS alternative",
    )
    ap.add_argument("--json", action="store_true", help="emit JSON payload")
    ap.add_argument("--status", action="store_true", help="emit only the compact hosted/local-only status matrix")
    args = ap.parse_args(argv)
    if args.readiness and (args.live or args.published or args.require_https):
        ap.error("--readiness runs all modes; do not combine it with --live, --published, or --require-https")
    if args.require_https and not args.live:
        ap.error("--require-https needs --live so HTTPS alternatives are actually probed")

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    if args.readiness:
        payload = collect_readiness(workspace, doc=args.doc, timeout_s=args.timeout)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render_readiness(payload))
        return 0 if payload.get("ok") else 1

    payload = collect(
        workspace,
        doc=args.doc,
        live=args.live,
        timeout_s=args.timeout,
        published=args.published,
        require_https=args.require_https,
    )
    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.status:
        print(render_status(payload))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
