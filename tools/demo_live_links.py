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
    python tools/demo_live_links.py --hub
    python tools/demo_live_links.py --hub --live

In --published mode, ACTION (published_deployment_drift) means the repository's
docs/demos.html is already clean but GitHub Pages is still serving stale HTML.
Use --live --require-https when a launch gate needs hosted demos to be embeddable
from HTTPS pages instead of merely reachable through top-level HTTP navigation.

Use --hub to audit the hosted hub route inventory that lives beyond the demo
cards: the landing pages (/, /adjudicate.html, /chat.html, /compare.html,
/gallery.html), the OpenAI-compatible /v1/models list, /healthz, and /metrics,
plus the root /api/* routes that are intentionally excluded and the retired
demo mounts that must stay tombstoned. --hub alone reports the intended route
set statically; --hub --live probes each route and reports whether it is
healthy, intentionally excluded, tombstoned, or an actionable placeholder/stale
hub link. It is separate from the card audit so --live --status stays unchanged.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
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


def net_value_from_api(api_text: str) -> dict[str, Any]:
    """Fold a hosted model-demo API body into the net-value witness.

    The live model demos (ctxdemo, demorace) report a model ladder, and demorace
    also reports an exact, timing-free prefill-token work-elimination ratio. This
    extracts the rung the live run actually served (the smallest present rung) and
    the measured fak delta, so the audit can prove the public card names the same
    model/rung the deployment serves and records a net-of-cost witness. Non-model
    demos (no "models" list) return an empty, defect-free witness so the check is a
    no-op for them.
    """
    try:
        data = json.loads(api_text)
    except (ValueError, TypeError):
        return {"rung": "", "present_rungs": [], "ratio": None, "defects": ["<invalid-json>"]}
    if not isinstance(data, dict):
        return {"rung": "", "present_rungs": [], "ratio": None, "defects": ["<non-object-json>"]}
    models = data.get("models")
    if not isinstance(models, list):
        return {"rung": "", "present_rungs": [], "ratio": None, "defects": []}
    present: list[str] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = m.get("name") or m.get("Name") or ""
        is_present = m.get("present")
        if is_present is None:
            is_present = m.get("Present")
        if name and is_present:
            present.append(str(name))
    ratio_block = data.get("prefill_tok_ratio_a_over_c")
    ratio = ratio_block if isinstance(ratio_block, dict) else None
    defects: list[str] = []
    if not present:
        defects.append("no model rung present; live run cannot back a net-true headline")
    return {
        "rung": present[0] if present else "",
        "present_rungs": present,
        "ratio": ratio,
        "defects": defects,
    }


def card_rung_defects(
    links: list[dict[str, Any]], witnesses: list[dict[str, Any]]
) -> list[str]:
    """Cross-check that each hosted model-demo CARD names the rung its live API served.

    This binds the public card copy to the live witness: the headline can neither
    drift ahead of nor lag behind the deployed model rung. Cards for non-model demos
    (no live rung in the witness) are skipped.
    """
    by_href = {w.get("href"): w for w in witnesses}
    defects: list[str] = []
    for row in links:
        if not row.get("card"):
            continue
        witness = by_href.get(row.get("href"))
        if not witness:
            continue
        rung = (witness.get("net_value") or {}).get("rung") or ""
        if not rung:
            continue
        if rung not in (row.get("text") or ""):
            defects.append(
                f"hosted card {row['href']} does not name its live model rung "
                f"{rung}; card copy and live API disagree"
            )
    return defects


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
    net_value: dict[str, Any] | None = None
    if api_url:
        api = fetcher(api_url, timeout_s)
        if not api.get("ok"):
            defects.append(f"api witness unreachable {api_url} ({api.get('status', 0)} {api.get('error', '')})")
        else:
            missing = missing_json_keys(str(api.get("text", "")), list(spec.get("api_keys", [])))
            for key in missing:
                defects.append(f"api witness {api_url} missing key: {key}")
            net_value = net_value_from_api(str(api.get("text", "")))
            for defect in net_value.get("defects", []):
                defects.append(f"net-value witness {api_url}: {defect}")

    return {
        "href": url,
        "ok": not defects,
        "skipped": False,
        "page_status": page.get("status", 0),
        "api": api_url,
        "api_status": api.get("status", 0) if api else 0,
        "net_value": net_value,
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
        live_defects.extend(card_rung_defects(audit["links"], witnesses))
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
    nv_lines = net_value_lines(audit)
    if nv_lines:
        lines.append("")
        lines.append("net-value witnesses:")
        for line in nv_lines:
            lines.append(f"  {line}")
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


def net_value_lines(audit: dict[str, Any]) -> list[str]:
    """One net-value witness line per live model demo: the served rung and the
    measured, timing-free prefill-token work-elimination delta."""
    lines: list[str] = []
    for witness in audit.get("witnesses") or []:
        net_value = witness.get("net_value") or {}
        rung = net_value.get("rung")
        if not rung:
            continue
        ratio = net_value.get("ratio") or {}
        ratio_part = ""
        raw = ratio.get("ratio")
        if isinstance(raw, (int, float)):
            ratio_part = (
                f"; prefill work {ratio.get('a')}->{ratio.get('c')} tok "
                f"= {raw:.1f}x eliminated"
            )
        lines.append(f"net-value: {witness['href']} rung={rung}{ratio_part}")
    return lines


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
    for line in net_value_lines(audit):
        lines.append(line)
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


# --- Hosted hub route inventory (#1745) ------------------------------------
#
# The card audit above watches the demo cards linked from docs/demos.html. The
# hosted hub root also serves routes that are not demo cards: the landing pages,
# the OpenAI-compatible model list, health, and Prometheus metrics. A manual
# probe (2026-06-30, #1745) found these live while the card audit only witnessed
# the root page marker. This inventory declares the intended hub route set and,
# under --live, reports whether each route is healthy, intentionally excluded,
# tombstoned (a proven-stale path that must stay gone), archived, or local-only.
#
# It is deliberately conservative so it does not turn transient content into
# flaky assertions: HTML routes assert a <title> marker (substring), JSON APIs
# assert object SHAPE (never volatile values), and /metrics asserts Prometheus
# HELP/TYPE structure plus metric NAMES only. Non-goals for v1 (per #1745): no
# volatile counter values, no latency/perf numbers, and root /api/* stays an
# intentionally-excluded 404 because the per-demo APIs live under their mounts.

HUB_SCHEMA = "fak-demo-hub-routes/1"

HUB_CLASS_PUBLIC = "public"
HUB_CLASS_EXCLUDED = "excluded"
HUB_CLASS_TOMBSTONED = "tombstoned"
HUB_CLASS_ARCHIVED = "archived"
HUB_CLASS_LOCAL_ONLY = "local_only"


@dataclass(frozen=True)
class HubRoute:
    path: str
    kind: str  # "html" | "json" | "prometheus"
    classification: str
    markers: tuple[str, ...] = ()  # HTML <title> substrings
    json_keys: tuple[str, ...] = ()  # top-level keys that must be present
    json_equals: tuple[tuple[str, Any], ...] = ()  # top-level key must equal value
    json_id_list: str = ""  # a list-valued key whose items must each carry a non-blank id
    metric_names: tuple[str, ...] = ()  # Prometheus metric names that must appear
    note: str = ""


# Page markers and API shapes are evidence-backed: HTML titles come from the
# #1745 maintainer probe; /v1/models, /healthz, and /metrics shapes match the
# gateway (cmd/fak api_host_test.go, claude_mac_fak_test.go, and
# internal/gateway/metrics_render.go's fak_gateway_up gauge). Tombstoned mounts
# mirror KNOWN_STALE_PREFIXES so a retired demo that reappears is actionable.
HUB_ROUTES: tuple[HubRoute, ...] = (
    HubRoute("/", "html", HUB_CLASS_PUBLIC,
             markers=("<title>fak — the agent kernel · live demos</title>",)),
    HubRoute("/adjudicate.html", "html", HUB_CLASS_PUBLIC, markers=("<title>fak · Adjudication",)),
    HubRoute("/chat.html", "html", HUB_CLASS_PUBLIC, markers=("<title>fak · Live Chat",)),
    HubRoute("/compare.html", "html", HUB_CLASS_PUBLIC, markers=("<title>fak · in-kernel",)),
    HubRoute("/gallery.html", "html", HUB_CLASS_PUBLIC, markers=("<title>fak · How it works",)),
    HubRoute("/v1/models", "json", HUB_CLASS_PUBLIC,
             json_equals=(("object", "list"),), json_id_list="data"),
    HubRoute("/healthz", "json", HUB_CLASS_PUBLIC,
             json_keys=("engine", "model"), json_equals=(("ok", True),)),
    HubRoute("/metrics", "prometheus", HUB_CLASS_PUBLIC, metric_names=("fak_gateway_up",)),
    HubRoute("/api/ladder", "json", HUB_CLASS_EXCLUDED,
             note="per-demo APIs live under their demo mount, not the hub root"),
    HubRoute("/api/scenarios", "json", HUB_CLASS_EXCLUDED,
             note="per-demo APIs live under their demo mount, not the hub root"),
    HubRoute("/api/suites", "json", HUB_CLASS_EXCLUDED,
             note="per-demo APIs live under their demo mount, not the hub root"),
    HubRoute("/guarddemo/", "html", HUB_CLASS_TOMBSTONED, note="retired guard demo; must stay 404"),
    HubRoute("/turntax/", "html", HUB_CLASS_TOMBSTONED, note="retired turntax mount; must stay gone"),
    HubRoute("/ctxdemo/", "html", HUB_CLASS_TOMBSTONED, note="retired ctxdemo mount; must stay gone"),
    HubRoute("/unsee/", "html", HUB_CLASS_TOMBSTONED, note="retired unsee mount; must stay gone"),
)


def hub_route_url(route: HubRoute, host: str = HOSTED_HOST) -> str:
    return f"http://{host}{route.path}"


def hub_witness_defects(route: HubRoute, text: str) -> list[str]:
    """Shape-only witness for a reachable public hub route. Never asserts
    volatile values: HTML by <title> marker, JSON by object shape, /metrics by
    Prometheus HELP/TYPE structure plus metric NAMES."""
    defects: list[str] = []
    if route.kind == "html":
        for marker in missing_markers(text, list(route.markers)):
            defects.append(f"missing page marker: {marker}")
    elif route.kind == "json":
        try:
            data = json.loads(text)
        except (ValueError, TypeError):
            return ["<invalid-json>"]
        if not isinstance(data, dict):
            return ["<non-object-json>"]
        for key in route.json_keys:
            if key not in data:
                defects.append(f"missing api key: {key}")
        for key, expected in route.json_equals:
            if data.get(key) != expected:
                defects.append(f"api key {key}={data.get(key)!r} expected {expected!r}")
        if route.json_id_list:
            items = data.get(route.json_id_list)
            if not isinstance(items, list) or not items:
                defects.append(f"api {route.json_id_list} is not a non-empty list")
            elif not all(isinstance(it, dict) and str(it.get("id", "")).strip() for it in items):
                defects.append(f"api {route.json_id_list}[].id missing or blank")
    elif route.kind == "prometheus":
        if "# HELP" not in text and "# TYPE" not in text:
            defects.append("metrics missing Prometheus HELP/TYPE lines")
        for name in route.metric_names:
            if name not in text:
                defects.append(f"metrics missing metric name: {name}")
    return defects


def _snippet(text: str, limit: int = 80) -> str:
    collapsed = " ".join(text.split())
    return collapsed if len(collapsed) <= limit else collapsed[:limit].rstrip() + "..."


def hub_placeholder_defects(text: str, host: str = HOSTED_HOST) -> list[str]:
    """Actionable hub-page problems independent of route health: placeholder
    card anchors (href '#' or empty) and links to proven-stale hub paths. The
    card label is truncated to a short snippet — the '#' href is the finding."""
    defects: list[str] = []
    for link in extract_links(text):
        target = link.href.strip()
        if link.is_card and target in {"", "#"}:
            defects.append(f"placeholder card: text={_snippet(link.text)!r} href={link.href!r}")
        abs_href = target if "://" in target else (f"http://{host}{target}" if target.startswith("/") else target)
        stale = known_stale_match(abs_href)
        if stale:
            defects.append(f"stale hub link to {stale['demo']}: {link.href}")
    return defects


def probe_hub_route(route: HubRoute, host: str = HOSTED_HOST, timeout_s: float = 8.0, *,
                    fetcher: Any = fetch_url_text, prober: Any = probe_url) -> dict[str, Any]:
    url = hub_route_url(route, host)
    row: dict[str, Any] = {
        "path": route.path,
        "url": url,
        "kind": route.kind,
        "classification": route.classification,
        "note": route.note,
        "checked": True,
        "http_status": 0,
        "state": "",
        "defects": [],
    }
    if route.classification == HUB_CLASS_PUBLIC:
        fetched = fetcher(url, timeout_s)
        row["http_status"] = int(fetched.get("status", 0))
        defects: list[str] = []
        if not fetched.get("ok"):
            defects.append(f"route unreachable ({fetched.get('status', 0)} {fetched.get('error', '')})")
            row["state"] = "hub-route-action"
        else:
            text = str(fetched.get("text", ""))
            defects.extend(hub_witness_defects(route, text))
            placeholder = hub_placeholder_defects(text, host) if route.path == "/" else []
            if placeholder:
                row["state"] = "hub-placeholder-action"
            elif defects:
                row["state"] = "hub-route-action"
            else:
                row["state"] = "hub-route-ok"
            defects.extend(placeholder)
        row["defects"] = defects
        return row

    # Excluded / tombstoned / archived: witness ABSENCE. We only need the status
    # code, so a HEAD/GET probe is enough. A route that starts serving again is
    # the actionable case.
    probe = prober(url, timeout_s)
    row["http_status"] = int(probe.get("status", 0))
    served = bool(probe.get("ok"))
    if route.classification == HUB_CLASS_EXCLUDED:
        if served:
            row["state"] = "hub-route-action"
            row["defects"] = [f"excluded root route unexpectedly serves ({row['http_status']}); per-demo APIs belong under their mount"]
        else:
            row["state"] = "hub-route-excluded"
    else:  # tombstoned / archived
        label = "hub-route-tombstoned" if route.classification == HUB_CLASS_TOMBSTONED else "hub-route-archived"
        if served:
            row["state"] = "hub-route-action"
            row["defects"] = [f"{route.classification} route is reachable again ({row['http_status']}) — the hub still exposes a retired path"]
        else:
            row["state"] = label
    return row


def hub_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "total": len(rows),
        "public": 0, "excluded": 0, "tombstoned": 0, "archived": 0, "local_only": 0,
        "ok": 0, "action": 0, "placeholder_action": 0,
    }
    for row in rows:
        cls = row.get("classification", "")
        if cls in summary:
            summary[cls] += 1
        state = row.get("state", "")
        if state == "hub-route-ok":
            summary["ok"] += 1
        elif state == "hub-placeholder-action":
            summary["placeholder_action"] += 1
            summary["action"] += 1
        elif state == "hub-route-action":
            summary["action"] += 1
    return summary


def collect_hub(host: str = HOSTED_HOST, *, live: bool = False, timeout_s: float = 8.0,
                fetcher: Any = fetch_url_text, prober: Any = probe_url) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    defects: list[str] = []
    for route in HUB_ROUTES:
        if not live or route.classification == HUB_CLASS_LOCAL_ONLY:
            declared = ("hub-route-local-only" if route.classification == HUB_CLASS_LOCAL_ONLY
                        else f"hub-route-{route.classification.replace('_', '-')}")
            rows.append({
                "path": route.path,
                "url": hub_route_url(route, host),
                "kind": route.kind,
                "classification": route.classification,
                "note": route.note,
                "checked": False,
                "http_status": 0,
                "state": declared,
                "defects": [],
            })
            continue
        row = probe_hub_route(route, host, min(timeout_s, 8.0), fetcher=fetcher, prober=prober)
        rows.append(row)
        for d in row.get("defects", []):
            defects.append(f"{route.path}: {d}")
    summary = hub_summary(rows)
    return build_hub_payload(host, rows, summary, defects, live=live)


def build_hub_payload(host: str, rows: list[dict[str, Any]], summary: dict[str, Any],
                      defects: list[str], *, live: bool) -> dict[str, Any]:
    ok = not defects
    if not live:
        verdict, finding = "INVENTORY", "hub_route_inventory"
        reason = f"{summary['total']} intended hub route(s) declared for {host}; rerun with --live to witness health"
        next_action = "run tools/demo_live_links.py --hub --live to probe the hosted hub"
    elif ok:
        verdict, finding = "OK", "hub_routes_clean"
        reason = (f"{summary['public']} public route(s) healthy, {summary['excluded']} excluded, "
                  f"{summary['tombstoned']} tombstoned; no placeholder or stale hub links")
        next_action = "rerun after a hub route or deployment change"
    else:
        verdict, finding = "ACTION", "hub_route_debt"
        reason = f"{len(defects)} hub route defect(s) on {host}"
        next_action = "fix the unhealthy route, remove the placeholder/stale hub link, or update the intended hub route set"
    return {
        "schema": HUB_SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "host": host,
        "live": live,
        "hub": {"routes": rows, "summary": summary, "defects": defects},
    }


def render_hub(payload: dict[str, Any]) -> str:
    """Compact hub-route inventory: one row per intended hub route, its
    classification, and (under --live) its witnessed state."""
    hub = payload.get("hub") or {}
    rows = hub.get("routes") or []
    summary = hub.get("summary") or {}
    title = "demo-hub-live-status" if payload.get("live") else "demo-hub-inventory"
    lines = [
        f"{title}: {payload['verdict']} ({payload['finding']})",
        f"  {payload['reason']}",
        (
            "summary: "
            f"public={summary.get('public', 0)} "
            f"excluded={summary.get('excluded', 0)} "
            f"tombstoned={summary.get('tombstoned', 0)} "
            f"archived={summary.get('archived', 0)} "
            f"local-only={summary.get('local_only', 0)} "
            f"ok={summary.get('ok', 0)} "
            f"action={summary.get('action', 0)} "
            f"placeholder={summary.get('placeholder_action', 0)}"
        ),
        "route                 kind        class        state                 http",
        "--------------------- ----------- ------------ --------------------- ------",
    ]
    for row in rows:
        http_cell = str(row.get("http_status", 0)) if row.get("checked") else "-"
        lines.append(
            f"{row['path']:<21} {row['kind']:<11} {row['classification']:<12} "
            f"{row['state']:<21} {http_cell}"
        )
        for defect in row.get("defects", [])[:2]:
            lines.append(f"  - {row['path']}: {defect}")
    if hub.get("defects"):
        lines.append("defects:")
        for defect in hub["defects"]:
            lines.append(f"  - {defect}")
    return "\n".join(lines)


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _use_utf8_stdout() -> None:
    """Live hub pages carry emoji/arrows a Windows cp1252 console cannot encode;
    the audit must never crash while REPORTING a finding. StringIO (tests) has no
    reconfigure, so guard for it and fall back to a lossy-but-safe replace."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _use_utf8_stdout()
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
    ap.add_argument(
        "--hub",
        action="store_true",
        help="audit the hosted hub route inventory (landing pages, /v1/models, /healthz, /metrics) beyond the demo cards; add --live to probe",
    )
    args = ap.parse_args(argv)
    if args.readiness and (args.live or args.published or args.require_https):
        ap.error("--readiness runs all modes; do not combine it with --live, --published, or --require-https")
    if args.require_https and not args.live:
        ap.error("--require-https needs --live so HTTPS alternatives are actually probed")
    if args.hub and (args.published or args.readiness or args.require_https):
        ap.error("--hub audits the hosted hub route set; do not combine it with --published, --readiness, or --require-https")

    if args.hub:
        payload = collect_hub(HOSTED_HOST, live=args.live, timeout_s=args.timeout)
        if args.json:
            print(json.dumps(payload, indent=2))
        else:
            print(render_hub(payload))
        return 0 if payload.get("ok") else 1

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
