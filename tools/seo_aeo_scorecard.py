#!/usr/bin/env python3
"""SEO/AEO scorecard — the measuring stick that makes "more discoverable" provable.

`tools/docs_scorecard.py` answers "are the docs *correct* for a reader who already
found us". This tool answers the orthogonal question: **will a reader — or an
answer engine — find us at all, and cite us correctly when they do.** Same
discipline (deterministic, content-only, no model, no network), aimed at the
discoverability surface instead of the prose:

  SEO  (Search Engine Optimization) — the signals Google/Bing rank on: a unique
       title tag and meta description on every published page, a canonical URL,
       a sitemap, a crawlable robots.txt, an Open Graph / social card, a clean
       heading hierarchy, no dead links bleeding crawl budget.
  AEO  (Answer Engine Optimization) — the signals an LLM answer engine
       (ChatGPT, Perplexity, Google AI Overviews, Claude) ingests and cites:
       JSON-LD structured data (SoftwareApplication, FAQPage, WebSite,
       BreadcrumbList), an `llms.txt` + `llms-full.txt` corpus, a self-contained
       "what is X" answer on the first screen, an FAQ in real question/answer
       structure, and generated machine artifacts that match the visible docs.

Like the docs scorecard, the headline metric is an integer you drive toward zero:
**seo-debt** — the count of concrete, re-derivable discoverability defects (a
published page with no meta description, a missing JSON-LD type, a stale
`llms-full.txt`, a dead link, a missing social card). "Make us more discoverable"
was an unfalsifiable claim; seo-debt is the number that makes it checkable, so an
improvement program can state an honest target ("cut seo-debt 10×, from N to
N/10") instead of a vibe.

Two layers fold into one payload:

  PER-PAGE (each published Pages surface, 0-100, weighted into a score + A-F):
    title          front-matter `title:` present, a sane title-tag length
    description    front-matter `description:` present, in the 70-160 char band
    headings       exactly one H1, no skipped heading level, real sections
    links          every local link target resolves on disk (crawl integrity)
    links_crawlable a local .md link must resolve to a Jekyll-PUBLISHED page a
                   crawler can actually fetch — not a disk file the site excludes
    answerability  the first screen is a self-contained plain-language answer
                   (the chunk an answer engine quotes), not bare scaffolding
    alt_text       every image (markdown ![alt] or a raw <img>) carries usable alt
                   text — image-search SEO + the caption a screen reader/engine reads

  SITE-LEVEL (the once-per-corpus infrastructure, each a unit of debt if absent):
    robots_ok · ai_crawlers (named answer-engine bots are explicitly welcomed)
    sitemap_plugin · seo_tag_plugin · canonical_url · og_image
    structured_data (one defect per missing expected JSON-LD @type)
    llms_txt · llms_full (present AND fresh vs llms.txt) · llms_full_sources
    faq_structured · faq_jsonld_sync · breadcrumb_jsonld_shape
    citation_links (every llms.txt-map + self-repo github link resolves live)

  PRESENCE vs SUCCESS. The first generation of KPIs above asked "is the meta/
  link/JSON-LD PRESENT" — which tops out the instant a field exists. The newer
  SUCCESS KPIs ask "does it WORK for the consumer who relies on it": does a link
  resolve to a page a crawler can FETCH (links_crawlable), is the title/meta
  UNIQUE across the corpus so an engine can tell the pages apart (meta_distinct,
  folded corpus-wide), does a citation link point to a LIVE target
  (citation_links). A page can score a perfect 100 on the presence checks while a
  reader who follows its links lands on a 404 — that gap is what these close.

`ok` is False iff any HARD defect exists — these are the work-list. Soft signals
(a too-short title, a thin page, an optional schema type) lower a score but never
gate, exactly the split the docs scorecard draws.

Read-only by construction: it reads the published `.md` surfaces, `docs/_config.yml`,
`docs/robots.txt`, the Pages `<head>` include, `llms.txt`/`llms-full.txt`, and the
social image; it edits nothing. Scores are STRATEGIC (go-to-market positioning),
so the committed report and the baseline JSON belong in the PRIVATE repo, not the
public tree — `--transfer` writes them there (see `--private-out`); the public
`.gitignore` blocks them from ever landing in a public commit.

Run from the repo ROOT::

    python tools/seo_aeo_scorecard.py                      # human scorecard
    python tools/seo_aeo_scorecard.py --json               # machine payload
    python tools/seo_aeo_scorecard.py --markdown           # the report body
    python tools/seo_aeo_scorecard.py --compare base.json  # prove the debt moved
    python tools/seo_aeo_scorecard.py --transfer --stamp DATE   # -> private repo
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Any

SCHEMA = "fak-seo-aeo-scorecard/4"

# Repo-root-relative inputs (best-effort; a missing one degrades a check, never errors).
CONFIG_REL = "docs/_config.yml"
ROBOTS_REL = "docs/robots.txt"
HEAD_INCLUDE_REL = "docs/_includes/head-custom.html"
INDEX_REL = "docs/index.md"
SHOWCASE_REL = "docs/showcase.html"
LLMS_REL = "llms.txt"
LLMS_FULL_REL = "llms-full.txt"
FAQ_REL = "docs/FAQ.md"

# Where strategic scores go: the private fleet repo, mirroring tools/dos_sync.py's
# `../fleet/.dos-archive/fak`. A `fak` clone sits beside its private `fleet`
# source; the score archive lands in a dir the public tree's .gitignore blocks,
# so a baseline can never become a public commit by accident.
DEFAULT_PRIVATE_OUT = os.path.join("..", "fleet", ".seo-archive", "fak")
REPORT_NAME = "SEO-AEO-SCORECARD.md"
BASELINE_NAME = "seo-aeo-baseline.json"   # the PINNED reference (before); --compare target
CURRENT_NAME = "seo-aeo-current.json"     # the latest payload (after), rewritten each run

# ---------------------------------------------------------------------------
# The DISCOVERY surface (the --scope core gate) is DERIVED, not hand-listed: it is
# every Jekyll-published page link-reachable from the front doors, minus the
# evidence/appendix subtrees. Deriving it (vs a curated CORE list) is deliberate —
# a hand-edited set can be quietly trimmed to drop a low-scoring peer and make the
# headline look clean (an adversarial review caught exactly that risk). A reviewer
# re-runs the same BFS and gets the same set; nothing can be excluded by hand.
# ---------------------------------------------------------------------------
# Front doors: where a reader/crawler enters. Reachability is link-BFS from these.
FRONT_DOORS = ["README.md", "llms.txt", "docs/index.md", "INDEX.md", "START-HERE.md"]

# Evidence/appendix subtrees: deep-technical proof docs, dated benchmark results,
# and working notes. They ARE published (indexable long-tail, scored under
# --scope published) but they are not the reader/answer-engine DISCOVERY surface,
# so --scope core excludes them. A directory rule — fully reproducible, not a
# hand-picked per-page exclusion.
EVIDENCE_DIRS = {"proofs", "benchmarks", "notes"}

# Expected JSON-LD @types for AEO. Each missing HARD type is one unit of seo-debt;
# answer engines lean on these to identify and cite the project. SoftwareApplication
# (what the project IS), FAQPage (the Q&A an engine quotes), WebSite (the site
# identity + search action), and BreadcrumbList (where a crawler places the docs
# surface in the site hierarchy) are required; Organization is a bonus.
JSONLD_TYPES_HARD = ["SoftwareApplication", "FAQPage", "WebSite", "BreadcrumbList"]
JSONLD_TYPES_SOFT = ["Organization"]

# Per-KPI weights for the per-page score. description + title weigh most — they ARE
# the search result (the blue link + the snippet) and the first thing an engine
# reads. links + links_crawlable together carry crawl integrity (a real link that
# 404s on the live site is as bad as a dead one). answerability + alt_text weigh
# least: answerability is a voice judgment call, and alt_text only bites pages that
# carry images. The sum MUST be 1.0 (asserted in the tests).
KPI_WEIGHTS: dict[str, float] = {
    "title": 0.21,
    "description": 0.25,
    "headings": 0.12,
    "links": 0.16,
    "links_crawlable": 0.12,
    "answerability": 0.08,
    "alt_text": 0.06,
}
assert abs(sum(KPI_WEIGHTS.values()) - 1.0) < 1e-9, "per-page KPI weights must sum to 1.0"

# Title-tag and meta-description length laws (chars). Outside the band is a SOFT
# nudge (it still renders, just sub-optimally); ABSENT is the HARD defect.
TITLE_MIN, TITLE_MAX = 15, 70
DESC_MIN, DESC_MAX = 70, 160

MIN_FAQ_QUESTIONS = 6  # below this, FAQ.md is too thin to seed a FAQPage

# Dirs under docs/ that Jekyll does NOT publish as reader pages (mirrors the
# _config.yml `exclude` list + Jekyll's own `_`-prefixed special dirs). A page in
# one of these is not an indexable HTML surface, so it carries no SEO meta duty.
# `launch` holds go-to-market distribution drafts (paste-ready Reddit/HN/X/YouTube
# posts, the positioning brief, landscape research) — launch ops, not reader docs,
# and excluded from publishing in _config.yml for the same reason planning/releases
# are; this set must stay in sync with that `exclude` list.
NONPUBLISHED_DIRS = {
    "benchmark", "benchmarking", "planning", "testing", "releases",
    "stable-releases", "launch", "archive", "_includes", "_layouts", "_data", "_site",
}

_LINK_RE = re.compile(r"\[(?P<text>[^\]]+)\]\((?P<target>[^)]+)\)")
_H_RE = re.compile(r"^(#{1,6})\s+\S", re.MULTILINE)
_H1_RE = re.compile(r"^#\s+\S", re.MULTILINE)
_H2_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
_FENCE_RE = re.compile(r"```.*?```", re.DOTALL)
_JSONLD_BLOCK_RE = re.compile(
    r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
    re.IGNORECASE | re.DOTALL,
)
_QUESTION_RE = re.compile(
    r"(?i)^(how|what|why|when|where|who|which|is|are|can|does|do|should|will)\b")

# Self-referential links into the project's OWN repo (a github.com/.../fak blob,
# tree, or raw URL on a real branch). The per-page `links` KPI skips ALL http(s),
# so it is blind to a self-repo link that rots when a file moves — yet that link
# is in-repo and resolvable offline. The captured group is the repo-relative path;
# the URL is terminated on the first delimiter so it never swallows trailing HTML
# attribute junk (`...mp4">full-resolution`) from a <video>/<a> tag.
SELF_REPO_RE = re.compile(
    r"https?://(?:github\.com|raw\.githubusercontent\.com)/anthony-chaudhary/fak/"
    r"(?:(?:blob|tree|raw)/)?(?:main|master|HEAD)/([^)\s\"'<>`#?]+)")

# Image-alt regexes (the alt_text KPI). A markdown image and a raw <img> tag both
# owe non-empty alt text (image-search SEO + the caption a screen reader / answer
# engine reads). Inline `code` and ```fenced``` spans are stripped first, so a doc
# that SHOWS `![](x.svg)` syntax as an example is never scored as a real missing-alt
# image — the same fence-stripping discipline kpi_links / kpi_headings already use.
_MD_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\((?P<src>[^)]+)\)")
# Reference-style image (`![alt][id]`, collapsed `![alt][]`) — renders a live image
# just like the inline form, so an empty-alt reference is the same HARD defect. The
# `]\[` shape never overlaps the inline `]\(` form, so the two loops don't double-count.
_MD_REF_IMG_RE = re.compile(r"!\[(?P<alt>[^\]]*)\]\[[^\]]*\]")
_HTML_IMG_RE = re.compile(r"<img\b[^>]*>", re.IGNORECASE)
# Anchor `alt` on a tag/quote/space boundary, NOT a bare `\b`: `\b` also fires inside
# `data-alt=`, letting a non-alt attribute satisfy the check and downgrade a HARD
# missing-alt to a pass. The leading class requires alt to start an attribute.
_HTML_ALT_RE = re.compile(r'(?:^|[\s"\'])alt\s*=\s*["\']([^"\']*)["\']', re.IGNORECASE)
_INLINE_CODE_RE = re.compile(r"`[^`\n]*`")

# A lone generic filler word is "present" alt that is useless to image search and a
# screen reader (the alt analogue of a _degenerate title/description). It is a SOFT
# nudge, never a HARD defect — a real one-word caption is rare but legal.
_ALT_FILLER = {"image", "img", "picture", "photo", "screenshot", "graphic",
               "figure", "icon", "logo", "diagram", "chart", "svg", "png", "jpg"}

# Named answer-engine / AI crawlers an AEO-optimized robots.txt must explicitly
# welcome (2025-2026 landscape). A bare `User-agent: * / Allow: /` technically
# permits them, but the higher AEO bar — the one this harden enforces — is to NAME
# the major answer engines so the intent to be crawled-and-cited survives a future
# wildcard tightening, and so a single accidental `Disallow` can't silently delist
# the project from an answer engine. Blocking any of these makes the project
# invisible to that engine's citations.
AI_CRAWLER_UAS = {
    "GPTBot", "OAI-SearchBot", "ChatGPT-User",        # OpenAI
    "ClaudeBot", "anthropic-ai", "Claude-SearchBot",  # Anthropic
    "PerplexityBot", "Perplexity-User",               # Perplexity
    "Google-Extended",                                # Google Gemini / AI Overviews
    "Applebot-Extended",                              # Apple Intelligence
    "CCBot",                                          # Common Crawl (feeds many LLMs)
}
# The subset every AEO robots.txt MUST name explicitly (the four dominant answer
# engines). Naming all of AI_CRAWLER_UAS is a bonus; missing one of these is debt.
AI_CRAWLER_REQUIRED = ("GPTBot", "ClaudeBot", "PerplexityBot", "Google-Extended")


def _is_question(heading: str) -> bool:
    h = heading.strip()
    return h.endswith("?") or bool(_QUESTION_RE.match(h))


def _degenerate(s: str) -> bool:
    """A 'present' string with no usable content — filler, a single repeated
    character, or fewer than two real words. Closes the gaming vector the review
    flagged: a title of 'x'*100 or a description of '.'*30 must NOT score a clean
    100; degenerate meta is as useless to a searcher as missing meta."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]+", s)
    distinct = {w.lower() for w in words}
    has_real = any(len(w) >= 3 for w in words)
    single_char = len(set(s.strip().replace(" ", ""))) <= 1
    return single_char or not has_real or len(distinct) < 2


def _degenerate_alt(alt: str) -> bool:
    """True for alt text that is 'present' but useless for image SEO / a screen
    reader: no real word, or a single generic filler word ('image', 'diagram')."""
    words = re.findall(r"[A-Za-z][A-Za-z'’-]+", alt)
    if not words:
        return True
    return len(words) == 1 and words[0].lower() in _ALT_FILLER


def _collect_jsonld_types(data: Any) -> set[str]:
    """Walk a parsed JSON-LD value collecting every @type (handles a bare object,
    a top-level array, and an @graph). Only valid-parsed blocks reach here, so a
    malformed <script> never contributes a phantom type."""
    out: set[str] = set()
    if isinstance(data, dict):
        t = data.get("@type")
        if isinstance(t, str):
            out.add(t)
        elif isinstance(t, list):
            out |= {x for x in t if isinstance(x, str)}
        for v in data.values():
            out |= _collect_jsonld_types(v)
    elif isinstance(data, list):
        for v in data:
            out |= _collect_jsonld_types(v)
    return out


def _jsonld_has_type(data: Any, typ: str) -> bool:
    if not isinstance(data, dict):
        return False
    t = data.get("@type")
    if isinstance(t, str):
        return t == typ
    if isinstance(t, list):
        return typ in {x for x in t if isinstance(x, str)}
    return False


def _iter_jsonld_objects(data: Any):
    """Yield every dict object in a parsed JSON-LD value."""
    if isinstance(data, dict):
        yield data
        for v in data.values():
            yield from _iter_jsonld_objects(v)
    elif isinstance(data, list):
        for v in data:
            yield from _iter_jsonld_objects(v)


def _jsonld_objects_with_type(values: list[Any], typ: str) -> list[dict[str, Any]]:
    return [obj for data in values for obj in _iter_jsonld_objects(data)
            if _jsonld_has_type(obj, typ)]


def _jsonld_url(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("@id", "url", "item"):
            v = value.get(key)
            if isinstance(v, str):
                return v
    return ""


def breadcrumb_shape_ok(values: list[Any]) -> tuple[bool, str]:
    """A BreadcrumbList must be more than a present @type: it needs ordered
    ListItems with absolute URLs, or crawlers cannot place the page reliably."""
    for bc in _jsonld_objects_with_type(values, "BreadcrumbList"):
        items = bc.get("itemListElement")
        if not isinstance(items, list) or len(items) < 2:
            continue
        ok = True
        for want_pos, item in enumerate(items, 1):
            if not isinstance(item, dict) or not _jsonld_has_type(item, "ListItem"):
                ok = False
                break
            if item.get("position") != want_pos:
                ok = False
                break
            name = item.get("name")
            url = _jsonld_url(item.get("item"))
            if not isinstance(name, str) or not name.strip() or not url.startswith(("https://", "http://")):
                ok = False
                break
        if ok:
            return True, f"BreadcrumbList has {len(items)} ordered absolute ListItem entries"
    return False, "BreadcrumbList JSON-LD missing or structurally invalid"


def faq_jsonld_sync_ok(values: list[Any], faq_text: str) -> tuple[bool, str]:
    """The FAQPage block must mirror the visible FAQ questions, not merely exist."""
    visible = [h.strip() for h in _H2_RE.findall(faq_text) if _is_question(h)]
    questions: list[str] = []
    answers_ok = True
    for faq in _jsonld_objects_with_type(values, "FAQPage"):
        entities = faq.get("mainEntity")
        if not isinstance(entities, list):
            continue
        for ent in entities:
            if not isinstance(ent, dict) or not _jsonld_has_type(ent, "Question"):
                answers_ok = False
                continue
            q = ent.get("name")
            if isinstance(q, str):
                questions.append(q.strip())
            ans = ent.get("acceptedAnswer")
            if isinstance(ans, dict):
                text = ans.get("text")
                if not isinstance(text, str) or len(text.strip()) < 20:
                    answers_ok = False
            else:
                answers_ok = False
    if visible and questions == visible and answers_ok:
        return True, f"FAQPage JSON-LD mirrors {len(visible)} visible FAQ questions"
    return False, (f"FAQPage JSON-LD stale or incomplete "
                   f"({len(questions)} schema questions vs {len(visible)} visible questions)")


def llms_full_source_audit(root: Path) -> dict[str, Any]:
    """Every local markdown target in llms.txt must appear as a Source block in
    llms-full.txt. This makes the one-fetch corpus a coverage check, not just a
    freshness substring check."""
    llms = _safe_read(root / LLMS_REL)
    llms_full = _safe_read(root / LLMS_FULL_REL)
    targets: list[str] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(llms):
        target = m.group("target").strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        rel = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not rel.endswith(".md") or rel in seen:
            continue
        seen.add(rel)
        targets.append(rel.replace("\\", "/"))
    sources = set(re.findall(r"^> Source: `([^`]+)`\s*$", llms_full, re.MULTILINE))
    missing = [t for t in targets if t not in sources]
    return {"targets": targets, "sources": sorted(sources), "missing": missing}


# ---------------------------------------------------------------------------
# Front-matter parsing (zero external deps — a minimal title/description reader).
# ---------------------------------------------------------------------------

def parse_front_matter(text: str) -> dict[str, str]:
    """Extract scalar `title:` and `description:` from a leading `---` YAML block.

    Handles the three forms the repo uses: a quoted one-liner (`title: "x"`), a
    bare one-liner (`title: x`), and a folded block scalar (`description: >-`
    followed by indented continuation lines). Returns {} if there is no front
    matter. Intentionally tiny: it reads only the two keys the scorecard needs,
    so it never has to be a full YAML parser (the no-deps rule).
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    lines = block.splitlines()
    out: dict[str, str] = {}
    i = 0
    while i < len(lines):
        line = lines[i]
        m = re.match(r"^(title|description)\s*:\s*(.*)$", line)
        if not m:
            i += 1
            continue
        key, rest = m.group(1), m.group(2).strip()
        if rest in (">", "|", ">-", "|-", ">+", "|+"):
            # Folded/literal block scalar: gather indented continuation lines.
            parts: list[str] = []
            i += 1
            while i < len(lines) and (lines[i].startswith((" ", "\t")) or not lines[i].strip()):
                parts.append(lines[i].strip())
                i += 1
            out[key] = " ".join(p for p in parts if p).strip()
            continue
        out[key] = _unquote(rest)
        i += 1
    return out


def _unquote(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    return s


# ---------------------------------------------------------------------------
# Per-page KPIs. Each takes already-read text (+ context) and returns
#   {kpi, score (0-100 int), detail, defects: [str], soft: [str]}
# every item in `defects` is one HARD unit of seo-debt; `soft` never gates.
# ---------------------------------------------------------------------------

def _clamp(score: float) -> int:
    return int(max(0, min(100, round(score))))


def kpi_title(fm: dict[str, str]) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    title = fm.get("title", "").strip()
    if not title:
        defects.append("no front-matter title: (no <title> tag / no blue-link text)")
        return {"kpi": "title", "score": 0, "detail": "missing title",
                "defects": defects, "soft": soft}
    if _degenerate(title):
        defects.append("degenerate title (filler/repeat — no usable <title> text)")
        return {"kpi": "title", "score": 0, "detail": "degenerate title",
                "defects": defects, "soft": soft}
    n = len(title)
    score = 100
    if n < TITLE_MIN:
        soft.append(f"title is thin ({n} chars; aim {TITLE_MIN}-{TITLE_MAX})")
        score -= 20
    elif n > TITLE_MAX:
        soft.append(f"title is long ({n} chars; search truncates past ~{TITLE_MAX})")
        score -= 15
    return {"kpi": "title", "score": _clamp(score),
            "detail": f"title present ({n} chars)", "defects": defects, "soft": soft}


def kpi_description(fm: dict[str, str]) -> dict[str, Any]:
    defects: list[str] = []
    soft: list[str] = []
    desc = fm.get("description", "").strip()
    if not desc:
        defects.append("no front-matter description: (no meta description / SERP snippet)")
        return {"kpi": "description", "score": 0, "detail": "missing description",
                "defects": defects, "soft": soft}
    if _degenerate(desc):
        defects.append("degenerate description (filler/repeat — no usable SERP snippet)")
        return {"kpi": "description", "score": 0, "detail": "degenerate description",
                "defects": defects, "soft": soft}
    n = len(desc)
    score = 100
    if n < DESC_MIN:
        soft.append(f"description is thin ({n} chars; aim {DESC_MIN}-{DESC_MAX})")
        score -= 20
    elif n > DESC_MAX:
        soft.append(f"description is long ({n} chars; search truncates past ~{DESC_MAX})")
        score -= 15
    return {"kpi": "description", "score": _clamp(score),
            "detail": f"description present ({n} chars)", "defects": defects, "soft": soft}


def kpi_headings(text: str) -> dict[str, Any]:
    """SEO heading hygiene: exactly one H1, no skipped level, real sections."""
    defects: list[str] = []
    soft: list[str] = []
    # Strip fenced code first: a shell comment (`# install deps`) or a `### step`
    # inside a ```code``` fence is NOT a markdown heading. Counting it inflated H1
    # tallies on 40 core pages and fabricated H1->H3 "skip" signals (17 of 18 were
    # fence noise). kpi_links already strips fences; headings must too.
    body = _FENCE_RE.sub(" ", _strip_front_matter(text))
    levels = [len(m.group(1)) for m in _H_RE.finditer(body)]
    h1s = [lvl for lvl in levels if lvl == 1]
    score = 100
    if len(h1s) == 0:
        defects.append("no H1 heading (a '# Title' line)")
        score -= 30
    elif len(h1s) > 1:
        soft.append(f"{len(h1s)} H1 headings (expected exactly one per page)")
        score -= 10
    # Skipped level: a jump from Hn to Hn+2 (e.g. H1 -> H3) confuses crawlers.
    prev = 0
    for lvl in levels:
        if prev and lvl > prev + 1:
            soft.append(f"heading level jumps H{prev}->H{lvl} (skips H{prev+1})")
            score -= 8
            break
        prev = lvl
    n_lines = body.count("\n") + 1
    if n_lines > 40 and len(levels) <= 1:
        defects.append("long page with no section headings (## …)")
        score -= 20
    return {"kpi": "headings", "score": _clamp(score),
            "detail": f"{len(h1s)} H1 / {len(levels)} headings",
            "defects": defects, "soft": soft}


def kpi_links(text: str, root: Path, doc_rel: str) -> dict[str, Any]:
    """Resolve every LOCAL link relative to the page's own dir — a dead link
    bleeds crawl budget and breaks an answer engine following a citation."""
    base = (root / doc_rel).parent
    # Strip fenced code blocks first: an illustrative path inside a ```code``` fence
    # is an example, not a real link, and validating it yields false-positive dead
    # links (the review flagged this). Markdown links in prose are still scanned.
    text = _FENCE_RE.sub(" ", text)
    dead: list[str] = []
    seen: set[str] = set()
    total = 0
    for m in _LINK_RE.finditer(text):
        target = m.group("target").strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        path_part = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not path_part or path_part in seen:
            continue
        seen.add(path_part)
        total += 1
        resolved = (root / path_part.lstrip("/")) if path_part.startswith("/") \
            else (base / path_part)
        if not resolved.exists():
            dead.append(path_part)
    defects = [f"dead link: {d}" for d in sorted(dead)]
    score = 100 - 20 * len(dead)
    return {"kpi": "links", "score": _clamp(score),
            "detail": (f"all {total} local link(s) resolve" if not dead
                       else f"{len(dead)}/{total} local link(s) dead"),
            "defects": defects, "soft": []}


def kpi_links_crawlable(text: str, root: Path, doc_rel: str) -> dict[str, Any]:
    """SUCCESS check, the crawl-integrity twin of `links`: a local link that
    EXISTS on disk is not the same as one a crawler can FETCH. A markdown link to
    a `.md` page under docs/ that Jekyll EXCLUDES from publishing (it lives in a
    NONPUBLISHED_DIRS subtree) resolves on disk — so `links` scores it 100 — yet
    it is a hard 404 on the live site and a dangling citation for an answer engine
    following it. That is the HARD defect. A link that resolves to a directory
    (no canonical page on the published site, though GitHub renders a listing) is
    a SOFT nudge: it works on one surface, not the other."""
    base = (root / doc_rel).parent
    text = _FENCE_RE.sub(" ", text)
    crawl404: list[str] = []
    dirlinks: list[str] = []
    seen: set[str] = set()
    for m in _LINK_RE.finditer(text):
        target = m.group("target").strip()
        if target.startswith(("http://", "https://", "mailto:", "#", "tel:")):
            continue
        pp = target.split("#", 1)[0].split("?", 1)[0].strip()
        if not pp or pp in seen:
            continue
        seen.add(pp)
        resolved = (root / pp.lstrip("/")) if pp.startswith("/") else (base / pp)
        if not resolved.exists():
            continue  # dead-on-disk is the existing `links` KPI's job, not double-counted
        try:
            tgt = resolved.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if resolved.is_dir():
            dirlinks.append(pp)
            continue
        if pp.endswith(".md") and tgt.startswith("docs/") and not _published(root, tgt):
            crawl404.append(f"{pp} (target {tgt} is excluded from publishing — 404 on the live site)")
    defects = [f"crawl-404: {c}" for c in sorted(crawl404)]
    soft = [f"directory link (no canonical published page): {d}/" for d in sorted(dirlinks)]
    score = 100 - 25 * len(crawl404)
    detail = ("every local link is crawlable on the published site" if not crawl404
              else f"{len(crawl404)} link(s) resolve on disk but 404 on the live site")
    return {"kpi": "links_crawlable", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


def kpi_answerability(text: str) -> dict[str, Any]:
    """AEO: the first screen should be a self-contained, plain-language answer —
    the span an answer engine lifts and quotes. A page that opens with only
    scaffolding (headings, tables, code) gives the engine nothing to cite. Voice
    is a judgment call, so this KPI emits NO hard defects — only soft nudges."""
    soft: list[str] = []
    score = 100
    if not _has_prose_opener(text):
        soft.append("no plain-language opening sentence before the first heading/code/table")
        score -= 25
    # An entity-defining opener ("X is a …") is the single most-quoted AEO pattern.
    body = _strip_front_matter(text)
    head = "\n".join(body.splitlines()[:25])
    if not re.search(r"\b(is|are)\b\s+\w", head):
        soft.append("first screen states no definition ('X is a …') for an answer engine to quote")
        score -= 10
    return {"kpi": "answerability", "score": _clamp(score),
            "detail": ("first screen is quotable prose" if score >= 90
                       else "first screen is thin for answer-engine quoting"),
            "defects": [], "soft": soft}


def kpi_alt_text(text: str) -> dict[str, Any]:
    """Image SEO + accessibility: every image — a markdown ![alt](src) or a raw
    <img> tag — must carry non-empty alt text. Empty alt is invisible to image
    search and silent to a screen reader / answer engine, so it is a HARD defect; a
    lone generic-filler caption ('image', 'diagram') is a SOFT nudge. A page with NO
    images scores a clean 100 (nothing to caption). Inline + fenced code is stripped
    first so a syntax example (`![](x.svg)`) is never counted as a real image."""
    body = _INLINE_CODE_RE.sub(" ", _FENCE_RE.sub(" ", text))
    defects: list[str] = []
    soft: list[str] = []
    n_img = 0

    def _check(alt: str, label: str, where: str) -> None:
        if not alt.strip():
            defects.append(f"{label} has no alt text: {where[:60]}")
        elif _degenerate_alt(alt):
            soft.append(f"{label} alt is generic filler ('{alt.strip()}'): {where[:50]}")

    for m in _MD_IMG_RE.finditer(body):
        n_img += 1
        _check(m.group("alt"), "image", m.group("src").strip())
    for m in _MD_REF_IMG_RE.finditer(body):
        n_img += 1
        _check(m.group("alt"), "image", m.group(0).strip())
    for m in _HTML_IMG_RE.finditer(body):
        n_img += 1
        a = _HTML_ALT_RE.search(m.group(0))
        _check(a.group(1) if a else "", "<img>", m.group(0))

    score = 100 - 25 * len(defects) - 8 * len(soft)
    detail = ("no images" if not n_img
              else f"{n_img} image(s), all captioned" if not defects and not soft
              else f"{len(defects)} missing + {len(soft)} weak alt of {n_img} image(s)")
    return {"kpi": "alt_text", "score": _clamp(score), "detail": detail,
            "defects": defects, "soft": soft}


# ---------------------------------------------------------------------------
# Small pure helpers
# ---------------------------------------------------------------------------

def _strip_front_matter(text: str) -> str:
    if not text.startswith("---"):
        return text
    end = text.find("\n---", 3)
    return text[end + 4:] if end != -1 else text


def _has_prose_opener(text: str) -> bool:
    body = _strip_front_matter(text)
    seen_h1 = False
    for raw in body.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("# "):
            seen_h1 = True
            continue
        if not seen_h1:
            # some pages open with prose before any heading — accept that too
            if re.search(r"[A-Za-z]", line) and not line.startswith(("#", "```", "|", "<", "!", "-", "*", ">")):
                if len(line) > 40 or line.endswith((".", ":", "!")):
                    return True
            continue
        if line.startswith(("#", "```", "|")):
            return False
        if line.startswith(("<", "!", "-", "*", ">", "1.")):
            continue
        if re.search(r"[A-Za-z]", line) and (len(line) > 40 or line.endswith((".", ":", "!"))):
            return True
    return False


# ---------------------------------------------------------------------------
# Per-page fold
# ---------------------------------------------------------------------------

def grade_letter(score: float) -> str:
    if score >= 90:
        return "A"
    if score >= 80:
        return "B"
    if score >= 70:
        return "C"
    if score >= 60:
        return "D"
    return "F"


def score_page(text: str, doc_rel: str, root: Path) -> dict[str, Any]:
    fm = parse_front_matter(text)
    kpis = [
        kpi_title(fm),
        kpi_description(fm),
        kpi_headings(text),
        kpi_links(text, root, doc_rel),
        kpi_links_crawlable(text, root, doc_rel),
        kpi_answerability(text),
        kpi_alt_text(text),
    ]
    by_name = {k["kpi"]: k for k in kpis}
    score = sum(KPI_WEIGHTS[name] * by_name[name]["score"] for name in KPI_WEIGHTS)
    defects = [f"{k['kpi']}: {d}" for k in kpis for d in k["defects"]]
    soft = [f"{k['kpi']}: {s}" for k in kpis for s in k["soft"]]
    return {
        "path": doc_rel,
        "score": round(score, 1),
        "grade": grade_letter(score),
        "kpis": {k["kpi"]: k["score"] for k in kpis},
        "kpi_detail": {k["kpi"]: k["detail"] for k in kpis},
        # raw meta strings the corpus-level meta_distinct pass dedups across pages.
        "meta": {"title": fm.get("title", "").strip(),
                 "description": fm.get("description", "").strip()},
        "defects": defects,
        "soft": soft,
        "n_defects": len(defects),
    }


def missing_page_entry(doc_rel: str) -> dict[str, Any]:
    return {
        "path": doc_rel, "score": 0.0, "grade": "F",
        "kpis": {k: 0 for k in KPI_WEIGHTS}, "kpi_detail": {},
        "meta": {"title": "", "description": ""},
        "defects": [f"missing: core page {doc_rel} does not exist on disk"],
        "soft": [], "n_defects": 1,
    }


def apply_corpus_meta_distinct(pages: list[dict[str, Any]]) -> int:
    """Cross-page SUCCESS check, folded AFTER per-page scoring: every page's title
    AND description must be UNIQUE across the scored corpus. Duplicate `<title>` /
    meta-description is the canonical "present but unsuccessful" defect a per-page
    check is STRUCTURALLY blind to — it scores each page in isolation, so two pages
    that ship the identical `<title>` each score a clean 100, yet search dedups
    them (it can't tell them apart, picks one, drops the rest). Each side of a
    duplicate gets one HARD defect, so it folds into seo_debt_in_pages. Returns the
    number of duplicate-defects added."""
    added = 0
    for field in ("title", "description"):
        groups: dict[str, list[dict[str, Any]]] = {}
        for d in pages:
            v = (d.get("meta") or {}).get(field, "").strip().lower()
            if v:
                groups.setdefault(v, []).append(d)
        for group in groups.values():
            if len(group) < 2:
                continue
            paths = [g["path"] for g in group]
            for d in group:
                peers = [p for p in paths if p != d["path"]]
                shown = ", ".join(peers[:3]) + ("…" if len(peers) > 3 else "")
                d["defects"].append(
                    f"meta_distinct: {field} is not unique — duplicates {shown} "
                    "(search can't tell the pages apart)")
                d["n_defects"] = len(d["defects"])
                added += 1
    return added


def citation_link_audit(root: Path) -> dict[str, Any]:
    """AEO citation integrity (the SUCCESS twin of the `llms_txt` presence check):
    every link an answer engine FOLLOWS from the curated surfaces must resolve to
    a LIVE target. Three reads, all offline:

      dead_map   — a link in the curated llms.txt map that no longer resolves.
      dead_self  — a self-referential github.com/.../fak blob|tree|raw link,
                   anywhere in the published corpus, that points at a moved/renamed
                   path (the per-page `links` KPI skips ALL http(s), so it is blind
                   to self-repo link rot — yet the path is in-repo and checkable).
      llms_full_unresolved — inlined links in llms-full.txt that don't resolve in
                   the flat one-fetch corpus. This is a SOFT advisory, not a gate:
                   the fix lives in gen_llms_full.py (rewrite inlined relative
                   links to absolute), and a forced regen here would sweep a peer's
                   uncommitted docs into a scoped commit on the shared trunk."""
    def _local_dead(text: str, base: Path) -> list[str]:
        out: list[str] = []
        for m in _LINK_RE.finditer(text):
            u = m.group("target").strip()
            if u.startswith(("http://", "https://", "mailto:", "#", "tel:")):
                continue
            pp = u.split("#", 1)[0].split("?", 1)[0].strip()
            if not pp:
                continue
            res = (root / pp.lstrip("/")) if pp.startswith("/") else (base / pp)
            if not res.exists():
                out.append(pp)
        return out

    dead_map = sorted(set(_local_dead(_safe_read(root / LLMS_REL), root)))

    dead_self: set[str] = set()
    for rel in enumerate_pages(root, "published"):
        txt = _FENCE_RE.sub(" ", _safe_read(root / rel))
        for m in SELF_REPO_RE.finditer(txt):
            p = m.group(1).rstrip("/")
            if not (root / p).exists():
                dead_self.add(f"{rel} -> {p}")

    llms_full = _safe_read(root / LLMS_FULL_REL)
    llms_full_unresolved = len(_local_dead(llms_full, root))

    return {"dead_map": dead_map, "dead_self": sorted(dead_self),
            "llms_full_unresolved": llms_full_unresolved}


# ---------------------------------------------------------------------------
# Site-level SEO/AEO infrastructure checks (each returns hard defects + soft).
# ---------------------------------------------------------------------------

def _robots_groups(robots: str) -> dict[str, list[str]]:
    """Map each `User-agent:` to its directive lines. A record is a run of one or
    more consecutive User-agent lines followed by directives; consecutive UA lines
    share the directives that follow (the robots.txt grouping rule). Comments and
    blank lines are ignored. Tiny on purpose — it only needs Allow/Disallow per UA."""
    groups: dict[str, list[str]] = {}
    pending: list[str] = []
    after_directive = False
    for raw in robots.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            # A blank line ENDS a record (robots.txt grouping rule): clear the
            # pending UA group so a trailing global directive after the blank
            # (`Disallow: /`, `Sitemap:`) can't bleed into the last named UA and
            # flip ai_crawlers to a false block.
            pending = []
            after_directive = False
            continue
        m = re.match(r"(?i)^user-agent:\s*(\S+)", line)
        if m:
            if after_directive:       # a UA line after a directive starts a new record
                pending = []
                after_directive = False
            ua = m.group(1)
            pending.append(ua)
            groups.setdefault(ua, [])
            continue
        after_directive = True
        for ua in pending:
            groups[ua].append(line)
    return groups


def ai_crawlers_ok(robots: str) -> tuple[bool, str]:
    """AEO: an answer-engine-friendly robots.txt must explicitly NAME the major
    answer-engine crawlers (so the intent to be cited survives a wildcard tighten)
    AND must not Disallow any of them. A bare `User-agent: * / Allow: /` allows them
    by default but names none — that is the HARD defect this harden surfaces."""
    if not robots.strip():
        return False, "robots.txt missing — no welcome for answer-engine crawlers"
    groups = _robots_groups(robots)
    named = set(groups)
    # Block-ALL forms: `Disallow: /`, `Disallow: /*`, `Disallow: *` all forbid every
    # path (the `*` wildcard matches everything). A partial path (`Disallow: /private/`,
    # `Disallow: /*.json`) is NOT a full block and must still pass — the bot is welcome.
    disallow_all = re.compile(r"(?i)^disallow:\s*(?:/\*?|\*)\s*$")
    blocked = sorted(
        ua for ua, lines in groups.items()
        if (ua in AI_CRAWLER_UAS or ua == "*") and any(disallow_all.match(line) for line in lines))
    if blocked:
        return False, (f"robots.txt Disallows answer-engine crawler(s): {', '.join(blocked)} "
                       "(delists the project from that engine's citations)")
    missing = [ua for ua in AI_CRAWLER_REQUIRED if ua not in named]
    if missing:
        return False, (f"robots.txt does not explicitly welcome {len(missing)} answer-engine "
                       f"crawler(s): {', '.join(missing)} (name + Allow each for AEO)")
    bonus = sorted((named & AI_CRAWLER_UAS) - set(AI_CRAWLER_REQUIRED))
    extra = f" (+{len(bonus)} more named)" if bonus else ""
    return True, (f"robots.txt explicitly welcomes all {len(AI_CRAWLER_REQUIRED)} major "
                  f"answer-engine crawlers{extra}")


def site_checks(root: Path) -> dict[str, Any]:
    config = _safe_read(root / CONFIG_REL)
    robots = _safe_read(root / ROBOTS_REL)
    head = _safe_read(root / HEAD_INCLUDE_REL)
    index = _safe_read(root / INDEX_REL)
    showcase = _safe_read(root / SHOWCASE_REL)
    llms = _safe_read(root / LLMS_REL)
    faq = _safe_read(root / FAQ_REL)

    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, hard: bool, detail_ok: str, detail_bad: str) -> None:
        checks.append({
            "name": name, "ok": ok,
            "detail": detail_ok if ok else detail_bad,
            "defect": (None if ok else detail_bad),
            "hard": hard,
        })

    # robots.txt: present, allows crawl, names a sitemap.
    robots_ok = bool(robots) and "Allow: /" in robots and "Sitemap:" in robots \
        and "Disallow: /\n" not in robots
    add("robots", robots_ok, True,
        "robots.txt allows crawl + names sitemap",
        "robots.txt missing / blocks crawl / names no Sitemap:")

    # ai_crawlers (AEO): the named answer-engine bots must be explicitly welcomed.
    ai_ok, ai_detail = ai_crawlers_ok(robots)
    add("ai_crawlers", ai_ok, True, ai_detail, ai_detail)

    # jekyll-sitemap + jekyll-seo-tag plugins emit /sitemap.xml + canonical/OG/JSON-LD.
    add("sitemap_plugin", "jekyll-sitemap" in config, True,
        "jekyll-sitemap enabled (auto /sitemap.xml)",
        "jekyll-sitemap not in docs/_config.yml plugins")
    add("seo_tag_plugin", "jekyll-seo-tag" in config, True,
        "jekyll-seo-tag enabled (canonical/OG/Twitter)",
        "jekyll-seo-tag not in docs/_config.yml plugins")

    # canonical: a url + baseurl must be set or canonical/sitemap URLs are wrong.
    canonical_ok = bool(re.search(r"^url:\s*\S", config, re.MULTILINE))
    add("canonical_url", canonical_ok, True,
        "site url set (canonical + absolute sitemap URLs resolve)",
        "no url: in docs/_config.yml (canonical/sitemap URLs break)")

    # og_image: a default social image declared AND the file exists on disk.
    og_match = re.search(r"image:\s*\"?([^\"\n]+)", config)
    og_file_ok = False
    if og_match:
        # an absolute raw URL pointing at a repo path -> check the in-repo file exists
        url = og_match.group(1)
        m = re.search(r"/main/(.+)$", url)
        og_file_ok = bool(m) and (root / m.group(1)).exists()
        if not m:  # a relative path
            og_file_ok = (root / url.lstrip("/")).exists()
    add("og_image", bool(og_match) and og_file_ok, True,
        "Open Graph social image declared and present",
        "no og:image default in _config.yml or the image file is missing")

    # structured_data: scan every surface JSON-LD can live on for @types. FAQPage
    # markup belongs on the FAQ page itself (Google's guideline), so FAQ.md is in
    # the blob alongside the global <head> include + the landing pages. Each block
    # is PARSED with json.loads — a @type is only counted from a block that is
    # valid JSON, because an answer engine / Rich Results rejects malformed JSON-LD.
    # (The review caught the old regex-only scan certifying broken blocks as present.)
    blob = "\n".join([head, index, showcase, faq])
    jsonld_values: list[Any] = []
    present_types: set[str] = set()
    invalid_blocks = 0
    for m in _JSONLD_BLOCK_RE.finditer(blob):
        body = m.group(1).strip()
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            invalid_blocks += 1
            continue
        jsonld_values.append(data)
        present_types |= _collect_jsonld_types(data)
    add("jsonld_valid", invalid_blocks == 0, True,
        "every JSON-LD block parses as valid JSON",
        f"{invalid_blocks} invalid JSON-LD block(s) — answer engines reject malformed JSON-LD")
    for t in JSONLD_TYPES_HARD:
        add(f"jsonld_{t}", t in present_types, True,
            f"JSON-LD {t} present",
            f"no JSON-LD {t} (answer engines can't identify/cite the project)")
    for t in JSONLD_TYPES_SOFT:
        add(f"jsonld_{t}", t in present_types, False,
            f"JSON-LD {t} present (bonus)",
            f"no JSON-LD {t} (optional, a citation bonus)")

    breadcrumb_ok, breadcrumb_detail = breadcrumb_shape_ok(jsonld_values)
    add("breadcrumb_jsonld_shape", breadcrumb_ok, True,
        breadcrumb_detail,
        breadcrumb_detail)

    # llms.txt: present + carries an explicit facts block answer engines anchor on.
    llms_ok = bool(llms) and re.search(r"(?i)key facts", llms) is not None
    add("llms_txt", llms_ok, True,
        "llms.txt present with a Key-facts block",
        "llms.txt missing or has no 'Key facts' anchor block")

    # llms-full.txt: present AND fresh — the one-fetch corpus. Freshness is
    # CONTENT-based, not mtime-based: gen_llms_full.py inlines the whole llms.txt
    # verbatim as its Index section, so a fresh llms-full.txt must CONTAIN the
    # current llms.txt body. Content beats mtime because git does not preserve
    # mtimes across a clone — an mtime check gives false STALE/FRESH in CI.
    llms_full = _safe_read(root / LLMS_FULL_REL)
    if not llms_full:
        add("llms_full", False, True, "", "llms-full.txt missing (no one-fetch corpus; run tools/gen_llms_full.py)")
    else:
        fresh = (not llms.strip()) or (llms.strip() in llms_full)
        add("llms_full", fresh, True,
            "llms-full.txt present and fresh (inlines current llms.txt)",
            "llms-full.txt is STALE (does not contain current llms.txt; re-run tools/gen_llms_full.py)")
    llms_sources = llms_full_source_audit(root)
    missing_sources = llms_sources["missing"]
    add("llms_full_sources", not missing_sources, True,
        f"llms-full.txt includes all {len(llms_sources['targets'])} llms.txt source documents",
        f"llms-full.txt misses {len(missing_sources)} llms.txt source document(s): "
        f"{', '.join(missing_sources[:3])}{'...' if len(missing_sources) > 3 else ''}")

    # faq_structured: FAQ.md present with enough QUESTION-SHAPED H2s to seed a
    # FAQPage. Only headings that read as questions count (end with '?' or lead
    # with how/what/why/…) — a page with six '## Notes'-style H2s is not an FAQ.
    q = sum(1 for h in _H2_RE.findall(faq) if _is_question(h)) if faq else 0
    add("faq_structured", q >= MIN_FAQ_QUESTIONS, True,
        f"FAQ.md has {q} question sections (seeds FAQPage)",
        f"FAQ.md missing or thin ({q} question H2s; need >= {MIN_FAQ_QUESTIONS})")
    faq_sync_ok, faq_sync_detail = faq_jsonld_sync_ok(jsonld_values, faq)
    add("faq_jsonld_sync", faq_sync_ok, True,
        faq_sync_detail,
        faq_sync_detail)

    # citation_links (SUCCESS): every link an answer engine follows from the
    # curated surfaces must resolve to a LIVE target — the llms.txt map AND every
    # self-referential github link in the corpus. One HARD defect if any is dead.
    cit = citation_link_audit(root)
    n_dead = len(cit["dead_map"]) + len(cit["dead_self"])
    add("citation_links", n_dead == 0, True,
        "every llms.txt-map + self-repo github link resolves to a live target",
        f"{n_dead} citation link(s) dead — a stale link sends an answer engine/reader "
        "to a 404 (see corpus.citation)")
    # llms-full.txt navigability (SUCCESS): the flat one-fetch corpus must not
    # contain local links whose original source-document base path was lost during
    # inlining. The generator rewrites inlined local links to absolute repo URLs,
    # so any remaining unresolved local link is a real answer-engine citation gap.
    add("llms_full_navigable", cit["llms_full_unresolved"] == 0, True,
        "llms-full.txt inlined links all resolve in the flat corpus",
        f"{cit['llms_full_unresolved']} inlined link(s) in llms-full.txt don't resolve in the "
        "flat one-fetch corpus (fix = gen_llms_full.py rewrites inlined relative links "
        "to absolute, then regenerate)")

    hard_defects = [c["defect"] for c in checks if not c["ok"] and c["hard"]]
    soft = [c["defect"] for c in checks if not c["ok"] and not c["hard"]]
    n_ok = sum(1 for c in checks if c["ok"])
    score = round(100 * n_ok / max(1, len(checks)), 1)
    return {
        "checks": checks,
        "score": score,
        "n_ok": n_ok,
        "n_total": len(checks),
        "defects": hard_defects,
        "soft": soft,
        "present_jsonld": sorted(present_types),
        "citation": cit,
        "llms_full_sources": llms_sources,
    }


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Published-set enumeration
# ---------------------------------------------------------------------------

@lru_cache(maxsize=8)
def _config_excludes(root_str: str) -> tuple[frozenset[str], frozenset[str]]:
    """Parse docs/_config.yml's `exclude:` list into (dir_rules, file_rules),
    both repo-`docs/`-relative. Reading the live config (instead of trusting only
    the hardcoded NONPUBLISHED_DIRS) keeps the scorecard's published set in sync
    with what Jekyll actually publishes BY CONSTRUCTION — a reviewer re-runs and
    gets the same answer, and a single-file exclude (e.g. an orphaned internal
    snapshot) is honored without a second hand-edited list drifting from the first.
    A trailing-slash entry is a directory rule; a `.md` entry is a file rule.
    Glob/quote/non-doc entries (`*.py`, `*.pdf`) are ignored — those never name a
    published reader page anyway."""
    cfg = _safe_read(Path(root_str) / CONFIG_REL)
    m = re.search(r"^exclude:\s*\n((?:[ \t]+-.*\n?|[ \t]*#.*\n?)+)", cfg, re.MULTILINE)
    dirs: set[str] = set()
    files: set[str] = set()
    if m:
        for line in m.group(1).splitlines():
            line = line.strip()
            if not line.startswith("-"):
                continue
            entry = line[1:].strip().strip('"').strip("'")
            if not entry or "*" in entry:
                continue
            if entry.endswith("/"):
                dirs.add(entry.rstrip("/"))
            elif entry.endswith(".md"):
                files.add(entry)
    return frozenset(dirs), frozenset(files)


def _published(root: Path, rel: str) -> bool:
    """True if Jekyll would publish docs/<…>.md as an indexable HTML page.

    A page is unpublished if any path segment hits a non-published directory
    (the hardcoded NONPUBLISHED_DIRS default OR a `dir/` rule in _config.yml's
    exclude list), or if its docs-relative path is a `*.md` file rule there."""
    p = Path(rel)
    if p.suffix.lower() != ".md":
        return False
    parts = p.parts
    if not parts or parts[0] != "docs":
        return False
    seg = set(parts[1:])
    cfg_dirs, cfg_files = _config_excludes(str(root.resolve()))
    if seg & (NONPUBLISHED_DIRS | cfg_dirs):
        return False
    docs_rel = Path(*parts[1:]).as_posix()
    return docs_rel not in cfg_files


def _discovery(rel: str) -> bool:
    """A published page that is part of the reader/answer-engine DISCOVERY surface
    (not the proofs/benchmarks/notes evidence appendix)."""
    parts = Path(rel).parts
    return _published(Path(), rel) and not (set(parts[1:]) & EVIDENCE_DIRS)


def reachable_published(root: Path) -> set[str]:
    """Link-BFS from the front doors; returns every repo-relative path reached.
    The discovery surface is the published+discovery-class subset of this — a
    reproducible derivation, so the scored set can't be hand-curated."""
    seeds = [(root / s) for s in FRONT_DOORS if (root / s).exists()]
    visited: set[str] = set()
    q: deque[Path] = deque(seeds)
    while q:
        cur = q.popleft()
        try:
            rel = cur.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
        if rel in visited:
            continue
        visited.add(rel)
        try:
            txt = cur.read_text(encoding="utf-8")
        except OSError:
            continue
        for m in _LINK_RE.finditer(txt):
            t = m.group("target").strip()
            if t.startswith(("http://", "https://", "mailto:", "#", "tel:")):
                continue
            pp = t.split("#", 1)[0].split("?", 1)[0]
            if not pp.endswith(".md"):
                continue
            nxt = (root / pp.lstrip("/")) if pp.startswith("/") else (cur.parent / pp)
            if nxt.exists():
                q.append(nxt)
    return visited


def discovery_orphans(root: Path) -> list[str]:
    """Discovery-class published pages NOT reachable from any front door — present
    but undiscoverable. Reported (not silently dropped) so an excluded page can't
    hide: this is the visibility the review demanded for peers like cursor.md."""
    reach = reachable_published(root)
    all_disc = {p.relative_to(root).as_posix()
                for p in (root / "docs").rglob("*.md")
                if _discovery(p.relative_to(root).as_posix())}
    return sorted(all_disc - reach)


def enumerate_pages(root: Path, scope: str) -> list[str]:
    """The page set for a scope:
      core      — the DERIVED discovery surface: published pages link-reachable
                  from the front doors, minus the evidence subtrees. Reproducible.
      published — every Jekyll-published .md (the full indexable tree, incl. the
                  proofs/benchmarks/notes long tail).
    """
    if scope == "core":
        reach = reachable_published(root)
        return sorted(r for r in reach if _discovery(r))
    if scope == "published":
        out: list[str] = []
        for p in sorted((root / "docs").rglob("*.md")):
            rel = p.relative_to(root).as_posix()
            if _published(root, rel):
                out.append(rel)
        return out
    raise ValueError(f"unknown scope {scope!r} (core|published)")


# ---------------------------------------------------------------------------
# Grader: fold per-page scores + site checks into the control-pane payload
# ---------------------------------------------------------------------------

def build_payload(*, workspace: str, pages: list[dict[str, Any]],
                  site: dict[str, Any], scope: str,
                  orphans: list[str] | None = None) -> dict[str, Any]:
    orphans = orphans or []
    n = len(pages)
    scores = [d["score"] for d in pages]
    page_defects = sum(d["n_defects"] for d in pages)
    site_defects = len(site.get("defects", []))
    total_defects = page_defects + site_defects
    mean_score = round(sum(scores) / max(1, n), 1)
    grades = {g: 0 for g in "ABCDF"}
    for d in pages:
        grades[d["grade"]] = grades.get(d["grade"], 0) + 1
    worst = sorted(pages, key=lambda d: (d["score"], -d["n_defects"]))[:8]
    # Meta coverage: fraction of published pages with BOTH a title and a description.
    full_meta = sum(1 for d in pages if d["kpis"].get("title", 0) > 0
                    and d["kpis"].get("description", 0) > 0)
    meta_pct = round(100 * full_meta / max(1, n), 1)
    # The headline 0-100: half the page mean, half the site infrastructure score.
    overall = round(0.5 * mean_score + 0.5 * site["score"], 1)
    # SUCCESS-KPI breakdown (the presence-not-success defects the deepened scorecard
    # surfaces): links that 404 on the live site, non-unique meta, dead citations.
    crawl_404 = sum(1 for d in pages for x in d["defects"] if "links_crawlable: crawl-404" in x)
    meta_duplicates = sum(1 for d in pages for x in d["defects"] if x.startswith("meta_distinct:"))
    citation = site.get("citation") or {}
    citation_dead = len(citation.get("dead_map", [])) + len(citation.get("dead_self", []))

    corpus = {
        "n_pages": n,
        "overall_score": overall,
        # Corpus-level letter on the shared 90/80/70/60 ladder (#1269) so the
        # control pane reads the real grade, not a score-derived inference.
        "grade": grade_letter(overall),
        "page_mean_score": mean_score,
        "site_score": site["score"],
        "site_checks_ok": f"{site['n_ok']}/{site['n_total']}",
        "meta_coverage_pct": meta_pct,
        "median_score": round(sorted(scores)[n // 2], 1) if n else 0.0,
        "min_score": round(min(scores), 1) if scores else 0.0,
        "grade_distribution": grades,
        "seo_debt": total_defects,
        "seo_debt_in_pages": page_defects,
        "seo_debt_in_site": site_defects,
        "crawl_404": crawl_404,
        "meta_duplicates": meta_duplicates,
        "citation_dead": citation_dead,
        "llms_full_unresolved": citation.get("llms_full_unresolved", 0),
        "present_jsonld": site.get("present_jsonld", []),
        "discovery_orphans": len(orphans),
        "orphan_pages": orphans,
        "worst": [{"path": d["path"], "score": d["score"], "grade": d["grade"],
                   "n_defects": d["n_defects"]} for d in worst],
    }

    if total_defects == 0:
        ok, verdict, finding = True, "OK", "discoverable"
        reason = (f"discoverability clean: {n} pages, overall {overall}/100, "
                  f"meta coverage {meta_pct}%, site {site['n_ok']}/{site['n_total']}, zero seo-debt")
        next_action = "no required edit; re-run after the next docs/site change"
    else:
        ok, verdict, finding = False, "ACTION", "seo_debt"
        reason = (f"{total_defects} unit(s) of seo-debt across {n} pages "
                  f"({page_defects} in-page + {site_defects} site); overall {overall}/100, "
                  f"meta coverage {meta_pct}%, site {site['n_ok']}/{site['n_total']}")
        next_action = ("retire seo-debt worst-first (corpus.worst + site defects): add missing "
                       "front-matter title/description, JSON-LD types, llms-full.txt; repoint a "
                       "crawl-404 link to a published page or an absolute URL; dedup a non-unique "
                       "title/description; fix a dead citation link; re-run to prove the drop")

    return {
        "schema": SCHEMA, "ok": ok, "verdict": verdict, "finding": finding,
        "reason": reason, "next_action": next_action, "workspace": workspace,
        "scope": scope, "corpus": corpus, "site": site, "pages": pages,
    }


def collect(workspace: Path, *, scope: str = "core") -> dict[str, Any]:
    root = workspace.resolve()
    rels = enumerate_pages(root, scope)
    pages: list[dict[str, Any]] = []
    for rel in rels:
        path = root / rel
        if not path.exists():
            pages.append(missing_page_entry(rel))
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            d = missing_page_entry(rel)
            d["defects"] = [f"unreadable: {rel}: {exc}"]
            pages.append(d)
            continue
        pages.append(score_page(text, rel, root))
    # Corpus-level SUCCESS pass: dedup title/description across the whole scored
    # set (a per-page KPI can't see another page's meta). Must run BEFORE
    # build_payload folds n_defects into seo-debt.
    apply_corpus_meta_distinct(pages)
    site = site_checks(root)
    orphans = discovery_orphans(root)
    return build_payload(workspace=str(root), pages=pages, site=site, scope=scope,
                         orphans=orphans)


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def render(payload: dict[str, Any]) -> str:
    c = payload.get("corpus") or {}
    site = payload.get("site") or {}
    lines = [
        f"seo-aeo-scorecard: {payload.get('verdict')} ({payload.get('finding')})  [scope={payload.get('scope')}]",
        f"  {payload.get('reason')}",
        "",
        (f"corpus: {c.get('n_pages', 0)} pages · overall {c.get('overall_score', 0)}/100 "
         f"(pages {c.get('page_mean_score', 0)} · site {c.get('site_score', 0)}) "
         f"· SEO-DEBT {c.get('seo_debt', 0)}"),
        (f"meta coverage: {c.get('meta_coverage_pct', 0)}%  ·  "
         f"site checks: {c.get('site_checks_ok', '0/0')}  ·  "
         f"JSON-LD present: {', '.join(c.get('present_jsonld', [])) or 'none'}"),
        (f"discovery orphans (published, not front-door-reachable): {c.get('discovery_orphans', 0)}"),
        (f"success KPIs: crawl-404 {c.get('crawl_404', 0)}  ·  meta-duplicates {c.get('meta_duplicates', 0)}"
         f"  ·  dead citations {c.get('citation_dead', 0)}  ·  llms-full unresolved {c.get('llms_full_unresolved', 0)} (advisory)"),
        ("grades: " + " ".join(f"{g}:{c.get('grade_distribution', {}).get(g, 0)}" for g in "ABCDF")),
        f"next: {payload.get('next_action')}",
        "",
        "per-page (worst first):",
        f"  {'score':>5} {'gr':>2} {'def':>3}  {'ttl':>3} {'dsc':>3} {'hdg':>3} {'lnk':>3} {'crl':>3} {'ans':>3} {'alt':>3}  path",
    ]
    for d in sorted(payload.get("pages", []), key=lambda x: (x["score"], -x["n_defects"])):
        k = d.get("kpis", {})
        lines.append(
            f"  {d['score']:>5} {d['grade']:>2} {d['n_defects']:>3}  "
            f"{k.get('title','-'):>3} {k.get('description','-'):>3} {k.get('headings','-'):>3} "
            f"{k.get('links','-'):>3} {k.get('links_crawlable','-'):>3} "
            f"{k.get('answerability','-'):>3} {k.get('alt_text','-'):>3}  {d['path']}")
    lines.append("")
    lines.append("site-level checks:")
    for ch in site.get("checks", []):
        mark = "ok " if ch["ok"] else ("!! " if ch["hard"] else "~  ")
        lines.append(f"  [{mark}] {ch['name']:20} {ch['detail']}")
    lines.append("")
    lines.append("seo-debt detail (top defectful pages):")
    worst = sorted(payload.get("pages", []), key=lambda x: -x["n_defects"])[:8]
    for d in worst:
        if not d["defects"]:
            continue
        lines.append(f"  {d['path']} ({d['n_defects']}):")
        for it in d["defects"][:6]:
            lines.append(f"      - {it}")
    if site.get("defects"):
        lines.append("  site:")
        for it in site["defects"]:
            lines.append(f"      - {it}")
    return "\n".join(lines)


def render_markdown(payload: dict[str, Any], *, stamp: str | None = None) -> str:
    """The PRIVATE SEO-AEO-SCORECARD.md body — strategic, never committed public."""
    c = payload.get("corpus") or {}
    site = payload.get("site") or {}
    scope = payload.get("scope", "core")
    gd = c.get("grade_distribution", {})
    out: list[str] = []
    out.append("# SEO / AEO scorecard (PRIVATE)")
    out.append("")
    out.append("> **Private — do not publish.** Discoverability scores are go-to-market "
               "positioning (the same strategic class as the ICP memo). The measuring "
               "TOOL is public (`tools/seo_aeo_scorecard.py`); these SCORES live only in "
               "the private repo. The public `.gitignore` blocks this file from a public commit.")
    out.append("")
    if stamp:
        out.append(f"<!-- seo-aeo-scorecard: {stamp} · scope={scope} · process: tools/seo_aeo_scorecard.py -->")
        out.append("")
    out.append(f"> Scope: **{scope}**. Regenerate + transfer: "
               f"`python tools/seo_aeo_scorecard.py --scope {scope} --transfer --stamp DATE`")
    out.append("")
    out.append("Headline metric is **seo-debt**: the count of concrete, re-derivable "
               "discoverability defects (a page with no meta description, a missing "
               "JSON-LD type, a stale `llms-full.txt`, a dead link). Drive it toward zero.")
    out.append("")
    out.append("## Corpus")
    out.append("")
    out.append("| Metric | Value |")
    out.append("|---|---|")
    out.append(f"| Published pages scored | {c.get('n_pages', 0)} |")
    out.append(f"| **SEO-debt (total defects)** | **{c.get('seo_debt', 0)}** "
               f"({c.get('seo_debt_in_pages', 0)} in-page + {c.get('seo_debt_in_site', 0)} site) |")
    out.append(f"| Overall score | {c.get('overall_score', 0)}/100 |")
    out.append(f"| — page mean / site | {c.get('page_mean_score', 0)} / {c.get('site_score', 0)} |")
    out.append(f"| Meta coverage (title+desc) | {c.get('meta_coverage_pct', 0)}% |")
    out.append(f"| Site checks passing | {c.get('site_checks_ok', '0/0')} |")
    out.append(f"| JSON-LD types present | {', '.join(c.get('present_jsonld', [])) or 'none'} |")
    out.append(f"| Discovery orphans (not front-door-reachable) | {c.get('discovery_orphans', 0)} |")
    out.append(f"| Grade distribution | A:{gd.get('A',0)} B:{gd.get('B',0)} C:{gd.get('C',0)} D:{gd.get('D',0)} F:{gd.get('F',0)} |")
    out.append("")
    out.append("## Per-page scores")
    out.append("")
    out.append("| Score | Grade | Debt | title | desc | head | link | crawl | ans | alt | Page |")
    out.append("|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|")
    for d in sorted(payload.get("pages", []), key=lambda x: (x["score"], -x["n_defects"])):
        k = d.get("kpis", {})
        out.append(
            f"| {d['score']} | {d['grade']} | {d['n_defects']} | "
            f"{k.get('title','-')} | {k.get('description','-')} | {k.get('headings','-')} | "
            f"{k.get('links','-')} | {k.get('links_crawlable','-')} | "
            f"{k.get('answerability','-')} | {k.get('alt_text','-')} | `{d['path']}` |")
    out.append("")
    out.append("## Site-level checks")
    out.append("")
    out.append("| State | Check | Detail |")
    out.append("|:--:|---|---|")
    for ch in site.get("checks", []):
        mark = "✅" if ch["ok"] else ("❌" if ch["hard"] else "⚠️")
        out.append(f"| {mark} | `{ch['name']}` | {ch['detail']} |")
    out.append("")
    out.append("## SEO-debt work-list")
    out.append("")
    any_defect = False
    for d in sorted(payload.get("pages", []), key=lambda x: -x["n_defects"]):
        if not d["defects"]:
            continue
        any_defect = True
        out.append(f"### `{d['path']}` — {d['n_defects']} defect(s), score {d['score']} ({d['grade']})")
        for it in d["defects"]:
            out.append(f"- {it}")
        out.append("")
    if site.get("defects"):
        any_defect = True
        out.append("### site")
        for it in site["defects"]:
            out.append(f"- {it}")
        out.append("")
    if not any_defect:
        out.append("No seo-debt: every published page + site check is clean. 🎉")
        out.append("")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Compare (prove the debt moved) + private transfer
# ---------------------------------------------------------------------------

def render_compare(baseline: dict[str, Any], current: dict[str, Any]) -> str:
    b = (baseline.get("corpus") or {})
    cur = (current.get("corpus") or {})
    bd, cd = b.get("seo_debt", 0), cur.get("seo_debt", 0)
    bo, co = b.get("overall_score", 0), cur.get("overall_score", 0)
    ratio = (bd / cd) if cd else float("inf")
    ratio_s = "∞ (zero)" if cd == 0 else f"{ratio:.1f}×"
    lines = [
        f"seo-debt: {bd} -> {cd}   ({ratio_s} fewer defects)",
        f"overall:  {bo}/100 -> {co}/100   (+{round(co - bo, 1)})",
        f"meta cov: {b.get('meta_coverage_pct', 0)}% -> {cur.get('meta_coverage_pct', 0)}%",
        f"site:     {b.get('site_checks_ok','?')} -> {cur.get('site_checks_ok','?')}",
        f"JSON-LD:  {', '.join(b.get('present_jsonld', [])) or 'none'} -> "
        f"{', '.join(cur.get('present_jsonld', [])) or 'none'}",
    ]
    if cd <= max(1, bd) / 10:
        lines.append(f"VERDICT: >=10x debt reduction achieved ({bd} -> {cd}).")
    else:
        need = max(1, bd // 10)
        lines.append(f"VERDICT: not yet 10x — need seo-debt <= {need} (now {cd}).")
    return "\n".join(lines)


def _atomic_write(path: str, data: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".seosync.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, path)
    except OSError:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def transfer(payload: dict[str, Any], private_out: str, *, stamp: str | None,
             rebaseline: bool = False) -> int:
    """Write the report + score JSON to the PRIVATE repo. Git-free, fail-soft.

    The baseline is PINNED on first transfer (the `--compare` reference, the
    "before"); later transfers refresh the report + a `current` snapshot (the
    "after") but never clobber the pinned baseline unless `--rebaseline`. So the
    private archive keeps the before/after record that proves the 10x.

    Returns 0 on write, 2 if the private repo is unreachable (a soft skip, the
    same contract as tools/dos_sync.py).
    """
    out = os.path.abspath(private_out)
    # The private REPO is the grandparent of the default `<repo>/.seo-archive/fak`.
    # Require it to exist (so we never conjure a whole tree where there is no
    # private repo), but DO create the `.seo-archive/fak` subdir within it — the
    # same shape tools/dos_sync.py uses for `../fleet/.dos-archive/fak`.
    private_repo = os.path.dirname(os.path.dirname(out))
    if not os.path.isdir(private_repo):
        print(f"transfer skipped: private repo not found: {private_repo}", file=sys.stderr)
        print("  (clone/point the private repo, or pass --private-out <dir>)", file=sys.stderr)
        return 2
    report_path = os.path.join(out, REPORT_NAME)
    baseline_path = os.path.join(out, BASELINE_NAME)
    current_path = os.path.join(out, CURRENT_NAME)
    payload_json = json.dumps(payload, indent=2) + "\n"
    _atomic_write(report_path, render_markdown(payload, stamp=stamp))
    _atomic_write(current_path, payload_json)
    wrote = [report_path, current_path]
    if rebaseline or not os.path.exists(baseline_path):
        _atomic_write(baseline_path, payload_json)
        wrote.append(baseline_path + ("  (re-pinned)" if rebaseline else "  (pinned)"))
    else:
        print(f"baseline pinned (kept): {baseline_path}  (use --rebaseline to overwrite)")
    print("transferred scores to PRIVATE repo:")
    for w in wrote:
        print(f"  {w}")
    print("  (commit them in the PRIVATE repo:")
    print(f"     git -C {private_repo} add {out} && git -C {private_repo} commit -s -m "
          "'chore(seo): archive SEO/AEO scorecard')")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="SEO/AEO discoverability scorecard (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--scope", default="core", choices=["core", "published"],
                    help="page set to score (default: core)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--markdown", action="store_true", help="emit the SEO-AEO-SCORECARD.md body")
    ap.add_argument("--stamp", default="", help="date stamp for the markdown header")
    ap.add_argument("--compare", default="", metavar="BASELINE.json",
                    help="print the seo-debt delta vs a prior baseline JSON (proves 10x)")
    ap.add_argument("--transfer", action="store_true",
                    help="write report + score JSON to the PRIVATE repo (scores never go public)")
    ap.add_argument("--rebaseline", action="store_true",
                    help="transfer: overwrite the pinned baseline (the --compare reference)")
    ap.add_argument("--private-out", default="",
                    help=f"private archive dir (default: <root>/{DEFAULT_PRIVATE_OUT})")
    args = ap.parse_args(argv)

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    payload = collect(workspace, scope=args.scope)

    if args.compare:
        try:
            baseline = json.loads(Path(args.compare).read_text(encoding="utf-8"))
        except OSError as exc:
            print(f"error: cannot read baseline {args.compare}: {exc}", file=sys.stderr)
            return 1
        print(render_compare(baseline, payload))
        return 0

    if args.transfer:
        private_out = args.private_out or os.path.join(str(workspace), DEFAULT_PRIVATE_OUT)
        rc = transfer(payload, private_out, stamp=args.stamp or None, rebaseline=args.rebaseline)
        # transfer still reports the verdict on stdout for the operator.
        print("")
        print(render(payload))
        return rc if rc == 2 else (0 if payload.get("ok") else 1)

    if args.json:
        print(json.dumps(payload, indent=2))
    elif args.markdown:
        print(render_markdown(payload, stamp=args.stamp or None))
    else:
        print(render(payload))

    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
