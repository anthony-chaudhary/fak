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
       JSON-LD structured data (SoftwareApplication, FAQPage, WebSite), an
       `llms.txt` + `llms-full.txt` corpus, a self-contained "what is X" answer
       on the first screen, an FAQ in real question/answer structure.

Like the docs scorecard, the headline metric is an integer you drive toward zero:
**seo-debt** — the count of concrete, re-derivable discoverability defects (a
published page with no meta description, a missing JSON-LD type, a stale
`llms-full.txt`, a dead link, a missing social card). "Make us more discoverable"
was an unfalsifiable claim; seo-debt is the number that makes it checkable, so an
improvement program can state an honest target ("cut seo-debt 10×, from N to
N/10") instead of a vibe.

Two layers fold into one payload:

  PER-PAGE (each published Pages surface, 0-100, weighted into a score + A-F):
    title         front-matter `title:` present, a sane title-tag length
    description   front-matter `description:` present, in the 70-160 char band
    headings      exactly one H1, no skipped heading level, real sections
    links         every local link target resolves on disk (crawl integrity)
    answerability the first screen is a self-contained plain-language answer
                  (the chunk an answer engine quotes), not bare scaffolding

  SITE-LEVEL (the once-per-corpus infrastructure, each a unit of debt if absent):
    robots_ok · sitemap_plugin · seo_tag_plugin · canonical_url · og_image
    structured_data (one defect per missing expected JSON-LD @type)
    llms_txt · llms_full (present AND fresh vs llms.txt) · faq_structured

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
from pathlib import Path
from typing import Any

SCHEMA = "fak-seo-aeo-scorecard/1"

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
# identity + search action) are required; Organization/BreadcrumbList are a bonus.
JSONLD_TYPES_HARD = ["SoftwareApplication", "FAQPage", "WebSite"]
JSONLD_TYPES_SOFT = ["Organization", "BreadcrumbList"]

# Per-KPI weights for the per-page score. description + title weigh most — they ARE
# the search result (the blue link + the snippet) and the first thing an engine
# reads. answerability is real but a judgment call, so it weighs least.
KPI_WEIGHTS: dict[str, float] = {
    "title": 0.25,
    "description": 0.30,
    "headings": 0.15,
    "links": 0.20,
    "answerability": 0.10,
}

# Title-tag and meta-description length laws (chars). Outside the band is a SOFT
# nudge (it still renders, just sub-optimally); ABSENT is the HARD defect.
TITLE_MIN, TITLE_MAX = 15, 70
DESC_MIN, DESC_MAX = 70, 160

MIN_FAQ_QUESTIONS = 6  # below this, FAQ.md is too thin to seed a FAQPage

# Dirs under docs/ that Jekyll does NOT publish as reader pages (mirrors the
# _config.yml `exclude` list + Jekyll's own `_`-prefixed special dirs). A page in
# one of these is not an indexable HTML surface, so it carries no SEO meta duty.
NONPUBLISHED_DIRS = {
    "benchmark", "benchmarking", "planning", "testing", "releases",
    "stable-releases", "_includes", "_layouts", "_data", "_site",
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
    body = _strip_front_matter(text)
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
        kpi_answerability(text),
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
        "defects": defects,
        "soft": soft,
        "n_defects": len(defects),
    }


def missing_page_entry(doc_rel: str) -> dict[str, Any]:
    return {
        "path": doc_rel, "score": 0.0, "grade": "F",
        "kpis": {k: 0 for k in KPI_WEIGHTS}, "kpi_detail": {},
        "defects": [f"missing: core page {doc_rel} does not exist on disk"],
        "soft": [], "n_defects": 1,
    }


# ---------------------------------------------------------------------------
# Site-level SEO/AEO infrastructure checks (each returns hard defects + soft).
# ---------------------------------------------------------------------------

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
    present_types: set[str] = set()
    invalid_blocks = 0
    for m in _JSONLD_BLOCK_RE.finditer(blob):
        body = m.group(1).strip()
        try:
            data = json.loads(body)
        except (ValueError, TypeError):
            invalid_blocks += 1
            continue
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

    # faq_structured: FAQ.md present with enough QUESTION-SHAPED H2s to seed a
    # FAQPage. Only headings that read as questions count (end with '?' or lead
    # with how/what/why/…) — a page with six '## Notes'-style H2s is not an FAQ.
    q = sum(1 for h in _H2_RE.findall(faq) if _is_question(h)) if faq else 0
    add("faq_structured", q >= MIN_FAQ_QUESTIONS, True,
        f"FAQ.md has {q} question sections (seeds FAQPage)",
        f"FAQ.md missing or thin ({q} question H2s; need >= {MIN_FAQ_QUESTIONS})")

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
    }


def _safe_read(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# Published-set enumeration
# ---------------------------------------------------------------------------

def _published(root: Path, rel: str) -> bool:
    """True if Jekyll would publish docs/<…>.md as an indexable HTML page."""
    p = Path(rel)
    if p.suffix.lower() != ".md":
        return False
    parts = p.parts
    if not parts or parts[0] != "docs":
        return False
    return not (set(parts[1:]) & NONPUBLISHED_DIRS)


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

    corpus = {
        "n_pages": n,
        "overall_score": overall,
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
                       "front-matter title/description, JSON-LD types, llms-full.txt; re-run to prove the drop")

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
        ("grades: " + " ".join(f"{g}:{c.get('grade_distribution', {}).get(g, 0)}" for g in "ABCDF")),
        f"next: {payload.get('next_action')}",
        "",
        "per-page (worst first):",
        f"  {'score':>5} {'gr':>2} {'def':>3}  {'ttl':>3} {'dsc':>3} {'hdg':>3} {'lnk':>3} {'ans':>3}  path",
    ]
    for d in sorted(payload.get("pages", []), key=lambda x: (x["score"], -x["n_defects"])):
        k = d.get("kpis", {})
        lines.append(
            f"  {d['score']:>5} {d['grade']:>2} {d['n_defects']:>3}  "
            f"{k.get('title','-'):>3} {k.get('description','-'):>3} {k.get('headings','-'):>3} "
            f"{k.get('links','-'):>3} {k.get('answerability','-'):>3}  {d['path']}")
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
    out.append("| Score | Grade | Debt | title | desc | head | link | ans | Page |")
    out.append("|---:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|")
    for d in sorted(payload.get("pages", []), key=lambda x: (x["score"], -x["n_defects"])):
        k = d.get("kpis", {})
        out.append(
            f"| {d['score']} | {d['grade']} | {d['n_defects']} | "
            f"{k.get('title','-')} | {k.get('description','-')} | {k.get('headings','-')} | "
            f"{k.get('links','-')} | {k.get('answerability','-')} | `{d['path']}` |")
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
