#!/usr/bin/env python3
r"""Daily idea-scout — surface RELATED ideas from arXiv, GitHub, Hacker News and
Reddit and file them as issues, deduped and capped. The research-to-backlog arm of
the fleet loop.

The issue-dispatch loop (docs/dispatch-loop.md) RESOLVES the open backlog; nothing
FEEDS it. This tool is the feeder: once a day it searches the outside world for
work adjacent to what fak is — an agent kernel that adjudicates tool calls (a
default-deny capability gate) and reuses cross-turn setup work (a KV/prefix-cache
gate) — and turns the genuinely-new, genuinely-relevant hits into GitHub issues a
human can triage. Sources, both keyless-or-already-authed (no new secret):

  * arXiv      the Atom export API (http://export.arxiv.org/api/query) — no key.
  * GitHub     `gh search repos` on the SAME authed CLI the dispatch loop uses,
               walked on two lanes from the same topic query: a STARS lane
               (all-time popular, floored at min_stars) and a FRESH lane
               (fresh_per_topic repos sorted most-recently-updated, floored at the
               lower fresh_min_stars) so newly-created / trending / recently-pushed
               repos surface instead of only incumbents. fresh_per_topic: 0 disables it.
  * Hacker News  Algolia's HN search API (https://hn.algolia.com/api/v1) — no key.
  * Reddit     the public search JSON (https://www.reddit.com/search.json) — no key.

The HN and Reddit lanes are points-scored rather than star-scored: a post under
`min_points` is dropped pre-score, and `points` earns the same shape of bonus
`stars` does. Both lanes long existed ONLY in the Go port (internal/ideascout),
so a topic naming `hn`/`reddit` here gathered nothing and still reported success
(#5549). The lane vocabulary is now declared once in SOURCE_LANES, a topic key no
lane reads is REFUSED rather than ignored, and
internal/ideascout/testdata/source_corpus.json pins the vocabulary and the parsers
against the Go implementation so the two cannot drift apart again in silence.

The hard part of an UNATTENDED issue filer is not fetching — it is NOT spamming.
Four dedup rungs gate every candidate before it can become an issue:

  1. seen-cache   .idea-scout/seen.json — a node-local {source_id: record} of every
                  candidate this machine FILED. A pure fast path: the cache is
                  git-ignored, so it can be lost, and losing it must not (and no
                  longer does) cost the guarantee. Rung 2 is what makes the promise.
  2. filed-stamp  the candidate's source_id read back out of the
                  `<!-- idea-scout-source: … -->` stamp on EVERY issue the scout has
                  ever filed. THE DURABLE RUNG: a source filed once is never filed
                  again, even years later. The index is built from a query TARGETED
                  at the `idea-scout` label (`gh issue list --label idea-scout`), not
                  from a fixed-size window of recent issues — so its completeness is
                  a function of how many issues the SCOUT has filed (capped at
                  --max-issues/day) and never of how fast the tracker as a whole
                  grows. GitHub is the replicated store; nothing local is trusted.
  3. issue-body   the candidate's source URL appears verbatim in some existing issue
                  body ⇒ a human already opened it by hand. Best-effort: scanned over
                  a recent-issue window (`issue_scan_limit`), so it is a bonus catch,
                  never the guarantee.
  4. title-near   token-overlap (Jaccard) with any existing issue title ⇒ a near-dup
                  a human already opened by hand. Best-effort, same window as rung 3.

Rung 2 is MANDATORY: if its index cannot be built — `gh` failed, or the label scan
came back saturated at `scout_scan_limit` and may therefore be truncated — the run
REFUSES (exit 2) instead of filing blind. A dedup index that silently degrades as
the tracker grows is exactly how already-triaged sources get re-filed, so growth has
to trip a loud refusal, not a quiet re-file.

And a hard CAP: at most --max-issues per run (default 3), top-scored first, so even
a pathological day cannot storm the tracker. Relevance is a TRANSPARENT integer
score (title/abstract term hits + recency + GitHub stars), surfaced on every
candidate so the ranking is auditable, never a black box — the same discipline as
tools/issue_triage.py.

SAFE BY DEFAULT: dry-run. The tool prints exactly the issues it WOULD file and
mutates nothing (not even the seen-cache). `--live` is the explicit opt-in that
actually creates issues via `gh issue create` and records them in the cache — the
same dry-run-first contract as the dispatch tools.

    python tools/idea_scout.py                  # dry-run: plan the issues, file nothing
    python tools/idea_scout.py --json           # machine-readable plan
    python tools/idea_scout.py --max-issues 3 --live   # file at most 3, record them
    python tools/idea_scout.py --config tools/idea_scout_topics.example.json

Exit codes: 0 = ran clean · 2 = infra error (gh missing / not authed / not a repo /
network down with no cache to fall back on).
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except (AttributeError, ValueError):
    pass

_CREATE_NO_WINDOW = 0x08000000


def _win_creationflags() -> int:
    return _CREATE_NO_WINDOW if os.name == "nt" else 0


SCHEMA = "fleet-idea-scout/1"
CACHE_DIRNAME = ".idea-scout"
CACHE_FILENAME = "seen.json"
SCOUT_LABEL = "idea-scout"
TRIAGE_LABEL = "needs-triage"
TRIAGE_ONLY_LABEL = "triage-only"
ARXIV_API = "http://export.arxiv.org/api/query"
HN_ALGOLIA_API = "https://hn.algolia.com/api/v1/search_by_date"
REDDIT_SEARCH_API = "https://www.reddit.com/search.json"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

# ---- The source vocabulary (#5549) -------------------------------------------
# One declaration of every gathering lane: the label it stamps on per-lane fetch
# errors, the topic-config key that arms it, and the human name the run report
# prints. Two lanes may share a topic key — `github` and `github-fresh` run the
# same query on different sorts.
#
# Three things read this, so a lane cannot be half-added: load_config REFUSES a
# topic key that is not here, the run report names the lanes it walked, and
# internal/ideascout/testdata/source_corpus.json pins the list against the Go
# scout's `sourceLanes`. Before #5549 the two implementations disagreed here and
# nothing said so: `hn` and `reddit` existed only in Go, and a topic naming them
# on THIS path — the scheduled one (tools/register_idea_scout.ps1) — gathered
# zero candidates and reported a normal success.
SOURCE_LANES: list[dict[str, str]] = [
    {"label": "arxiv", "topic_key": "arxiv", "display": "arXiv"},
    {"label": "github", "topic_key": "github", "display": "GitHub"},
    {"label": "github-fresh", "topic_key": "github", "display": "GitHub"},
    {"label": "hn", "topic_key": "hn", "display": "Hacker News"},
    {"label": "reddit", "topic_key": "reddit", "display": "Reddit"},
]

# Topic-config keys that name no source lane: the topic's identity, its relevance
# terms, and the area label filed issues hang under.
TOPIC_META_KEYS: list[str] = ["key", "terms", "area"]


def source_topic_keys() -> list[str]:
    """Topic-config keys that arm a lane, in declaration order, de-duplicated
    (`github` arms two lanes)."""
    out: list[str] = []
    for lane in SOURCE_LANES:
        if lane["topic_key"] and lane["topic_key"] not in out:
            out.append(lane["topic_key"])
    return out


def source_display_list() -> str:
    """The lane vocabulary as the run report says it: "arXiv + GitHub + Hacker
    News + Reddit". Derived, not spelled out, so the report cannot claim a lane
    the gatherer does not walk."""
    out: list[str] = []
    for lane in SOURCE_LANES:
        if lane["display"] not in out:
            out.append(lane["display"])
    return " + ".join(out)


def source_label(source: str) -> str:
    """Display name for a candidate's `source` field."""
    return {"arxiv": "arXiv", "github": "GitHub", "hackernews": "Hacker News",
            "reddit": "Reddit"}.get(source, source)

# ---- Topics (baked-in defaults; override the whole set via --config) ----------
# Each topic maps fak's domain onto a concrete arXiv query, a GitHub repo query,
# the relevance terms that earn score, and the GitHub area-label to hang the issue
# under. arXiv `arxiv` strings use the API query language (all:/ti:/abs: + boolean
# AND/OR); they are URL-encoded at fetch time. `area` MUST be an existing repo
# label or it is dropped (a non-existent label would make `gh issue create` fail).
DEFAULT_TOPICS: list[dict[str, Any]] = [
    {
        "key": "prompt-injection-defense",
        "arxiv": 'abs:"prompt injection" AND (abs:agent OR abs:LLM OR abs:tool)',
        # GitHub repo search ANDs every term, so a long query matches ~nothing —
        # keep it 2-3 high-signal words and let score+min_stars+dedup narrow it.
        "github": "prompt injection defense",
        "hn": "prompt injection",
        "reddit": "prompt injection agent",
        "terms": ["prompt injection", "indirect", "jailbreak", "guardrail",
                  "defense", "tool", "agent", "untrusted", "quarantine"],
        "area": "security",
    },
    {
        "key": "tool-call-adjudication",
        "arxiv": '(abs:"tool use" OR abs:"function calling") AND '
                 '(abs:safety OR abs:permission OR abs:capability OR abs:policy)',
        "github": "agent tool security",
        "hn": "agent tool permissions",
        "reddit": "agent tool sandbox permission",
        "terms": ["tool call", "function calling", "capability", "permission",
                  "policy", "adjudicat", "default-deny", "sandbox", "syscall"],
        "area": "trust-floor",
    },
    {
        "key": "agent-gateway-serving",
        "arxiv": '(abs:LLM OR abs:agent) AND (abs:gateway OR abs:proxy OR '
                 'abs:serving OR abs:router)',
        "github": "llm gateway proxy",
        "hn": "llm gateway",
        "reddit": "llm gateway proxy router",
        "terms": ["gateway", "proxy", "serving", "router", "openai", "api",
                  "multi-agent", "shared cache", "audit"],
        "area": "agentic-serving",
    },
    {
        "key": "kv-prefix-cache-reuse",
        "arxiv": '(abs:"KV cache" OR abs:"prefix cache" OR abs:"prompt cache") AND '
                 '(abs:reuse OR abs:sharing OR abs:inference)',
        "github": "llm kv cache",
        "hn": "prompt caching",
        "reddit": "kv cache prompt caching inference",
        "terms": ["kv cache", "prefix cache", "prompt cache", "reuse", "radix",
                  "paged", "sharing", "turn", "prefill", "speculative"],
        "area": "prompt-caching",
    },
    {
        "key": "mcp-security",
        "arxiv": 'abs:"model context protocol" OR (abs:agent AND abs:"tool '
                 'poisoning")',
        "github": "MCP security",
        "hn": "model context protocol",
        "reddit": "model context protocol mcp",
        "terms": ["model context protocol", "mcp", "tool poisoning", "server",
                  "manifest", "untrusted", "supply chain"],
        "area": "mcp",
    },
    {
        "key": "agent-model-arch",
        "arxiv": '(abs:agent OR abs:"tool use") AND (abs:"function calling" OR '
                 'abs:fine-tuning OR abs:training) AND ti:LLM',
        "github": "function calling agent",
        "hn": "open source llm agent",
        "reddit": "local llm function calling agent",
        "terms": ["function calling", "tool use", "fine-tun", "training",
                  "checkpoint", "qwen", "llama", "reasoning"],
        "area": "model-arch",
    },
]

# ---- Scoring + dedup thresholds (override via flags) -------------------------
DEFAULTS = {
    "recent_days": 180,   # arXiv submitted within this → recency bonus
    "min_score": 25,      # a candidate below this is not worth an issue
    "max_issues": 3,      # hard cap on issues filed per run (anti-storm)
    "arxiv_per_topic": 8,  # arXiv results fetched per topic
    "github_per_topic": 6,  # GitHub repos fetched per topic (stars lane)
    "hn_per_topic": 8,    # Hacker News stories fetched per topic
    "reddit_per_topic": 8,  # Reddit posts fetched per topic
    "min_stars": 25,      # stars-lane repos under this many stars are dropped pre-score
    # Coarse GitHub KiB proxy: rejects tiny scaffolds, but can drop dense small
    # repos and admit large hollow ones; it is not a quality signal.
    "min_repo_size_kb": 500,
    "min_points": 10,     # HN/Reddit posts under this many points are dropped pre-score
    "fresh_per_topic": 6,  # recency-sorted GitHub repos fetched per topic (0 disables the fresh lane)
    "fresh_min_stars": 3,  # fresh-lane star floor: admits young repos the min_stars floor would drop
    "fresh_window_days": 45,  # pushed within this window earns the strong "actively updated" bonus
    "dup_jaccard": 0.55,  # title token-overlap to call a near-duplicate
    # Rung 3/4 only (URL-in-body + title-Jaccard vs HUMAN-opened issues): a recency
    # window over the whole tracker. Deliberately NOT the anti-re-file guarantee —
    # that is scout_scan_limit below, because this one shrinks in coverage every time
    # the tracker gets busier.
    "issue_scan_limit": 800,  # recent issues fetched for the soft near-dup rungs
    # Rung 2, the durable one: every issue carrying the `idea-scout` LABEL, i.e. the
    # scout's own filing history. Bounded by max_issues/day (≤3), not by tracker
    # growth, so one number covers years. Saturating it is a REFUSAL, never a
    # silent truncation — see the scout-index gate in main().
    "scout_scan_limit": 5000,  # cap on the label-targeted filed-issue index
    "milestone": "",      # assign filed issues to this milestone title (empty = none)
    "project": "",        # ProjectsV2 number to add filed issues to (empty = none)
    "project_owner": "",  # owner login for --project (empty = repo owner / viewer)
}

# Term-hit weights for the transparent relevance score.
W_TITLE_HIT = 10
W_BODY_HIT = 3
W_RECENT_180 = 12
W_RECENT_30 = 22       # additive on top of the 180 bonus → very fresh = +34
STAR_DIVISOR = 100     # +1 per 100 stars …
STAR_CAP = 30          # … capped
HN_POINT_DIV = 20      # +1 per 20 HN/Reddit points …
HN_POINT_CAP = 25      # … capped
W_RECENT_PUSH = 10     # GitHub repo pushed within 90d
W_FRESH_PUSH = 15      # … or +15 if pushed within fresh_window_days (actively updated)
TRENDING_CAP = 20      # cap on the star-velocity (stars/day) trending bonus


# ============================================================================
# Pure helpers (no I/O) — unit-tested directly.
# ============================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> set[str]:
    """Lowercase alnum tokens of length ≥ 3 (drops 'the', 'a', punctuation)."""
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) >= 3}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / len(a | b)


def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_iso(s: str) -> dt.datetime | None:
    """Parse an ISO-8601 stamp (arXiv '…Z', GitHub '…Z') to aware UTC."""
    if not s:
        return None
    try:
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def score_candidate(cand: dict[str, Any], topic: dict[str, Any],
                    cfg: dict[str, Any], now: dt.datetime | None = None,
                    ) -> tuple[int, list[str]]:
    """Transparent integer relevance score + the reasons that earned it.

    Term hits in the title weigh more than the abstract; fresh arXiv papers and
    well-starred / recently-pushed repos earn bonuses. Returned reasons make the
    ranking auditable in the plan output."""
    now = now or _now_utc()
    title = (cand.get("title") or "").lower()
    body = (cand.get("summary") or "").lower()
    score = 0
    reasons: list[str] = []

    hit_terms: list[str] = []
    for term in topic.get("terms", []):
        t = term.lower()
        if t in title:
            score += W_TITLE_HIT
            hit_terms.append(term + "(title)")
        elif t in body:
            score += W_BODY_HIT
            hit_terms.append(term)
    if hit_terms:
        reasons.append("terms: " + ", ".join(hit_terms))

    published = _parse_iso(cand.get("published", ""))
    if published is not None:
        age = (now - published).days
        if 0 <= age <= cfg["recent_days"]:
            score += W_RECENT_180
            reasons.append(f"recent ({age}d)")
            if age <= 30:
                score += W_RECENT_30
                reasons.append("very fresh (≤30d)")

    stars = int(cand.get("extra", {}).get("stars", 0) or 0)
    if stars:
        bonus = min(stars // STAR_DIVISOR, STAR_CAP)
        if bonus:
            score += bonus
            reasons.append(f"{stars} stars (+{bonus})")
    # HN/Reddit carry `points` where GitHub carries `stars`: the crowd's own
    # signal on the same 0..cap scale, so a heavily-upvoted post can clear
    # min_score the way a well-starred repo does.
    points = int(cand.get("extra", {}).get("points", 0) or 0)
    if points:
        bonus = min(points // HN_POINT_DIV, HN_POINT_CAP)
        if bonus:
            score += bonus
            reasons.append(f"{points} points (+{bonus})")
    pushed = _parse_iso(cand.get("extra", {}).get("pushed_at", ""))
    if pushed is not None:
        days = (now - pushed).days
        window = cfg.get("fresh_window_days", 45) or 45
        if 0 <= days <= window:
            score += W_FRESH_PUSH
            reasons.append(f"pushed ≤{window}d (actively updated)")
        elif days <= 90:
            score += W_RECENT_PUSH
            reasons.append("pushed ≤90d")

    # Trending: a young repo already gathering stars (high stars/day) is on the
    # rise; an old repo with the same stars accrued them slowly and scores ~0.
    if stars:
        created = _parse_iso(cand.get("published", ""))
        if created is not None:
            age_days = max((now - created).days, 1)
            raw_vel = stars // age_days
            if raw_vel > 0:
                bonus = min(raw_vel, TRENDING_CAP)
                score += bonus
                reasons.append(f"trending ({raw_vel}★/day, +{bonus})")

    return score, reasons


def parse_arxiv_atom(xml_text: str, topic_key: str) -> list[dict[str, Any]]:
    """Parse an arXiv Atom feed into candidate dicts. Tolerant of a malformed
    feed (returns [] rather than raising) so one bad topic can't sink the run."""
    out: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id = (entry.findtext("atom:id", default="", namespaces=ATOM_NS) or "").strip()
        if not raw_id:
            continue
        # http://arxiv.org/abs/2401.12345v2 → 2401.12345 (strip scheme + version)
        abs_id = raw_id.rsplit("/", 1)[-1]
        abs_id = re.sub(r"v\d+$", "", abs_id)
        title = " ".join((entry.findtext("atom:title", default="",
                                          namespaces=ATOM_NS) or "").split())
        summary = " ".join((entry.findtext("atom:summary", default="",
                                            namespaces=ATOM_NS) or "").split())
        published = (entry.findtext("atom:published", default="",
                                    namespaces=ATOM_NS) or "").strip()
        authors = [a.findtext("atom:name", default="", namespaces=ATOM_NS)
                   for a in entry.findall("atom:author", ATOM_NS)]
        authors = [a for a in authors if a]
        out.append({
            "source": "arxiv",
            "source_id": f"arxiv:{abs_id}",
            "url": f"https://arxiv.org/abs/{abs_id}",
            "title": title,
            "summary": summary,
            "published": published,
            "topic": topic_key,
            "extra": {"authors": authors[:6]},
        })
    return out


def parse_github_repos(items: list[dict[str, Any]], topic_key: str,
                       ) -> list[dict[str, Any]]:
    """Map `gh search repos --json …` items to candidate dicts."""
    out: list[dict[str, Any]] = []
    for it in items or []:
        full = it.get("fullName") or it.get("name") or ""
        if not full:
            continue
        url = it.get("url") or f"https://github.com/{full}"
        # GitHub repo names are CASE-INSENSITIVE (github.com/Acme/Repo ==
        # github.com/acme/repo), but the API can return either casing run-to-run.
        # Lower-case the dedup key so the same repo can't slip the seen-cache /
        # body-stamp rungs on a casing flip; the display URL keeps its casing.
        out.append({
            "source": "github",
            "source_id": f"github:{full.lower()}",
            "url": url,
            "title": full,
            "summary": it.get("description") or "",
            "published": it.get("createdAt", ""),
            "topic": topic_key,
            "extra": {
                "stars": it.get("stargazersCount", 0),
                "pushed_at": it.get("pushedAt", "") or it.get("updatedAt", ""),
                "language": it.get("language", "") or "",
                "size": it.get("size", 0),
            },
        })
    return out


_HTML_TAG_RE = re.compile(r"<[^>]*>")


def _squash(s: str) -> str:
    """Collapse every run of whitespace to one space (Go: `squashSpace`)."""
    return " ".join((s or "").split())


def _strip_tags(s: str) -> str:
    """Remove the light HTML (<p>, <a>, <i>) the HN API leaves in `story_text`
    so the summary is plain prose (Go: `stripTags`)."""
    return _HTML_TAG_RE.sub(" ", s or "")


def _first_non_blank(*values: str) -> str:
    for v in values:
        if (v or "").strip():
            return v
    return ""


def parse_hackernews_json(json_text: str, topic_key: str) -> list[dict[str, Any]]:
    """Turn an Algolia HN search response into candidate dicts.

    A pure fold over the wire JSON: no network, no clock, so it replays from a
    fixture. Link stories keep their outbound URL; text/self posts fall back to
    the HN item permalink so the candidate always resolves to something a triager
    can open. Mirrors internal/ideascout/parse.go `ParseHackerNewsJSON` — the two
    are pinned against the same fixture bytes by
    internal/ideascout/testdata/source_corpus.json."""
    out: list[dict[str, Any]] = []
    try:
        doc = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(doc, dict):
        return out
    for hit in doc.get("hits") or []:
        if not isinstance(hit, dict):
            continue
        hid = str(hit.get("objectID") or "").strip()
        if not hid:
            continue
        title = _squash(_first_non_blank(hit.get("title") or "",
                                         hit.get("story_title") or ""))
        if not title:
            continue
        permalink = f"https://news.ycombinator.com/item?id={hid}"
        url = _first_non_blank(hit.get("url") or "", hit.get("story_url") or "",
                               permalink)
        out.append({
            "source": "hackernews",
            "source_id": f"hn:{hid}",
            "url": url,
            "title": title,
            "summary": _squash(_strip_tags(hit.get("story_text") or "")),
            "published": (hit.get("created_at") or "").strip(),
            "topic": topic_key,
            "extra": {
                "points": int(hit.get("points", 0) or 0),
                "num_comments": int(hit.get("num_comments", 0) or 0),
                "discussion": permalink,
                "author": hit.get("author", "") or "",
            },
        })
    return out


def parse_reddit_json(json_text: str, topic_key: str) -> list[dict[str, Any]]:
    """Turn a Reddit listing/search response into candidate dicts.

    Like the other parsers, a pure fold over the wire JSON. Reddit stamps posts
    with a Unix `created_utc` float rather than an ISO string, so it is converted
    to RFC3339 here (a deterministic transform, no wall clock) to match the shared
    freshness path. Self/text posts carry the permalink in `url`; link posts carry
    the outbound target, and the permalink is always kept as the discussion link.
    Mirrors internal/ideascout/parse.go `ParseRedditJSON`."""
    out: list[dict[str, Any]] = []
    try:
        doc = json.loads(json_text)
    except (json.JSONDecodeError, TypeError):
        return out
    if not isinstance(doc, dict):
        return out
    children = ((doc.get("data") or {}).get("children")
                if isinstance(doc.get("data"), dict) else None) or []
    for child in children:
        if not isinstance(child, dict):
            continue
        h = child.get("data") or {}
        if not isinstance(h, dict):
            continue
        pid = str(h.get("id") or "").strip()
        if not pid:
            continue
        title = _squash(h.get("title") or "")
        if not title:
            continue
        permalink = f"https://www.reddit.com{h['permalink']}" if h.get("permalink") else ""
        url = _first_non_blank(h.get("url") or "", permalink)
        if not url:
            url = f"https://www.reddit.com/comments/{pid}"
        created = float(h.get("created_utc", 0) or 0)
        published = (dt.datetime.fromtimestamp(int(created), dt.timezone.utc)
                     .strftime("%Y-%m-%dT%H:%M:%SZ")) if created > 0 else ""
        out.append({
            "source": "reddit",
            "source_id": f"reddit:{pid}",
            "url": url,
            "title": title,
            "summary": _squash(_strip_tags(h.get("selftext") or "")),
            "published": published,
            "topic": topic_key,
            "extra": {
                "points": int(h.get("score", 0) or 0),
                "num_comments": int(h.get("num_comments", 0) or 0),
                "discussion": _first_non_blank(permalink, url),
                "subreddit": h.get("subreddit", "") or "",
            },
        })
    return out


_STAMP_RE = re.compile(r"idea-scout-source:\s*([^\s>]+)")


def stamp_index(issues: list[dict[str, Any]]) -> set[str]:
    """Every `idea-scout-source:` stamp carried by `issues`, lower-cased.

    This is rung 2's whole payload. Case is folded on BOTH sides (here and in
    ``is_duplicate``) because GitHub repo names are case-insensitive while the
    search API hands back whichever casing it feels like — an un-folded compare
    lets `Acme/Repo` slip past a stamp that reads `acme/repo`."""
    out: set[str] = set()
    for iss in issues:
        for m in _STAMP_RE.findall(iss.get("body") or ""):
            out.add(m.strip().lower())
    return out


def existing_issue_index(issues: list[dict[str, Any]],
                         ) -> tuple[set[str], list[set[str]], str]:
    """Build the dedup index from existing issues: every source_id already
    stamped in a body, and the title token-sets for near-dup detection. Returns
    (stamped_source_ids, title_token_sets, joined_bodies_lower)."""
    title_sets: list[set[str]] = []
    bodies: list[str] = []
    for iss in issues:
        bodies.append((iss.get("body") or "").lower())
        title_sets.append(tokenize(iss.get("title") or ""))
    return stamp_index(issues), title_sets, "\n".join(bodies)


def is_duplicate(cand: dict[str, Any], seen: dict[str, Any],
                 stamped: set[str], title_sets: list[set[str]],
                 bodies_joined: str, dup_jaccard: float) -> str | None:
    """Return the dedup rung that fires ('seen-cache' / 'filed-stamp' /
    'issue-body' / 'title-near'), or None if the candidate is genuinely new.

    'filed-stamp' and 'issue-body' are reported separately on purpose: the first
    is the durable, complete filing history (rung 2) and the second is a
    best-effort URL sighting inside a recency window (rung 3). Collapsing them
    would make a windowed guess indistinguishable from the guarantee in the
    run report."""
    sid = cand["source_id"]
    sid_l = sid.lower()
    if sid in seen or sid_l in seen:
        return "seen-cache"
    if sid_l in stamped:
        return "filed-stamp"
    url = cand["url"].lower()
    if url and url in bodies_joined:
        return "issue-body"
    ctoks = tokenize(cand["title"])
    for tset in title_sets:
        if jaccard(ctoks, tset) >= dup_jaccard:
            return "title-near"
    return None


def render_issue(cand: dict[str, Any], score: int, reasons: list[str],
                 topic: dict[str, Any], today: str) -> dict[str, Any]:
    """Build the {title, body, labels} an issue is created from. The source_id
    stamp in an HTML comment is the load-bearing dedup anchor (rung 2)."""
    src = cand["source"]
    raw_title = cand["title"].strip().rstrip(".")
    if len(raw_title) > 100:
        raw_title = raw_title[:97].rstrip() + "…"
    title = f"idea-scout: {raw_title}"

    summary = cand.get("summary", "").strip()
    if len(summary) > 700:
        summary = summary[:697].rstrip() + "…"
    extra = cand.get("extra", {})
    facts = []
    if src == "arxiv":
        if extra.get("authors"):
            facts.append("**Authors:** " + ", ".join(extra["authors"]))
        if cand.get("published"):
            facts.append(f"**Submitted:** {cand['published'][:10]}")
    elif src in ("hackernews", "reddit"):
        if extra.get("subreddit"):
            facts.append(f"**Subreddit:** r/{extra['subreddit']}")
        if extra.get("points"):
            facts.append(f"**Points:** {extra['points']}")
        if extra.get("num_comments"):
            facts.append(f"**Comments:** {extra['num_comments']}")
        if extra.get("discussion"):
            facts.append(f"**Discussion:** {extra['discussion']}")
        if cand.get("published"):
            facts.append(f"**Posted:** {cand['published'][:10]}")
    else:  # github
        if extra.get("stars"):
            facts.append(f"**Stars:** {extra['stars']}")
        if extra.get("language"):
            facts.append(f"**Language:** {extra['language']}")
        if extra.get("pushed_at"):
            facts.append(f"**Last push:** {extra['pushed_at'][:10]}")

    body = (
        f"> Auto-filed by the daily **idea-scout** "
        f"(`tools/idea_scout.py`, {today}). A candidate RELATED idea found on "
        f"{source_label(src)}; **needs human triage** — close as `wontfix`/`duplicate` if it is "
        f"not worth pursuing.\n\n"
        f"**Source:** {cand['url']}\n\n"
        + ("\n".join(facts) + "\n\n" if facts else "")
        + f"**Why surfaced** (topic `{topic['key']}`, score {score}): "
        + ("; ".join(reasons) if reasons else "matched topic query")
        + "\n\n"
        "### Dispatchability\n"
        "- dispatchability: `triage_only`\n"
        "- reason: idea-scout candidates need human scope, lane, witness, and "
        "acceptance criteria before they become worker-ready leaves.\n\n"
        + (f"**Summary**\n\n{summary}\n\n" if summary else "")
        + "---\n"
        + "_Triage hint: is this a capability fak should adopt, a threat it should "
        + "defend against, or prior art to cite? If none, close it._\n"
        + f"<!-- idea-scout-source: {cand['source_id']} -->"
    )

    labels = [SCOUT_LABEL, TRIAGE_LABEL, TRIAGE_ONLY_LABEL, "research"]
    area = topic.get("area")
    if area:
        labels.append(area)
    return {"title": title, "body": body, "labels": labels,
            "source_id": cand["source_id"], "url": cand["url"],
            "score": score, "topic": topic["key"]}


def plan_issues(candidates: list[dict[str, Any]], topics_by_key: dict[str, dict],
                seen: dict[str, Any], stamped: set[str],
                title_sets: list[set[str]], bodies_joined: str,
                cfg: dict[str, Any], today: str, now: dt.datetime,
                ) -> tuple[list[dict[str, Any]], dict[str, int],
                           list[dict[str, str]]]:
    """Score → dedup → threshold → CAP. Returns (issues_to_file, skip_stats,
    dropped). `dropped` names the rung that stopped each individual source_id —
    aggregate counts alone cannot answer "was THIS already-triaged source caught,
    and by which rung", which is the only question that matters after a re-file.
    Deterministic: candidates are de-duplicated by source_id within the run and
    sorted by (score desc, source_id) before the cap so the plan is stable."""
    stats = {"seen-cache": 0, "filed-stamp": 0, "issue-body": 0, "title-near": 0,
             "below-min": 0, "within-run-dup": 0}
    dropped: list[dict[str, str]] = []
    scored: list[dict[str, Any]] = []
    run_seen: set[str] = set()
    for cand in candidates:
        sid = cand["source_id"]
        if sid in run_seen:
            stats["within-run-dup"] += 1
            dropped.append({"source_id": sid, "rung": "within-run-dup"})
            continue
        run_seen.add(sid)
        topic = topics_by_key.get(cand["topic"], {"key": cand["topic"], "terms": []})
        rung = is_duplicate(cand, seen, stamped, title_sets, bodies_joined,
                            cfg["dup_jaccard"])
        if rung:
            stats[rung] += 1
            dropped.append({"source_id": sid, "rung": rung})
            continue
        score, reasons = score_candidate(cand, topic, cfg, now)
        if score < cfg["min_score"]:
            stats["below-min"] += 1
            dropped.append({"source_id": sid, "rung": "below-min"})
            continue
        scored.append(render_issue(cand, score, reasons, topic, today))

    scored.sort(key=lambda r: (-r["score"], r["source_id"]))
    dropped.sort(key=lambda d: (d["rung"], d["source_id"]))
    return scored[: cfg["max_issues"]], stats, dropped


# ============================================================================
# I/O boundary — network + gh. Thin wrappers so the logic above stays testable.
# ============================================================================
def fetch_arxiv(query: str, max_results: int, timeout: int = 30) -> str:
    params = urllib.parse.urlencode({
        "search_query": query,
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "max_results": str(max_results),
    })
    req = urllib.request.Request(f"{ARXIV_API}?{params}",
                                 headers={"User-Agent": "fak-idea-scout/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https/http arXiv)
        return resp.read().decode("utf-8", "replace")


def fetch_hackernews(query: str, limit: int, timeout: int = 30) -> str:
    """Algolia's HN search API — keyless, like arXiv. Mirrors
    internal/ideascout/fetch.go `LiveFetcher.FetchHackerNews`."""
    params = urllib.parse.urlencode({
        "query": query,
        "tags": "story",
        "hitsPerPage": str(limit),
    })
    req = urllib.request.Request(f"{HN_ALGOLIA_API}?{params}",
                                 headers={"User-Agent": "fak-idea-scout/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https Algolia)
        return resp.read().decode("utf-8", "replace")


def fetch_reddit(query: str, limit: int, timeout: int = 30) -> str:
    """Reddit's public search JSON — keyless. Mirrors
    internal/ideascout/fetch.go `LiveFetcher.FetchReddit`."""
    params = urllib.parse.urlencode({
        "q": query,
        "sort": "new",
        "t": "week",
        "limit": str(limit),
    })
    # Reddit rejects requests without a descriptive, non-default User-Agent.
    req = urllib.request.Request(
        f"{REDDIT_SEARCH_API}?{params}",
        headers={"User-Agent": "fak-idea-scout/1.0 "
                               "(+https://github.com/anthony-chaudhary/fak)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (https Reddit)
        return resp.read().decode("utf-8", "replace")


def gh_json(args: list[str], timeout: int = 60) -> Any:
    """Run a `gh` subcommand that emits JSON; return the parsed value. Raises
    RuntimeError on a non-zero exit, or subprocess.TimeoutExpired if `gh` hangs
    past `timeout` — both are caught by the caller so a stuck CLI can't wedge the
    daily run (the same defensive bound fetch_arxiv has)."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout,
                          creationflags=_win_creationflags())
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_github(query: str, limit: int) -> list[dict[str, Any]]:
    return gh_json([
        "search", "repos", query, "--limit", str(limit), "--sort", "stars",
        "--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,"
        "createdAt,language,size",
    ])


def fetch_github_fresh(query: str, limit: int) -> list[dict[str, Any]]:
    """The recency-first companion to fetch_github: the SAME topic query (so the
    neighborhood stays "relative to ours") sorted by most-recently-updated instead
    of all-time stars, surfacing newly-created / trending / freshly-pushed repos
    the stars sort would bury under incumbents."""
    return gh_json([
        "search", "repos", query, "--limit", str(limit), "--sort", "updated",
        "--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,"
        "createdAt,language,size",
    ])


def fetch_existing_issues(limit: int) -> list[dict[str, Any]]:
    """Rung 3/4 corpus: the `limit` most recent issues, whoever opened them. A
    RECENCY WINDOW — it answers "did a human already write this up lately", and
    that is all it is allowed to answer."""
    return gh_json([
        "issue", "list", "--state", "all", "--limit", str(limit),
        "--json", "number,title,body",
    ])


def fetch_scout_issues(limit: int) -> list[dict[str, Any]]:
    """Rung 2 corpus: every issue the scout has EVER filed, open or closed.

    TARGETED, not windowed. `--label idea-scout` is a server-side filter, so the
    result set is the scout's own filing history — it does not thin out because
    unrelated issues were opened this week, which is precisely how the recency
    window in ``fetch_existing_issues`` lost the guarantee. Every filed issue
    carries the label (``render_issue`` always emits SCOUT_LABEL) and the
    matching `<!-- idea-scout-source: … -->` stamp, so label ⊇ stamped-by-us.

    `--state all` is load-bearing: a source whose issue was triaged and CLOSED is
    the exact case that must not come back."""
    return gh_json([
        "issue", "list", "--state", "all", "--label", SCOUT_LABEL,
        "--limit", str(limit), "--json", "number,title,body",
    ], timeout=180)


def ensure_scout_label() -> None:
    """Idempotently create marker/triage labels so `gh issue create` never fails on
    a missing label. Best-effort, but NOT silent: a real failure here
    (auth/permission) would otherwise resurface as a confusing per-issue 'label
    not found', so warn loudly to stderr."""
    wanted = [
        (SCOUT_LABEL, "8a63d2",
         "Auto-filed by the daily idea-scout (tools/idea_scout.py); needs human triage"),
        (TRIAGE_LABEL, "d4c5f9",
         "Needs human scoping before an agent dispatch can take it"),
        (TRIAGE_ONLY_LABEL, "d4c5f9",
         "Useful issue, but not a worker-ready dispatch leaf"),
    ]
    for name, color, desc in wanted:
        try:
            proc = subprocess.run(
                ["gh", "label", "create", name, "--color", color,
                 "--description", desc, "--force"],
                capture_output=True, text=True, encoding="utf-8", timeout=30,
                creationflags=_win_creationflags())
        except (OSError, subprocess.TimeoutExpired) as e:
            print(f"warning: could not run `gh label create {name}`: {e}",
                  file=sys.stderr)
            continue
        if proc.returncode != 0:
            print(f"warning: could not ensure '{name}' label "
                  f"(issue creation may fail): {proc.stderr.strip()[:200]}",
                  file=sys.stderr)


def create_issue(issue: dict[str, Any], *, milestone: str = "") -> str:
    """`gh issue create` → the new issue URL.

    When ``milestone`` is set, file the issue straight into it so scouted work joins
    the milestone backlog the dispatch fleet selects from (the milestone must already
    exist — gh errors otherwise). Empty leaves the issue milestone-less.
    """
    args = ["issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in issue["labels"]:
        args += ["--label", lab]
    if milestone:
        args += ["--milestone", milestone]
    try:
        proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                              encoding="utf-8", timeout=60,
                              creationflags=_win_creationflags())
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("gh issue create timed out after 60s") from e
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


def add_to_project(url: str, number: str, owner: str = "") -> None:
    """Best-effort: add a freshly-filed issue to a ProjectsV2 board.

    Warn-don't-die (mirrors ``ensure_scout_label``): a board/scope failure must never
    lose the created issue or skip the seen-cache write. Needs the gh ``project`` write
    scope and the integer project number — both operator prerequisites — so this is
    opt-in: an empty ``number`` skips it entirely.
    """
    if not number:
        return
    cmd = ["gh", "project", "item-add", str(number), "--url", url]
    if owner:
        cmd += ["--owner", owner]
    proc = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        print(f"warn: gh project item-add {number} (issue filed but not on the board): "
              f"{proc.stderr.strip()[:200]}", file=sys.stderr)


# ============================================================================
# Cache + config I/O.
# ============================================================================
def cache_path(workspace: Path) -> Path:
    return workspace / CACHE_DIRNAME / CACHE_FILENAME


def load_seen(workspace: Path) -> dict[str, Any]:
    p = cache_path(workspace)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("seen", data) if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_seen(workspace: Path, seen: dict[str, Any]) -> None:
    p = cache_path(workspace)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"schema": SCHEMA, "seen": seen}, indent=2,
                            ensure_ascii=False), encoding="utf-8")


def load_config(path: str | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (topics, cfg). Without --config: baked-in defaults. With it: a JSON
    file {"topics": [...], "thresholds": {...}} overrides either or both."""
    cfg = dict(DEFAULTS)
    topics = [dict(t) for t in DEFAULT_TOPICS]
    if path:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(raw.get("topics"), list) and raw["topics"]:
            topics = raw["topics"]
        supplied = raw.get("thresholds") or {}
        _check_threshold_keys(supplied, cfg)
        for k, v in supplied.items():
            cfg[k] = v
    # Validate up front so a malformed --config fails clean (exit 2) instead of
    # silently scoring every candidate 0 (missing `terms`) or KeyError-ing at
    # render time (missing `key`). A topic must name itself, carry relevance
    # terms, name only keys some lane reads, and query at least one source.
    source_keys = source_topic_keys()
    for i, t in enumerate(topics):
        if not isinstance(t, dict) or not t.get("key"):
            raise ValueError(f"topic[{i}] missing non-empty 'key'")
        if not isinstance(t.get("terms"), list) or not t["terms"]:
            raise ValueError(f"topic '{t.get('key', i)}' missing non-empty 'terms' list")
        _check_topic_keys(t)
        if not any(t.get(k) for k in source_keys):
            raise ValueError(f"topic '{t['key']}' must set at least one source: "
                             + ", ".join(repr(k) for k in source_keys))
    return topics, cfg


def _check_topic_keys(topic: dict[str, Any]) -> None:
    """Refuse a topic key no lane reads (#5549).

    A key that is merely ignored is the defect this repo actually hit: `hn` and
    `reddit` existed only in the Go scout, so a topic naming them on the scheduled
    Python path gathered zero candidates and still exited 0. Nothing said the lane
    was missing. Refusing by name is the loud alternative — a config that names a
    lane the running implementation cannot serve now fails clean (exit 2)."""
    source_keys = source_topic_keys()
    known = set(source_keys) | set(TOPIC_META_KEYS)
    unknown = sorted(k for k in topic if k not in known)
    if not unknown:
        return
    raise ValueError(
        f"topic '{topic.get('key')}' names unknown key(s) "
        + ", ".join(repr(k) for k in unknown)
        + ": no source lane reads them, so they would gather nothing silently. "
        "Known source keys: " + ", ".join(repr(k) for k in source_keys)
        + "; other topic keys: " + ", ".join(repr(k) for k in TOPIC_META_KEYS))


def _check_threshold_keys(supplied: dict[str, Any], cfg: dict[str, Any]) -> None:
    """Refuse a threshold no knob reads, for the same reason `_check_topic_keys`
    refuses an unknown topic key: a setting that appears to take and does not is
    the silent failure. Previously unknown thresholds were dropped by an
    `if k in cfg` filter, so `min_points` against an implementation with no points
    floor read as accepted."""
    unknown = sorted(k for k in supplied if k not in cfg)
    if not unknown:
        return
    raise ValueError(
        "thresholds name unknown key(s) " + ", ".join(repr(k) for k in unknown)
        + ": no knob reads them. Known thresholds: "
        + ", ".join(repr(k) for k in sorted(cfg)))


# ============================================================================
# Driver.
# ============================================================================
def gather_candidates(topics: list[dict[str, Any]], cfg: dict[str, Any],
                      errors: list[str]) -> list[dict[str, Any]]:
    """Fetch + parse every topic across every lane it arms. A failing lane/topic
    is logged to `errors` as `label[topic]: …` and skipped — one dead query never
    sinks the run.

    The lanes are spelled out longhand rather than folded over `SOURCE_LANES` —
    they are not uniform enough to table without obscuring them — so `SOURCE_LANES`
    stays the declared VOCABULARY and this stays the implementation. The two cannot
    drift apart in either direction: `load_config` refuses a topic key no lane
    declares (so a lane here with no vocabulary row is unreachable), and
    internal/ideascout/testdata/source_corpus.json requires every declared key to
    actually admit a candidate (so a row here with no lane body reds)."""
    cands: list[dict[str, Any]] = []
    for topic in topics:
        key = topic.get("key", "?")
        if topic.get("arxiv"):
            try:
                xml = fetch_arxiv(topic["arxiv"], cfg["arxiv_per_topic"])
                cands += parse_arxiv_atom(xml, key)
            except Exception as e:  # noqa: BLE001
                errors.append(f"arxiv[{key}]: {e}")
        if topic.get("github"):
            try:
                items = fetch_github(topic["github"], cfg["github_per_topic"])
                items = [it for it in items
                         if int(it.get("size", 0) or 0) >= cfg["min_repo_size_kb"]
                         and int(it.get("stargazersCount", 0) or 0) >= cfg["min_stars"]]
                cands += parse_github_repos(items, key)
            except Exception as e:  # noqa: BLE001
                errors.append(f"github[{key}]: {e}")
        if topic.get("github") and cfg.get("fresh_per_topic", 0) > 0:
            # The fresh lane: same topic query, sorted most-recently-updated, with a
            # low star floor so young/trending repos the min_stars floor would drop
            # enter the pool. Recency is rewarded in scoring (which has a clock);
            # here we only admit and tag provenance (extra.lane = "fresh").
            try:
                items = fetch_github_fresh(topic["github"], cfg["fresh_per_topic"])
                items = [it for it in items
                         if int(it.get("size", 0) or 0) >= cfg["min_repo_size_kb"]
                         and int(it.get("stargazersCount", 0) or 0) >= cfg["fresh_min_stars"]]
                fresh = parse_github_repos(items, key)
                for c in fresh:
                    c["extra"]["lane"] = "fresh"
                cands += fresh
            except Exception as e:  # noqa: BLE001
                errors.append(f"github-fresh[{key}]: {e}")
        # The points-scored social lanes. Same lane order as the Go gatherer
        # (internal/ideascout/gather.go) so the two walk topics identically.
        if topic.get("hn"):
            _append_points_lane(cands, errors, "hn", key, cfg,
                                lambda t=topic: fetch_hackernews(
                                    t["hn"], cfg["hn_per_topic"]),
                                parse_hackernews_json)
        if topic.get("reddit"):
            _append_points_lane(cands, errors, "reddit", key, cfg,
                                lambda t=topic: fetch_reddit(
                                    t["reddit"], cfg["reddit_per_topic"]),
                                parse_reddit_json)
    return cands


def _append_points_lane(cands: list[dict[str, Any]], errors: list[str],
                        label: str, topic_key: str, cfg: dict[str, Any],
                        fetch: Any, parse: Any) -> None:
    """Run one points-scored social lane (Hacker News, Reddit): fetch, record a
    `label[topic]: …` error and admit nothing on failure, else admit the parsed
    candidates that clear the shared `min_points` floor. The lanes differ only in
    which fetch and which parser they name. Mirrors internal/ideascout/gather.go
    `appendPointsLane`."""
    try:
        raw = fetch()
    except Exception as e:  # noqa: BLE001
        errors.append(f"{label}[{topic_key}]: {e}")
        return
    floor = int(cfg.get("min_points", 0) or 0)
    for c in parse(raw, topic_key):
        if int(c["extra"].get("points", 0) or 0) >= floor:
            cands.append(c)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Daily idea-scout: " + source_display_list()
                    + " → deduped, capped GitHub issues.")
    ap.add_argument("--workspace", default=".",
                    help="repo root (holds .idea-scout/ cache). Default: cwd.")
    ap.add_argument("--config", help="JSON file overriding topics/thresholds.")
    ap.add_argument("--max-issues", type=int,
                    help=f"hard cap on issues filed (default {DEFAULTS['max_issues']}).")
    ap.add_argument("--min-score", type=int,
                    help=f"drop candidates below this (default {DEFAULTS['min_score']}).")
    ap.add_argument("--live", action="store_true",
                    help="actually create issues + record them (default: dry-run).")
    ap.add_argument("--json", action="store_true", help="machine-readable output.")
    ap.add_argument("--milestone", default=None,
                    help="assign filed issues to this milestone title (default: none; "
                         "the milestone must already exist).")
    ap.add_argument("--project", default=None,
                    help="ProjectsV2 number to add filed issues to (default: none; "
                         "needs the gh `project` write scope). Best-effort.")
    ap.add_argument("--project-owner", default=None,
                    help="owner login for --project (default: repo owner / viewer).")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve()
    today = _now_utc().strftime("%Y-%m-%d")
    now = _now_utc()

    try:
        topics, cfg = load_config(args.config)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    if args.max_issues is not None:
        cfg["max_issues"] = args.max_issues
    if args.min_score is not None:
        cfg["min_score"] = args.min_score
    if args.milestone is not None:
        cfg["milestone"] = args.milestone
    if args.project is not None:
        cfg["project"] = args.project
    if args.project_owner is not None:
        cfg["project_owner"] = args.project_owner
    topics_by_key = {t.get("key", f"t{i}"): t for i, t in enumerate(topics)}

    errors: list[str] = []
    candidates = gather_candidates(topics, cfg, errors)

    seen = load_seen(workspace)

    # ---- Rung 2, the durable one --------------------------------------------
    # The scout's OWN filing history, pulled by label so the query is targeted at
    # exactly the population being deduped. This is what makes "filed once, never
    # filed again" true without trusting the git-ignored local cache. It is
    # MANDATORY: if it cannot be built completely we refuse, because a partial
    # index is indistinguishable from "this source is new" and re-files an
    # already-triaged source.
    scout_limit = int(cfg["scout_scan_limit"])
    try:
        scout_issues = fetch_scout_issues(scout_limit)
    except Exception as e:  # noqa: BLE001
        errors.append(f"scout-index: {e}")
        print(f"refuse: cannot build the filed-issue index (`gh issue list --label "
              f"{SCOUT_LABEL}`); filing now could re-file an already-triaged "
              f"source ({e})", file=sys.stderr)
        return 2
    if len(scout_issues) >= scout_limit:
        # Saturation is ambiguous — gh returns exactly `limit` both when that is
        # all there is and when it truncated. Refuse loudly rather than let the
        # guarantee rot silently the way the 800-issue window did.
        print(f"refuse: the filed-issue index came back saturated at "
              f"scout_scan_limit={scout_limit}, so it may be truncated and a "
              f"previously-filed source could be re-filed. Raise "
              f"`thresholds.scout_scan_limit` (--config) above the number of "
              f"issues the scout has ever filed.", file=sys.stderr)
        return 2
    stamped = stamp_index(scout_issues)

    # ---- Rungs 3/4: the soft, windowed corpus of everything else -------------
    # Human-opened issues that reference the same URL, or carry a near-identical
    # title. Nice to have, never the guarantee. The pre-existing refusal is kept
    # as-is (nothing here is relaxed on the back of rung 2): degrading these rungs
    # onto a bare local cache is still a worse run than no run.
    window_scanned = 0
    try:
        issues = fetch_existing_issues(cfg["issue_scan_limit"])
        win_stamped, title_sets, bodies_joined = existing_issue_index(issues)
        stamped |= win_stamped  # catches a filed issue whose label a human stripped
        window_scanned = len(issues)
    except Exception as e:  # noqa: BLE001
        errors.append(f"issues: {e}")
        if not seen:
            print(f"refuse: cannot fetch existing issues and no seen-cache to fall "
                  f"back on ({e})", file=sys.stderr)
            return 2
        title_sets, bodies_joined = [], ""

    if not candidates and errors:
        print("refuse: every source failed:\n  " + "\n  ".join(errors),
              file=sys.stderr)
        return 2

    to_file, skip_stats, dropped = plan_issues(
        candidates, topics_by_key, seen, stamped, title_sets, bodies_joined,
        cfg, today, now)

    filed: list[dict[str, Any]] = []
    if args.live and to_file:
        ensure_scout_label()
        for issue in to_file:
            try:
                url = create_issue(issue, milestone=cfg.get("milestone", ""))
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['source_id']}]: {e}")
                continue
            if cfg.get("project"):
                add_to_project(url, cfg["project"], cfg.get("project_owner", ""))
            seen[issue["source_id"]] = {
                "filed_at": today, "issue_url": url, "score": issue["score"],
                "topic": issue["topic"]}
            filed.append({**issue, "issue_url": url})
        if filed:
            save_seen(workspace, seen)

    result = {
        "schema": SCHEMA, "date": today, "mode": "live" if args.live else "dry-run",
        "candidates_gathered": len(candidates),
        # Make the durable rung's coverage auditable in the run record: a reader
        # can see the filed-issue index was COMPLETE (label-targeted, unsaturated)
        # rather than having to trust that it was.
        "dedup_index": {
            "filed_issues_scanned": len(scout_issues),
            "filed_stamps": len(stamped),
            "scout_scan_limit": scout_limit,
            "scout_index_complete": True,
            "window_issues_scanned": window_scanned,
            "issue_scan_limit": cfg["issue_scan_limit"],
        },
        "skipped": skip_stats,
        # Per-source attribution, not just counts: which rung stopped which
        # source_id. Lets an auditor re-run the dry-run and check by name that a
        # known-already-triaged source is being caught, and by which rung.
        "dropped": dropped,
        "planned": [
            {"title": i["title"], "labels": i["labels"], "url": i["url"],
             "source_id": i["source_id"], "score": i["score"], "topic": i["topic"]}
            for i in to_file],
        "filed": [{"title": f["title"], "issue_url": f.get("issue_url", "")}
                  for f in filed],
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0

    # Human report.
    print(f"idea-scout {today} — {result['mode']}")
    print(f"  gathered {len(candidates)} candidates from "
          f"{len(topics)} topics × ({source_display_list()})")
    print(f"  dedup index: {len(stamped)} source stamps from "
          f"{len(scout_issues)} filed issue(s) (label-targeted, complete) "
          f"+ {window_scanned} recent issue(s) for the near-dup rungs")
    sk = ", ".join(f"{k}={v}" for k, v in skip_stats.items() if v) or "none"
    print(f"  deduped/dropped: {sk}")
    if not to_file:
        print("  → nothing new worth filing today.")
    else:
        verb = "FILED" if args.live else "would file"
        print(f"  → {verb} {len(to_file)} issue(s) (cap {cfg['max_issues']}, "
              f"min-score {cfg['min_score']}):")
        for i in to_file:
            mark = ""
            if args.live:
                f = next((x for x in filed if x["source_id"] == i["source_id"]), None)
                mark = f"  {f['issue_url']}" if f else "  (create failed)"
            print(f"     [{i['score']:>3}] {i['title']}")
            print(f"           {i['url']}  labels={','.join(i['labels'])}{mark}")
    if errors:
        print("  errors:")
        for e in errors:
            print(f"     ! {e}")
    if not args.live and to_file:
        print("\n  dry-run — file these for real with:  "
              "python tools/idea_scout.py --live")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
