#!/usr/bin/env python3
r"""Daily idea-scout — surface RELATED ideas from arXiv + GitHub and file them as
issues, deduped and capped. The research-to-backlog arm of the fleet loop.

The issue-dispatch loop (docs/dispatch-loop.md) RESOLVES the open backlog; nothing
FEEDS it. This tool is the feeder: once a day it searches the outside world for
work adjacent to what fak is — an agent kernel that adjudicates tool calls (a
default-deny capability gate) and reuses cross-turn setup work (a KV/prefix-cache
gate) — and turns the genuinely-new, genuinely-relevant hits into GitHub issues a
human can triage. Sources, both keyless-or-already-authed (no new secret):

  * arXiv      the Atom export API (http://export.arxiv.org/api/query) — no key.
  * GitHub     `gh search repos` on the SAME authed CLI the dispatch loop uses.

The hard part of an UNATTENDED issue filer is not fetching — it is NOT spamming.
Three dedup rungs gate every candidate before it can become an issue:

  1. seen-cache   .idea-scout/seen.json — a persistent {source_id: record} of every
                  candidate ever FILED (or explicitly skipped). A source filed once
                  is never filed again, even years later. This is the durable rung.
  2. issue-body   the candidate's source URL / source_id stamped in any existing
                  issue body ⇒ already filed (survives a lost cache).
  3. title-near   token-overlap (Jaccard) with any existing issue title ⇒ a near-dup
                  a human already opened by hand.

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

SCHEMA = "fleet-idea-scout/1"
CACHE_DIRNAME = ".idea-scout"
CACHE_FILENAME = "seen.json"
SCOUT_LABEL = "idea-scout"
ARXIV_API = "http://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

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
        "terms": ["prompt injection", "indirect", "jailbreak", "guardrail",
                  "defense", "tool", "agent", "untrusted", "quarantine"],
        "area": "security",
    },
    {
        "key": "tool-call-adjudication",
        "arxiv": '(abs:"tool use" OR abs:"function calling") AND '
                 '(abs:safety OR abs:permission OR abs:capability OR abs:policy)',
        "github": "agent tool security",
        "terms": ["tool call", "function calling", "capability", "permission",
                  "policy", "adjudicat", "default-deny", "sandbox", "syscall"],
        "area": "trust-floor",
    },
    {
        "key": "agent-gateway-serving",
        "arxiv": '(abs:LLM OR abs:agent) AND (abs:gateway OR abs:proxy OR '
                 'abs:serving OR abs:router)',
        "github": "llm gateway proxy",
        "terms": ["gateway", "proxy", "serving", "router", "openai", "api",
                  "multi-agent", "shared cache", "audit"],
        "area": "agentic-serving",
    },
    {
        "key": "kv-prefix-cache-reuse",
        "arxiv": '(abs:"KV cache" OR abs:"prefix cache" OR abs:"prompt cache") AND '
                 '(abs:reuse OR abs:sharing OR abs:inference)',
        "github": "llm kv cache",
        "terms": ["kv cache", "prefix cache", "prompt cache", "reuse", "radix",
                  "paged", "sharing", "turn", "prefill", "speculative"],
        "area": "prompt-caching",
    },
    {
        "key": "mcp-security",
        "arxiv": 'abs:"model context protocol" OR (abs:agent AND abs:"tool '
                 'poisoning")',
        "github": "MCP security",
        "terms": ["model context protocol", "mcp", "tool poisoning", "server",
                  "manifest", "untrusted", "supply chain"],
        "area": "mcp",
    },
    {
        "key": "agent-model-arch",
        "arxiv": '(abs:agent OR abs:"tool use") AND (abs:"function calling" OR '
                 'abs:fine-tuning OR abs:training) AND ti:LLM',
        "github": "function calling agent",
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
    "github_per_topic": 6,  # GitHub repos fetched per topic
    "min_stars": 25,      # GitHub repos under this many stars are dropped pre-score
    "dup_jaccard": 0.55,  # title token-overlap to call a near-duplicate
    "issue_scan_limit": 800,  # existing issues fetched for the dedup index
}

# Term-hit weights for the transparent relevance score.
W_TITLE_HIT = 10
W_BODY_HIT = 3
W_RECENT_180 = 12
W_RECENT_30 = 22       # additive on top of the 180 bonus → very fresh = +34
STAR_DIVISOR = 100     # +1 per 100 stars …
STAR_CAP = 30          # … capped
W_RECENT_PUSH = 10     # GitHub repo pushed within 90d


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
    pushed = _parse_iso(cand.get("extra", {}).get("pushed_at", ""))
    if pushed is not None and (now - pushed).days <= 90:
        score += W_RECENT_PUSH
        reasons.append("pushed ≤90d")

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
            },
        })
    return out


def existing_issue_index(issues: list[dict[str, Any]],
                         ) -> tuple[set[str], list[set[str]], str]:
    """Build the dedup index from existing issues: every URL/source_id already
    stamped in a body, and the title token-sets for near-dup detection. Returns
    (stamped_strings, title_token_sets, joined_bodies_lower)."""
    stamped: set[str] = set()
    title_sets: list[set[str]] = []
    bodies: list[str] = []
    for iss in issues:
        body = (iss.get("body") or "")
        bodies.append(body.lower())
        title_sets.append(tokenize(iss.get("title") or ""))
        for m in re.findall(r"idea-scout-source:\s*([^\s>]+)", body):
            stamped.add(m.strip())
    return stamped, title_sets, "\n".join(bodies)


def is_duplicate(cand: dict[str, Any], seen: dict[str, Any],
                 stamped: set[str], title_sets: list[set[str]],
                 bodies_joined: str, dup_jaccard: float) -> str | None:
    """Return the dedup rung that fires ('seen-cache' / 'issue-body' / 'title-near'),
    or None if the candidate is genuinely new."""
    sid = cand["source_id"]
    if sid in seen:
        return "seen-cache"
    if sid in stamped:
        return "issue-body"
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
        f"{src}; **needs human triage** — close as `wontfix`/`duplicate` if it is "
        f"not worth pursuing.\n\n"
        f"**Source:** {cand['url']}\n\n"
        + ("\n".join(facts) + "\n\n" if facts else "")
        + f"**Why surfaced** (topic `{topic['key']}`, score {score}): "
        + ("; ".join(reasons) if reasons else "matched topic query")
        + "\n\n"
        + (f"**Summary**\n\n{summary}\n\n" if summary else "")
        + "---\n"
        + "_Triage hint: is this a capability fak should adopt, a threat it should "
        + "defend against, or prior art to cite? If none, close it._\n"
        + f"<!-- idea-scout-source: {cand['source_id']} -->"
    )

    labels = [SCOUT_LABEL, "research"]
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
                ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Score → dedup → threshold → CAP. Returns (issues_to_file, skip_stats).
    Deterministic: candidates are de-duplicated by source_id within the run and
    sorted by (score desc, source_id) before the cap so the plan is stable."""
    stats = {"seen-cache": 0, "issue-body": 0, "title-near": 0,
             "below-min": 0, "within-run-dup": 0}
    scored: list[dict[str, Any]] = []
    run_seen: set[str] = set()
    for cand in candidates:
        sid = cand["source_id"]
        if sid in run_seen:
            stats["within-run-dup"] += 1
            continue
        run_seen.add(sid)
        topic = topics_by_key.get(cand["topic"], {"key": cand["topic"], "terms": []})
        rung = is_duplicate(cand, seen, stamped, title_sets, bodies_joined,
                            cfg["dup_jaccard"])
        if rung:
            stats[rung] += 1
            continue
        score, reasons = score_candidate(cand, topic, cfg, now)
        if score < cfg["min_score"]:
            stats["below-min"] += 1
            continue
        scored.append(render_issue(cand, score, reasons, topic, today))

    scored.sort(key=lambda r: (-r["score"], r["source_id"]))
    return scored[: cfg["max_issues"]], stats


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


def gh_json(args: list[str], timeout: int = 60) -> Any:
    """Run a `gh` subcommand that emits JSON; return the parsed value. Raises
    RuntimeError on a non-zero exit, or subprocess.TimeoutExpired if `gh` hangs
    past `timeout` — both are caught by the caller so a stuck CLI can't wedge the
    daily run (the same defensive bound fetch_arxiv has)."""
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8", timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"gh {' '.join(args)} -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    out = proc.stdout.strip()
    return json.loads(out) if out else []


def fetch_github(query: str, limit: int) -> list[dict[str, Any]]:
    return gh_json([
        "search", "repos", query, "--limit", str(limit), "--sort", "stars",
        "--json", "fullName,description,url,stargazersCount,pushedAt,updatedAt,"
        "createdAt,language",
    ])


def fetch_existing_issues(limit: int) -> list[dict[str, Any]]:
    return gh_json([
        "issue", "list", "--state", "all", "--limit", str(limit),
        "--json", "number,title,body",
    ])


def ensure_scout_label() -> None:
    """Idempotently create the marker label so `gh issue create` never fails on a
    missing label. `--force` updates if it already exists. Best-effort, but NOT
    silent: a real failure here (auth/permission) would otherwise resurface as a
    confusing per-issue 'label not found', so warn loudly to stderr."""
    try:
        proc = subprocess.run(
            ["gh", "label", "create", SCOUT_LABEL, "--color", "8a63d2",
             "--description",
             "Auto-filed by the daily idea-scout (tools/idea_scout.py); "
             "needs human triage",
             "--force"],
            capture_output=True, text=True, encoding="utf-8", timeout=30)
    except (OSError, subprocess.TimeoutExpired) as e:
        print(f"warning: could not run `gh label create {SCOUT_LABEL}`: {e}",
              file=sys.stderr)
        return
    if proc.returncode != 0:
        print(f"warning: could not ensure '{SCOUT_LABEL}' label "
              f"(issue creation may fail): {proc.stderr.strip()[:200]}",
              file=sys.stderr)


def create_issue(issue: dict[str, Any]) -> str:
    """`gh issue create` → the new issue URL."""
    args = ["issue", "create", "--title", issue["title"], "--body", issue["body"]]
    for lab in issue["labels"]:
        args += ["--label", lab]
    proc = subprocess.run(["gh", *args], capture_output=True, text=True,
                          encoding="utf-8")
    if proc.returncode != 0:
        raise RuntimeError(f"gh issue create -> {proc.returncode}: "
                           f"{proc.stderr.strip()[:300]}")
    return proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""


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
        for k, v in (raw.get("thresholds") or {}).items():
            if k in cfg:
                cfg[k] = v
    # Validate up front so a malformed --config fails clean (exit 2) instead of
    # silently scoring every candidate 0 (missing `terms`) or KeyError-ing at
    # render time (missing `key`). A topic must name itself, carry relevance
    # terms, and query at least one source.
    for i, t in enumerate(topics):
        if not isinstance(t, dict) or not t.get("key"):
            raise ValueError(f"topic[{i}] missing non-empty 'key'")
        if not isinstance(t.get("terms"), list) or not t["terms"]:
            raise ValueError(f"topic '{t.get('key', i)}' missing non-empty 'terms' list")
        if not t.get("arxiv") and not t.get("github"):
            raise ValueError(f"topic '{t['key']}' must set 'arxiv' and/or 'github'")
    return topics, cfg


# ============================================================================
# Driver.
# ============================================================================
def gather_candidates(topics: list[dict[str, Any]], cfg: dict[str, Any],
                      errors: list[str]) -> list[dict[str, Any]]:
    """Fetch + parse every topic from both sources. A failing source/topic is
    logged to `errors` and skipped — one dead query never sinks the run."""
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
                         if int(it.get("stargazersCount", 0) or 0) >= cfg["min_stars"]]
                cands += parse_github_repos(items, key)
            except Exception as e:  # noqa: BLE001
                errors.append(f"github[{key}]: {e}")
    return cands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Daily idea-scout: arXiv + GitHub → deduped, capped GitHub issues.")
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
    topics_by_key = {t.get("key", f"t{i}"): t for i, t in enumerate(topics)}

    errors: list[str] = []
    candidates = gather_candidates(topics, cfg, errors)

    # The existing-issue dedup index is mandatory: with no way to know what is
    # already filed, --live could re-storm the tracker. If gh fails AND there is
    # no cache, refuse rather than risk a spam run.
    seen = load_seen(workspace)
    try:
        issues = fetch_existing_issues(cfg["issue_scan_limit"])
        stamped, title_sets, bodies_joined = existing_issue_index(issues)
    except Exception as e:  # noqa: BLE001
        errors.append(f"issues: {e}")
        if not seen:
            print(f"refuse: cannot fetch existing issues and no seen-cache to fall "
                  f"back on ({e})", file=sys.stderr)
            return 2
        stamped, title_sets, bodies_joined = set(), [], ""

    if not candidates and errors:
        print("refuse: every source failed:\n  " + "\n  ".join(errors),
              file=sys.stderr)
        return 2

    to_file, skip_stats = plan_issues(
        candidates, topics_by_key, seen, stamped, title_sets, bodies_joined,
        cfg, today, now)

    filed: list[dict[str, Any]] = []
    if args.live and to_file:
        ensure_scout_label()
        for issue in to_file:
            try:
                url = create_issue(issue)
            except Exception as e:  # noqa: BLE001
                errors.append(f"create[{issue['source_id']}]: {e}")
                continue
            seen[issue["source_id"]] = {
                "filed_at": today, "issue_url": url, "score": issue["score"],
                "topic": issue["topic"]}
            filed.append({**issue, "issue_url": url})
        if filed:
            save_seen(workspace, seen)

    result = {
        "schema": SCHEMA, "date": today, "mode": "live" if args.live else "dry-run",
        "candidates_gathered": len(candidates),
        "skipped": skip_stats,
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
          f"{len(topics)} topics × (arXiv + GitHub)")
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
