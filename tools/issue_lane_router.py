#!/usr/bin/env python3
"""Route OPEN GitHub issues to the dos.toml lane whose file-tree they touch.

The DOS supervisor dispatches by LANE (from dos.toml), and a lane-worker picks
work from the plan portfolio — issues are invisible to it. So a live supervisor
run only resolves tickets that happen to ride along on plan-lane work; it cannot
TARGET the backlog. The closure auditor (`tools/issue_closure_audit.py`) proved
the cost: closure_rate sits near zero because nothing aims the fleet at tickets.

This tool is the missing aim. For each open issue it picks the lane whose `trees`
globs the issue most likely touches, with a confidence ladder
(path-confirmed > exact-scope > alias > label > keyword > none) so the supervisor can
prefer high-confidence routes and the worker can fold its lane's issues into the
dispositions sidecar it already builds. UNROUTED is a first-class, surfaced
output — an issue with no defensible lane is never force-fit (and an exclusive
lane is never auto-handed to a heuristic-driven worker).

Read-only. It never edits, labels, or closes an issue. The dispatch worker (and
the operator) consume the map; writing sidecars / closing tickets stays gated.

Run from the repo ROOT (dos resolves its lane taxonomy from the nearest dos.toml;
from fak/ it scaffolds a throwaway wrong-lane config).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fleet-issue-lane-router/1"

# Mirrored from issue_triage.py (small; triage is not an importable package).
_SCOPE_RE = re.compile(r"\b(\w+)\(([^)]+)\)")  # feat(scope), fix(scope), ...
# Bare `prefix:` start form (e.g. `abi: ...`, `RSI: ...`) — the loose convention
# used for scopeless types. Only the leading token before the first colon.
_BARE_PREFIX_RE = re.compile(r"^([A-Za-z][\w-]*):\s")

# Concrete repo paths an issue body/title may name (the strongest routing signal).
# The root alternation splits on its leading char: word-rooted roots (tools, docs,
# fak/...) keep a `\b` so a partial prefix like `mytools/` never matches; DOT-rooted
# roots (.github, .claude) cannot use `\b` — a `\b` needs a word char on one side, but
# `.github` is non-word-then-word preceded by a backtick/space (both non-word), so the
# boundary is absent and `\b\.github` silently never matched a `.github/...` path in its
# natural `\`.github/...\`` position. A `(?<![\w.])` lookbehind anchors the dotted root
# instead: it matches when preceded by a non-word/non-dot char (start, space, backtick)
# and still rejects an embedded `x.github`. Without this a workflow-only finding (e.g. a
# scheduled `.github/workflows/security-audit.yml` gate) path-confirmed NO lane.
_PATH_RE = re.compile(
    r"((?:\b(?:fak/(?:internal|cmd|experiments)|tools|docs|visuals)"
    r"|(?<![\w.])\.(?:github|claude))"
    r"/[\w./-]+)"
)

# Lanes that are exclusive in dos.toml — NEVER auto-route a worker onto these from
# a heuristic; that is exactly the collision the arbiter exists to prevent.
EXCLUSIVE_LANES = {"abi", "release", "global"}

# Scope token -> lane, when the scope is not itself a lane name. Conservative and
# derived from real issue scopes vs the real lane roster. Override via --config.
SCOPE_ALIAS: dict[str, str] = {
    "cuda": "compute", "gpu": "compute", "vulkan": "compute", "metal": "compute",
    "serve": "gateway", "anthropic": "gateway",
    "inkernel": "engine",
    "qwen35": "model", "qwen36": "model",
    "loader": "ggufload",
    "swebench": "experiments", "demo": "experiments", "simpledemo": "experiments",
    "fanbench": "bench",
    "terminal-bench": "bench",
    "testing": "ci",
    "simd": "model",
    "rehydrate": "sessionimage",
    "devex": "devindex",
    "readme": "docs", "getting-started": "docs", "fak": "docs",
    "adopt": "docs", "licensing": "docs",
    "dashboard": "metrics", "observability": "metrics",
    "dos": "tools", "control-pane": "tools", "rsi": "tools",
    "dispatch": "tools", "scrub": "tools", "ops": "tools",
    "grafana": "tools", "support-maturity": "tools", "cachevalue": "tools",
    "tooling": "tools",
    "mobile": "examples", "edge": "examples",
    "install": "cmd",
    "adjudication": "adjudicator",
}

# Label name -> lane (rung 4, weakest). Labels are coarse, so confidence is low.
LABEL_ALIAS: dict[str, str] = {
    "gpu": "compute", "compute": "compute", "performance": "compute",
    "documentation": "docs",
    "model": "model", "model-arch": "model",
    "loader": "ggufload",
    "security": "policy", "trust-floor": "policy",
    "build": "ci",
    "rsi": "tools", "dispatch": "tools",
    "agentic-serving": "gateway",
}

# Last-rung lexical fallback for unscoped issue titles/bodies. This is deliberately
# small and lane-named: it catches obvious routing words that were otherwise rotting
# with no conventional scope, without dumping ambiguous work into a catch-all lane.
KEYWORD_ALIAS: dict[str, str] = {
    "promptmmu": "promptmmu",
    "cuda": "compute",
    "a100": "compute",
    "gpu": "compute",
    "benchmark": "bench",
    "dashboard": "metrics",
    "observability": "metrics",
    "telemetry": "metrics",
    "tooling": "tools",
    "backlog": "tools",
}

# Confidence ordering (higher wins; used for sort + override decisions).
CONFIDENCE_RANK = {
    "path-confirmed": 5,
    "exact-scope": 4,
    "alias": 3,
    "label": 2,
    "keyword": 1,
    "none": 0,
}

# Issues a required HUMAN/EXTERNAL action genuinely blocks (e.g. a legal trademark
# filing) — nothing an agent can land. Carrying the `blocked-by-human` GitHub label,
# they are dropped from the DISPATCH candidate set in collect() — the one chokepoint
# BOTH dispatch lanes read (issue_dispatch.pick_lane and issue_resolve_dispatch via
# this router) — so a worker never re-spins on an issue no agent can close. The label
# keeps them VISIBLE to a human, the skipped count is surfaced (never a silent drop),
# and the closure auditor is untouched (the issue still closes when the human acts).
# Apply the label RARELY — only when truly human-blocked, never as a difficulty dodge.
BLOCKED_BY_HUMAN_LABEL = "blocked-by-human"


def repo_root(start: Path | None = None) -> Path:
    here = (start or Path(__file__)).resolve()
    return here.parent.parent


def run_text(cmd: list[str], cwd: Path, *, timeout: int = 60) -> dict[str, Any]:
    """Run a command, return {stdout, stderr, returncode}; UTF-8/replace decode.

    git/dos/gh emit non-ASCII prose that the Windows default cp1252 codec cannot
    decode, which otherwise crashes the subprocess reader thread.
    """
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"stdout": "", "stderr": str(exc), "returncode": 1, "_error": str(exc)}
    return {"stdout": proc.stdout, "stderr": proc.stderr, "returncode": proc.returncode}


def read_json_from_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if text:
        try:
            obj = json.loads(text)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return obj
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return {}


# ---------------------------------------------------------------------------
# Lane taxonomy (from dos doctor) + glob matching
# ---------------------------------------------------------------------------

def lane_taxonomy(workspace: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Return (concurrent_lanes, {lane: [tree globs]}) from `dos doctor --json`."""
    payload = read_json_from_text(
        run_text(["dos", "doctor", "--workspace", str(workspace), "--json"], workspace)["stdout"]
    )
    lanes = payload.get("lanes") or {}
    concurrent = [str(x) for x in (lanes.get("concurrent") or [])]
    trees = {str(k): [str(g) for g in v] for k, v in (lanes.get("trees") or {}).items()}
    return concurrent, trees


def _glob_to_re(glob: str) -> re.Pattern[str]:
    """Translate a dos tree glob to a regex with proper segment boundaries.

    `**` matches any depth (including zero segments); `*` matches within one
    path segment only — so `fak/internal/gateway/**` matches
    `fak/internal/gateway/x.go` and `fak/internal/gateway/sub/x.go` but NOT
    `fak/internal/gatewayx/...` (no partial-segment match).
    """
    g = glob.replace("\\", "/")
    out = []
    i = 0
    while i < len(g):
        if g.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif g.startswith("**", i):
            out.append(".*")
            i += 2
        elif g[i] == "*":
            out.append("[^/]*")
            i += 1
        else:
            out.append(re.escape(g[i]))
            i += 1
    return re.compile("^" + "".join(out) + "$")


def path_matches_lane(path: str, trees: dict[str, list[str]]) -> list[str]:
    """All lanes whose tree globs match `path` (normalized, repo-relative)."""
    # Strip ONLY a leading `./` prefix — NOT `str.lstrip("./")`, which is a
    # character-set strip that also eats the leading `.` of a dotted root like
    # `.github/...`, leaving `github/...` that the `.github/**` glob then never
    # matches (so a workflow-only #978 finding path-confirmed NO lane). Same trap
    # gate_signal.py's `where`-path handling already fixed.
    p = path.replace("\\", "/")
    if p.startswith("./"):
        p = p[2:]
    # Issue text names files in the `fak/internal/...` doc-link convention (AGENTS.md
    # writes the repo as `fak/`), but the Go module is the repository ROOT and the
    # dos.toml trees are repo-relative (`internal/...`, `cmd/...`). Strip a leading
    # `fak/` so a doc-link path matches the real-layout tree. Without this, the
    # 2026-06-22 dos.toml prefix correction (fak/internal/** -> internal/**) would
    # have made path-confirmed routing silently go dark.
    if p.startswith("fak/"):
        p = p[len("fak/"):]
    hits = []
    for lane, globs in trees.items():
        for glob in globs:
            if _glob_to_re(glob).match(p):
                hits.append(lane)
                break
    return hits


# ---------------------------------------------------------------------------
# The router (pure)
# ---------------------------------------------------------------------------

def _scope_token(title: str) -> str | None:
    """The scope from `type(scope):` if present, else a bare `prefix:` start."""
    title = title or ""
    m = _SCOPE_RE.search(title)
    if m:
        return m.group(2).strip().lower()
    bare = _BARE_PREFIX_RE.match(title)
    return bare.group(1).strip().lower() if bare else None


def _type_token(title: str) -> str | None:
    """The leading type from `type(scope):` for nonstandard issue families."""
    m = _SCOPE_RE.search(title or "")
    return m.group(1).strip().lower() if m else None


def _label_names(issue: dict[str, Any]) -> set[str]:
    return {str(lab.get("name", "")) for lab in issue.get("labels", [])}


def _has_keyword(text: str, keyword: str) -> bool:
    """Whole-token keyword match, treating '-' and '_' as part of a token."""
    return bool(re.search(r"(?<![\w-])" + re.escape(keyword.lower()) + r"(?![\w-])",
                          text.lower()))


def is_blocked_by_human(issue: dict[str, Any], *, label: str = BLOCKED_BY_HUMAN_LABEL) -> bool:
    """True when the issue carries the human/external-blocked label — see
    BLOCKED_BY_HUMAN_LABEL. Such issues are kept out of the dispatch candidate set."""
    return label in _label_names(issue)


# An epic is a PARENT/tracking issue (an umbrella over child issues), not a single
# closeable change. A worker told to close it "with the smallest correct change"
# burns a slot for minutes and ships a partial commit at best while the epic stays
# open — measured waste (~7% of open issues, ~1 in 6 spawns landed on one). Skipping
# the parent never strands a lane: collect() just routes the next closeable issue,
# and the epic still closes normally when its children do. Detected by the `epic`
# label OR the repo's `epic(scope):` / `epic:` commit-style title convention.
_EPIC_TITLE_RE = re.compile(r"^\s*epic\b\s*[\(:]", re.IGNORECASE)


def is_epic(issue: dict[str, Any]) -> bool:
    """True when the issue is an epic parent (label `epic` or an `epic(...)`/`epic:`
    title). Kept out of the dispatch candidate set like a human-blocked issue."""
    if "epic" in _label_names(issue):
        return True
    return bool(_EPIC_TITLE_RE.match(str(issue.get("title") or "")))


def is_dispatchable(issue: dict[str, Any]) -> bool:
    """A worker can be pointed at this issue: not human-blocked, not an epic parent."""
    return not is_blocked_by_human(issue) and not is_epic(issue)


def route_issue(
    issue: dict[str, Any],
    concurrent: list[str],
    trees: dict[str, list[str]],
    *,
    scope_alias: dict[str, str] | None = None,
    label_alias: dict[str, str] | None = None,
    keyword_alias: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Route one issue to a lane via the confidence ladder. Pure + deterministic."""
    scope_alias = scope_alias or SCOPE_ALIAS
    label_alias = label_alias or LABEL_ALIAS
    keyword_alias = keyword_alias or KEYWORD_ALIAS
    title = str(issue.get("title") or "")
    body = str(issue.get("body") or "")
    lane_set = set(concurrent)

    # Rung 1: path-grep probe (strongest). Run first so it can override a wrong scope.
    path_lanes: list[str] = []
    for p in _PATH_RE.findall(title + "\n" + body):
        for lane in path_matches_lane(p, trees):
            if lane in lane_set and lane not in path_lanes:
                path_lanes.append(lane)
    path_lane: str | None = None
    path_ambiguous = False
    if len(path_lanes) == 1:
        path_lane = path_lanes[0]
    elif len(path_lanes) > 1:
        path_ambiguous = True

    scope = _scope_token(title)
    typ = _type_token(title)

    # Rung 2: exact scope == lane.
    scope_lane = None
    scope_conf = None
    if scope and scope in lane_set and scope not in EXCLUSIVE_LANES:
        scope_lane, scope_conf = scope, "exact-scope"
    # Rung 3: alias scope -> lane.
    elif scope and scope in scope_alias and scope_alias[scope] in lane_set:
        scope_lane, scope_conf = scope_alias[scope], "alias"
    elif typ and typ in scope_alias and scope_alias[typ] in lane_set:
        scope_lane, scope_conf = scope_alias[typ], "alias"

    # Rung 4: label -> lane.
    label_lane = None
    for lab in sorted(_label_names(issue)):
        if lab in label_alias and label_alias[lab] in lane_set:
            label_lane = label_alias[lab]
            break

    # Rung 5: explicit keyword -> lane. Weakest automatic signal, but better than
    # leaving obviously-lane-named issues permanently unrouted.
    keyword_lane = None
    keyword = None
    searchable = title + "\n" + body
    for key in sorted(keyword_alias):
        lane = keyword_alias[key]
        if lane in lane_set and _has_keyword(searchable, key):
            keyword, keyword_lane = key, lane
            break

    # Exclusive-lane scope is operator-gated, never auto-routed.
    if scope in EXCLUSIVE_LANES:
        return _route(issue, None, "none", f"exclusive-scope:{scope}", False,
                      unrouted_reason=f"exclusive-lane scope '{scope}'; operator-gated")

    # Resolve with override: path_lane wins outright (it's the non-forgeable signal).
    if path_lane is not None:
        weaker = scope_lane or label_lane
        conflict = weaker is not None and weaker != path_lane
        signal = f"path:{path_lane}" + (f" (overrode {weaker})" if conflict else "")
        return _route(issue, path_lane, "path-confirmed", signal, conflict)

    if path_ambiguous:
        # Tie-break: prefer the lane also matching scope/label; else lexicographic.
        prefer = scope_lane if scope_lane in path_lanes else (label_lane if label_lane in path_lanes else None)
        pick = prefer or sorted(path_lanes)[0]
        return _route(issue, pick, "path-confirmed", f"path-ambiguous:{'|'.join(sorted(path_lanes))}",
                      True, unrouted_reason=None)

    if scope_lane is not None:
        token = scope if scope in lane_set or scope in scope_alias else typ
        return _route(issue, scope_lane, scope_conf, f"scope:{token}->{scope_lane}", False)
    if label_lane is not None:
        return _route(issue, label_lane, "label", f"label->{label_lane}", False)
    if keyword_lane is not None:
        return _route(issue, keyword_lane, "keyword", f"keyword:{keyword}->{keyword_lane}", False)

    reason = "no scope, no repo-path, no aliasable label" if scope else "no scope/path/label signal"
    return _route(issue, None, "none", "unrouted", False, unrouted_reason=reason)


def _route(issue, lane, confidence, signal, conflict, *, unrouted_reason=None) -> dict[str, Any]:
    return {
        "number": issue.get("number"),
        "title": str(issue.get("title") or "")[:80],
        "lane": lane,
        "confidence": confidence,
        "signal": signal,
        "signal_conflict": bool(conflict),
        "unrouted_reason": unrouted_reason,
    }


# ---------------------------------------------------------------------------
# Fetch + payload
# ---------------------------------------------------------------------------

IssueFetcher = Callable[[Path], list[dict[str, Any]]]


def fetch_issues(workspace: Path, *, limit: int = 1000) -> list[dict[str, Any]]:
    # body IS required — the path-grep rung reads it. (closure-audit omits body;
    # do not blind-copy that fetcher.)
    res = run_text(
        ["gh", "issue", "list", "--state", "open", "--limit", str(limit),
         "--json", "number,title,labels,body"],
        workspace,
    )
    text = (res.get("stdout") or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        return []
    return data if isinstance(data, list) else []


def load_injected_issues(source: str) -> list[dict[str, Any]]:
    """Read the open-issue set from a ``gh issue list --json`` array instead of the
    built-in gh fetch — ``source`` is a file path, or ``-`` for stdin.

    This is what lets a NAMED VIEW drive routing: pipe
    ``issue_views.py show --view <slug> --json --fields number,title,labels,body``
    into ``--issues -`` and the view IS the backlog the router folds, rather than
    every tool re-fetching the whole open set (issue-views is the default
    selection surface; this makes it the literal one for the router too).

    Field-tolerant: the router only reads number/title/labels/body, and any field a
    view omits degrades gracefully — an absent ``body`` merely weakens the
    path-grep rung (scope/label/keyword routing still fire), so pass
    ``--fields …,body`` upstream when full path-confirmation fidelity matters.
    Raises ValueError on non-array / invalid JSON.
    """
    raw = sys.stdin.read() if source == "-" else Path(source).read_text(encoding="utf-8")
    text = raw.strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError as exc:
        raise ValueError(f"--issues input is not valid JSON: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError("--issues input must be a JSON array of gh issue objects")
    return data


def build_payload(
    *,
    workspace: str,
    routes: list[dict[str, Any]],
    trees: dict[str, list[str]],
    max_unrouted_frac: float = 0.25,
    fetch_error: str | None = None,
    coverage: dict[str, Any] | None = None,
    skipped_blocked: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    skipped_blocked = skipped_blocked or []
    skipped = [
        {"number": i.get("number"), "title": str(i.get("title") or "")[:80]}
        for i in skipped_blocked
    ]
    by_conf: dict[str, int] = {k: 0 for k in CONFIDENCE_RANK}
    lanes: dict[str, dict[str, Any]] = {}
    for r in routes:
        by_conf[r["confidence"]] = by_conf.get(r["confidence"], 0) + 1
        lane = r["lane"]
        if lane:
            grp = lanes.setdefault(lane, {"tree": trees.get(lane, []), "count": 0, "issues": []})
            grp["count"] += 1
            grp["issues"].append(r["number"])

    total = len(routes)
    unrouted = by_conf["none"]
    routed = total - unrouted
    frac = round(unrouted / total, 4) if total else 0.0

    coverage = coverage or {"complete": True, "notes": []}
    incomplete = not coverage.get("complete", True)
    coverage_note = "; ".join(coverage.get("notes") or [])

    if fetch_error:
        ok, verdict, finding = False, "FETCH_ERROR", "fetch_error"
        reason = fetch_error
        next_action = "fix the gh/dos read-back error, then re-run the lane router"
    elif incomplete:
        # The gh fetch hit its cap, so some open issues were never routed. Routing
        # only a slice of the backlog is itself an ACTION — surface it loudly.
        ok, verdict, finding = False, "ACTION", "incomplete_coverage"
        reason = (
            f"routed {routed}/{total} fetched, but the open-issue fetch was truncated, "
            f"so some open issues were never routed — {coverage_note}"
        )
        next_action = (
            "re-run with a higher --issue-limit (the wired loop now defaults above the "
            "current open-issue count) so every open issue is routed"
        )
    elif total and frac > max_unrouted_frac:
        ok, verdict, finding = False, "ACTION", "high_unrouted"
        reason = f"{unrouted}/{total} open issues UNROUTED (frac={frac} > {max_unrouted_frac})"
        next_action = "operator: add scopes/labels or extend SCOPE_ALIAS so workers can target these"
    else:
        ok, verdict, finding = True, "OK", "routed"
        blocked_note = f"; {len(skipped)} human-blocked skipped" if skipped else ""
        reason = (f"{routed}/{total} open issues routed to {len(lanes)} lane(s); "
                  f"{unrouted} UNROUTED{blocked_note}")
        next_action = "dos-dispatch workers fold lanes[<their lane>].issues into the dispositions sidecar"

    return {
        "schema": SCHEMA,
        "ok": ok,
        "verdict": verdict,
        "finding": finding,
        "reason": reason,
        "next_action": next_action,
        "workspace": workspace,
        "coverage": coverage,
        "counts": {
            "open": total, "routed": routed, "unrouted": unrouted,
            "unrouted_frac": frac, "by_confidence": by_conf,
            "skipped_human_blocked": len(skipped),
        },
        "lanes": dict(sorted(lanes.items(), key=lambda kv: (-kv[1]["count"], kv[0]))),
        "issues": sorted(routes, key=_route_sort_key),
        "skipped_human_blocked": sorted(skipped, key=lambda s: -int(s.get("number") or 0)),
    }


def _route_sort_key(r: dict[str, Any]) -> tuple[int, int, int]:
    # UNROUTED first (operator triage), then by confidence desc, then issue number desc.
    unrouted_first = 0 if r["lane"] is None else 1
    return (unrouted_first, -CONFIDENCE_RANK.get(r["confidence"], 0), -int(r.get("number") or 0))


def compute_coverage(*, issues_fetched: int, issue_limit: int) -> dict[str, Any]:
    """Flag when the open-issue fetch hit its cap (so some issues went unrouted).

    `gh issue list` returns newest-first; a fetch that returned exactly the limit
    almost certainly dropped older open issues, which then never reach a lane.
    Routing a *slice* of the backlog while reporting `routed/open` as if it were
    the whole open set is the silent-truncation failure this surfaces.
    """
    truncated = issues_fetched >= issue_limit
    notes: list[str] = []
    if truncated:
        notes.append(
            f"gh fetch returned {issues_fetched} open issue(s) = the --issue-limit cap; "
            f"older open issues may be unrouted — raise --issue-limit"
        )
    return {
        "complete": not truncated,
        "truncated": truncated,
        "issues_fetched": issues_fetched,
        "issue_limit": issue_limit,
        "notes": notes,
    }


def collect(
    workspace: Path,
    *,
    issue_limit: int = 1000,
    max_unrouted_frac: float = 0.25,
    fetcher: IssueFetcher | None = None,
    scope_alias: dict[str, str] | None = None,
    label_alias: dict[str, str] | None = None,
    keyword_alias: dict[str, str] | None = None,
    injected: bool = False,
) -> dict[str, Any]:
    root = workspace.resolve()
    concurrent, trees = lane_taxonomy(root)
    fetch = fetcher or (lambda ws: fetch_issues(ws, limit=issue_limit))
    issues = fetch(root)
    if injected:
        # The issue set was supplied explicitly (a named-view slice piped in via
        # --issues), not fetched from gh — so the silent-truncation guard does not
        # apply: the slice IS the intended backlog, complete by construction. (A
        # view deliberately narrows the open set; flagging it "truncated" and
        # advising "raise --issue-limit" would misfire.)
        coverage = {
            "complete": True, "truncated": False, "injected": True,
            "issues_fetched": len(issues), "issue_limit": issue_limit,
            "notes": ["issues injected via --issues (a named-view slice); coverage "
                      "reflects the provided slice, not a full gh fetch"],
        }
    else:
        coverage = compute_coverage(issues_fetched=len(issues), issue_limit=issue_limit)
    # Drop non-dispatchable issues (human/external-blocked AND epic parents) from the
    # candidate set — the one chokepoint both lanes read — but surface them so the skip
    # is never silent. Epics stay visible to humans and still close when their children do.
    blocked = [i for i in issues if not is_dispatchable(i)]
    routable = [i for i in issues if is_dispatchable(i)]
    fetch_error = None
    if not concurrent:
        fetch_error = "dos doctor returned no lanes — run from the repo root (not fak/)"
    elif not issues and not injected:
        fetch_error = "gh returned no open issues (auth/network?)"
    routes = [
        route_issue(i, concurrent, trees, scope_alias=scope_alias, label_alias=label_alias,
                    keyword_alias=keyword_alias)
        for i in routable
    ]
    return build_payload(
        workspace=str(root), routes=routes, trees=trees,
        max_unrouted_frac=max_unrouted_frac, fetch_error=fetch_error, coverage=coverage,
        skipped_blocked=blocked,
    )


def render(payload: dict[str, Any]) -> str:
    c = payload.get("counts") or {}
    lines = [
        f"issue-lane router: {payload.get('verdict')} ({payload.get('finding')})",
        f"routed={c.get('routed')}/{c.get('open')} unrouted={c.get('unrouted')} "
        f"(frac={c.get('unrouted_frac')})  by_conf={c.get('by_confidence')}",
        f"next: {payload.get('next_action')}",
    ]
    skipped = payload.get("skipped_human_blocked") or []
    if skipped:
        nums = ", ".join(f"#{s['number']}" for s in skipped[:10])
        lines.append(f"  human-blocked (skipped, not dispatched): {len(skipped)}  {nums}")
    coverage = payload.get("coverage") or {}
    if not coverage.get("complete", True):
        for note in coverage.get("notes") or []:
            lines.append(f"  ! partial coverage: {note}")
    lines.append("  lanes with ticket backlog:")
    for lane, grp in list((payload.get("lanes") or {}).items())[:20]:
        nums = ",".join(f"#{n}" for n in grp["issues"][:10])
        lines.append(f"    {lane:<14} {grp['count']:>2}  {nums}")
    unrouted = [r for r in payload.get("issues", []) if r["lane"] is None]
    if unrouted:
        lines.append(f"  UNROUTED ({len(unrouted)}) — operator triage:")
        for r in unrouted[:10]:
            lines.append(f"    #{r['number']:<5} {r['unrouted_reason']}: {r['title']}")
    return "\n".join(lines)


def render_md(payload: dict[str, Any], *, date: str) -> str:
    c = payload.get("counts") or {}
    out = [
        f"# Issue → lane routing — {date}",
        "",
        f"- routed **{c.get('routed')}/{c.get('open')}**, UNROUTED {c.get('unrouted')} "
        f"(frac {c.get('unrouted_frac')})",
        f"- by confidence: `{c.get('by_confidence')}`",
        f"- verdict: `{payload.get('verdict')}` — {payload.get('reason')}",
    ]
    coverage = payload.get("coverage") or {}
    if not coverage.get("complete", True):
        out.append(f"- **coverage**: ⚠ partial — {'; '.join(coverage.get('notes') or [])}")
    elif coverage:
        out.append(f"- **coverage**: complete (issues_fetched={coverage.get('issues_fetched')})")
    out += [
        "",
        "## Per-lane backlog",
        "",
        "| lane | count | issues |",
        "|---|---|---|",
    ]
    for lane, grp in (payload.get("lanes") or {}).items():
        out.append(f"| {lane} | {grp['count']} | {', '.join('#'+str(n) for n in grp['issues'])} |")
    out += ["", "## UNROUTED (operator triage)", "", "| # | reason | title |", "|---|---|---|"]
    for r in payload.get("issues", []):
        if r["lane"] is None:
            out.append(f"| #{r['number']} | {r['unrouted_reason']} | {r['title']} |")
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Route open GitHub issues to dos.toml lanes (read-only).")
    ap.add_argument("--workspace", default="", help="workspace root (default: repo root)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument("--issue-limit", type=int, default=1000, help="max open issues to fetch")
    ap.add_argument("--max-unrouted-frac", type=float, default=0.25,
                    help="ACTION verdict when UNROUTED fraction exceeds this")
    ap.add_argument("--config", default="", help="JSON file overriding scope_alias / label_alias")
    ap.add_argument("--md", default="", help="write a dated markdown report to this path")
    ap.add_argument("--issues", default="", metavar="PATH|-",
                    help="route a gh-issue-list JSON array read from a file or '-' "
                         "(stdin) instead of fetching via gh — lets a named view drive "
                         "routing, e.g. `python tools/issue_views.py show --view "
                         "ready-leaves --json --fields number,title,labels,body | "
                         "python tools/issue_lane_router.py --issues - --json`")
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    scope_alias = label_alias = None
    keyword_alias = None
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        scope_alias = cfg.get("scope_alias")
        label_alias = cfg.get("label_alias")
        keyword_alias = cfg.get("keyword_alias")

    fetcher: IssueFetcher | None = None
    injected = False
    if args.issues:
        try:
            _injected_rows = load_injected_issues(args.issues)
        except (ValueError, OSError) as exc:
            print(f"ERROR: --issues: {exc}", file=sys.stderr)
            return 2
        fetcher = lambda ws: _injected_rows  # noqa: E731
        injected = True

    payload = collect(
        workspace, issue_limit=args.issue_limit, max_unrouted_frac=args.max_unrouted_frac,
        fetcher=fetcher, injected=injected,
        scope_alias=scope_alias, label_alias=label_alias,
        keyword_alias=keyword_alias,
    )

    if args.md:
        date = run_text(["git", "log", "-1", "--format=%cs"], workspace)["stdout"].strip() or "unknown"
        Path(args.md).write_text(render_md(payload, date=date), encoding="utf-8")

    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(render(payload))
    return 0 if payload.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
