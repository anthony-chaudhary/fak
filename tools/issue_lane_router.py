#!/usr/bin/env python3
"""Route OPEN GitHub issues to the dos.toml lane whose file-tree they touch.

The DOS supervisor dispatches by LANE (from dos.toml), and a lane-worker picks
work from the plan portfolio — issues are invisible to it. So a live supervisor
run only resolves tickets that happen to ride along on plan-lane work; it cannot
TARGET the backlog. The closure auditor (`tools/issue_closure_audit.py`) proved
the cost: closure_rate sits near zero because nothing aims the fleet at tickets.

This tool is the missing aim. For each open issue it picks the lane whose `trees`
globs the issue most likely touches, with a confidence ladder
(path-confirmed > exact-scope > alias > label > none) so the supervisor can
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
from pathlib import Path
from typing import Any, Callable

SCHEMA = "fleet-issue-lane-router/1"

# Mirrored from issue_triage.py (small; triage is not an importable package).
_SCOPE_RE = re.compile(r"\b(\w+)\(([^)]+)\)")  # feat(scope), fix(scope), ...
# Bare `prefix:` start form (e.g. `abi: ...`, `RSI: ...`) — the loose convention
# used for scopeless types. Only the leading token before the first colon.
_BARE_PREFIX_RE = re.compile(r"^([A-Za-z][\w-]*):\s")

# Concrete repo paths an issue body/title may name (the strongest routing signal).
_PATH_RE = re.compile(
    r"\b((?:fak/(?:internal|cmd|experiments)|tools|docs|visuals|\.github|\.claude)"
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
    "readme": "docs", "getting-started": "docs", "fak": "docs",
    "adopt": "docs", "licensing": "docs",
    "dos": "tools", "control-pane": "tools", "rsi": "tools",
    "dispatch": "tools", "scrub": "tools", "ops": "tools",
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

# Confidence ordering (higher wins; used for sort + override decisions).
CONFIDENCE_RANK = {"path-confirmed": 4, "exact-scope": 3, "alias": 2, "label": 1, "none": 0}


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
    p = path.replace("\\", "/").lstrip("./")
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


def _label_names(issue: dict[str, Any]) -> set[str]:
    return {str(lab.get("name", "")) for lab in issue.get("labels", [])}


def route_issue(
    issue: dict[str, Any],
    concurrent: list[str],
    trees: dict[str, list[str]],
    *,
    scope_alias: dict[str, str] | None = None,
    label_alias: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Route one issue to a lane via the confidence ladder. Pure + deterministic."""
    scope_alias = scope_alias or SCOPE_ALIAS
    label_alias = label_alias or LABEL_ALIAS
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

    # Rung 2: exact scope == lane.
    scope_lane = None
    scope_conf = None
    if scope and scope in lane_set and scope not in EXCLUSIVE_LANES:
        scope_lane, scope_conf = scope, "exact-scope"
    # Rung 3: alias scope -> lane.
    elif scope and scope in scope_alias and scope_alias[scope] in lane_set:
        scope_lane, scope_conf = scope_alias[scope], "alias"

    # Rung 4: label -> lane.
    label_lane = None
    for lab in sorted(_label_names(issue)):
        if lab in label_alias and label_alias[lab] in lane_set:
            label_lane = label_alias[lab]
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
        return _route(issue, scope_lane, scope_conf, f"scope:{scope}->{scope_lane}", False)
    if label_lane is not None:
        return _route(issue, label_lane, "label", f"label->{label_lane}", False)

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


def build_payload(
    *,
    workspace: str,
    routes: list[dict[str, Any]],
    trees: dict[str, list[str]],
    max_unrouted_frac: float = 0.25,
    fetch_error: str | None = None,
    coverage: dict[str, Any] | None = None,
) -> dict[str, Any]:
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
        reason = f"{routed}/{total} open issues routed to {len(lanes)} lane(s); {unrouted} UNROUTED"
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
        },
        "lanes": dict(sorted(lanes.items(), key=lambda kv: (-kv[1]["count"], kv[0]))),
        "issues": sorted(routes, key=_route_sort_key),
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
) -> dict[str, Any]:
    root = workspace.resolve()
    concurrent, trees = lane_taxonomy(root)
    fetch = fetcher or (lambda ws: fetch_issues(ws, limit=issue_limit))
    issues = fetch(root)
    coverage = compute_coverage(issues_fetched=len(issues), issue_limit=issue_limit)
    fetch_error = None
    if not concurrent:
        fetch_error = "dos doctor returned no lanes — run from the repo root (not fak/)"
    elif not issues:
        fetch_error = "gh returned no open issues (auth/network?)"
    routes = [
        route_issue(i, concurrent, trees, scope_alias=scope_alias, label_alias=label_alias)
        for i in issues
    ]
    return build_payload(
        workspace=str(root), routes=routes, trees=trees,
        max_unrouted_frac=max_unrouted_frac, fetch_error=fetch_error, coverage=coverage,
    )


def render(payload: dict[str, Any]) -> str:
    c = payload.get("counts") or {}
    lines = [
        f"issue-lane router: {payload.get('verdict')} ({payload.get('finding')})",
        f"routed={c.get('routed')}/{c.get('open')} unrouted={c.get('unrouted')} "
        f"(frac={c.get('unrouted_frac')})  by_conf={c.get('by_confidence')}",
        f"next: {payload.get('next_action')}",
    ]
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
    args = ap.parse_args(argv)

    workspace = Path(args.workspace).resolve() if args.workspace else repo_root()
    scope_alias = label_alias = None
    if args.config:
        cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
        scope_alias = cfg.get("scope_alias")
        label_alias = cfg.get("label_alias")

    payload = collect(
        workspace, issue_limit=args.issue_limit, max_unrouted_frac=args.max_unrouted_frac,
        scope_alias=scope_alias, label_alias=label_alias,
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
