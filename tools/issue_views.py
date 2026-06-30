#!/usr/bin/env python3
"""issue_views.py — the named-issue-view selection surface.

The default "what should I work on" entry for fak dispatch and triage. It reads
``.github/issue-views.json`` (named views derived from the repo's real label
taxonomy) and resolves a view into a concrete ``gh issue list --search`` query.

This is the API-readable mirror of the GitHub saved views at
``https://github.com/<repo>/issues/views``, which hydrate client-side and cannot
be read through ``gh``/REST/GraphQL. Edit the JSON to track the GitHub UI; this
tool turns a view name into the same backlog the dispatch family consumes, so
selection stops being scattered ad-hoc ``--label`` queries.

Read-only: it fetches and prints; it never edits, labels, or closes an issue.
Pure stdlib — ``python tools/issue_views.py`` resolves with no venv.

Subcommands::

    list                   every view (slug · title · query); --counts adds live counts
    show   --view <slug>   run the view's query; --json emits the gh issue array
    query  --view <slug>   print the resolved gh command only (offline; no network)
    default                print the default view slug

A ``--view`` omitted (or ``default``) resolves to the config's ``default`` view.
The ``--json`` output of ``show`` is the raw ``gh issue list --json`` array
(``number,title,labels,updatedAt,assignees,url``) — pipe it straight into
``issue_lane_router`` / ``issue_triage``.
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

DEFAULT_FIELDS = "number,title,labels,updatedAt,assignees,url"


def repo_root(start: Path | None = None) -> Path:
    """Walk up from ``start`` (cwd) to the dir holding ``.github/issue-views.json``.

    Falls back to the git toplevel, then to ``start`` itself — so the tool works
    whether invoked from the repo root or a subdir.
    """
    here = (start or Path.cwd()).resolve()
    for cand in (here, *here.parents):
        if (cand / ".github" / "issue-views.json").is_file():
            return cand
        if (cand / ".git").exists():
            return cand
    return here


def default_config_path(root: Path) -> Path:
    return root / ".github" / "issue-views.json"


def load_config(path: Path) -> dict[str, Any]:
    """Load + minimally validate the views config. Raises ValueError on a bad shape."""
    try:
        cfg = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"no issue-views config at {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be an object")
    views = cfg.get("views")
    if not isinstance(views, list) or not views:
        raise ValueError(f"{path}: `views` must be a non-empty list")
    seen: set[str] = set()
    for v in views:
        slug = v.get("slug") if isinstance(v, dict) else None
        query = v.get("query") if isinstance(v, dict) else None
        if not slug or not isinstance(slug, str):
            raise ValueError(f"{path}: every view needs a string `slug`")
        if slug in seen:
            raise ValueError(f"{path}: duplicate view slug {slug!r}")
        seen.add(slug)
        if not query or not isinstance(query, str):
            raise ValueError(f"{path}: view {slug!r} needs a non-empty `query`")
    default = cfg.get("default")
    if default is not None and default not in seen:
        raise ValueError(f"{path}: default {default!r} is not a defined view slug")
    return cfg


def view_map(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {v["slug"]: v for v in cfg["views"]}


def resolve_view(cfg: dict[str, Any], slug: str | None) -> dict[str, Any]:
    """Return the view dict for ``slug``; the config default when ``slug`` is falsy.

    Raises KeyError (with the available slugs) on an unknown name.
    """
    vm = view_map(cfg)
    if not slug:
        slug = cfg.get("default")
        if not slug:
            raise KeyError("no --view given and config has no `default`")
    if slug not in vm:
        raise KeyError(f"unknown view {slug!r}; defined: {', '.join(sorted(vm))}")
    return vm[slug]


def build_gh_args(
    cfg: dict[str, Any],
    view: dict[str, Any],
    *,
    limit: int | None = None,
    json_fields: str | None = None,
) -> list[str]:
    """Assemble the ``gh issue list`` argv that materializes ``view``.

    Deterministic and side-effect-free — the offline witness for ``query``.
    """
    repo = cfg.get("repo")
    lim = limit if limit is not None else int(cfg.get("limit", 300))
    args = ["gh", "issue", "list"]
    if repo:
        args += ["--repo", str(repo)]
    args += ["--search", str(view["query"]), "--limit", str(lim)]
    if json_fields:
        args += ["--json", json_fields]
    return args


def run_gh(args: list[str], *, parse_json: bool) -> Any:
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or f"gh exited {proc.returncode}\n")
        raise SystemExit(proc.returncode)
    out = proc.stdout
    return json.loads(out) if parse_json else out


def _count(cfg: dict[str, Any], view: dict[str, Any]) -> int:
    rows = run_gh(build_gh_args(cfg, view, json_fields="number"), parse_json=True)
    return len(rows)


# ---- subcommand handlers -------------------------------------------------


def cmd_list(cfg: dict[str, Any], ns: argparse.Namespace) -> int:
    default = cfg.get("default")
    for v in cfg["views"]:
        slug = v["slug"]
        marker = " *" if slug == default else ""
        count = f"  [{_count(cfg, v)}]" if ns.counts else ""
        print(f"{slug}{marker}{count}\n    {v.get('title', '')}\n    {v['query']}")
    if ns.counts:
        print("\n(* = default view; [N] = live open-issue count)")
    else:
        print("\n(* = default view; pass --counts for live counts)")
    return 0


def cmd_query(cfg: dict[str, Any], ns: argparse.Namespace) -> int:
    view = resolve_view(cfg, ns.view)
    print(shlex.join(build_gh_args(cfg, view, limit=ns.limit, json_fields=ns.fields or None)))
    return 0


def cmd_show(cfg: dict[str, Any], ns: argparse.Namespace) -> int:
    view = resolve_view(cfg, ns.view)
    if ns.json:
        rows = run_gh(build_gh_args(cfg, view, limit=ns.limit, json_fields=ns.fields), parse_json=True)
        json.dump(rows, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    rows = run_gh(build_gh_args(cfg, view, limit=ns.limit, json_fields=DEFAULT_FIELDS), parse_json=True)
    print(f"# {view['slug']} — {view.get('title', '')}  ({len(rows)} open)")
    print(f"# {view['query']}")
    for r in rows:
        labs = ",".join(lab.get("name", "") for lab in r.get("labels", []))
        assignee = (r.get("assignees") or [{}])[0].get("login", "") if r.get("assignees") else ""
        who = f" @{assignee}" if assignee else ""
        print(f"#{r['number']:<6} {r.get('title', '')}{who}   [{labs}]")
    return 0


def cmd_default(cfg: dict[str, Any], ns: argparse.Namespace) -> int:
    print(cfg.get("default", ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="issue_views.py", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=None,
                   help="path to issue-views.json (default: .github/issue-views.json)")
    sub = p.add_subparsers(dest="cmd")

    pl = sub.add_parser("list", help="list every view")
    pl.add_argument("--counts", action="store_true", help="add live open-issue counts (one gh call per view)")
    pl.set_defaults(func=cmd_list)

    pq = sub.add_parser("query", help="print the resolved gh command (offline)")
    pq.add_argument("--view", default=None, help="view slug (default: config default)")
    pq.add_argument("--limit", type=int, default=None)
    pq.add_argument("--fields", default="", help="gh --json fields; empty omits --json")
    pq.set_defaults(func=cmd_query)

    ps = sub.add_parser("show", help="run a view's query and print the issues")
    ps.add_argument("--view", default=None, help="view slug (default: config default)")
    ps.add_argument("--limit", type=int, default=None)
    ps.add_argument("--json", action="store_true", help="emit the raw gh issue array")
    ps.add_argument("--fields", default=DEFAULT_FIELDS, help="gh --json fields for --json output")
    ps.set_defaults(func=cmd_show)

    pd = sub.add_parser("default", help="print the default view slug")
    pd.set_defaults(func=cmd_default)
    return p


def main(argv: list[str] | None = None) -> int:
    # gh issue titles carry non-ASCII prose (Θ, em-dash) that the Windows default
    # cp1252 stdout codec cannot encode — force UTF-8 like the sibling tools do.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
    ns = build_parser().parse_args(argv)
    if not getattr(ns, "func", None):
        build_parser().print_help()
        return 2
    root = repo_root()
    cfg_path = ns.config or default_config_path(root)
    try:
        cfg = load_config(cfg_path)
    except ValueError as exc:
        sys.stderr.write(f"issue_views: {exc}\n")
        return 2
    try:
        return ns.func(cfg, ns)
    except KeyError as exc:
        sys.stderr.write(f"issue_views: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
