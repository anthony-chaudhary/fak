# Named issue views as the default selection surface (2026-06-29)

> **The default "what should I work on" entry is now a named view, not an ad-hoc
> `--label` query.** `tools/issue_views.py` reads `.github/issue-views.json` and
> resolves a view name into a concrete `gh issue list --search` query. It is the
> API-readable mirror of the GitHub saved views at
> [`/issues/views`](https://github.com/anthony-chaudhary/fak/issues/views).

## Why this exists

GitHub's repository-level **saved issue views** (the `/issues/views` page) are the
curated way to slice the backlog — but they are **not reachable through any API**.
That page hydrates client-side and needs a real browser session: `gh api
.../issues/views` 404s, the `Repository` GraphQL type exposes no `views` field, and
an authenticated `curl` of the page renders logged-out (the `viewsListWrapper` div
ships empty with `aria-busy="true"`). So an agent or a cron loop **cannot read the
saved views** to drive selection from them.

Meanwhile the dispatch tooling selected issues through ~10 scattered, hand-built
`gh issue list --label X` / `--search` queries (one shape per tool), each its own
source of truth. There was no single named-views surface, so "use the saved views
by default" had nothing to bind to.

This note records the fix: a small JSON config of named views derived from the
repo's **real label taxonomy**, plus a read-only helper that turns a view name
into the same backlog the dispatch family already consumes.

## What shipped

- **`.github/issue-views.json`** — the named views. Each carries a `slug`, a
  `title`, a GitHub-search `query`, and a `note`. A top-level `default` names the
  view used when none is given (`ready-leaves`). Edit this to track the GitHub UI;
  reconcile by hand (the GitHub views can't be machine-diffed against it).
- **`tools/issue_views.py`** — pure-stdlib, read-only. Subcommands:
  - `list [--counts]` — every view (slug · title · query), optionally with live
    open-issue counts.
  - `show --view <slug> [--json] [--limit N]` — run the view's query and print the
    issues; `--json` emits the raw `gh issue list --json` array.
  - `query --view <slug>` — print the resolved `gh` command only (offline; the
    deterministic witness).
  - `default` — print the default view slug.
- **`tools/issue_views_test.py`** — 13 hermetic tests (config validation, view
  resolution, query assembly, plus a structural check of the shipped config). In
  the CI no-blackhole hermetic set, so it auto-gates.
- Registered as the `issue_views` helper in `.claude/project.yaml`.

## The views (derived from the real label taxonomy)

Grounded in the actual open backlog on 2026-06-29 (219 open issues):

| slug | query | what it is |
| --- | --- | --- |
| `ready-leaves` ★ | `is:open -label:epic -label:research no:assignee` | **default** — unclaimed, dispatchable leaves (not umbrellas, not design-shaped) |
| `p0-p1` | `is:open label:priority/P0,priority/P1 -label:epic sort:created-asc` | prioritized leaves, oldest-first |
| `epics` | `is:open label:epic` | umbrellas — decompose via `/dos-replan`, don't dispatch as a leaf |
| `needs-triage` | `is:open label:idea-scout` | auto-filed by the daily idea-scout; the `/issue-triage` queue |
| `help-wanted` | `is:open label:"help wanted" -label:epic` | good hand-offs / extra attention |
| `good-first-issue` | `is:open label:"good first issue"` | newcomer-friendly, smallest blast radius |
| `agentic-serving` | `is:open label:agentic-serving -label:epic` | area cluster (largest) |
| `substrate` | `is:open label:substrate -label:epic` | area cluster — ABI / standard |
| `trust-floor` | `is:open label:trust-floor -label:epic` | area cluster — Decide-vs-Syscall |
| `gpu` | `is:open label:gpu,compute,multi-gpu,metal -label:epic` | area cluster — GPU/kernel (OR over four labels) |

`label:a,b` (comma) is **OR**; two separate `label:` tokens are AND. Every query is
always paired with an explicit `--limit` because `gh` defaults to 30.

## How to use it by default

When selecting work, start from a view instead of inventing a `--label` query:

```bash
# the default "what's dispatchable now" surface
python tools/issue_views.py show --view ready-leaves

# any named view, e.g. the prioritized leaves (oldest-first)
python tools/issue_views.py show --view p0-p1
```

The `--json` output is the raw `gh issue list --json
number,title,labels,updatedAt,assignees,url` array, so a view feeds any tool
that consumes that canonical payload — for example the CI blockers roll-up
(`cmd/fak/blockers.go` reads `--issues -` from stdin):

```bash
python tools/issue_views.py show --view p0-p1 --json | fak blockers feed --issues -
```

> Note: `fak blockers feed` posts the roll-up to a Slack channel — run it only
> when you mean to publish, not as a casual check.

**Honest integration limit:** `issue_lane_router` and `issue_triage` fetch their
*own* backlog internally — they take no injected-issue stdin — so for those a
view is the conceptual slice you're working (read its `--counts` to know what a
pass will cover), not a literal pipe. Wiring an `--issues -` injection into those
two so a view can scope them directly is the named follow-on.

## Honest boundary — what is NOT yet wired

The **scheduled dispatch cron** (`FleetIssueDispatch` → `tools/issue_dispatch.py`)
and `issue_lane_router` still build their own backlog fetch; this ship makes the
named-views surface the *documented + helper-registered* default and proves it
end-to-end, but it does **not** rewrite those cron-owned hot paths to call it
(they churn under peer/cron edits — flipping them is a separate, lane-disjoint
change). Folding `issue_dispatch.py`'s backlog fetch onto `issue_views.py
show --view <default>` is the named follow-on.

The local config is a **manual** mirror of the GitHub saved views — there is no
machine sync (the GitHub side is unreadable). When the saved views change, edit
`.github/issue-views.json` to match.
