---
title: "Witness curated front-door navigation"
description: "A link witness proves more than file existence: a curated route must resolve, name its audience, reach the right authority, and end in an action you can start."
---

# Witness curated front-door navigation

**Primary audience:** maintainers verifying current public and contributor navigation.
**Lifecycle:** current process authority. **Generation:** `gen/now`.
**Authority:** the [documentation audience inventory](../project/documentation-audience-inventory.json) selects curated routes; this page defines their navigation witness. Runtime and support claims remain authoritative in code, tests, and [`docs/supported/`](../supported/README.md).
**Next action:** enumerate the inventory entries marked `front_door_linked: true`, resolve every local target and anchor, then follow each route to its stated checkable action and record the result.

A link witness proves more than file existence. Every curated route must be reachable, identify its audience and current context, lead to the authority appropriate to its job, and end in an action a reader can start.

## Curated set

At commit `cc76cf4329a4652e7ebbad9e2f610790891229ce`, the inventory selects six routes:

| Route | Audience | Navigation job |
|---|---|---|
| `README.md` | public front door | Explain fak, offer the shortest proof, and route the next goal. |
| `START-HERE.md` | public front door | Match a reader goal to its shortest current path. |
| `llms.txt` | contributor/agent | Map tasks to maintained authorities. |
| `AGENTS.md` | contributor/agent | Supply workspace rules and executable build/ship routes. |
| `CONTRIBUTING.md` | contributor | Select and start the correct contribution workflow. |
| `SECURITY.md` | operator/reporter | Explain the capability floor and start confidential disclosure. |

The full repository commit is the inventory revision. A changed inventory, curated route, or direct target makes this result stale and requires a rerun.

## Two-part witness

### 1. Resolution

For every Markdown link in each curated route:

1. Record source path, visible label, raw destination, and optional fragment.
2. Resolve relative paths from the source directory; resolve root-relative paths from the repository root.
3. For local Markdown/text targets, verify the file and heading anchor. Account for generated duplicate-heading suffixes.
4. Record external HTTP, HTTPS, and mail destinations separately. Validate their syntax and, when network access is available, response or mail-route ownership; an offline run must mark them `external-unwitnessed`, not silently pass them.
5. Reject private-only destinations from public routes and record redirects or superseding routes explicitly.

A reachable file with a missing anchor fails resolution. A successful resolution says nothing yet about whether the destination is useful or current.

### 2. Action read-back

Read each route as its declared audience and record:

- its page job and current `gen/now`/lifecycle boundary;
- the choice or default presented to that audience;
- the first checkable action (command, selection, report route, or authority lookup);
- the destination needed to start that action;
- whether the destination matches the source label and preserves audience, mode, support, and generation context;
- the observable completion signal.

Classify each route:

| Verdict | Rule |
|---|---|
| **Pass** | All local targets and anchors resolve; external links are witnessed or explicitly marked; the route reaches a matching current authority and a checkable action. |
| **Repair link** | A target, anchor, redirect, or public/private boundary fails. |
| **Repair route** | Links resolve, but labels, choices, generation/support context, or destination authority mismatch. |
| **Add action** | Navigation reaches information but supplies no checkable next action or completion signal. |

One failure fails that route; do not average it into an overall score. File or link a deduplicated issue for every unresolved failure.

## Current bounded result

The six routes at `cc76cf4329a4652e7ebbad9e2f610790891229ce` contain **168 Markdown links**: **160 local** and **8 external**. All 160 local targets and applicable anchors resolve. The external links require the disposition field in a repeat run; this local witness does not claim network reachability.

Structured action read-back found each route has a startable action matching its inventory role: run the offline proof (`README.md`/`START-HERE.md`), select a maintained authority (`llms.txt`), run the named workspace command (`AGENTS.md`), choose and verify a contribution route (`CONTRIBUTING.md`), or use the confidential reporting path (`SECURITY.md`). Result: **6 pass, 0 repair-link, 0 repair-route, 0 add-action** for local navigation and action shape.

This result is bounded to the stated revision. It does not certify every repository document or the live availability of external services.

## Witness record

Use one row per link failure and one action row per curated route:

| Field | Required value |
|---|---|
| Revision | Full commit containing inventory, sources, and targets |
| Source | Curated route and declared audience |
| Link | Label, raw destination, resolved destination/anchor, local or external |
| Resolution | pass, failure reason, or `external-unwitnessed` |
| Route read-back | Page job, choice/default, first action, destination, completion signal |
| Context | Mode, lifecycle, generation, support boundary, and destination authority |
| Verdict | Pass, repair link, repair route, or add action |
| Disposition | Passing evidence, fixed commit, or open issue |

A complete run includes:

```text
inventory_revision: <full commit>
curated_routes: <ordered paths>
links: {total: <n>, local: <n>, external: <n>}
resolution_failures: <rows or []>
external_dispositions: <witnesses or external-unwitnessed rows>
route_verdicts: <one per route>
unresolved_findings: <issue links or []>
independent_reader:
  audience: <restated audience>
  page_job: <restated witness>
  applicable_choice: <verdict rule>
  next_action: <restated first step>
  ambiguities: []
```

After repair, rerun the changed source, its target, and direct inbound curated routes. Use the [front-door clarity scorecard](frontdoor-clarity-scorecard.md) when a route—not merely a link—changes, and the [generation drift check](generation-drift-check.md) when authority context changes.
