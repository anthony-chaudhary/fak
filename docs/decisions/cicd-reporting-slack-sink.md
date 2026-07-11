---
title: "CI/CD reporting Slack sink — one channel for status, issues, blockers, and work"
description: "The decision to fold fak's CI/CD reporting Slack feeders (scoreboard, blockers, bench, cachevalue, capacity, node-usage, backlog, dojo, product, releases, steering) onto a single sink channel C0BGQ411TCJ, with per-surface and family-wide override preserved."
---

# CI/CD reporting Slack sink

fak reports its CI/CD state to Slack through a **family of status feeders** — one
GitHub Actions workflow per surface (`.github/workflows/*-feed.yml` plus the
`ci.yml` scorecard producer). Each folds a CI-derivable signal into a card and
posts it to a channel on a cadence:

| surface | what it reports | feeder |
| --- | --- | --- |
| scoreboard | scorecard portfolio total-debt / grade / verdict | `scoreboard-feed.yml` |
| blockers | open `blocked`-labelled issue backlog (status vs paged) | `blockers-feed.yml` |
| bench | benchmark rollups / run-requests | `bench-feed.yml` |
| cachevalue | cache-value P&L roll-up (witnessed kernel reuse) | `cachevalue-feed.yml`, `cachevalue-weekly.yml` |
| capacity | fleet capacity snapshot | `capacity-feed.yml` |
| node-usage | compute-node usage snapshots | `node-usage-feed.yml` |
| backlog | issue triage + bottleneck digest | `backlog-feed.yml` |
| dojo | RSI calibration rollups / trends | `dojo-feed.yml`, `dojo-rsi-feed.yml` |
| product | product direction / persona findings | `product-feed.yml` |
| steering | steering-guard governance surface | `steering-guard.yml` |
| releases | cut-release announcements + release-observability | `release-artifacts.yml`, `release-notify.yml` |

Each feeder resolves its channel the same way: the workflow reads
`${{ vars.FAK_<SURFACE>_CHANNEL || … }}` and passes it as `--channel`, and the Go
CLI mirrors that with a per-package `ChannelDefault` + `ResolveChannel()` that
reads only its own `FAK_<SURFACE>_CHANNEL` key. Historically each surface carried
a **distinct** default channel, so the CI/CD story was scattered across a dozen
near-silent rooms.

## Decision

Fold the whole CI/CD reporting family onto **one sink channel — `C0BGQ411TCJ`** —
by default, so an operator watches one timeline for CI/CD status, issues,
blockers, and work instead of a dozen rooms. This mirrors the `#scorecards`
catch-all routing decision ([scorecards-channel-routing.md](scorecards-channel-routing.md)):
discoverability beats partition at this volume.

The sink id lives in exactly one place in code —
`scoreboard.CICDReportChannel` (`internal/scoreboard/scoreboard.go`) — and every
reporting surface's `ChannelDefault` references it. It is a **public**, non-secret
channel id in the scoreboard Slack workspace (team `T0BDEJF1HGB`): the id grants
nothing without the `FAK_SCOREBOARD_TOKEN` bot token.

## Overrides are preserved

Routing is a **default**, not a hard sink. The resolution order at both layers
(workflow YAML and Go `ResolveChannel`) is:

1. the surface's own `FAK_<SURFACE>_CHANNEL` (env / repo variable / `.env.slack.local`) — wins;
2. the family-wide `FAK_CICD_REPORT_CHANNEL` (`scoreboard.CICDReportChannelEnv`) — repoints every reporting feeder at once;
3. the built-in sink default `C0BGQ411TCJ`.

So a metric that earns its own room (per the graduation rule in the scorecards
routing decision) is split back out by setting its own key, and the whole family
moves together by setting `FAK_CICD_REPORT_CHANNEL` — without editing code.

## Scope: what is NOT on the sink

Surfaces that are not CI/CD status/issue/blocker/work reporting keep their own
channels: `#news` (external research), `#marketing`, the `grafana`/`alerts` ops
surfaces, per-session `guard-sessions` threads, and the `chatrelay` bridge. The
per-push `ci.yml` scorecard producer also stays on its explicit
`FAK_SCOREBOARD_CHANNEL` operator secret (an intentional override) rather than
flooding the sink on every trunk push — the daily `scoreboard-feed` already carries
the portfolio card to the sink.
