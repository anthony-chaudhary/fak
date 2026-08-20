---
title: "Harness operator homepage render witness — 2026-08-20"
description: "Before/after and executable evidence for the local harness operational homepage."
---

# Harness operator homepage render witness — 2026-08-20

## Verdict

The local harness now opens on agent state, goals, and real gateway destinations. The
previous implementation-led headline and run-first layout are retired.

- Before: [`normal.png`](normal.png) leads with “Local, bounded, yours.” and exposes no
  agent totals, goal registry, or gateway navigation.
- After: [`operator-home-2026-08-20.png`](operator-home-2026-08-20.png) shows 2 live
  agents, 3 stored runs, 3 active goals, the Web gateway action, and 8 gateway surfaces
  before the run composer.

## Captured inputs

The browser and UI process were real. The screenshot's gateway state came from a bounded
loopback contract fixture serving `/healthz` and `/debug/vars`: 2 session rows and a
3-machine/7-session fleet. Three synthetic goals were written through `fak goal create`.
No private goal registry or workspace path was captured.

The browser read returned:

```text
mode=live reachable=True live_agents=2 goals=3 dashboards=8
```

The command-level selfcheck returned:

```text
HARNESS_WEB_SELFCHECK ok protocol=fak.harness.run/v1 normal=8 resumed=2 approval=4 failure=3 skins=2 runs=3 goals=1 dashboards=8 html_sha256=b35f9fbb8e90b78f83d1b312bc887f24efc79c939b927f6732338a9ee3d5f21b
```

## Reproduction gates

```text
wsl.exe bash -lc 'cd /mnt/c/work/fak && go test ./internal/harnessweb -count=1'
fak-dev buildcheck --vet ./internal/harnessweb
go run ./cmd/harnesswebdemo --selfcheck
```

The render witness in `TestCapturedPageRendersOperatingStatesAndSecondSkin` asserts the
operator sections and rejects the retired promotional headline. The status witness in
`TestStatusProjectsAgentStatsGoalsAndLiveDashboardLinks` drives the real HTTP handler
with canonical goal-registry records and `/debug/vars`-shaped session/fleet input.

The PNG was captured from the running loopback UI with system Chrome headless at
1440×1650. The gateway link sanitizer test independently proves that embedded userinfo,
query credentials, and fragments never enter browser-visible URLs.
