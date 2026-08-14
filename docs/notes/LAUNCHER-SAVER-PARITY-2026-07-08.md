---
title: "Token-savers stay on by default across every launch surface — headless and ultracode too (2026-07-08)"
description: "The token-defaults scorecard locks the on-by-default savers for the interactive fak manage / fak serve front doors. This note extends that guarantee to the launch surfaces the fleet's own automated sessions use — the headless dispatch worker, the account-switch / ultracode launch, and the codex launcher — with a behavioral regression lock proving none of them strip a saver."
---

# The savers stay on for our own automated sessions, not just a human at the keyboard

> **`fak token-defaults-scorecard` proves the token-savers are ON by default — but only for the
> two front doors it reads (`cmd/fak/guard.go`, `cmd/fak/serve.go`).** The fleet's own automated
> sessions don't reach guard through those front doors: each assembles its OWN `fak manage … --`
> argv. This note closes that gap with a behavioral invariant + regression lock so "on by default"
> holds for **headless dispatch and ultracode sessions** too, not just an interactive launch.

## The gap

The [token-defaults scorecard](../serving/token-defaults-scorecard.md) derives each saver's
on/off state from the entrypoint source and locks it against regression
(`cmd/fak/token_defaults.go` → `TestTokenDefaultsSnapshotFresh`). Its six default-on levers are
`provider_cache`, `toolfloor`, `vdso` (lossless) and `compacthistory`, `elideresult`, `ctxview`
(bounded). Every check reads `guard.go` / `serve.go` — the **interactive** front doors.

But the sessions that matter most for "self usage" launch guard a different way, each building its
own guard argv:

| Surface | Builder | What it is |
|---|---|---|
| **Headless dispatch worker** | `dispatchtick.GuardedLaunchCommand` → `guardedDispatchCommand` (`cmd/fak/dispatch_tick.go`) | the unattended fleet turn — `fak manage -- claude -p …` |
| **Account-switch / ultracode launch** | `buildLaunchArgv` (`cmd/fak/accounts_launch.go`), `ultracode:true` ⇒ `--settings '{"ultracode":true}'` | the `f` shortcut / `fak accounts launch` — workflow mode |
| **Codex launcher** | `buildCodexLaunchArgv` (`cmd/fak/codex_launcher.go`) | `fak codex` — guarded Codex |

All three front the *same* guard binary and inject **no** saver-disabling flag, so today they
inherit guard's full default-on stack for free. Nothing pinned that. A future edit that spliced
`--ctx-view-budget 0` or `--vdso=false` into any of these builders would silently strip a saver
from every headless/ultracode session **while the front-door scorecard stayed green** — the
savers would still be "on by default" for a human, and quietly off for the fleet.

## The lock

`guardArgvDisabledSavers` (`cmd/fak/launcher_saver_parity.go`) is the missing invariant: given a
guard-fronting argv it scans guard's own flag segment (the tokens before the first standalone
`--`, exactly what guard parses) and returns which on-by-default savers the argv would turn OFF —
a budget flag (`--compact-history-budget` / `--elide-result-bytes` / `--ctx-view-budget`) set to
`<= 0`, or `--vdso=<falsey>` / `--no-vdso`. It is pure, so every launcher's real output is checked
without spawning anything.

- **`TestLaunchSurfacesPreserveDefaultSavers`** runs each launch surface's real builder — including
  the ultracode launch and the managed-cache-posture-spliced headless worker — and asserts
  `guardArgvDisabledSavers` is empty. If a launcher edit strips a saver, this fails while the
  scorecard passes.
- **`TestGuardArgvDisabledSaversDetects`** proves the detector actually fires: a saver-disabling
  override in every flag form guard accepts is caught, and the boundaries it must *not* trip on
  (a positive budget, a bare `--vdso`, a `--ctx-view-budget 0` that appears *after* `--` and so
  belongs to the agent, not guard) read clean. A lock whose detector never triggers is no lock.

`debug-stats` is deliberately excluded: it is the observable per-turn cache/token layer,
legitimately silenced by `--quiet` on a headless worker, and silencing it costs zero tokens — it
is observability, not a saver.

## Honest scope

This locks that **no launch surface disables a saver guard turns on** — the savers reach every
automated session. It does not touch the **managed-cache** (1h-TTL prompt-cache upgrade) posture,
which stays `auto` (passive) on a subscription-OAuth seat by deliberate design
([`MANAGED-CACHE-PROVING-GROUND-2026-07-03.md`](MANAGED-CACHE-PROVING-GROUND-2026-07-03.md)) and
is carried, not forced, by every launcher's
`--managed-cache` flag (`$FAK_MANAGED_CACHE` ⇒ `on` to activate). Sibling default-on witnesses:
[`CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md`](CTXVIEW-DEFAULT-ON-WITNESS-2026-06-28.md),
[`COMPRESS-DEFAULT-ON-WITNESS-2026-07-06.md`](COMPRESS-DEFAULT-ON-WITNESS-2026-07-06.md).
