---
title: "What's new in fak — recent changes witnessed by commit"
description: "A generated, commit-witnessed summary of recent fak changes: grouped themes, an explicit freshness boundary, and a link that verifies every line."
---

# What's new in fak

> **In one breath:** This page groups recent witnessed fak changes into human-readable themes.
> Each item names its shipped or planned scope and links to proof. It is current through
> commit `9a0fb818e` on 2026-09-01.

Every entry below is one commit that carries a per-leaf ship stamp, so each line is traceable
to repository evidence rather than to a hand-written claim. The page is refreshed by one
committed command, and it summarizes — it is not a complete changelog and it promises no
release cadence.

## What does this page cover?

It covers the 7 days of repository history ending 2026-09-01: **1424 stamped changes** across **1470 non-merge commits**, grouped
into the themes below. The repository release marker at that commit is `0.45.0`.

A stamped change is a commit whose subject carries a `(fak <leaf>)` trailer — the same
witness the commit-message gate and the marketing feeds bind to. Every entry carries the
scope it may be claimed at:

| Scope | What it means | How it is decided |
|---|---|---|
| `shipped` | The change landed on trunk and its subsystem is claimed as shipped. | Stamped commit whose leaf and issue pass the [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) honesty ledger. |
| `research/plan` | A plan, note, or experiment — not a capability you can use. | Every path the commit touched is a research, planning, or notes surface. |
| `landed, not claimed shipped` | Real work landed, but the capability is still `[SIMULATED]`/`[STUB]`. | [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) still tags the commit's leaf or issue as unshipped. |

## How fresh is this page, and how is it refreshed?

The freshness boundary is explicit and machine-checkable — the page carries the commit it
was folded from, so staleness is a computation rather than a judgement call.

| Field | Value |
|---|---|
| Current through | `9a0fb818ed8880e1b010b6d76e0d0bf49ff50024` |
| Commit date | 2026-09-01 |
| Range folded | `2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024` |
| Counted | 1424 stamped changes / 1470 non-merge commits |
| Generator module | `internal/marketing@r28+g3be545f40` |
| Source of truth | `git log` + linked GitHub issues + [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md), folded by the derived generator module above |
| Refresh | `fak marketing whats-new --write` |
| Staleness check | `fak marketing whats-new --check --json` (default ceiling: 7 commit-date days or 250 non-merge commits behind) |

<!-- fak:recent-changes anchor=9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 date=2026-09-01 range=2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 ships=1424 commits=1470 version=0.45.0 generator-module=internal/marketing@r28+g3be545f40 per-theme=4 days=7 generator=fak-marketing-whats-new -->

`fak marketing whats-new --check` reads that anchor, measures how far `HEAD` has moved past it, and
regenerates the page over the recorded range. It exits non-zero when the page is older than
the allowed window or no longer matches what the repository says, so a hand edit or a
rotted snapshot is reported instead of trusted. Nothing on this page is maintained by
hand: the only curated content in the generator is the theme titles and their "why it
matters" lines, which describe subsystems rather than individual changes.

## What changed recently?

Each theme names why its subsystems matter, then lists its most user-facing changes
newest-first. Features and fixes are listed before supporting work, and every theme states
how many changes it counted so the summary never hides its own truncation.

### Micro-agents and micro-context

**Why it matters:** These subsystems decide what a model actually sees: which context is resident, what is paged out, and how a small agent is handed only the slice it needs. Work here changes token cost and answer quality on the same hardware.

**In this window:** 33 stamped changes across `ctxplan`, `headroom`, `trajectory` — 13 feature(s), 14 fix(es), 6 supporting change(s).

- **[`2e6e578f`](https://github.com/anthony-chaudhary/fak/commit/2e6e578f)** 2026-09-01 — feat(trajectory): publish live token destination snapshots — subsystem `trajectory`, scope `shipped`, issue [#9556](https://github.com/anthony-chaudhary/fak/issues/9556)
- **[`71749a5a`](https://github.com/anthony-chaudhary/fak/commit/71749a5a)** 2026-09-01 — feat(trajectory): expose incident burst report — subsystem `trajectory`, scope `shipped`, issue [#10547](https://github.com/anthony-chaudhary/fak/issues/10547)
- **[`fd0e5475`](https://github.com/anthony-chaudhary/fak/commit/fd0e5475)** 2026-08-29 — feat(ctxplan): add retention labels for context pages — subsystem `ctxplan`, scope `shipped`, issue [#9897](https://github.com/anthony-chaudhary/fak/issues/9897)
- **[`24b3696e`](https://github.com/anthony-chaudhary/fak/commit/24b3696e)** 2026-08-28 — feat(headroom): add native compression promotion decision — subsystem `headroom`, scope `shipped`, issue [#9658](https://github.com/anthony-chaudhary/fak/issues/9658)

29 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

### Context filtering and tool integration

**Why it matters:** The gateway, capability index, and tool-process seams are where another agent or client plugs into fak, and where results are screened before they reach the model. Work here changes what integrations are possible and what a tool call is allowed to return.

**In this window:** 26 stamped changes across `capindex`, `gateway`, `mcp` — 10 feature(s), 12 fix(es), 4 supporting change(s).

- **[`aee8c853`](https://github.com/anthony-chaudhary/fak/commit/aee8c853)** 2026-08-30 — feat(gateway): add admission batch budgets — subsystem `gateway`, scope `shipped`
- **[`17668b5c`](https://github.com/anthony-chaudhary/fak/commit/17668b5c)** 2026-08-29 — feat(gateway): buffer split UTF-8 incrementally — subsystem `gateway`, scope `shipped`
- **[`72a2a465`](https://github.com/anthony-chaudhary/fak/commit/72a2a465)** 2026-08-29 — feat(gateway): expose rejected tier observations — subsystem `gateway`, scope `shipped`
- **[`14642877`](https://github.com/anthony-chaudhary/fak/commit/14642877)** 2026-08-29 — feat(capabilities): add performance observability — subsystem `capindex`, scope `shipped`

22 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

### Fleet, dispatch, and account routing

**Why it matters:** These subsystems run many sessions across many machines and accounts: which worker picks up which unit of work, which seat or endpoint serves it, and how a stalled or leaked lease is recovered. Work here changes throughput and operational safety of a fleet.

**In this window:** 48 stamped changes across `accounts`, `dispatchtick`, `dos`, `fleetaccounts`, `launchshim`, `leaseref` and 2 more — 12 feature(s), 18 fix(es), 18 supporting change(s).

- **[`e820d741`](https://github.com/anthony-chaudhary/fak/commit/e820d741)** 2026-09-01 — feat(workerworktree): add lifecycle status surfaces — subsystem `workerworktree`, scope `shipped`, issue [#10551](https://github.com/anthony-chaudhary/fak/issues/10551)
- **[`819b3916`](https://github.com/anthony-chaudhary/fak/commit/819b3916)** 2026-09-01 — feat(workerworktree): expose shared lifecycle collector — subsystem `workerworktree`, scope `shipped`, issue [#10551](https://github.com/anthony-chaudhary/fak/issues/10551)
- **[`6f162653`](https://github.com/anthony-chaudhary/fak/commit/6f162653)** 2026-09-01 — feat(workerworktree): add lifecycle status projection — subsystem `workerworktree`, scope `shipped`, issue [#10551](https://github.com/anthony-chaudhary/fak/issues/10551)
- **[`f50ac083`](https://github.com/anthony-chaudhary/fak/commit/f50ac083)** 2026-09-01 — feat(worktree): publish scrubbed host snapshots — subsystem `workerworktree`, scope `shipped`, issue [#10507](https://github.com/anthony-chaudhary/fak/issues/10507)

44 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

### Model runtime, kernels, and caching

**Why it matters:** The loader, compute kernels, and cache tiers are the hot path: they decide how fast a token is produced and how much of a prior context is reused instead of recomputed. Work here shows up as latency and cost, not as new surface.

**In this window:** 147 stamped changes across `cachevalueledger`, `cachevaluereport`, `compute`, `ggufload`, `metrics`, `model` and 3 more — 55 feature(s), 26 fix(es), 66 supporting change(s).

- **[`cdea03fe`](https://github.com/anthony-chaudhary/fak/commit/cdea03fe)** 2026-09-01 — feat(modver): add historical snapshot support — subsystem `modver`, scope `shipped`
- **[`555b3d34`](https://github.com/anthony-chaudhary/fak/commit/555b3d34)** 2026-08-31 — feat(model): add retained Q4_K MTP execution — subsystem `model`, scope `shipped`, issue [#9985](https://github.com/anthony-chaudhary/fak/issues/9985)
- **[`742d0a74`](https://github.com/anthony-chaudhary/fak/commit/742d0a74)** 2026-08-31 — feat(compute): add exact Qwen3.8 Vulkan decode receipts — subsystem `compute`, scope `shipped`, issue [#10013](https://github.com/anthony-chaudhary/fak/issues/10013)
- **[`d6899ca6`](https://github.com/anthony-chaudhary/fak/commit/d6899ca6)** 2026-08-31 — feat(model): add Qwen3.8 MTP net-cost receipts — subsystem `model`, scope `shipped`, issue [#9991](https://github.com/anthony-chaudhary/fak/issues/9991)

143 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

### Policy, adjudication, and audit

**Why it matters:** Policy and adjudication are the boundary that blocks a poisoned result or a destructive operation, and the audit trail is what proves the boundary held. Work here changes what an agent is allowed to do and what evidence you keep afterwards.

**In this window:** 41 stamped changes across `adjudicator`, `architest`, `guard`, `hooks`, `quality`, `usagelog` — 6 feature(s), 21 fix(es), 14 supporting change(s).

- **[`e038be67`](https://github.com/anthony-chaudhary/fak/commit/e038be67)** 2026-08-30 — feat(usagelog): add lazy structured diagnostics — subsystem `usagelog`, scope `shipped`
- **[`4cd71de6`](https://github.com/anthony-chaudhary/fak/commit/4cd71de6)** 2026-08-29 — feat(quality): add endpoint capability preflight — subsystem `quality`, scope `shipped`, issue [#10030](https://github.com/anthony-chaudhary/fak/issues/10030)
- **[`ed49cd3f`](https://github.com/anthony-chaudhary/fak/commit/ed49cd3f)** 2026-08-29 — feat(quality): add endpoint capability preflight — subsystem `quality`, scope `shipped`, issue [#10030](https://github.com/anthony-chaudhary/fak/issues/10030)
- **[`9d2f2a32`](https://github.com/anthony-chaudhary/fak/commit/9d2f2a32)** 2026-08-28 — feat(adjudicator): add leased dev-path attestation — subsystem `adjudicator`, scope `shipped`, issue [#9850](https://github.com/anthony-chaudhary/fak/issues/9850)

37 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

### Documentation and discoverability

**Why it matters:** These changes maintain the pages, indexes, and machine-readable feeds a reader or answer engine uses to find the right authority. Work here changes what is findable, not what the kernel does.

**In this window:** 261 stamped changes across `devindex`, `docs` — 2 feature(s), 15 fix(es), 244 supporting change(s), 104 research/plan.

- **[`e8add47f`](https://github.com/anthony-chaudhary/fak/commit/e8add47f)** 2026-08-28 — feat(build): add phase-timed build receipts — subsystem `devindex`, scope `shipped`, issue [#9661](https://github.com/anthony-chaudhary/fak/issues/9661)
- **[`936bf3fe`](https://github.com/anthony-chaudhary/fak/commit/936bf3fe)** 2026-08-28 — feat(devindex): add runtime extraction candidate report — subsystem `devindex`, scope `shipped`, issue [#9716](https://github.com/anthony-chaudhary/fak/issues/9716)
- **[`661b7e13`](https://github.com/anthony-chaudhary/fak/commit/661b7e13)** 2026-09-01 — fix(devindex): regenerate the dev-handoff manifest and curate the verb catalog — subsystem `devindex`, scope `shipped`
- **[`a6fa45cd`](https://github.com/anthony-chaudhary/fak/commit/a6fa45cd)** 2026-08-27 — fix(plan): expose ten Mac Qwen units — subsystem `docs`, scope `research/plan`, issue [#9430](https://github.com/anthony-chaudhary/fak/issues/9430)

257 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

### Command line and developer workflow

**Why it matters:** The `fak` verbs and the repository's own tooling are how both humans and automated contributors drive everything above. Work here changes the commands you run, not the behavior they govern.

**In this window:** 240 stamped changes across `agent`, `cmd`, `gitdaily`, `tools` — 64 feature(s), 104 fix(es), 72 supporting change(s).

- **[`4f89bffe`](https://github.com/anthony-chaudhary/fak/commit/4f89bffe)** 2026-09-01 — feat(gitdaily): expose Go cache lifecycle controls — subsystem `gitdaily`, scope `shipped`, issue [#10556](https://github.com/anthony-chaudhary/fak/issues/10556)
- **[`4af04504`](https://github.com/anthony-chaudhary/fak/commit/4af04504)** 2026-09-01 — feat(flowmetrics): enforce overload at native admission — subsystem `cmd`, scope `shipped`, issue [#10416](https://github.com/anthony-chaudhary/fak/issues/10416)
- **[`ae4c1752`](https://github.com/anthony-chaudhary/fak/commit/ae4c1752)** 2026-09-01 — feat(worktree): expose worker intent in lifecycle list — subsystem `cmd`, scope `shipped`, issue [#10528](https://github.com/anthony-chaudhary/fak/issues/10528)
- **[`a0bb0df4`](https://github.com/anthony-chaudhary/fak/commit/a0bb0df4)** 2026-08-31 — feat(cmd): expose run-first Grafana operations dashboards — subsystem `cmd`, scope `shipped`, issue [#10361](https://github.com/anthony-chaudhary/fak/issues/10361)

236 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

### Other maintained subsystems

**Why it matters:** These are witnessed changes in subsystems no theme above claims yet. They are listed rather than dropped so the totals on this page reconcile with git history; the theme table in `internal/marketing/recentchanges.go` is where a leaf graduates into a named theme.

**In this window:** 628 stamped changes across `abi`, `affectedtests`, `agentic`, `agenticbench`, `agenticbenchcoverage`, `agentquery` and 209 more — 250 feature(s), 176 fix(es), 202 supporting change(s), 14 research/plan.

- **[`9d62dc1d`](https://github.com/anthony-chaudhary/fak/commit/9d62dc1d)** 2026-09-01 — feat(extensions): add fault isolation spine — subsystem `extensionfault`, scope `shipped`, issue [#10428](https://github.com/anthony-chaudhary/fak/issues/10428)
- **[`b5a3a229`](https://github.com/anthony-chaudhary/fak/commit/b5a3a229)** 2026-09-01 — feat(examples): add portable replay fixture — subsystem `examples`, scope `shipped`, issue [#6804](https://github.com/anthony-chaudhary/fak/issues/6804)
- **[`17dab75f`](https://github.com/anthony-chaudhary/fak/commit/17dab75f)** 2026-09-01 — feat(examples): add embedded custom harness — subsystem `examples`, scope `shipped`, issue [#10563](https://github.com/anthony-chaudhary/fak/issues/10563)
- **[`bd817040`](https://github.com/anthony-chaudhary/fak/commit/bd817040)** 2026-08-27 — fix(plan-audit): route project skill to native reconciler — subsystem `claude`, scope `research/plan`, issue [#9328](https://github.com/anthony-chaudhary/fak/issues/9328)

624 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak "
```

## Where can I verify each item?

Every line above is a commit sha, and the sha is the verification. From a checkout:

```bash
git show <sha>                                   # the exact diff behind any line above
git log --no-merges --oneline 2523ce149c36a3cf5c9f8338ddc743090f97c8bc..9a0fb818ed8880e1b010b6d76e0d0bf49ff50024 --grep "(fak <leaf>)"   # every counted change in one subsystem
fak version modules --only internal/marketing    # current derived module@rev identity
fak marketing whats-new --check --json                # this page's own freshness, as JSON
```

Each entry also links its commit on GitHub and, when the subject names one, the issue that
scoped it. Scope labels come from [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md), which records what is shipped,
simulated, or stubbed; tagged release notes live in the
[release-notes index](https://github.com/anthony-chaudhary/fak/blob/main/docs/releases/README.md). The generator identity above uses the repository's derived
[`module@rev` contract](notes/VERSION-EVERYTHING-SPINE-2026-07-03.md), pinned to the page's
anchor rather than borrowed from whatever binary performs the refresh.

## What this page does not claim

- **No release cadence.** A window with many changes and a window with few are both honest;
  nothing here promises when the next tagged release lands.
- **No unwitnessed claims.** A commit without a per-leaf ship stamp is counted in the
  non-merge commit total but is never listed as a change, and a capability still tagged
  `[SIMULATED]`/`[STUB]` in `CLAIMS.md` is downgraded rather than advertised.
- **Not a complete changelog.** Each theme shows a bounded slice and states how many more it
  counted; the `git log` recipe above is the complete list.
- **Not a concept explanation.** The themes name subsystems and route onward; they do not
  re-explain what the kernel is.

## Where to go next

This page is a recent-changes view, not a general entry point. For anything other than
"what changed lately", use the authority that owns it:

- **Tagged releases and upgrade notes:** the [release-notes index](https://github.com/anthony-chaudhary/fak/blob/main/docs/releases/README.md).
- **What is shipped, simulated, or stubbed:** [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md).
- **Plans and research, before they ship:** [`docs/research/`](research/README.md).
- **The machine-readable version of this feed:** [`docs/marketing/`](marketing/README.md),
  which publishes the same witnessed ships as JSON-LD and plain text for answer engines.
