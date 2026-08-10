---
title: "What's new in fak — recent changes witnessed by commit"
description: "A generated, commit-witnessed summary of recent fak changes: grouped themes, an explicit freshness boundary, and a link that verifies every line."
---

# What's new in fak

> **In one breath:** This page groups recent witnessed fak changes into human-readable themes.
> Each item names its shipped or planned scope and links to proof. It is current through
> commit `6fc671b23` on 2026-08-09.

Every entry below is one commit that carries a per-leaf ship stamp, so each line is traceable
to repository evidence rather than to a hand-written claim. The page is refreshed by one
committed command, and it summarizes — it is not a complete changelog and it promises no
release cadence.

## What does this page cover?

It covers the 7 days of repository history ending 2026-08-09: **745 stamped changes** across **746 non-merge commits**, grouped
into the themes below. The repository release marker at that commit is `0.43.0`.

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
| Current through | `6fc671b2346cddbd97522f0934d54593bbd7751c` |
| Commit date | 2026-08-09 |
| Range folded | `32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c` |
| Counted | 745 stamped changes / 746 non-merge commits |
| Generator module | `internal/marketing@r22+g3b57441796` |
| Source of truth | `git log` + linked GitHub issues + [`CLAIMS.md`](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md), folded by the derived generator module above |
| Refresh | `fak marketing whats-new --write` |
| Staleness check | `fak marketing whats-new --check --json` (default ceiling: 7 commit-date days or 250 non-merge commits behind) |

<!-- fak:recent-changes anchor=6fc671b2346cddbd97522f0934d54593bbd7751c date=2026-08-09 range=32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c ships=745 commits=746 version=0.43.0 generator-module=internal/marketing@r22+g3b57441796 per-theme=4 days=7 generator=fak-marketing-whats-new -->

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

**In this window:** 39 stamped changes across `ctxmmu`, `ctxplan`, `ctxresidency`, `headroom`, `memq`, `microagent` and 1 more — 28 feature(s), 7 fix(es), 4 supporting change(s), 9 landed but not claimed shipped.

- **[`b4efa463`](https://github.com/anthony-chaudhary/fak/commit/b4efa463)** 2026-08-09 — feat(headroom): add same-corpus compressor comparison — subsystem `headroom`, scope `shipped`, issue [#6064](https://github.com/anthony-chaudhary/fak/issues/6064)
- **[`3e8b1c75`](https://github.com/anthony-chaudhary/fak/commit/3e8b1c75)** 2026-08-09 — feat(microcontextdemo): implement large-input operator spine — subsystem `microcontextdemo`, scope `shipped`, issue [#6029](https://github.com/anthony-chaudhary/fak/issues/6029)
- **[`77ef8ac7`](https://github.com/anthony-chaudhary/fak/commit/77ef8ac7)** 2026-08-08 — feat(microcontextdemo): add controlled 10k soak — subsystem `microcontextdemo`, scope `shipped`, issue [#5792](https://github.com/anthony-chaudhary/fak/issues/5792)
- **[`d6d6f313`](https://github.com/anthony-chaudhary/fak/commit/d6d6f313)** 2026-08-08 — feat(microagent): bound the budget park queue and shed with backpressure — subsystem `microagent`, scope `landed, not claimed shipped`, issue [#2021](https://github.com/anthony-chaudhary/fak/issues/2021)

35 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

### Context filtering and tool integration

**Why it matters:** The gateway, capability index, and tool-process seams are where another agent or client plugs into fak, and where results are screened before they reach the model. Work here changes what integrations are possible and what a tool call is allowed to return.

**In this window:** 36 stamped changes across `capindex`, `capindexgw`, `egressfloor`, `gateway`, `promptlint`, `toolproc` — 17 feature(s), 10 fix(es), 9 supporting change(s).

- **[`99381b26`](https://github.com/anthony-chaudhary/fak/commit/99381b26)** 2026-08-09 — feat(promptlint): add counted one-breath doc gate — subsystem `promptlint`, scope `shipped`, issue [#5939](https://github.com/anthony-chaudhary/fak/issues/5939)
- **[`7b0b804c`](https://github.com/anthony-chaudhary/fak/commit/7b0b804c)** 2026-08-08 — feat(egressfloor): implement poisoned result quarantine — subsystem `egressfloor`, scope `shipped`, issue [#2844](https://github.com/anthony-chaudhary/fak/issues/2844)
- **[`22d498fb`](https://github.com/anthony-chaudhary/fak/commit/22d498fb)** 2026-08-08 — feat(gateway): add roster models to catalog — subsystem `gateway`, scope `shipped`, issue [#5634](https://github.com/anthony-chaudhary/fak/issues/5634)
- **[`83f3d850`](https://github.com/anthony-chaudhary/fak/commit/83f3d850)** 2026-08-08 — feat(capindexgw): refuse catalog-reachable core tools — subsystem `capindexgw`, scope `shipped`, issue [#2926](https://github.com/anthony-chaudhary/fak/issues/2926)

32 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

### Fleet, dispatch, and account routing

**Why it matters:** These subsystems run many sessions across many machines and accounts: which worker picks up which unit of work, which seat or endpoint serves it, and how a stalled or leaked lease is recovered. Work here changes throughput and operational safety of a fleet.

**In this window:** 72 stamped changes across `accounts`, `dgxbridge`, `dispatch`, `dispatchtick`, `dos`, `fleetaccounts` and 8 more — 24 feature(s), 27 fix(es), 21 supporting change(s).

- **[`21212a28`](https://github.com/anthony-chaudhary/fak/commit/21212a28)** 2026-08-09 — feat(sessionaudit): add cold-tool-defer refault audit — subsystem `sessionaudit`, scope `shipped`, issue [#3625](https://github.com/anthony-chaudhary/fak/issues/3625)
- **[`e8f9a805`](https://github.com/anthony-chaudhary/fak/commit/e8f9a805)** 2026-08-09 — feat(workerworktree): add detached land-ready queue — subsystem `workerworktree`, scope `shipped`, issue [#5994](https://github.com/anthony-chaudhary/fak/issues/5994)
- **[`c9de72e7`](https://github.com/anthony-chaudhary/fak/commit/c9de72e7)** 2026-08-08 — feat(workerworktree): add remote recovery mirrors — subsystem `workerworktree`, scope `shipped`, issue [#5997](https://github.com/anthony-chaudhary/fak/issues/5997)
- **[`7da041ff`](https://github.com/anthony-chaudhary/fak/commit/7da041ff)** 2026-08-08 — feat(stallscan): add CPU saturation and spinner attribution — subsystem `stallscan`, scope `shipped`

68 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

### Model runtime, kernels, and caching

**Why it matters:** The loader, compute kernels, and cache tiers are the hot path: they decide how fast a token is produced and how much of a prior context is reused instead of recomputed. Work here shows up as latency and cost, not as new surface.

**In this window:** 63 stamped changes across `bench`, `cachemeta`, `cachevaluereport`, `compute`, `ggufload`, `livecodebench` and 5 more — 38 feature(s), 13 fix(es), 12 supporting change(s).

- **[`8d430ad1`](https://github.com/anthony-chaudhary/fak/commit/8d430ad1)** 2026-08-09 — feat(nativebench): expose comparison obligations — subsystem `nativebench`, scope `shipped`, issue [#6036](https://github.com/anthony-chaudhary/fak/issues/6036)
- **[`a48b7eec`](https://github.com/anthony-chaudhary/fak/commit/a48b7eec)** 2026-08-09 — feat(metrics): add SOTA parity performance dashboard model — subsystem `metrics`, scope `shipped`, issue [#196](https://github.com/anthony-chaudhary/fak/issues/196)
- **[`9cf1ebb1`](https://github.com/anthony-chaudhary/fak/commit/9cf1ebb1)** 2026-08-08 — feat(cachevaluereport): add recall injection debit — subsystem `cachevaluereport`, scope `shipped`, issue [#5975](https://github.com/anthony-chaudhary/fak/issues/5975)
- **[`3920eb49`](https://github.com/anthony-chaudhary/fak/commit/3920eb49)** 2026-08-08 — feat(cachevalue): implement Track-2 fidelity SLO gate — subsystem `cachevaluereport`, scope `shipped`, issue [#3644](https://github.com/anthony-chaudhary/fak/issues/3644)

59 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

### Policy, adjudication, and audit

**Why it matters:** Policy and adjudication are the boundary that blocks a poisoned result or a destructive operation, and the audit trail is what proves the boundary held. Work here changes what an agent is allowed to do and what evidence you keep afterwards.

**In this window:** 62 stamped changes across `adjudicator`, `architest`, `guard`, `hooks`, `journal`, `policy` and 4 more — 22 feature(s), 20 fix(es), 20 supporting change(s).

- **[`21e3edf9`](https://github.com/anthony-chaudhary/fak/commit/21e3edf9)** 2026-08-09 — feat(architest): align new-leaf with composite tiers — subsystem `architest`, scope `shipped`, issue [#4042](https://github.com/anthony-chaudhary/fak/issues/4042)
- **[`8bd4a824`](https://github.com/anthony-chaudhary/fak/commit/8bd4a824)** 2026-08-08 — feat(policy): add auditable suppressions — subsystem `policy`, scope `shipped`, issue [#5681](https://github.com/anthony-chaudhary/fak/issues/5681)
- **[`72be81b0`](https://github.com/anthony-chaudhary/fak/commit/72be81b0)** 2026-08-08 — feat(adjudicator): scope action rules to sink-bearing arguments — subsystem `adjudicator`, scope `shipped`
- **[`38949a1d`](https://github.com/anthony-chaudhary/fak/commit/38949a1d)** 2026-08-08 — feat(adjudicator): declare supplemental inline eval specs — subsystem `adjudicator`, scope `shipped`, issue [#3963](https://github.com/anthony-chaudhary/fak/issues/3963)

58 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

### Documentation and discoverability

**Why it matters:** These changes maintain the pages, indexes, and machine-readable feeds a reader or answer engine uses to find the right authority. Work here changes what is findable, not what the kernel does.

**In this window:** 107 stamped changes across `devindex`, `docs`, `ideascout` — 6 feature(s), 15 fix(es), 86 supporting change(s), 6 research/plan.

- **[`6b6eccf0`](https://github.com/anthony-chaudhary/fak/commit/6b6eccf0)** 2026-08-09 — feat(devindex): add source-derived verb refusal surface — subsystem `devindex`, scope `shipped`, issue [#5934](https://github.com/anthony-chaudhary/fak/issues/5934)
- **[`bc845a32`](https://github.com/anthony-chaudhary/fak/commit/bc845a32)** 2026-08-08 — feat(ideascout): add GitHub repository substance gate — subsystem `ideascout`, scope `shipped`
- **[`4ffecdc4`](https://github.com/anthony-chaudhary/fak/commit/4ffecdc4)** 2026-08-07 — feat(devindex): add the fak index execaudit verb and tree witness — subsystem `devindex`, scope `shipped`, issue [#5648](https://github.com/anthony-chaudhary/fak/issues/5648)
- **[`bb523ff6`](https://github.com/anthony-chaudhary/fak/commit/bb523ff6)** 2026-08-06 — fix(docs): correct the claim-space note's compute-admission gap rungs — subsystem `docs`, scope `research/plan`, issue [#3269](https://github.com/anthony-chaudhary/fak/issues/3269)

103 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

### Command line and developer workflow

**Why it matters:** The `fak` verbs and the repository's own tooling are how both humans and automated contributors drive everything above. Work here changes the commands you run, not the behavior they govern.

**In this window:** 205 stamped changes across `agent`, `cmd`, `gitdaily`, `tools`, `wipref` — 73 feature(s), 98 fix(es), 34 supporting change(s).

- **[`9c1c132e`](https://github.com/anthony-chaudhary/fak/commit/9c1c132e)** 2026-08-09 — feat(info): show guard fleet in default TUI — subsystem `cmd`, scope `shipped`, issue [#6060](https://github.com/anthony-chaudhary/fak/issues/6060)
- **[`a3c13719`](https://github.com/anthony-chaudhary/fak/commit/a3c13719)** 2026-08-09 — feat(agent): route delegated work through spawn placement — subsystem `agent`, scope `shipped`, issue [#5420](https://github.com/anthony-chaudhary/fak/issues/5420)
- **[`28a88a31`](https://github.com/anthony-chaudhary/fak/commit/28a88a31)** 2026-08-09 — feat(wipref): add remote checkpoint containment drain — subsystem `wipref`, scope `shipped`, issue [#5556](https://github.com/anthony-chaudhary/fak/issues/5556)
- **[`6ed7eb24`](https://github.com/anthony-chaudhary/fak/commit/6ed7eb24)** 2026-08-08 — feat(gitdaily): add calendar-day coverage — subsystem `cmd`, scope `shipped`, issue [#5987](https://github.com/anthony-chaudhary/fak/issues/5987)

201 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

### Other maintained subsystems

**Why it matters:** These are witnessed changes in subsystems no theme above claims yet. They are listed rather than dropped so the totals on this page reconcile with git history; the theme table in `internal/marketing/recentchanges.go` is where a leaf graduates into a named theme.

**In this window:** 161 stamped changes across `a2achan`, `ablate`, `accountobs`, `agentdojo`, `agentsindex`, `apihostprobe` and 104 more — 98 feature(s), 40 fix(es), 23 supporting change(s), 8 research/plan.

- **[`4eec7251`](https://github.com/anthony-chaudhary/fak/commit/4eec7251)** 2026-08-09 — feat(config): expose every fak.toml opinion disposition — subsystem `deploymanifest`, scope `shipped`, issue [#6069](https://github.com/anthony-chaudhary/fak/issues/6069)
- **[`24eba4c3`](https://github.com/anthony-chaudhary/fak/commit/24eba4c3)** 2026-08-09 — feat(archreport): expose reverse architecture edges — subsystem `archreport`, scope `shipped`, issue [#6080](https://github.com/anthony-chaudhary/fak/issues/6080)
- **[`1bf94748`](https://github.com/anthony-chaudhary/fak/commit/1bf94748)** 2026-08-09 — feat(toolshape): add exhaustive result contracts — subsystem `toolshape`, scope `shipped`, issue [#5661](https://github.com/anthony-chaudhary/fak/issues/5661)
- **[`a8c13fdf`](https://github.com/anthony-chaudhary/fak/commit/a8c13fdf)** 2026-08-08 — feat(experiments): implement measured negframe lever arms — subsystem `experiments`, scope `research/plan`, issue [#5851](https://github.com/anthony-chaudhary/fak/issues/5851)

157 further change(s) in this theme are counted but not listed. To read the complete stamped history for this range and filter by the subsystems above:
```bash
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak "
```

## Where can I verify each item?

Every line above is a commit sha, and the sha is the verification. From a checkout:

```bash
git show <sha>                                   # the exact diff behind any line above
git log --no-merges --oneline 32c71d74cbee5d2703b51e133157d1ddd18ee9a9..6fc671b2346cddbd97522f0934d54593bbd7751c --grep "(fak <leaf>)"   # every counted change in one subsystem
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
