---
title: "fak maturity scorecard — where each capability sits on its lifecycle ladder"
description: "Every declared fak capability (one per internal/<leaf> lane) placed on a closed lifecycle ladder — proposed → prototyped → tested → dogfooded → default, with a benchmarked badge — and the next work item that would mature it. Immaturity is not a defect; a ladder-skip (fak relies on a capability yet leaves it untested) is. Re-derived from dos.toml + the tree's import graph + the CLI reference."
---

# fak maturity scorecard — lifecycle, not just completeness

**maturity_debt (ladder-skips): 0**; maturity index **79/100 (C)** over **112** declared capabilities (35 carry the benchmarked badge).

> maturity: 112 capabilities, index 79/100 (C); no ladder-skips (every capability is at most as mature as its evidence); 1 proposed · 0 prototyped · 17 tested · 58 dogfooded · 36 default

A v1 prototype can be legitimately *complete* and still not be tested, dogfooded, benchmarked, or the default. This scorecard makes that lifecycle visible: it places every declared capability (one per `internal/<leaf>` lane in [`dos.toml`](../dos.toml) `[lanes.trees]`) on a closed ladder, and for each one names the next step that would mature it. Every rung is gated by evidence the author did not write — code on disk, a `*_test.go`, an edge in the running binary's transitive import graph (fak itself runs it), a documented verb — so the only way up the ladder is to change the real tree.

**Immaturity is not a defect.** A capability honestly at `prototyped` is a complete v1 that simply has not been matured yet — expected, and never counted against anyone. The one defect this refuses is a **ladder-skip**: a capability that looks more mature than its evidence — concretely, one fak relies on (dogfooded, a default surface, or benchmarked) yet leaves untested. That is the maturity sibling of the product scorecard's verdict-overclaim and the readiness ladder's `READINESS_OVERCLAIM` ([#582](https://github.com/anthony-chaudhary/fak/issues/582) / grammar G1).

## The lifecycle ladder

| # | Rung | Reached when (evidence the author did not write) |
|---|---|---|
| 0 | `proposed` | a declared capability with no code on disk yet |
| 1 | `prototyped` | a non-test `.go` file exists in the leaf — a complete v1 |
| 2 | `tested` | the leaf carries a `*_test.go` (the QA rung) |
| 3 | `dogfooded` | the leaf is on the running binary's transitive import graph — **fak itself runs it** |
| 4 | `default` | the capability is a documented `fak` verb (`docs/cli-reference.md`) — the default surface |
| · | `benchmarked` (badge) | a `func Benchmark*` in the leaf or a `BENCHMARK-AUTHORITY.md` row — the natural step after `default` |

## Distribution

| Rung | Capabilities |
|---|---|
| `proposed` | 1 |
| `prototyped` | 0 |
| `tested` | 17 |
| `dogfooded` | 58 |
| `default` | 36 |
| `benchmarked` (badge) | 35 |

## Next work — the agentic-culture backlog

Each gap is a concrete, checkable next work item. `fak maturity next` is the queue an agent (or the issue-dispatch loop) pulls from to advance the fleet one rung at a time. Ladder-skips first (they are the real debt), then the least-mature capabilities (the most leverage).

| | From → gap | Next work item | Witness |
|---|---|---|---|
|  | `proposed → prototyped` | prototype dgxbridge: land a v1 in internal/dgxbridge | a non-test .go file exists under internal/dgxbridge |
|  | `tested → dogfooded` | dogfood advmodel: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/advmodel imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood architest: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/architest imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood boundarylint: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/boundarylint imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood capindexgw: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/capindexgw imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood cohort: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/cohort imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood ctxresidency: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/ctxresidency imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood fakrpc: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/fakrpc imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood leakcheck: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/leakcheck imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood opttarget: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/opttarget imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood pathlint: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/pathlint imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood residency: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/residency imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood sharedtask: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/sharedtask imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood spec: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/spec imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood tracesink: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/tracesink imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood urllint: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/urllint imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood vcachewarm: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/vcachewarm imported by cmd/, internal/registrations, or internal/kernel |
|  | `tested → dogfooded` | dogfood windowgate: wire it onto the running binary's path so fak itself runs it | github.com/anthony-chaudhary/fak/internal/windowgate imported by cmd/, internal/registrations, or internal/kernel |
|  | `dogfooded → default` | default a2achan: promote it to a documented default surface (a fak verb) | a2achan documented in docs/cli-reference.md |
|  | `dogfooded → default` | default agenttopo: promote it to a documented default surface (a fak verb) | agenttopo documented in docs/cli-reference.md |
|  | `dogfooded → default` | default ailuminate: promote it to a documented default surface (a fak verb) | ailuminate documented in docs/cli-reference.md |
|  | `dogfooded → default` | default answershape: promote it to a documented default surface (a fak verb) | answershape documented in docs/cli-reference.md |
|  | `dogfooded → default` | default appversion: promote it to a documented default surface (a fak verb) | appversion documented in docs/cli-reference.md |
|  | `dogfooded → default` | default blob: promote it to a documented default surface (a fak verb) | blob documented in docs/cli-reference.md |
|  | `dogfooded → default` | default blobfs: promote it to a documented default surface (a fak verb) | blobfs documented in docs/cli-reference.md |
|  | `dogfooded → default` | default blobhttp: promote it to a documented default surface (a fak verb) | blobhttp documented in docs/cli-reference.md |
|  | `dogfooded → default` | default cachemeta: promote it to a documented default surface (a fak verb) | cachemeta documented in docs/cli-reference.md |
|  | `dogfooded → default` | default capindex: promote it to a documented default surface (a fak verb) | capindex documented in docs/cli-reference.md |
|  | `dogfooded → default` | default comm: promote it to a documented default surface (a fak verb) | comm documented in docs/cli-reference.md |
|  | `dogfooded → default` | default contextq: promote it to a documented default surface (a fak verb) | contextq documented in docs/cli-reference.md |
| | | _… and 59 more (run `fak maturity next`)_ | |

## Run it

```bash
go run ./cmd/fak maturity              # the lifecycle scorecard
go run ./cmd/fak maturity next         # the next-work backlog (ladder-skips first)
go run ./cmd/fak maturity route        # plan deduped public GitHub issues for the top public rows
go run ./cmd/fak maturity route --live # create/update public-routeable issues so dispatch can drain them
go run ./cmd/fak maturity --markdown    # regenerate this doc
go run ./cmd/fak maturity --json        # machine payload (control-pane / dispatch loop)
go test ./internal/maturity/...        # prove the ladder + skip detection + next-work fold
```

`maturity route` keeps private-boundary lanes visible in `maturity next`, but reports them as skipped instead of filing public issues.

**Next:** advance the fleet one rung: `fak maturity next` lists 89 next work item(s); the least-mature capability is the most leverage
