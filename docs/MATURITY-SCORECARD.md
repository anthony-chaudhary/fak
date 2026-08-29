---
title: "fak maturity scorecard — where each capability sits on its lifecycle ladder"
description: "Every declared fak capability (one per internal/<leaf> lane) placed on a closed lifecycle ladder — proposed → prototyped → tested → dogfooded → default, with a benchmarked badge — and the next work item that would mature it. Immaturity is not a defect; a ladder-skip (fak relies on a capability yet leaves it untested) is. Re-derived from dos.toml + the tree's import graph + the CLI reference."
---

# fak maturity scorecard — lifecycle, not just completeness

**maturity_debt (ladder-skips): 0**; maturity index **50/100 (F)** over **583** declared capabilities (102 carry the benchmarked badge).

> maturity: 583 capabilities, index 50/100 (F); no ladder-skips (every capability is at most as mature as its evidence); 2 proposed · 1 prototyped · 575 tested · 5 dogfooded · 0 default

A v1 prototype can be legitimately *complete* and still not be tested, dogfooded, benchmarked, or the default. This scorecard makes that lifecycle visible: it places every declared capability (one per `internal/<leaf>` lane in [`dos.toml`](../dos.toml) `[lanes.trees]`) on a closed ladder, and for each one names the next step that would mature it. Every rung is gated by evidence the author did not write — code on disk, a `*_test.go`, an edge in the running binary's transitive import graph (fak itself runs it), a documented verb — so the only way up the ladder is to change the real tree.

**Immaturity is not a defect.** A capability honestly at `prototyped` is a complete v1 that simply has not been matured yet — expected, and never counted against anyone. The one defect this refuses is a **ladder-skip**: a capability that looks more mature than its evidence — concretely, one fak relies on (dogfooded, a default surface, or benchmarked) yet leaves untested. That is the maturity sibling of the product scorecard's verdict-overclaim and the readiness ladder's `READINESS_OVERCLAIM` ([#582](https://github.com/anthony-chaudhary/fak/issues/582) / grammar G1).

## The lifecycle ladder

| # | Rung | Reached when (evidence the author did not write) |
|---|---|---|
| 0 | `proposed` | a declared capability with no code on disk yet |
| 1 | `prototyped` | a non-test `.go` file exists in the leaf — a complete v1 |
| 2 | `tested` | the leaf carries a `*_test.go` (the QA rung) |
| 3 | `dogfooded` | the leaf is on the running binary's transitive import graph — **fak itself runs it** |
| 4 | `default` | a passing runtime proof declares the capability active without an opt-in action (`default_on=true`) |
| · | `benchmarked` (badge) | a `func Benchmark*` in the leaf or a `BENCHMARK-AUTHORITY.md` row — the natural step after `default` |

## Distribution

| Rung | Capabilities |
|---|---|
| `proposed` | 2 |
| `prototyped` | 1 |
| `tested` | 575 |
| `dogfooded` | 5 |
| `default` | 0 |
| `benchmarked` (badge) | 102 |

## Next work — the agentic-culture backlog

Each gap is a concrete, checkable next work item. `fak maturity next` is the queue an agent (or the issue-dispatch loop) pulls from to advance the fleet one rung at a time. Ladder-skips first (they are the real debt), then the least-mature capabilities (the most leverage).

| | From → gap | Next work item | Witness |
|---|---|---|---|
|  | `proposed → prototyped` | prototype framebus: land a v1 in internal/framebus | a non-test .go file exists under internal/framebus |
|  | `proposed → prototyped` | prototype kimik3page: land a v1 in internal/kimik3page | a non-test .go file exists under internal/kimik3page |
|  | `prototyped → tested` | test xprobe: add unit tests covering internal/xprobe | a *_test.go in internal/xprobe (go test ./internal/xprobe/... passes) |
|  | `tested → dogfooded` | dogfood a2achan: exercise a real runtime path and record its passing command | a passing runtime command recorded for a2achan in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood abi: exercise a real runtime path and record its passing command | a passing runtime command recorded for abi in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood accountobs: exercise a real runtime path and record its passing command | a passing runtime command recorded for accountobs in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood accountprobe: exercise a real runtime path and record its passing command | a passing runtime command recorded for accountprobe in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood accounts: exercise a real runtime path and record its passing command | a passing runtime command recorded for accounts in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood adjudicator: exercise a real runtime path and record its passing command | a passing runtime command recorded for adjudicator in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood advmodel: exercise a real runtime path and record its passing command | a passing runtime command recorded for advmodel in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood affectedtests: exercise a real runtime path and record its passing command | a passing runtime command recorded for affectedtests in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agent: exercise a real runtime path and record its passing command | a passing runtime command recorded for agent in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agentdemo: exercise a real runtime path and record its passing command | a passing runtime command recorded for agentdemo in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agentdojo: exercise a real runtime path and record its passing command | a passing runtime command recorded for agentdojo in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agenticbench: exercise a real runtime path and record its passing command | a passing runtime command recorded for agenticbench in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agentreadinessscore: exercise a real runtime path and record its passing command | a passing runtime command recorded for agentreadinessscore in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agentsindex: exercise a real runtime path and record its passing command | a passing runtime command recorded for agentsindex in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agenttest: exercise a real runtime path and record its passing command | a passing runtime command recorded for agenttest in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood agenttopo: exercise a real runtime path and record its passing command | a passing runtime command recorded for agenttopo in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood ailuminate: exercise a real runtime path and record its passing command | a passing runtime command recorded for ailuminate in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood amdgpu: exercise a real runtime path and record its passing command | a passing runtime command recorded for amdgpu in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood answershape: exercise a real runtime path and record its passing command | a passing runtime command recorded for answershape in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood antipattern: exercise a real runtime path and record its passing command | a passing runtime command recorded for antipattern in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood apihostprobe: exercise a real runtime path and record its passing command | a passing runtime command recorded for apihostprobe in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood appversion: exercise a real runtime path and record its passing command | a passing runtime command recorded for appversion in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood architest: exercise a real runtime path and record its passing command | a passing runtime command recorded for architest in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood archreport: exercise a real runtime path and record its passing command | a passing runtime command recorded for archreport in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood assumecheck: exercise a real runtime path and record its passing command | a passing runtime command recorded for assumecheck in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood astquery: exercise a real runtime path and record its passing command | a passing runtime command recorded for astquery in internal/maturity/runtime-proofs.json |
|  | `tested → dogfooded` | dogfood atif: exercise a real runtime path and record its passing command | a passing runtime command recorded for atif in internal/maturity/runtime-proofs.json |
| | | _… and 553 more (run `fak maturity next`)_ | |

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

`maturity route` keeps private-boundary lanes visible in `maturity next`, but reports them as skipped instead of filing public issues. The default-on `maturity-continuity` cycle runs the same full-portfolio invariant weekly.

**Next:** advance the fleet one rung: `fak maturity next` lists 583 next work item(s); the least-mature capability is the most leverage
