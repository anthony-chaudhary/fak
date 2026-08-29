---
title: "Go-native performance system: compiler, runtime, concurrency, and placement"
description: "A source-pinned field-borrow pass over Go 1.26+ that separates capabilities fak already uses from measured gaps, files seven scoped issues under #10178, and excludes style-only migrations."
---

# Go-native performance system (2026-08-29)

This pass asks a narrower question than "use more Go features": which Go compiler, runtime,
concurrency, and placement mechanisms can materially improve fak through a real seam and a
falsifiable receipt? The result is epic [#10178](https://github.com/anthony-chaudhary/fak/issues/10178),
seven new dispatchable children, four existing owners to augment, and an explicit list of
features that should not be adopted without new evidence.

Durable field-borrow receipt:
`study_9e9bd3223b3133fd493c62f9d3cfac010e87cf596cca84977b76d0fec9b6bc61`.

## Self-query witness

Observed 2026-08-29 against the live tree.

- `fak capabilities "container CPU quota GOMAXPROCS scheduler runtime metrics"` returned no
  matching capability.
- `fak capabilities "profile guided optimization PGO Go compiler build"` returned only the
  adjacent native-inference bottleneck card.
- `fak capabilities "flight recorder goroutine scheduler diagnostics"` returned only an
  adjacent capability-floor card.
- `fak study search "Go PGO"` returned `[]`.
- `fak dev index docs` surfaced adjacent runtime/build notes; raw symbol search confirmed
  typed atomics, generics, build tags, and selected `testing/synctest` use, but no release
  PGO input or `runtime/trace.FlightRecorder` use.

These null and adjacent results are the field-borrow gap witness. They do not by themselves
prove code absence, so every row below was cross-checked against the raw tree and all-state
GitHub issue search.

## Dated source ledger and license

| Source | Event/state | Immutable anchor | What changed here | Refresh trigger |
|---|---|---|---|---|
| Go 1.26.6 | released 2026-08-13 | `golang/go@1ea5a71ad8ceb7b9f16b4b6f8ea4739a4327dd6e` | pins scheduler metrics, compiler behavior, and runtime contracts used by this study | fak toolchain bump |
| Go PGO | released/general-use; observed 2026-08-29 | [PGO guide](https://go.dev/doc/pgo) paired with the pinned compiler source | supports a representative-profile experiment, not a promised gain | PGO format/compiler change |
| Container-aware `GOMAXPROCS` | shipped in Go 1.25; accepted proposal #73193 | implementation `e6dacf91ffb0a356aa692ab5c46411e2eef913f3` | exposes a mismatch with fak's package-init model pool | scheduler/cgroup contract change |
| Scheduler-state metrics | shipped in Go 1.26 | commits `13df972f6885ebdeba1ea38f0acd99ea0f2bfb49` and `78a3968c2c9f2d6e8eb6dc263b4a2517c72d71be` | makes runnable/waiting/thread pressure observable | metric rename/removal |
| Flight Recorder | shipped in Go 1.25; proposal #63185 | implementation `83df0afc4e5c3719a6aca08a798460d38e78fc95` | enables bounded pre-event traces at typed stall seams | trace API/format change |
| Arenas proposal | on indefinite hold; production warning dated 2023-01-17 | [golang/go#51317](https://github.com/golang/go/issues/51317) | excludes production arena adoption | proposal reactivation/release |

Go source, docs, and tests at the pinned revision use the BSD-3-Clause license. Every active
candidate below is **ADAPT** at the public-contract or test-shape level; no upstream
implementation bytes are copied. Arenas are **EXCLUDE**. Iterators, weak pointers, `unique`,
and generic specialization without a profile are **WATCH**.

## Present, partial, absent

| Axis | Verdict | fak seam | Portfolio disposition |
|---|---|---|---|
| Typed atomics, generics, build tags | PRESENT | `internal/abi/registry.go`, `internal/model/parallel.go`, platform leaves | retain; no generic adoption ticket |
| Bounded goroutine scheduling | PARTIAL | `internal/microagent/slotsched.go`, `budgetqueue.go`; unbounded/native gaps remain | reuse the shipped patterns |
| `testing/synctest` | PARTIAL | present in `internal/bgloop`; absent from scheduler lifecycle tests | DEFAULT where virtual time models the primitive |
| Representative release PGO | ABSENT | `scripts/build.sh:47-74`, `cmd/fak/profile.go:38-109` | DEFAULT only after matched promotion evidence |
| Hierarchical load/dequant budget | PARTIAL | `internal/ggufload/gguf_parload.go`, `gguf_dequant.go` | DEFAULT after bounded-load witness |
| Dynamic runtime-capacity coupling | PARTIAL | `internal/model/parallel.go:27-36`, `budget.go:17-22` | DEFAULT for default-sourced workers; overrides fixed |
| Scheduler/GC receipt | PARTIAL | `internal/taskmgr/taskmgr.go:665-687`, `internal/harnessres/harnessres.go` | DEFAULT, low-cardinality and overhead-gated |
| Flight recording | ABSENT | native scheduler/debug stall seams | OPTIONAL-MODULE, default off |
| NUMA replicas/affinity | PARTIAL | components exist under `internal/compute`; ordinary caller absent | OPTIONAL-MODULE under existing #5127 |
| Arenas | ABSENT | no production seam | EXCLUDE |
| Iterators, weak, `unique`, blanket pooling/specialization | absent or partial | no material profile | WATCH |

## Filed work

- [#10179](https://github.com/anthony-chaudhary/fak/issues/10179) qualifies a representative
  Go PGO lifecycle through paired release artifacts.
- [#10180](https://github.com/anthony-chaudhary/fak/issues/10180) gives GGUF tensor loading
  and dequantization one hierarchical concurrency budget.
- [#10181](https://github.com/anthony-chaudhary/fak/issues/10181) makes default-sourced native
  CPU workers follow dynamic runtime capacity between dispatch generations.
- [#10182](https://github.com/anthony-chaudhary/fak/issues/10182) adds a version-tolerant
  scheduler/GC metrics receipt.
- [#10183](https://github.com/anthony-chaudhary/fak/issues/10183) fuzzes microagent
  Spawn/Drain/Close as a deterministic concurrent state machine.
- [#10184](https://github.com/anthony-chaudhary/fak/issues/10184) captures bounded Go flight
  recordings on typed stalls.
- [#10185](https://github.com/anthony-chaudhary/fak/issues/10185) reports toolchain-versioned
  escape, BCE, and inlining debt while keeping runtime evidence authoritative.

Existing owners remain the right place for their exact gaps: #9661 for build action and
`-p x GOMAXPROCS` evidence; #1915 for bounded native pipeline stages; #5127 for NUMA
production wiring; #2010 for profile-first bounded pooling with arenas removed; and #2054
for runtime resource read-back.

## Why the exclusions matter

"More Go" is not synonymous with "newer syntax everywhere." `sync.Pool` is not a capacity
bound, weak pointers are not deterministic resource ownership, higher `GOAMD64` levels may
not help this binary, generic symbol count is not final text size, and container CPU quota is
average throughput rather than a hard parallelism or affinity contract. Each tempting
migration stays unforked until a profile identifies a real fak cost and a matched experiment
can disconfirm the proposed benefit.

## Companions

- Epic: [#10178](https://github.com/anthony-chaudhary/fak/issues/10178)
- Build result owner: [#9661](https://github.com/anthony-chaudhary/fak/issues/9661)
- Native scheduling epic: [#1911](https://github.com/anthony-chaudhary/fak/issues/1911)
- Runtime resource dogfood: [#2054](https://github.com/anthony-chaudhary/fak/issues/2054)
- NUMA execution owner: [#5127](https://github.com/anthony-chaudhary/fak/issues/5127)
