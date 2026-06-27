# fak fleet partition manifest (`dos-plan-price` input)

The build is a witness-gated DOS dark-fleet wave over **disjoint file-trees**. This
manifest is what `dos-plan-price` scores before any worker launches: it computes the
collision graph, confirms collision-free max-concurrency, and refuses a colliding
partition with **zero agents launched**. `dos-arbitrate` then takes one lease per
tree. Each leaf is its **own directory**, so the file-tree collision graph
`dos-plan-price` scores is empty by construction — two workers editing two different
leaves never touch the same files — except `internal/abi` itself, which is **wave-0,
human-owned, and unleasable**. (The *import* graph is a layered DAG, not a star —
leaves import lower-tier leaves, enforced by `internal/architest`; see `fak/GROWTH.md`.
It is the file-tree disjointness, not import independence, that keeps the leases disjoint.)

> **Hour notation:** `hN` means `N` hours from the start of the build (e.g., `h6–h30` is the 6–30 hour window).

## Wave 0 — the serial gate (human, ~6h). NOT fanned out. {#h0}

| Tree | Deliverable | Witness (mechanical, non-author) |
|---|---|---|
| `internal/abi/**` | Freeze `types.go` + `registry.go` (this artifact). | `go build ./internal/abi/...` green **and** `testdata/abi_v0.1.golden` conformance test pins every wire struct (additive-only enforced). |
| `cmd/fak/`, `internal/registrations/` | Binary skeleton + the blank-import list; `go build ./...` green. | `go vet ./...` exit 0; `fak version` prints ABI 0.1. |
| `internal/kernel/**` | The frozen-walker impl (Submit/Reap/Syscall folds the chain, walks FastPaths, dispatches Ops, drives ProvisionalSinks). | unit test: a registered stub Adjudicator denies → call never dispatched; fold orders by `FoldRank`. |
| `testdata/fixtures/**` | Operator-authored hard fixtures: poison-result set, malformed-call set, curated pure/idempotent workload, frozen tau2-trace subset. | files committed; each fixture has an expected-verdict golden. |
| go.mod reconciliation | One toolchain ≥1.26; vendor AGT-Go + dos-preflake/go + metrics-service. | `go build ./...` links all three in one binary. |

Wave-0 also runs the **vDSO purity profile as a GATE**: extract the tau2 traces,
measure the provably-unchanged fraction; if <10–15%, scope the vDSO headline to the
read-only subset (recorded in the fixture) before fan-out.

## Wave 1 — the 4 independent leaves (parallel, h6–h30) {#h6} {#h30}

| Worker | Tree (lease) | Goal | Witness |
|---|---|---|---|
| **W1-engine** | `internal/engine/**` | `EngineDriver` over LiteLLM→remote; local↔remote by env. | same prompt → valid completion from stub-local + recorded-remote transport. |
| **W2-kernelpdp** | `internal/agent/**`, `internal/gpulease/**` | AGT semantic PDP + DOS lease PEP, each `RegisterAdjudicator`'d. | poison fixture denied; transform mutates Args; default-deny on empty policy. |
| **W3-vdso** *(strongest worker)* | `internal/vdso/**` | 3 `FastPath` tiers (pure/CAS/static). | bench: N pure/cached calls resolve with engine-counter==0 **and measured hit-rate>0 on the frozen workload**. |
| **W4-mmu** | `internal/mmu/**` | Write-time `QuarantinePayload` path + Go CAS blob store as `RegionBackend`+`PageOutBackend`+`ProvisionalSink`. | poison fixture absent from assembled context; paged-out result round-trips byte-identical; Rollback(txn) drops scratch. |

## Wave 2 — gated on wave-1 *confirmed* phases (`dos-witness-claim`, h30–h50) {#h30} {#h50}

| Worker | Tree | Goal | Witness |
|---|---|---|---|
| **W5-harness** | `internal/harness/**` | Tool loop: every call → `Syscall`; order vdso→adjudicate→dispatch. | trace proves order; 3-tool scripted task completes against real W1–W4. |
| **W6-preflight** | `internal/preflight/**` | Rungs 0–2 as ranked `Adjudicator`s + typed `LabelRow` emitter. | malformed fixture caught pre-fire; a `LabelRow` JSONL line emitted. |
| **W7-kpi-stewards** | `internal/kpi/**`, `internal/stewards/**` | metrics-service-shaped `Emitter`s; steward population + meta-steward prune. | seeded 4+1 stewards → meta-steward prunes exactly the dead one; counters scraped clean. |

## Serial tail (human-attended, h50–h72) {#h50} {#h72}

Real-impl integration checkpoint (~h44): re-run W5 against **real** W1–W4 (not
stubs) — the re-run is the witness. Then the A/B bench (`--vdso=off` vs `--vdso=on`,
**workload-hash-gated** so both arms provably ran the identical input), one
RSI-as-ship-gate shot on a trivial known-good tweak, `dos-witness-claim` fold, tag
v0.1.

## Growth slots — where the *next* ideas land (still disjoint)

These are pre-allocated trees + reserved number ranges (see `registry.go`), so the
post-v0.1 ideas attach without re-pricing the partition:

| Future idea | Tree | Reserved range | Lease needed |
|---|---|---|---|
| speculative exec | `internal/spec/**` | `OpsSpec`, `ExtSpec` | yes (side-effecting) |
| async / io_uring | `internal/async/**` | `OpsAsync`, `ExtAsync` | yes |
| zero-copy backend | `internal/zerocopy/**` | (RegionBackend, no opcode) | no |
| syscall-tuned model | `internal/labeler/**`, `internal/model/**` | `ExtLabel`, `EventsLabel` | no |
| headroom codec | `internal/headroom/**` | (PageOutBackend) | no |
| witness enforcement | `internal/witness/**` | (WitnessResolver) | yes |
| federated trust | `internal/fedtrust/**` | `ExtTrust`, `VerdictsVendor` | no |
| cross-agent result pool | `internal/sharepool/**` | (SharePolicy Adjudicator) | no |

Each is a new leaf in its **own tree** (importing lower tiers per the layering, see
`fak/GROWTH.md`); `dos-plan-price` stays empty-collision because no two leaves share a
file tree or a reserved number.
