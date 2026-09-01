# Study-repo: LMBenchmark → fak (2026-07-11)

A `study-repo` / `field-borrow` pass over **[LMCache/LMBenchmark](https://github.com/LMCache/LMBenchmark)** @ `c1e05a708a5a1fd04a9ec09d215edbdcccdf92cb` (HEAD 2026-01-28), **Apache-2.0**. LMBenchmark is the LMCache team's **load-generation + latency/throughput/cache benchmark harness** — the *client* that drives the serving stacks fak has already studied ([LMCache](CONCEPT-STUDY-LMCACHE-2026-07-08.md) #3366, [Dynamo](CONCEPT-STUDY-DYNAMO-2026-07-08.md) #3352, [Mooncake/MinIO-MemKV](CONCEPT-STUDY-MINIO-MEMKV-2026-07-08.md) #3309–#3320, SGLang RadixAttention via `cmd/radixbench`).

License is compatible, but every borrow is **`inspire`** (Python→Go clean-room; no bytes vendored).

## Headline

**13 candidates extracted → 1 filed (#3937), 0 monoliths, the rest PRESENT or mission-fenced.** This is the honest outcome for the load-gen *client* of serving stores fak already out-built: fak's own bench family (`cmd/radixbench` hit-rate-vs-SGLang, `cmd/fanbench`/`internal/turnbench` cross-agent dedup, `cmd/sessionbench`, `cmd/ctxdemo`/`cmd/demorace` warm-cache baselines, `internal/cachemeta.PlanFleetReuse`, gateway percentile TTFT/TPOT/e2e histograms) already covers the borrowable surface. LMBenchmark's *uncaptured* half is open-loop QPS / arrival-curve / SLO-throughput load generation — a surface fak **deliberately declines** (agent kernel, not a token engine; structural throughput loss ~0.75× peak → ~3% at saturation vs raw SGLang; "never point at a throughput leaderboard as a fak win").

## Method

- **Acquire + pin** the clone (SHA above), license-gate (Apache-2.0 → `inspire`).
- **Deep-read** the modules via parallel fan-out readers: `mooncake/`, `agentic/`, `synthetic-multi-round-qa/`, `sharegpt/`, `real-multi-round-qa/` → 13 candidate borrows, each grounded at a source `path:line@sha`.
- **Witness** each against fak (`fak_feature_query` + `fak index` + grep across `cmd/`+`internal/`); 7 dropped at first witness as PRESENT / mission-fenced / runtime-mismatch.
- **Adversarial re-witness** (a 7-agent workflow) of the 6 survivors: each skeptic was told to *prove fak already has it*, bias conservative, honor the mission fence, and ground the verdict in a fak `path:line`. Plus a completeness critic. This step **flipped the leading candidate (C1) to DO_NOT_FILE** and surfaced the one true gap (C4).

## Candidate ledger

| # | Borrow (source `path:line@c1e05a7`) | Verdict | fak anchor | Disposition |
|---|---|---|---|---|
| **C4** | agentic dependency-**DAG** workload: `input_from` producer edges + per-round join (`agentic/agentic-qa.py:57-59,227-230`) | **PARTIAL** | `cmd/topobench/main.go:176-184` collapses declared edges to a star genome; `internal/agenttopo/agenttopo.go:197` `CombineIn` join already present | **FILED #3937** |
| C1 | public prefix-sharing trace reconstruction: `hash_ids → f"{id}"+512×"hi"` (`mooncake/mooncake-qa.py:237-240`) | PARTIAL | ingestion seam already exists: `cmd/radixbench/main.go:203` `-workload`, `internal/compute/kvreplay_trace.go:61` `ParseKVReplayTrace` | DO_NOT_FILE — public trace is a **data fixture**, not a code gap; hit-rate is workload×algorithm-determined & fak runs the identical radix algorithm |
| C2 | open-loop arrival-curve / QPS replay with time-dilation | N/A | — | DROP — throughput load-gen, **mission-fenced** |
| C3 | offered vs achieved vs inflight three-number honesty | PARTIAL | `internal/gateway/metrics_render.go:138` (achieved) + inflight gauge & `_max_age_seconds` (`metrics.go:952/964`) | DO_NOT_FILE — anti-masquerade already delivered server-side; "offered" is a client open-loop artifact |
| C5 | warmup set is a **strict prefix** of the measured set | PRESENT | `cmd/radixbench/main.go:265-283,598-603` (`longestCommonPrefix` primes; seed-7 warmup disjoint from measured) | DROP |
| C6 | per-run cache isolation (fresh tree / salt vs cross-run contamination) | PRESENT | `cmd/radixbench/main.go:352` fresh `radixkv.New(0)` per rep; `internal/compute/kvreplay_oracle.go:26-33` pure-function replay | DROP |
| C7 | SLO operating-point selection over a sweep (max thpt s.t. P99 ≤ target) | PARTIAL | `internal/turnbench/fanledger.go:68` picks widest-fan, not SLO-argmax | DO_NOT_FILE — throughput-under-latency, **mission-fenced** |
| C8 | streaming TTFT vs steady-state split | PRESENT | gateway TTFT/TPOT histograms + streaming split | DROP |
| C9 | per-sample server-gauge snapshot stored **on each latency row** | PARTIAL | `internal/gateway/serving_metrics.go:167-215` snapshots the gauges but as aggregate per-worker rows | DO_NOT_FILE — per-request tail-attribution is an open-loop queueing diagnostic |
| C10 | offline reprocess/re-summarize of a saved raw-results file | N/A | fak benches emit structured JSON already | DROP — low-value tooling |
| C11 | decompression guard on downloaded/stored trace blobs | N/A | fak has no untrusted-compressed-trace hot path | DROP |
| C12 | allocator/env-knob tuning (GC pressure) | N/A | Go runtime — Python-GC-moot | DROP |
| C13 | discrete-event scheduler sim inside the load generator | N/A | — | DROP — open-loop load-gen, **mission-fenced** |

Completeness critic: *"No material technique left unmined."* The residual cache-value axes (offload-tier transfer latency, eviction-pressure hit-rate degradation, TTFT-vs-overlap) are **already explicit unfilled cells inside epic #2244** — filing would duplicate the epic, not add a borrowed technique.

## The one filed borrow — #3937

fak already has the DAG **primitive** (`internal/agenttopo` — `Linear`/`Star`/arbitrary `Declare` edges, in/out neighborhoods = `input_from`, and `CombineIn` = a multi-parent join keyed by producer ID) *and* a `cmd/topobench` harness — but the **reuse oracle discards the edges**: `topobench` scores a declared chain/diamond as `turnbench.TopologyGenome{Width, SubTurns, Lanes}` (edges kept only as a `declared_edges` summary string) and runs it through the pure-star `turnbench.RunFanoutCell`. So cross-agent prefix reuse over *data-dependent* graphs — sequential prefix-extension along a producer→consumer edge, and multi-parent join reuse at a `CombineIn` node — is never measured, even though the star (max sibling-sharing) case is. #3937 files the edge-honoring oracle as a ship-alone enrichment (first checkable step: a red-then-green `turnbench` test asserting a 3-node chain reuses more prefix than its star collapse). Honest value bound: the pure-chain case overlaps `radixbench`'s multi-turn growing-prefix reuse; the genuinely-new signal is the **diamond / multi-parent-join** reuse.

## Mission fence (why so much was declined)

LMBenchmark's centre of gravity is **serving throughput/latency under offered load** — offered-vs-achieved QPS, arrival-curve replay, SLO operating points, per-request queue-tail attribution. fak sits *in front of* the model as an admission/adjudication/cache-coherence layer and structurally loses on raw throughput by design; it competes on **cross-agent KV reuse / cache value**, provable isolation, and the same-model raw-vs-fak differential. So C2/C7/C9/C13 are not gaps — they are the axis fak refuses. Only C4, which is a *reuse* measurement over realistic agent graphs (not a throughput leaderboard), survived.

Feeds the **#2236** superset ranking (milestone *The KV cache value is owned, observed & 2×*). Sibling of #3366 (LMCache), #3352 (Dynamo), #3309–#3320 (Mooncake/MemKV).
