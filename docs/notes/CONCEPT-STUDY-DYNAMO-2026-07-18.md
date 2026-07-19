# Concept study — NVIDIA Dynamo (deep /study, 2026-07-18)

- **Repo:** https://github.com/ai-dynamo/dynamo
- **Pinned:** `ea89e8bdfcc8b9c95514fa9beabc5bd8296ec546` (`@ea89e8bd`) — HEAD `feat(runtime): add endpoint-scoped event transport (#11841)`
- **License:** Apache-2.0 (compatible; all borrows below are **INSPIRE** / clean-room Go — no vendored bytes). NOTICE carves out MIT test data under `lib/llm/tests/data/deepseek-v3.2`.
- **Method:** deep `/study` (`--deep`) over a local checkout at pinned HEAD; README as map, then 5 parallel sub-readers (one per subsystem) returning load-bearing `path:line@ea89e8bd` findings; every borrow second-witnessed against fak's actual Go on the specific axis before a verdict.
- **Clone limit:** depth-1 (single commit in log) — "why they removed X" history is unavailable; no borrow rests on a history read.

## Fan-out coverage

| Sub-reader | Dynamo subsystem | Depth |
|---|---|---|
| A | KV-aware router (`lib/kv-router/**`, `lib/llm/src/kv_router/**`, `lib/kv-hashing/**`) | selector cost fn, tiered overlap, softmax, side-index, hashing, indexer families — full/load-bearing |
| B | SLA planner / autoscaler (`components/src/dynamo/planner/**`, `global_planner`, `profiler`) | scaling loops, perf-model inversion, predictors, budget, advisory — full core |
| C | KVBM block manager (`lib/llm/src/block_manager/**`, `lib/kvbm-*`) | tiering, eviction, offload, registry, consolidator, S3 lock — full/load-bearing |
| D | Disagg + NIXL (`kv_router/prefill_router`, `block_manager/{connector,distributed}`, NIXL, migration) | conditional-disagg, NIXL descriptors, layer-wise store, rendezvous, migration — full |
| E | Runtime/discovery/frontend (`lib/runtime/**`, `lib/llm/src/http`, preprocessor) | discovery, etcd lease/lock, event plane, push-router fault classifier, HTTP disconnect — load-bearing |

### Completeness-critic residue (subsystems intentionally NOT deep-read, justified)
- **ModelExpress / AIConfigurator / Grove** — separate repos (weight-streaming, config sim, k8s gang-scheduler); out of the routing/planner/disagg scope of this pass.
- **Multimodal E/P/D + video-gen (`components/**/omni`, FastVideo/SGLang-Diffusion)** — encode-worker + diffusion paths; the E/P/D *split* mechanism mirrors the P/D glue already read in reader D.
- **`deploy/**`, `recipes/**`, `container/**`, helm/CRDs** — deployment packaging, not decision logic.
- **`lib/bindings/python`, `lib/mocker`, `lib/rl`, `lib/data-gen`, `lib/truthy`, `lib/memory` low-level NIXL** — FFI/test-harness/kernels; the load-bearing transfer semantics were captured from `block/transfer/nixl.rs`.
- **Indexer data-structure bodies** (`concurrent_radix_tree*`, `branch_sharded.rs`, `cuckoo/*`) — mechanism + intent captured from `indexer/README.md` + headers; borrows are at design grain, not byte grain.

## Reconstructed worldview (who Dynamo is built for)
Dynamo is the **orchestration layer above** vLLM/SGLang/TRT-LLM for **datacenter multi-node**
LLM serving. Its user is a **cluster operator** optimizing **cluster-wide goodput + TCO under
an SLA (TTFT/ITL)** across a fleet of **disaggregated GPU workers** it *owns and can scale*.
Every design choice follows: route by KV-overlap because redundant prefill is the dominant
waste; separate prefill/decode because they bottleneck on different resources; autoscale on a
live-fit perf model because GPUs cost minutes to provision and dollars to over-provision; move
KV over NIXL/RDMA because the fabric is fast and the alternative is recompute.

**The fak divergence that colors every verdict:** fak's primary world is a **governed gateway
routing coding-agent API traffic across provider accounts/seats/models** — it does *not*
generally own the GPU workers, so it optimizes **cost/capacity/authorization + replay
determinism for an audited fleet**, where Dynamo optimizes **GPU goodput for an owned cluster**.
BUT fak *also* has a real serving/kernel side (in-kernel model, `radixkv`, a P/D
`serving_autoscaler`, `residency_router`, disaggregation-provenance/pricing). The high-value
borrows land precisely where fak's serving side already exists but is **coarser on a specific
axis** than Dynamo's — enhancements to shipped seams, not new capabilities.

## Candidate table

Verdict legend: **FILED** (PARTIAL/ABSENT enhancement) · **PRESENT** (fak already does this on-axis → drop) · **DIVERGENT** (fak chose otherwise for a still-holding reason) · **DEDUP** (already filed) · **NOTE** (worldview/observation).

### Routing (fak headline PRESENT via `residency_router.go`; finer axes vary)
| Borrow | Source `@ea89e8bd` | Axis | fak witness / worldview reason | Verdict |
|---|---|---|---|---|
| Route to best cached-prefix worker, tie-broken by load | `kv-router/src/scheduling/selector.rs:203` | overlap-minus-load worker selection | `internal/gateway/residency_router.go` `CacheAwarePolicy` already does overlap×inv-load + P2C + skew-fallback (models Dynamo/SGLang) | **PRESENT** |
| **Tier-weighted overlap credit** (device 1.0/host .75/disk .25) | `selector.rs:199-202`, `overlap.rs:201` | value-per-hit by tier reached | fak scores plain prefix **length** (`residency_router.go:209,491`); has tier data (`radixkv/crosstier.go`) unused | **FILED #5272** |
| **Anticipatory output-block load** (expected-output scaled) | `sequences/prompt_registry.rs:334`, `scheduler.rs:397` | project decode-KV footprint into load | fak load = in-flight **count** (`residency_router.go:480`) | **FILED #5274** |
| Predict-on-route speculative short-TTL side-index (decide→observe blind spot) | `llm/src/kv_router/indexer/side.rs:29-105` | in-flight same-prefix burst stickiness | already filed from llm-d's kvblock speculative index; Dynamo `side.rs` = 2nd witness | **DEDUP #3888** |
| Smooth rational overlap-decay `1/(1+k·excess)` vs hard skew cutover | `selector.rs:186-198` | anti-herd without a hard cliff | fak uses a hard `SkewThreshold` (`residency_router.go:421`), deliberately mirroring SGLang | **DIVERGENT** (tuning, not a gap) |
| Temperature softmax + reservoir tie-break | `selector.rs:28-83,383` | load-spread among ties | fak's P2C already randomizes ties | **PRESENT** |
| Chained XXH3 block hash + per-request salt (LoRA/namespace isolation) | `kv-hashing/src/salt.rs:49`, `protocols.rs:148` | prefix-share iff correct | fak keys by segment identity + `radixkv/namespace.go`; model folded via `responsesPromptCacheKey` | **PRESENT-ish / NOTE** (LoRA-adapter salt not fak's world) |
| Event-free bucketed-TTL self-cleaning index | `indexer/pruning.rs:22-85` | route when engines emit no evictions | fak index is LRU-capacity, no time-TTL; pairs with #3888 | **NOTE** (fold into #3888) |
| Sticky per-worker write sharding; BranchShardedIndexer | `indexer/README.md:177,382` | index write-throughput at >1k workers | fak fleet ≪ Dynamo scale; no firehose | **DIVERGENT** (scale) |
| Three-axis prefill admission (requests/ISL-tokens/cached-tokens) typed 429 | `scheduling/policy_queue.rs:40-66` | multi-dimensional overload bound | fak has `gateway/admission.go`,`batchsched.go`; token-dim limits partial | **NOTE** (candidate, un-filed) |

### Planner / autoscaler (fak has `serving_autoscaler.go`, 724 lines)
| Borrow | Source `@ea89e8bd` | Axis | fak witness / worldview reason | Verdict |
|---|---|---|---|---|
| **Advisory (shadow) mode** — decide+journal+export, actuate nothing | `planner/core/base.py:118-122` | validate a policy on live traffic pre-actuation | fak `ScaleMode` only native/defer_to_dynamo; `ScaleDecisionJournal` exists → small add | **FILED #5276** |
| **Forecast-driven predictive MinReplicas floor** (dual-timescale) | `throughput_scaling.py:52-101`, `load_scaling.py:196` | proactive provisioning vs reactive precision | fak `MinReplicas` static; reactive-only `reconcileRole`; `loaddebounce` unwired | **FILED #5277** |
| Consolidation-aware scale-down (re-predict survivor load, veto flap) | `load_scaling.py:416-485` | kill N↔N-1 oscillation | fak `serving_autoscaler.go:618` `survivorWouldBreach` does exactly this | **PRESENT** |
| Hysteresis + cooldown + drain-before-remove | `state_machine.py:334`, planner cadence | anti-thrash + graceful shrink | fak `HysteresisTicks`/`Cooldown`/`FleetMembership.Drain` | **PRESENT** |
| KV-hit-rate demand discount (cap 0.95) on forecast prefill | `perf_model/base.py:26`, `prefill.py:85` | don't over-provision cache-served tokens | fak has `KVUtilization` signal + `cacheobs`; discount not wired into sizing | **NOTE** (fold into #5277) |
| Pluggable forecasters (ARIMA/Kalman/Prophet) + idle-skip + log1p | `planner/core/load/predictors.py:56` | robust short-horizon forecast | prerequisite of #5277; keep forecaster pluggable | **NOTE** (in #5277) |
| Perf-model *inversion* (binary-search batch→max SLA-meeting RPS); monotonicity guard; bucket-retirement | `perf_model/decode.py:116`, `base.py:227,75` | SLA→exact replica count | fak is **threshold/step + hysteresis**, not a live regression — simpler, no model-fit failure modes | **DIVERGENT** (fak's simpler control is a deliberate, still-holding choice) |
| Per-pool SLA inversion (TTFT→prefill, ITL→decode independently) | `throughput_scaling.py:165-227` | independent P/D right-size | fak `serving_autoscaler` already splits Prefill/Decode `PoolObjective` | **PRESENT** (split) / inversion **DIVERGENT** |
| Cross-deployment GPU transfer via intent-pairing, free-before-allocate | `global_planner/scale_handler.py:342` | hard cluster budget w/ inter-pool moves | fak seats under account budgets; no cross-deployment GPU pool | **NOTE** (worldview) |
| Profiler→planner cold-start FPM bootstrap | `monitoring/aic_interpolation.py:59` | day-0 sizing before live data | ties to perf-model path fak diverges from | **DIVERGENT** |

### KVBM / block manager (fak: `radixkv`, `l3kv`, `kvmmu`, `enginecache`, stripeload #4298)
| Borrow | Source `@ea89e8bd` | Axis | fak witness / worldview reason | Verdict |
|---|---|---|---|---|
| Leaf-only eviction over prefix-lineage graph (never evict a block with cached children) | `kvbm-logical/src/pools/inactive/backends/lineage/mod.rs:128`, `eviction.rs:40` | protect shared-prefix ancestors under pressure | fak `radixkv` evicts the LRU **leaf** collapsing parents upward (pkg header) | **PRESENT** |
| Frequency-gated offload admission (double-on-touch, decay) to throttle SSD writes | `block_manager/offload/filter.rs:96-125` | SSD write-amplification / endurance on host→disk | fak stripeload/`enginecache` — endurance filter likely absent | **NOTE** (candidate; route to #4296/#3419) |
| TinyLFU count-min sketch + 4-tier freq-bucketed LRU (scan-resistant) | `kvbm-logical/src/tinylfu.rs:4`, `multi_lru_backend.rs:20` | admit-by-frequency, scan resistance | fak has SLRU (`radixkv/eviction_slru_test.go`) — segmented, scan-resistant; TinyLFU sketch finer | **PRESENT-ish / NOTE** |
| Content-addressed registry keyed by (hash, **tier**) + refcount + RAII remove | `block_manager/block/registry.rs:37-226` | exact per-tier liveness of a cached block | fak `registrations.go`,`cachemeta.go` — per-tier keying partial | **NOTE** (candidate) |
| KV-event consolidator dedup across engine+offloader (STORE first, REMOVE last) | `kv_consolidator/tracker.rs:233-405` | one coherent cache-visibility stream from N producers | fak coherence bus + `engine/cacheevents.go` — cross-source dedup partial | **NOTE** (route to #3320 cache-coherence) |
| Power-of-two position sampling for fleet block location (log-index a long prefix) | `kvbm-logical/src/events/policy.rs:17-70` | fleet cache-index size vs locatability | fak fleet residency index emits per-segment; geometric checkpointing absent | **NOTE** (candidate) |
| Fleet single-writer dedup to S3 via conditional-PUT lease locks | `kvbm-engine/src/object/s3/lock.rs:19-135` | cross-node write dedup for remote object tier | fak remote-KV-to-object-store not a current tier | **NOTE** (worldview) |
| G1→G3 direct offload bypassing host DRAM (GPUDirect Storage) | `block_manager/offload.rs:360-395` | host-DRAM footprint on offload path | hardware-specific (GDS NVMe) | **DIVERGENT** (hw) |
| Priority offload queue w/ weak refs + monotonic FIFO tiebreak; 2-stage backpressure | `offload/request.rs:14`, `pending.rs:204` | bandwidth-fair, memory-bounded async copy | fak stripeload bandwidth splitter (#4298) covers part | **PRESENT-ish / NOTE** |

### Disaggregation + NIXL transfer (fak: `cacheprice.CheapestRoute`, disagg-provenance)
| Borrow | Source `@ea89e8bd` | Axis | fak witness / worldview reason | Verdict |
|---|---|---|---|---|
| **Migration non-transferable guard** — refuse token-replay for guided-decode / n>1 | `llm/src/migration.rs:214-241,260-308` | availability without silent structured-output corruption | fak migration = session-state (`session/descriptor.go`); mid-stream token-replay guard absent | **FILED #5278** |
| Conditional disaggregation: local vs remote prefill by cost | glossary `docs/reference/glossary.md:14`; **not implemented** (`docs/backends/trtllm/README.md:23` "not supported yet") | per-request agg-vs-disagg source choice | fak `cacheprice/route.go` `CheapestRoute` **already** decides local/remote/recompute by token cost — fak *leads* here | **NOTE** (fak ahead) |
| Layer-wise progress-gated KV store (ship a layer's KV as computed) | `block_manager/connector/protocol.rs:38`, `scheduler.rs:122` | prefill→decode transfer/compute overlap | fak kernel is CPU/SSD-offload-shaped, not layer-streaming RDMA | **DIVERGENT** (hw/world) |
| Two-source rendezvous barrier before any KV transfer | `connector/scheduler.rs:539-585` | correctness under async control/compute race | fak disagg-provenance primitives (in-flight, #disagg epic) | **NOTE** (candidate → #3259) |
| Contiguity-collapsed NIXL descriptor lists | `block/transfer/nixl.rs:25-73` | RDMA setup overhead (one big DMA) | NIXL/RDMA-specific | **DIVERGENT** (hw) |
| Orphan-transfer guard (don't abort decode routing on client cancel mid-transfer) | `kv_router/prefill_router/mod.rs:285` | no KV block leak on cancel-vs-complete race | niche to owned-GPU KV egress | **NOTE** |
| Decode-side router override (zero overlap credit / prefill tracking on decode leg) | `prefill_router/mod.rs:414-423` | don't double-count prefill work at decode | pairs with fak P/D routing if/when split lands (#28) | **NOTE** |

### Runtime / discovery / frontend
| Borrow | Source `@ea89e8bd` | Axis | fak witness / worldview reason | Verdict |
|---|---|---|---|---|
| Routable-vs-overloaded split (backpressure ≠ removal) + 5s auto-reconcile | `runtime/src/component/client.rs` (`RoutingInstances`) | temporary overload self-heals; faults quarantine | fak `dispatchApplyAccountCooldown` = 429 backpressure auto-expire (accounts); serving-fleet `FleetMembership` has Drain — overloaded-but-not-removed state for the serving fleet partial | **NOTE** (candidate → #3365) |
| 3-way fault classifier (fault-quarantine / backpressure-throttle / zombie inactivity-timeout) | `push_router.rs` `wrap_with_fault_detection` | typed recovery per failure class | fak route-health classifier (timeout/auth/model_unavailable/rate_limited/5xx) rich; zombie-timeout quarantine partial | **NOTE** (candidate → #3365) |
| Least-loaded / P2C selection w/ RAII OccupancyPermit (decrement once at stream EOF) | `push_router.rs` (occupancy) | in-flight charge that can't leak | fak routes by account availability + P2C in residency router; per-instance RAII occupancy charge partial | **NOTE** (candidate → #3365) |
| Phased lease-loss shutdown (drain in-flight, then teardown) | `runtime/src/transports/etcd/lease.rs` | lost discovery lease drains, not drops | fak worktree worker prepare/land/reap + session drain | **PRESENT-ish / NOTE** |
| Layered timeout ordering (request-plane fires before HTTP safety-net) + sanitized mid-stream error frames | `http/service/disconnect.rs` | the layer that owns quarantine fires first; no backend detail leaks | fak gateway error handling; ordering discipline is a NOTE | **NOTE** |
| Endpoint-scoped event transport (HEAD #11841) | `lib/runtime` (event plane) | per-endpoint pub/sub scoping | fak coherence bus | **NOTE** (worldview) |

## Worldview findings (no single axis to witness)
- **fak already *leads* on conditional-disaggregation source choice.** Dynamo lists per-request agg-vs-disagg as "not supported yet" (fixed per-worker role); fak's `cacheprice.CheapestRoute` decides recompute/local/remote by token cost *today*. Worth a public concept-comparison row, not a borrow.
- **Perf-model inversion is the road fak did not take.** Dynamo builds a live-fit regression to invert SLA→replica-count (with monotonicity guard, bucket retirement, cold-start bootstrap). fak's autoscaler is threshold/step + hysteresis — simpler, no model-fit failure surface. This is a deliberate, still-holding divergence; revisit only if reactive convergence proves too slow in practice.

## Filed this pass
- **#5272** — tier-weighted overlap credit in the fleet residency scorer (→ epic #50)
- **#5274** — anticipatory decode-footprint load term in the fleet scorer (→ epic #50)
- **#5276** — advisory (shadow) ScaleMode for the serving autoscaler (→ epic #50)
- **#5277** — forecast-driven predictive MinReplicas floor / dual-timescale (→ epic #50)
- **#5278** — refuse token-replay migration for guided-decode / n>1 (→ epic #3352)

## Companions
- **field-borrow** — the per-capability witness+file discipline this pass reused for the on-axis witnesses.
- **Epics:** #50 (serving — RIDE+NATIVE spine, home of the 4 routing/autoscaler leaves), #3352 (in-flight migration, home of #5278), #3365 (dynamo fleet-control — home for the un-filed runtime/fault NOTE candidates), #4296 (KV-mobility), #3320/#3259 (cache-coherence / disaggregation) for the KVBM/disagg NOTE candidates, #2238 (shipped KV-aware routing baseline these refine).
- **Dedup:** #3888 (speculative provisional cache-warm — Dynamo `side.rs` is a second witness).
- **Field-borrow candidates parked in NOTE rows** (un-filed, lower-priority or route-to-existing-epic): three-axis prefill admission, KVBM offload-endurance filter, per-tier registry keying, KV-event cross-source consolidator, power-of-two position sampling, two-source rendezvous barrier, routable-vs-overloaded split, 3-way fault classifier, RAII occupancy charge.
