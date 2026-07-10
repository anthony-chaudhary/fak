---
title: "study-repo: vLLM (M2 lens) → fak (2026-07-10)"
description: "A focused study-repo pass re-reading vLLM's KV-cache core through the M2 (owned/observed/2x) lens: 5 PRESENT/tracked, 2 ABSENT, 17 PARTIAL; filed 5 leaves (#3893-#3897) + 10 enrichment recs onto existing tickets."
---

# study-repo: vLLM (M2 lens) → fak (2026-07-10)

A **focused** `study-repo` pass over **[vllm-project/vllm](https://github.com/vllm-project/vllm)** — not the
whole engine again (the superset epic **#2236** and the vLLM adapter lane **#1729–#1734** already map it
broadly), but a re-read of the **KV-cache value core** through the lens of **milestone M2 — *the KV cache
value is owned, observed & 2×***. Shallow clone (load-bearing modules, current HEAD), pinned at
**`08dfd68`**.

## Why vLLM is a direct reference (and what the M2 lens excludes)

vLLM is a *GPU inference engine*; fak is a *prompt / tool-page / session / cache-value* layer that never
touches a KV tensor. So — exactly as in the LMCache pass ([note](CONCEPT-STUDY-LMCACHE-2026-07-08.md)) —
the transferable surface is the **cache-management pattern one level up**: prefix admission/pricing,
residency accounting, eviction attribution, hit/reuse observability, and tier fall-through *contracts*;
**not** the CUDA/attention/PagedAttention kernels. Structural mapping: KV block = prompt/tool-page chunk;
GPU/CPU/disk/remote tier = hot/spill/blob/remote tier; `num_computed_tokens` = resident-prefix coverage.
This pass deliberately stays on the M2 seam and leaves M3/M4/M6/M7 (tiering transport, routing, quant) to
their existing children (#2238–#2243).

## License gate

vLLM is **Apache-2.0**; fak is **Apache-2.0** — integration would be license-legal, but every candidate is
Python → Go, so **all are `inspire`** (clean-room reimplementation, source cited). **No bytes vendored.**

## What was read (code, not the pitch)

Six parallel readers over the M2-load-bearing subsystems, each grounding candidates at a real `path:line`,
then a per-candidate witness against fak (raw `grep` over `internal/` + `gh` backlog dedup — the lexical
`fak_feature_query` was bypassed as unusable here, it returns ~1 MB/call):

- **block hashing / keying** — `vllm/v1/core/kv_cache_utils.py` (block-hash chain, `NONE_HASH`, group-id suffix, reusability).
- **scheduler / admission** — `vllm/v1/core/sched/scheduler.py` (computed-tokens admission, preemption accounting, skip-queue).
- **cache manager / coordinator** — `vllm/v1/core/kv_cache_manager.py`, `kv_cache_coordinator.py`, `block_pool.py` (uncached-common-prefix, eviction ordering).
- **KV-cache metrics** — `vllm/v1/metrics/stats.py`, `loggers.py`, `kv_cache_metrics.py` (per-source split, preempted split, sampled residency sidecar).
- **KV offload / tiering** — `vllm/v1/kv_offload/{base,file_mapper,tiering/*,cpu/*}` (reserve→commit, 4-state lookup, coalesced promotion, parallel-agnostic mapper) — the whole `tiering/` subtree is **new** at this HEAD.
- **KV connector v1 / events** — `vllm/distributed/kv_transfer/kv_connector/v1/{base,multi_connector}.py`, `utils.py`, `offloading/*` (tri-state lookup, partial-load errors, witness quorum, per-reason latency series).

**~30 grounded candidates → deduped to 24 → witnessed against fak: 5 PRESENT/tracked, 2 ABSENT, 17 PARTIAL.**

## The decisive finding — fak owns the fundamentals; the gaps are *accounting*, not *mechanism*

Five candidates witnessed **PRESENT or already-tracked** and were dropped: block-hash-chain keying (`radixkv`
native, this epic's M2 row), SHA-256 content keys (`crypto/sha256`), atomic-publish spill (`l3kv/store.go:160`
temp+fsync+rename, `blobfs`), torn-read→clean-miss (`l3kv/store.go:28`), and chunked prefill (tracked #1912).
The honest headline: **the two genuinely-ABSENT borrows (#3894 almost-hit, #3897 partial-hit) and the three
PARTIAL leaves are all *accounting/attribution* deltas** — fak already moves the bytes correctly; what it
doesn't yet do is *price* residency into admission, *attribute* a self-inflicted eviction, *decompose* reuse
by source, or *disclaim* a bad sub-range instead of a whole miss. M2 is an observability milestone, and this
is where vLLM's newest code is sharpest.

## Filed this pass — 5 leaves (M2, parent #2236)

Every leaf carries a source `path:line@08dfd68`, a fak seam `path:line`, the witnessed "what fak does today",
and a first checkable step.

- **#3893** `feat(admission)` — price a turn by resident-prefix coverage (`num_computed_tokens` ⇒ cheaper schedule); wires `cacheprice` into `dispatch_tick_preflight`. PARTIAL. Feeds #2242.
- **#3894** `feat(cachemeta)` — the **"almost-hit"**: `uncached-common-prefix` (longest-cached − reusable-hit) as the tokens worth materializing once. **ABSENT.**
- **#3895** `feat(cacheobs)` — attribute **self-inflicted** evictions: split preempted/forced-recompute from cold miss. PARTIAL.
- **#3896** `feat(cacheobs)` — per-**source** reuse decomposition (local-compute / local-hit / external-transfer, `parts==total`). PARTIAL.
- **#3897** `feat(l3kv)` — **partial-hit surgical recompute**: serve the good sub-range, disclaim the exact bad blocks. **ABSENT.**

## Merged into existing issues (not re-filed — anti-re-file discipline)

Recorded as enrichment recommendations on the epic **#2236** ([comment](https://github.com/anthony-chaudhary/fak/issues/2236)); each is a *specific delta* against a live ticket:

- **#3393** evict→reuse-gap ← **sampled residency sidecar** (`sample_rate` + birth/idle/lifetime, drain-swap interval, separate from the cache path) `kv_cache_metrics.py:59`.
- **#3391** token hit-rate ratio ← **windowed** recent-N interval rate + empty-suppress/preserve-latest guards `metrics/stats.py:73`.
- **#3386** atomic batch reserve ← two-phase **reserve→commit** store returning `evicted_keys` atomically `kv_offload/base.py:95`.
- **#3382** multi-root spill sharder ← **parallel-agnostic** shared spill dir (content+config-digest collapses TP/PP/CP+rank) `file_mapper.py:112`.
- **#1469** restore-on-access ← **deferred coalesced** promotion (reserve-on-hit dedups, batch-flush one load per tier/request) `kv_offload/tiering/manager.py:326`.
- **#1732** NIXL leases ← pre-compute **tri-state** lookup HIT/HIT_PENDING/RETRY + `None`-defer before any byte moves `kv_connector/v1/base.py:454`; + `request_finished` custody handshake `:547`.
- **#2238** fleet KV-aware routing ← distributed "finished/hit" as a **witness quorum** (count every worker to zero) `kv_connector/utils.py:50`; `KVEventAggregator` count==num_workers.
- **#2239** cost-aware eviction ← eviction **ordering by reusability** (never-reusable/no-hash evicted first) `block_pool.py:717`.
- **#2242** planner admission ← reserve-full-ISL + in-flight reservation gating `scheduler.py:903`; non-strict-FCFS skip-queue `:530`.
- **#1729** cache_salt/identity ← group-id as a **recoverable non-hashed** key suffix `kv_cache_utils.py:66`; per-process random `NONE_HASH` anti-poison root `:111`.

## Full candidate table

| # | subsystem | borrow (technique) | source `path:line@08dfd68` | witness | route | disposition |
|---|---|---|---|---|---|---|
| 1 | keying | Block-hash chain — block key = hash(prev_hash, block_tokens, extra) | `v1/core/kv_cache_utils.py:617` | PRESENT | inspire | DROP (radixkv native) |
| 2 | keying | SHA-256 content keys as the cross-process default | `v1/config/cache.py:95` | PRESENT | inspire | DROP (crypto/sha256) |
| 3 | tiering | Atomic-publish spill — temp + fsync + os.replace, skip-if-exists | `v1/kv_offload/tiering/fs/io.py:66` | PRESENT | inspire | DROP (l3kv/blobfs) |
| 4 | residency | Torn-read guard — write-in-flight never served as a clean hit | `v1/kv_offload/base.py:55` | PRESENT | inspire | DROP (l3kv:28) |
| 5 | sched | Chunked prefill — admit prefix that fits, defer the tail | `v1/core/sched/scheduler.py:831` | TRACKED | inspire | DROP (#1912) |
| 6 | admission | Price a turn by resident coverage — bill only `total − num_computed_tokens` | `v1/core/sched/scheduler.py:810` | PARTIAL | inspire | **filed #3893** |
| 7 | cachemeta | "Almost-hit" — `num_uncached_common_prefix_tokens`, the tokens worth warming once | `v1/core/kv_cache_coordinator.py:763` | **ABSENT** | inspire | **filed #3894** |
| 8 | cacheobs | Self-inflicted eviction — preempted query/hit split, kept apart from cold miss | `v1/metrics/stats.py:133` | PARTIAL | inspire | **filed #3895** |
| 9 | cacheobs | Per-source reuse — local-compute / local-hit / external-transfer, `parts==total` | `v1/metrics/stats.py:292` | PARTIAL | inspire | **filed #3896** |
| 10 | l3kv | Partial-hit — disclaim the exact bad blocks, recompute only those | `kv_connector/v1/base.py:375` | **ABSENT** | inspire | **filed #3897** |
| 11 | observability | Sampled residency sidecar — `sample_rate`, birth/idle/lifetime, drain-swap | `v1/metrics/kv_cache_metrics.py:59` | PARTIAL | inspire | enrich #3393 |
| 12 | observability | Windowed recent-N interval hit-rate + empty-suppress/preserve-latest | `v1/metrics/stats.py:73` | PARTIAL | inspire | enrich #3391 |
| 13 | tiering | Two-phase reserve→commit store; `evicted_keys` returned atomically | `v1/kv_offload/base.py:95` | PARTIAL | inspire | enrich #3386 |
| 14 | tiering | Parallel-agnostic shared spill dir (content+config digest, hash-shard) | `v1/kv_offload/file_mapper.py:112` | PARTIAL | inspire | enrich #3382 |
| 15 | tiering | Deferred coalesced promotion — reserve-on-hit dedups, one batched load | `v1/kv_offload/tiering/manager.py:326` | PARTIAL | inspire | enrich #1469 |
| 16 | residency | Pre-compute tri-state lookup HIT/HIT_PENDING/RETRY + `None`-defer | `kv_connector/v1/base.py:454` | PARTIAL | inspire | enrich #1732 |
| 17 | trust | Distributed finished = witness quorum (count workers to zero) | `kv_connector/utils.py:50` | PARTIAL | inspire | enrich #2238 |
| 18 | eviction | Eviction ordering by reusability — never-reusable/no-hash first | `v1/core/block_pool.py:717` | PARTIAL | inspire | enrich #2239 |
| 19 | admission | Reserve-full-ISL + in-flight reservation gating; non-FCFS skip-queue | `v1/core/sched/scheduler.py:903,530` | PARTIAL | inspire | enrich #2242 |
| 20 | keying | Group-id as a recoverable non-hashed key suffix (pure digest stays shareable) | `v1/core/kv_cache_utils.py:66` | PARTIAL | inspire | enrich #1729 |
| 21 | keying | Per-process random `NONE_HASH` anti-poison root, opt-in determinism | `v1/core/kv_cache_utils.py:111` | PARTIAL | inspire | enrich #1729 |
| 22 | offload | `ref_cnt` single-int tri-state (-1 in-flight / 0 evictable / >0 pinned) | `v1/kv_offload/cpu/policies/base.py:25` | PARTIAL | inspire | note (#3384 two-axis) |
| 23 | tiering | Pluggable tier ABC — self-declares lookup/store + own granularity | `v1/kv_offload/tiering/base.py:90` | PARTIAL | inspire | note (M3 #2239/#985) |
| 24 | offload | Retention-interval sparse checkpointing (keep anchors + replay boundary) | `v1/core/single_type_kv_cache_manager.py:817` | ABSENT-niche | inspire | note (vcache_snapshot) |

## Honest limits

- The witness is **lexical + a snapshot**: raw `grep` over `internal/` + `gh` dedup, true only as of
  2026-07-10 at vLLM `08dfd68`. Re-witness before acting on an old row.
- **2 ABSENT / 17 PARTIAL / 5 PRESENT**: no borrow is a from-scratch subsystem; every PARTIAL is a *specific
  accounting delta* against an existing fak mechanism (the seam + "what fak does today" per issue).
- Shallow clone + six readers is **one pass over the M2-load-bearing modules**, not every file — a reader may
  have missed a subsystem (M3/M4 transport deliberately out of scope this pass).
- License reading is a good-faith Apache-2.0-vs-Apache-2.0 compatibility check, not legal advice.
- The 5 new issues sit as `orphan-note`-adjacent fresh tickets; `class:*` work-class labels are backfilled by
  the nightly `tools/issue_lane_router.py` (its snapshot predated these).

## Companions

- Filed: **#3893, #3894, #3895, #3896, #3897** (5 leaves, M2, parent #2236).
- Enriched (epic comment): **#3393, #3391, #3386, #3382, #1469, #1732, #2238, #2239, #2242, #1729**.
- Feeds: **#2236** (superset concept-by-concept ranking, M2); adapter lane **#1729–#1734**.
- Skills: [`study-repo`](../../.claude/skills/study-repo/SKILL.md) (this pass),
  [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) (the witness+file back-half).
- Sibling study notes: [LMCache](CONCEPT-STUDY-LMCACHE-2026-07-08.md),
  [Mooncake](CONCEPT-STUDY-MOONCAKE-2026-07-10.md), [SGLang caching](CONCEPT-STUDY-SGLANG-CACHING-2026-07-10.md).
