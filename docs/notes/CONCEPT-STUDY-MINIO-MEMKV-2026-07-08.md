---
title: "Study-repo: MinIO MemKV → fak (2026-07-08)"
description: "MemKV is a closed-source pitch with no public repo, so study-repo grounded its borrows in open-source KV engines (Mooncake, NIXL) and filed 25 fak issues."
---

# Study-repo: MinIO MemKV → fak (2026-07-08)

**Target:** <https://www.min.io/product/memkv> — a **product marketing page**, not a
repo. MinIO MemKV is a **closed-source commercial** appliance (sold via "Talk to a
Specialist"); there is **no public source repository** (the only public `memkv` repo,
[quangh33/memkv](https://github.com/quangh33/memkv), is an unrelated educational
Redis-clone). So study-repo's core loop — *clone → read → ground each borrow at
`path:line@sha`* — **cannot run against MemKV itself**. Filing "adopt MemKV" tickets
from the pitch would be the exact *borrowed-the-pitch* failure this skill exists to
kill.

**What MemKV claims** (all vendor figures UNMEASURED, never adopted as fak targets):
a petascale flash/NVMe KV-cache **memory tier** offloading GPU HBM KV over end-to-end
RDMA (no filesystem, no object protocol, no CPU in the data path); 2–16 MB GPU-native
blocks; a single ARM64 binary embedded in the MinIO STX appliance; NVIDIA
STX / Spectrum-X 800GbE / PCIe Gen6; "40–60% lower cost/token", "95%+ GPU util",
"~100x agentic scale".

## The move: study the open-source engines in the same band

MemKV competes with **Mooncake Store, LMCache, NVIDIA NIXL/Dynamo KVBM, SGLang
HiCache**. Those *are* cloneable, so the real study-repo pass ran against them:

- **Mooncake** `mooncake-transfer-engine` @ `c989668`
- **NIXL** @ `d9de2af`

fak witness (dogfooded `fak_feature_query` + `Grep` + tree read): the technique is
**already tracked and partially owned** — fak's residency ladder
(`internal/cachemeta/hardware.go`), durable L3 backend (`internal/l3kv/l3kv.go`),
radix prefix cache (`internal/radixkv`), the serving SOTA matrix
(`docs/serving/pd-disaggregation-kv-routing-sota.md`), and the transport-governance
seam (`docs/serving/kv-transport-governance-nixl-mooncake-lmcache.md`). fak's
deliberate posture on this band is **ride the transport/store, own the governance**
(admission/scope/taint/bit-exact eviction/deletion-cert). MemKV is a *ride-candidate
performance store with no governance* — market validation of fak's thesis, not a
design to reimplement. Every borrow is **INSPIRE** (clean-room; zero foreign bytes
copied).

## Candidate table — borrow · anchor · witness · route · filed #

| Borrow | Source anchor | Witness | Route | Filed |
|---|---|---|---|---|
| Rank MemKV in the serving SOTA matrix + inventory | pitch (no code) vs `pd-disaggregation-kv-routing-sota.md` §1 | PARTIAL (band tracked, MemKV unranked) | inspire | **#3309** |
| Borrow Mooncake/NIXL transport primitives for `StageTransport` | Mooncake@`c989668` `transfer_engine.h:113,145`; NIXL@`d9de2af` `nixl.h:216`; fak seam `internal/model/pipeline.go:224` | ABSENT (single-tensor sync send) | inspire | **#3310** |
| MemKV competitor row in the KV-transport-governance doc | pitch | PARTIAL | inspire | **#3311** |
| Name + profile the flash-KV-over-RDMA memory rung | `internal/cachemeta/hardware.go` tier ladder | PARTIAL | inspire | **#3312** |
| KV block/segment granularity (MiB vs token page) | `internal/radixkv/radixkv.go` (token-granular); L4 study MiB segments | PARTIAL | inspire | **#3313** |
| Cost/token as an OBSERVED competitor datapoint | pitch (UNMEASURED) + `internal/cachewitness` provenance | inspire | **#3314** |
| Positioning: validation-and-contrast | `L4-OBJECT-STORE-TIER-STUDY-2026-07-01.md:64-82` | inspire | **#3315** |
| Live-wire Mooncake Transfer Engine events | Mooncake `getNotifies`/`submitTransferWithNotify` | ABSENT | inspire | **#3316** |
| Live-wire LMCache offload/restore events | LMCache offload API | ABSENT | inspire | **#3317** |
| Explicit `Source` enum on CacheEvent/ServingEvent | `pd-disaggregation-kv-routing-sota.md` §2 (named gap) | ABSENT | inspire | **#3318** |
| Regenerable-KV plan → active epic | `docs/serving/regenerable-kv-plan.md` (no live ticket) | untracked doc | — | **#3319** |
| Agentic-caching plane extensions → active epic | `docs/notes/AGENTIC-CACHING-SOTA-2026-06-19.md` (no live ticket) | untracked doc | — | **#3320** |

## Ticket-audit of the corpus (make sure the `.md` items are active tickets)

Cross-referenced every issue number the ~21 KV/serving/cache research docs cite against
live GitHub state:

- **Live parents (re-homed onto these):** #50, #637, #751, #805, #809, #1469, #1472,
  #2169, #2236, #2240, #2669, #2671, #2674, #3199.
- **Closed parents the docs still point at (stale refs to fix):** #37
  (transport-bridge — work continues under **#3199** / **#50**), #79 (L3 epic),
  #903 (research SOTA), #493, #496, #819, #870, #1549–#1553. The L4 study's
  "#493 network StageTransport" / "#496 S7 durability" are **number-mismatches** —
  those IDs are unrelated closed issues.
- **Untracked docs → filed as epics:** `regenerable-kv-plan.md` (#3319),
  `AGENTIC-CACHING-SOTA-2026-06-19.md` (#3320).

## Honest limits

- MemKV itself was never read at a `@sha` (no source exists) — its rows are marked
  UNMEASURED and every concrete borrow is grounded in the **open-source** Mooncake/NIXL
  code instead.
- The `fak_feature_query` witness is lexical + a snapshot; re-witness before acting.
- Deeper `#79`/`#706`-family design items (L3 governance sidecars, multilevel-cache
  probes) trace to **closed** epics — they need a per-item shipped-vs-open check before
  re-filing, so they were **not** filed here (surfaced only).

## Round 2 — shelf shipped-vs-open audit (2026-07-08)

Witnessed the 31 deferred shelf candidates (design items whose docs cited **closed**
epics #79/#706/#905/#493/#134) against the live tree + GitHub. Result: **13 genuine
gaps filed (#3407–#3419)**; **18 dropped** — 15 already **SHIPPED**, 3 **open
duplicates**. The shipped-check is the point: it stopped 15 tickets that would have
duplicated landed code.

- **Filed #3407–#3419:** enginecache re-witness (→#1490), generation map
  machine-readable (→#1625) + ungroomed-issue report (#3409), DSA/recurrent KV-quant
  follow-on (→#2236), unify Hits+1 reuse seam (→#2236), fit KVReuseEstimate decay
  (→#2669), xenginekv co-residence transport (→#53), demote-before-drop PlanKVDemotion
  (→#2236), fix phantom `internal/kvbm` tag (#3415), 3× Metal decode/prefill levers
  (→#59), rank external KV-offload stores incl. MemKV (→#3199).
- **Dropped SHIPPED (grounded):** L3RegionBackend `internal/l3region/l3region.go:166`;
  referee sidecar `internal/gateway/l3referee.go:113`; durability promotion
  `internal/l3region/promotion.go:164`; cross-tenant share `internal/gateway/l3share.go:117`;
  deletion-cert `internal/gateway/l3deletioncert.go:125`; AWQ witness
  `internal/compute/cuda_awq_test.go:114`; SafeKV / mcpproxy / provenance / label-lattice
  already witnessed in `CONCEPT-FIELD-BORROW-LANDSCAPE-2026-07-08.md`.
- **Dropped OPEN_DUP:** AgenticFlict→#3202, agent-OS prior-art→#3203, EAGLE-3
  draft-head→#3197.
- **Re-home map (closed→live):** #79→#53, #706→#985, #905→#2236, #493/#134→#3199.

**Total for the MemKV study: 25 issues filed (#3309–#3320, #3407–#3419).**

## Companions

- [`study-repo`](../../.claude/skills/study-repo/SKILL.md) · [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md)
- Sibling study: [`CONCEPT-STUDY-HEADROOM-2026-07-08.md`](CONCEPT-STUDY-HEADROOM-2026-07-08.md)
- Parent epics: #2236 (superset ranking, sibling of #3144) · #3199 / #50 (KV-transport) ·
  #2169 (L4/object) · #2671 (min-cost tier) · #637 (throughput + trust/L3 spine)
- Read-corpus anchor: [`pd-disaggregation-kv-routing-sota.md`](../serving/pd-disaggregation-kv-routing-sota.md) ·
  [`kv-transport-governance-nixl-mooncake-lmcache.md`](../serving/kv-transport-governance-nixl-mooncake-lmcache.md) ·
  [`L4-OBJECT-STORE-TIER-STUDY-2026-07-01.md`](L4-OBJECT-STORE-TIER-STUDY-2026-07-01.md)
