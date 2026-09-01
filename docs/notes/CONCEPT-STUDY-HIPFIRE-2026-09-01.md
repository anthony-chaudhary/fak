# Concept study: warpfront/hipfire

**Verdict:** Hipfire's most transferable contribution is its evidence architecture, not a particular AMD kernel. At `0f1628241f751d9ee44922b5a1ca40aa756a82e0`, it treats native inference as a fail-closed system where physical admission, cache/session identity, retained-command safety, architecture-specific kernels, and route/hash-bound certification outrank isolated throughput. FAK should reuse two existing issue seams, watch two emerging mechanisms, and avoid duplicating its own benchmark, gate, and negative-evidence machinery.

- **Source:** `https://github.com/warpfront/hipfire`
- **Pin:** `0f1628241f751d9ee44922b5a1ca40aa756a82e0`
- **Observed:** 2026-09-01
- **Tracker:** #10591
- **Receipt:** `docs/research/inventory/warpfront-hipfire.json`

## Worldview

Hipfire treats RDNA-native inference as a fail-closed, evidence-bound system where physical admission, cache/session identity, retained-command safety, architecture-isolated kernels, and route/hash-bound certification matter more than isolated throughput. Its unusually large investigation corpus and retained failed kernels make negative knowledge part of the product. Open device-mesh and transactional rollback PRs are directions, not shipped facts.

Its validation doctrine is claim-specific: no-GPU CI is not GPU proof, serving success is not numerical or route proof, and production admission may not be inferred from a nearby configuration (`docs/VALIDATION.md:112@0f1628241f751d9ee44922b5a1ca40aa756a82e0`). Prompt bytes are experimental state, so receipts bind fixture identity rather than a prose description (`AGENTS.md:179@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).

## Shipped at the pin versus emerging

Shipped work includes real expert-parallel load acknowledgements and request thinking semantics (#671), MQ4V2 routing (#664), CK Asym3 FWHT D256 (#663), Qwen3.5 TP5 (#662), physical-capacity VRAM sizing (#661), MTP batched prompt KV projection (#660), and vision paths (#653/#654).

Open or post-pin direction is not credited as shipped: transactional EP rank constructors (#648), partial DFlash load rollback (#652), adaptive-KV post-ship integration (#647), BF16 KV tier defaults (#670), device-mesh tracking (#666), and the device-mesh weight-store pilot (#675).

## Deep inventory

The study fanned out across the whole tracked tree and history, including 1,200 `crates/**` files, 1,034 `kernels/**` files, 311 `benchmarks/**` files, 28 `tests/**` files, five `redline/**` corpus files, 108 `autoresearch/**` files, 43 `.agents/**` files, eight `.github/**` files, 21 `scripts/reap/**` files, 13 `tools/change_gate/**` files, 155 `docs/plans/**` files, and 1,086 `docs/investigations/**` files. It also inspected designs, specs, changelog, releases, issues/PRs and selected discussions, registry, license bundle, NOTICE, credits, prior art, and the pinned-but-uninitialized submodule.

Representative mechanisms:

- Strict admission rejects unsafe physical fits (`crates/hipfire-runtime/src/admission.rs:86@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).
- Adaptive KV performs transactional handoff and reset (`crates/hipfire-runtime/src/kv_adaptive.rs:271@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).
- Session-prefix reuse carries monotonic identity and excludes active generations (`crates/hipfire-runtime/src/session_table.rs:12@0f1628241f751d9ee44922b5a1ca40aa756a82e0`; `:235`).
- Retained command replay fails closed and defends against ABA reuse (`crates/rdna-compute/src/replay.rs:5@0f1628241f751d9ee44922b5a1ca40aa756a82e0`; `:307`).
- Separate graph caches encode distinct ownership/lifetime classes (`crates/rdna-compute/src/graph.rs:90@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).
- Architecture isolation is explicit in fused kernels (`kernels/src/attention_flash_q8_0_reduce_gated_mq_rotate.gfx1100.hip:11@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).
- Failed kernels remain with regression and diagnosis evidence (`kernels/src/attention_dflash_wmma_m32_kstg_FAILED.hip:6@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).
- Benchmark measurement is separate from production admission (`docs/BENCHMARKS.md:25@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).
- Redline binds registry/profile, source/daemon, contract, and PM4 identities (`docs/GOLDEN-REDLINE.md:24@0f1628241f751d9ee44922b5a1ca40aa756a82e0`; `:71`).
- Baseline promotion uses CAS while GPU occupancy is separately locked (`autoresearch/ar/driver.py:119@0f1628241f751d9ee44922b5a1ca40aa756a82e0`; `autoresearch/ar/gitpilot.py:74@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).
- Review capsules explicitly preserve incomplete/rejected states (`autoresearch/ar/review/capsule.py:217@0f1628241f751d9ee44922b5a1ca40aa756a82e0`).

## Candidate decisions

| Borrow | Axis | FAK witness | Route | Issue/outcome |
|---|---|---|---|---|
| Startup/steady physical memory admission | On-axis native safety | `internal/localadmission/reservation.go`; `reservation_test.go:20` | Inspire; `PARTIAL/DEFAULT` | #9587 already owns the missing aggregate Apple pressure seam. |
| Prefix reuse with monotonic session identity and active-generation exclusion | On-axis serving performance/correctness | Existing paged-KV/prefix serving work | Inspire; `PARTIAL/DEFAULT` | #8395; add these invariants to its acceptance envelope, no duplicate. |
| Transactional adaptive-KV handoff | On-axis cache ownership | `internal/model/cache_geometry.go:227` has transactional geometry rollback | Inspire; `PARTIAL/WATCH` | Watch under #8395 until a shipped adaptive-tier seam exists. |
| Fixture/hash/route/admission-separated certification | On-axis evidence | `BENCHMARK-AUTHORITY.md` | `PRESENT/EXCLUDE` | FAK already has the applicable contract. |
| Durable failed-optimization knowledge | On-axis native performance honesty | `docs/sota/`, `CLAIMS.md`, benchmark notes | `PRESENT/EXCLUDE` | Do not add another registry. |
| Host-capability-aware typed gate receipts | On-axis operations | FAK/DOS gate and receipt surfaces | `PRESENT/EXCLUDE` | No parallel gate framework. |
| Partial native resource rollback | On-axis native loading | Host file cleanup exists; device rollback is unproven | Inspire; `ABSENT/WATCH` | Upstream is emerging; file only after a concrete FAK device-allocation repro. |

No new implementation issue was filed: the two actionable shipped mechanisms map to #9587 and #8395, while the rest are present, insufficiently distinct, or emerging.

## Evidence limits and completeness critic

The tracked-tree coverage was exhaustive by major source class, but not every one of the 1,086 investigation documents was read line-by-line; workers clustered them and deep-read representative, decision-bearing records. Older benchmark rows often lack complete binary/model identity, and the statistical treatment is mostly medians without confidence intervals, randomized ordering, or power analysis. Selected PR discussions were inspected, but several active architecture PRs had no formal reviews or inline review comments. The submodule at `2ef91896bcdc4d26624f952e5c905c787cd9bc9e` was not initialized, so its exact license remains unverified. These are explicit limits, not evidence against the project.

## License and provenance

**Route: INSPIRE / clean-room adaptation only for this pass; no code copied.** The root carries MIT/Apache-2.0 texts, NOTICE, credits, and prior-art correction records, but GitHub classifies the bundle as “Other,” individual surfaces may carry distinct attribution, generated/vendor content requires review, and the pinned submodule license was not inspected. Direct integration therefore requires per-file header/provenance review, NOTICE preservation, and submodule/dependency verification.

## Companions

None. This study intentionally filed no new mechanism issue; #9587 and #8395 are pre-existing on-axis owners, while #10591 tracks the study itself.
