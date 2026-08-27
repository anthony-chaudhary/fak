---
title: "Current native-performance constraints"
description: "Generated operational snapshot of current fak-native bottlenecks, ready work, collisions, and the OSS-to-performance decision walk."
---

# Current native-performance constraints

**As of:** 2026-08-27  
**Authority:** generated from `internal/nativeperf.BuildCurrentSnapshot`; immutable receipts remain the measurement evidence.  
**Refresh:** `fak native-performance --current-md`

An active constraint is the presently evidenced condition limiting a named performance outcome. Its type, horizon, owner, next action, and exit condition make it actionable; review_by prevents a stale observation from silently remaining current.

## Current constraints

| Constraint | Type / horizon / state | Envelope and driver | Evidence and authority | Next action / exit |
|---|---|---|---|---|
| `measurement-control-loop` — Real profiling and regression control loop | `evidence` / `semi-durable` / `waiting-evidence`<br>Observed 2026-08-25; review by 2026-09-01 | `portfolio`<br>The classifier and receipt gates are built, but only synthetic Metal/CUDA profile bundles are committed and the scheduled workflow does not consume a returned campaign receipt. | [contract] Profile schema, classification, and one-lever selection are implemented. (`docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md#phaseprofile-bundles-and-bottleneck-selected-work`)<br>[open] The current acceptance table names both real profiler bundles OPEN. (`docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md#acceptance-status`)<br>[control-loop] The public workflow currently prints the private handoff and validator command; it does not fetch or gate a returned artifact. (`.github/workflows/native-performance-regression.yml`)<br>Owner: internal/nativeperf profile and gate contracts; issue #8848 | **Next:** Capture one real scrubbed Metal bundle and one real scrubbed CUDA bundle, then make the scheduled/manual workflow validate a returned request instead of only printing the handoff.<br>**Exit when:** Both native envelopes have accepted real profile bundles and one scheduled/manual run consumes a scrubbed request and records the gate verdict. |
| `metal-resident-decode` — Metal resident decode | `dependency` / `structural` / `ready`<br>Observed 2026-08-23; review by 2026-08-27 | `qwen38-27b-q4km-m3pro-p32-t64`<br>The near-matched native point is about 47% of the diagnostic llama.cpp comparison; repeated synchronous Q4_K submissions and an incomplete coarse hybrid token graph remain the issue-backed driver, pending a real profiler bundle.<br>Ready: `metal.command-buffer-amortization` | [accepted] The frozen full-run Metal envelope remains 2.3-2.9 decode tok/s with functional PASS. (`docs/_witnesses/qwen38-27b-2026-08-20/metal-native-run-summary.json`)<br>[approximate] The later observation is 3.3 vs 6.966061 tok/s, P31/T64 vs P32/T64, without a joint quality-complete receipt. (`https://github.com/anthony-chaudhary/fak/issues/8697`)<br>Owner: issue #8324 | **Next:** Run the exact P32/T64 one-lever command-buffer-amortization OFF/ON profile and receipt, then use it to confirm or revise the coarse resident graph work.<br>**Exit when:** A same-envelope quality-passing receipt proves the default fak-native path owns the coarse token submission and meets the issue's >=5 tok/s promotion floor, or a real profile selects a different driver. |
| `cuda-cold-decode` — CUDA cold decode | `dependency` / `structural` / `ready`<br>Observed 2026-08-25; review by 2026-09-01 | `q38-q4km-native-cuda-a100-cold-decode`<br>The exact Q4_K_M A100 cold arm is correct at 11.8-12.1 tok/s. A distinct P=1 optimization envelope still uses scalar f32 activation products before the proposed Q8_1/DP4A path; its A/B must retain that separate identity.<br>Ready: `cuda.q8_1-activation-quant` | [accepted] Five cold unique runs were 5/5 exact at 11.8-12.1 decode tok/s on A100-SXM4-40GB. (`docs/_witnesses/issue-8819-qwen38-cache-attribution/summary.json`)<br>[hypothesis] Q8_1 activation quantization followed by signed DP4A Q4_K MMVQ is the issue-owned P=1 sequence; no gain is assumed. (`https://github.com/anthony-chaudhary/fak/issues/8635`)<br>Owner: issue #8635 | **Next:** Run the strict Q8_1 OFF/ON numerical gate; only then run the DP4A MMVQ end-to-end A/B with Q8_1 fixed ON.<br>**Exit when:** The default fak-native CUDA decode path passes full-model quality with zero fallback and a repeated same-artifact end-to-end gain, or the measured profile selects another driver. |
| `cuda-cache-correctness` — CUDA cache restore correctness | `correctness` / `structural` / `held-correctness`<br>Observed 2026-08-25; review by 2026-09-01 | `q38-q4km-cuda-a100-cache-restore`<br>The identical-prompt cache arm restored the wrong output in 5/5 attempts; its approximately 0.2 tok/s is diagnostic and cannot be optimized or promoted as parity. | [diagnostic] The cache arm was 0/5 exact; four confirmed hits were about 0.2 tok/s. (`docs/_witnesses/issue-8819-qwen38-cache-attribution/summary.json`)<br>Owner: issue #8848 follow-on from closed attribution issue #8819 | **Next:** Instrument prefix snapshot, recurrent-state clone, host staging, restore, and backend-copy bytes/time; rerun five cold and five identical cache-hit requests with an unambiguous exact-output prompt.<br>**Exit when:** The identical-prompt arm is exact in every gated repetition with cache identity, restored state, zero fallback, and end-to-end timing in the same receipt. |
| `laptop-placement` — 36 GiB laptop placement | `capacity` / `structural` / `capacity-bound`<br>Observed 2026-08-26; review by 2026-09-09 | `q38-q4km-native-metal-m3pro-capacity`<br>The canonical no-FAK_Q4K_FREE_CPU control reached readiness and one native Metal token, but peak swap grew by 7,681,930,690 bytes; the fail-closed derived minimum is 44 GiB. | [witnessed-refusal] The 36 GiB control derived a 44 GiB minimum after positive swap growth and restored the prior service. (`docs/_witnesses/issue-8971-streamed-q4k-capacity/canonical-no-free-cpu.json`)<br>Owner: capacity receipt for closed issue #8971 | **Next:** Place this exact no-free-CPU serving envelope on hardware meeting the 44 GiB derived bound; keep the 36 GiB laptop as control/orchestration or use only a separately named supported envelope.<br>**Exit when:** A same-artifact, same-environment native receipt proves zero positive swap on admitted hardware, or a newer measured bound supersedes 44 GiB. |
| `native-serving-stack` — Native serving stack | `dependency` / `structural` / `ready`<br>Observed 2026-08-25; review by 2026-09-03 | `qwen38-27b-q4km-m3pro-p32-t64`<br>Paged KV, exact-prefix reuse, chunked prefill, and continuous batching lack isolated real Qwen3.8 receipts; combining them before isolated arms would hide attribution.<br>Ready: `metal.paged-kv`, `metal.chunked-prefill` | [contract] The graph records each serving mechanism as a separate absent lever with an exact witness requirement. (`docs/benchmarks/NATIVE-PERFORMANCE-HILLCLIMB.md#metal-raw-decode-and-serving-levers`)<br>Owner: issue #8395 | **Next:** Work dependency-ready paged-KV and chunked-prefill as separate arms; advance prefix reuse and continuous batching only after paged KV is enabled and witnessed.<br>**Exit when:** Each mechanism has a quality-passing isolated receipt and the composed serving campaign reports TTFT/ITL p50/p95, aggregate throughput, peak memory, prefix-hit rate, and fallback count. |

## Dependency-ready work and collisions

Every dependency-ready arm is shown. Metal and CUDA are independent waves and may progress in parallel; arms inside a matched envelope remain serial one-lever experiments.

| Wave | Envelope | Ready arms | Parallel with | Within-wave rule |
|---|---|---|---|---|
| `metal` | `qwen38-27b-q4km-m3pro-p32-t64` | `metal.command-buffer-amortization`, `metal.paged-kv`, `metal.chunked-prefill` | `cuda` | Every arm shares one matched Metal envelope and must retain one-lever attribution without device contention. |
| `cuda` | `qwen38-27b-q4k-a100-p1-decode` | `cuda.q8_1-activation-quant` | `metal` | The Q8_1 candidate explicitly toggles the conflicting scalar-f32 baseline OFF inside one matched A/B. |

| Collision | Kind | Members | Why |
|---|---|---|---|
| `metal-envelope` | `shared-envelope` | `metal.command-buffer-amortization`, `metal.paged-kv`, `metal.chunked-prefill` | These arms share the exact M3 Pro envelope; benchmark them serially and never combine them before each isolated receipt exists. |
| `cuda-activation-arm` | `experiment-toggle` | `cuda.scalar-f32-activation-baseline`, `cuda.q8_1-activation-quant` | The two activation-product arms conflict; Q8_1 evidence must name the scalar baseline as OFF, not enable both. |

## OSS-to-performance walk

The closed walk is **source -> seam -> measured constraint -> deduped issue -> matched A/B -> keep/reject**. An exhaustive source list is discovery input, not permission to create or implement every idea.

| State | Meaning | Required evidence |
|---|---|---|
| `watch` | Pinned source identity retained without active implementation work. | repository, revision, license, and why it may matter |
| `candidate` | A plausible source seam is named, but the exhaustive study or measured constraint binding is incomplete. | pinned source plus proposed seam; no performance issue inferred |
| `studied` | A bounded study and candidate matrix are recorded; incomplete source classes remain explicit and block mapping. | study note, inventory map, completeness critic, explicit coverage limits, and dedupe readback |
| `mapped` | One exact source seam is bound to a measured current constraint and a deduped issue. | path/line@revision, FAK seam, constraint ID, and issue readback |
| `mapped-needs-limiter` | The exhaustive join found mapped backlog, but no measured limiter has selected which mapped seam should consume the next performance slot. | complete/qualified study, disposition counts, mapped issue evidence, and an explicit limiter-selection gap |
| `experimenting` | A one-lever matched A/B is running or captured without a keep decision yet. | baseline/candidate receipts with quality, identity, memory, and end-to-end outcome |
| `kept` | The adapted native path passed the A/B and is selected for the default or next composition. | accepted receipt, attribution/license, default-path witness, and rollback |
| `rejected` | The source seam failed the A/B or no longer addresses the measured constraint. | negative receipt or superseding profile plus retained reason |

| # | Gate | Requirement | Exit |
|---:|---|---|---|
| 1 | source | Pin repository revision and license; inventory is discovery evidence, not a FAK gap. | `candidate or watch` |
| 2 | seam | Complete the exhaustive study and name one exact source path/algorithm and one exact fak-native seam. | `studied` |
| 3 | measured constraint | Bind the seam to a current constraint whose profiler/receipt evidence names that driver; otherwise return to watch. | `mapped or watch` |
| 4 | deduped issue | Read back open and closed issues, then attach the route to one owner with a one-lever done condition. | `mapped with issue` |
| 5 | A/B | Implement inside fak-native and capture a matched baseline/candidate with quality, identity, memory, fallback, and end-to-end accounting. | `experimenting` |
| 6 | keep/reject | Keep only a quality-passing end-to-end gain; otherwise retain the negative result and reject or re-profile. | `kept or rejected` |

### Current source queue projection

The complete discovery registry remains `docs/research/monitored-repositories.json`. These rows show only sources currently adjacent to a named performance constraint. `candidate`, `studied`, and `mapped-needs-limiter` are not implementation authorization; mapped backlog is not performance-closed.

| Source @ revision | State | Seam | Proposed constraint / deduped issue | Next |
|---|---|---|---|---|
| `vllm-project/vllm@f18d0ba90d972a852a351c98be3f42b31372cfe4` | `mapped-needs-limiter` | 193 joined mechanism clusters: 183 actionable, including 168 partial and 13 conflict rows | `measurement-control-loop`, `native-serving-stack` / — | Select from the 172 actionable partial/conflict rows using the measured limiter; the current prioritizer walking only five uncovered rows is backlog visibility, not performance closure. |
| `sgl-project/sglang@536f570e6692eec0656ef9689db7591ca1d0e0a7` | `studied` | 12 serving and compatibility candidates; forge-history coverage remains explicitly partial | `native-serving-stack` / #8395 | Resolve the partial forge fence, then promote only a candidate selected by the serving limiter and existing issue dedupe. |
| `flashinfer-ai/flashinfer@39b484f1ce2fff086c66f9a899a0a58ba7f0ec3e` | `mapped-needs-limiter` | 22 decision-changing CUDA/kernel candidates, all deduped to existing FAK work or dependency-boundary rejection | `measurement-control-loop`, `cuda-cold-decode` / — | Use the real CUDA profile to select one already-deduped seam; complete source accounting does not itself choose or close performance work. |
| `llm-d/llm-d@bc20f73bd344b5a0faad5afca93831088aeee957` | `mapped-needs-limiter` | 20 serving-control candidates; two unduplicated gaps filed after complete dedupe | `native-serving-stack` / #9385, #9386 | Keep #9385/#9386 visible, but schedule them only when measured serving control or recovery is the active limiter. |

## Update contract

Update the typed snapshot in the same change that accepts, rejects, or reclassifies evidence. Preserve immutable receipts; change a driver only from a compatible real profile or end-to-end receipt. Run `go test ./internal/nativeperf` and the focused `cmd/fak` native-performance tests, then regenerate this page with `fak native-performance --current-md`.
