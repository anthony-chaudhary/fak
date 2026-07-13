# Mac many-agent model selection: cache economics first

**Status:** host-verified selection model; real-Mac serving remains device-gated  
**Issue:** [#3810](https://github.com/anthony-chaudhary/fak/issues/3810), child of [#3809](https://github.com/anthony-chaudhary/fak/issues/3809)  
**Decision:** promote **Qwen2.5-7B Q8** as the provisional checkpoint for the Mac many-agent spine.

This is not a single-stream leaderboard pick. The target is the most useful long-horizon,
many-agent workload that fits on a MacBook after weights, shared-prefix KV, private-tail KV,
and concurrency headroom are charged. The numerical capacity result belongs to #3811; this
document fixes the selection rubric and the candidate set that calculation must evaluate.

## Rubric

A candidate advances only if its in-kernel forward path is verified first. The remaining
axes rank candidates; they do not excuse an unsupported architecture.

| Axis | Measurable proxy | Why it matters for many agents |
|---|---|---|
| In-kernel architecture support | `Config.ModelType` plus `ClassifyForwardPath` result, re-derived from `internal/model/arch_support.go` and its tests | A model that cannot load and name its forward path cannot anchor the runnable spine. |
| Weight footprint | GGUF file/RSS at the selected quantization; use Q8 for the provisional pick and report the exact artifact in the Mac smoke | Weights are the fixed unified-memory tax paid before the first agent. |
| Unified-memory per agent | `weights(quant) + 2 * layers * kv_heads * head_dim * bytes_per_KV_element * private_tail_tokens`; #3811 must calculate it from checkpoint metadata and report agents-that-fit on the target Mac | This charges both the fixed model residency and the marginal private-tail KV cost per concurrent agent. |
| Shared-prefix structure | Measured system+tool-prefix tokens divided by total input tokens, plus fak `reused_tokens` in the A/B run | This is the cross-agent KV surface stored once with caching on instead of once per agent. |
| Context usefulness | Model-card context limit, capped by fak's tested serve limit; report the bounded horizon used | Long useful horizons increase both agent utility and the value of sharing the stable system/tools prefix. |
| Agentic/tool-use floor | #2721-compatible bounded tool loop: completion rate, valid tool-call rate, and no-progress turns at the chosen quant, or `not yet` | A smaller model that fits more replicas but cannot sustain the workload is not a net win. |
| Existing Metal evidence | Captured fak Metal load/decode/serve artifact with model, quant, tok/s, and RSS | De-risks the device-gated smoke; modeled capacity is not substituted for measured serving. |
| Many-agent value | #3811 agents-that-fit with caching off/on, then measured TTFT and completed work under K-agent load in #3813/#3815 | This is the decision target. Single-stream tok/s is supporting evidence only. |

## Candidate matrix

`Load status` is deliberately conservative: **host-verified path** means the architecture
classifier is pinned by source/tests, not that this exact checkpoint has completed the
required Metal load+serve smoke. That device witness is #3814.

| Candidate | Representative architecture / scale | Verified in-kernel path | Load status today | Weight / KV / context assessment | Existing evidence | Selection verdict |
|---|---|---|---|---|---|---|
| **Qwen2.5-7B Q8** | dense `qwen2`, 7B GQA | `attnSeq-gqa` | Host-verified path; exact Metal load+serve still required | Mid-sized weight tax; GQA reduces KV versus full MHA; long-context candidate. #3811 must emit exact bytes from checkpoint metadata. | #67 reports the same Qwen2.5-7B-Q8 path at **0.988x llama.cpp Metal decode parity**; this is relative parity evidence, not a fresh tok/s claim. | **Provisional pick:** strongest combination of already-proven fak Metal proximity, useful 7B capability, and GQA cache economics. |
| Qwen2.5-Coder-7B Q8 | dense `qwen2`, 7B GQA | `attnSeq-gqa` | Host-verified path; exact checkpoint smoke pending | Same capacity class as Qwen2.5-7B; coding specialization may improve agent work but must be benchmarked. | `TestClassifyForwardPathSelectedSmallGQACandidates` pins the Qwen2.5-Coder mapping; no cited real-Mac smoke for this checkpoint yet. | First quality-specialized alternate if the agent benchmark beats the base model without a material footprint penalty. |
| Llama-3.2-3B Q8 | `llama`, 3B GQA | `attnSeq-gqa` | Host-verified path; exact checkpoint smoke pending | Lowest weight tax in the seeded set and favorable agents-per-GB; lower capability is the principal risk. | The classifier test pins `llama` to the standard GQA path; real-Mac quality and serving evidence are `not yet`. | Capacity-floor alternate; promote only if the bounded agent workload remains reliable. |
| Gemma-3-4B | `gemma3`, 4B GQA/QK-norm family | `attnSeq-gqa` (standard uniform geometry) | Generic path is present, but this exact checkpoint is **not** pinned by the selected-candidate classifier and has no load smoke | Attractive middle weight class; exact KV geometry/context must come from the chosen GGUF. | Resolver coverage exists for `gemma3`, but that is weaker than an exact load+serve witness. | Hold behind an explicit classifier/load witness; do not infer Gemma-4 support from the name. |
| Gemma-4-4B | `gemma4`, heterogeneous per-layer geometry | `gemma4` | Host-verified path; exact checkpoint smoke pending | Attractive middle weight class; heterogeneous head dimensions require checkpoint-derived KV accounting, not a uniform estimate. | `arch_support_gemma4.go` validates per-layer geometry and the selected-candidate test pins `gemma4`. | Supported alternate, but less de-risked on Metal than Qwen2.5-7B. |
| Qwen3-4B (dense) | expected dense Qwen GQA, 4B | **Not verified for the exact checkpoint** | Hold: no exact candidate mapping/load witness in the cited support seam | Attractive footprint on paper; KV/context values remain checkpoint-derived. | Current code explicitly names recognized `qwen35` hybrids and refuses unrecognized linear-attention signatures; that does not prove an arbitrary Qwen3-4B GGUF. | Do not select until #3814 records its exact architecture key and typed path/refusal. |
| Qwen3.5/3.6-27B | `qwen35`-family GDN hybrid, 27B baseline | `qwen35-gdn` only when `layer_types` identifies the hybrid | **The target GGUF refusal is the #934 state:** if metadata leaves `layer_types` empty while linear-attention tensors are present, `refuseUnsupportedHybridArch` rejects it at load time | Weight footprint consumes the memory budget needed for concurrent agents before private-tail KV is counted. | The typed refusal is pinned in `arch_support.go`; #4381 is a separate real-Mac serving-curve benchmark and no result is imported here. | **Explicitly excluded baseline:** refused for the unresolved metadata signature and too large for the many-agent fit target. |

## Code-derived architecture evidence

The support column is reproducible from the committed code rather than model-family
name matching:

1. `internal/model/arch_support.go` calls the typed hybrid-architecture refusal before
   classification, then dispatches `gemma4`, MLA/MoE, MiniMax, and recognized Qwen3.5
   hybrid paths; standard separate-projection GQA falls through to `attnSeq-gqa`.
2. `internal/model/arch_support_test.go` directly pins Llama-3.2-3B and
   Qwen2.5-Coder-7B to `attnSeq-gqa`, and Gemma-4-4B to `gemma4`.
3. `internal/model/arch_support_gemma4.go` rejects malformed Gemma-4 heterogeneous
   geometry at load time. Gemma-3 is therefore not labeled `gemma4`; its generic-path
   status and weaker exact-checkpoint evidence are shown separately.
4. A family resemblance is not a witness. Qwen3-4B remains held until its actual GGUF
   metadata classifies or returns a typed refusal. This avoids silently treating a
   linear-attention hybrid as dense GQA.

## Provisional pick and promotion gates

**Pick Qwen2.5-7B Q8 for the first real-Mac run.** It is the best *many-agent* value
candidate now because it combines a useful 7B capability class, GQA KV economics, and
#67's existing 0.988x llama.cpp Metal decode-parity result on the same model/quant class.
That ratio is the tuned comparative evidence available today; it is not presented as a
new absolute throughput measurement.

The pick is provisional until all of these are witnessed:

1. #3811 reports weights, checkpoint-derived KV bytes/token, and agents-that-fit with
   caching off/on for the target Mac memory budget.
2. #3814 loads this exact checkpoint through `ClassifyForwardPath`, serves a real
   `/v1/chat/completions` turn on Metal, and captures tok/s plus RSS.
3. #3813 measures fak-on versus fak-off shared-prefix reuse under concurrent agents.
4. #3815 runs the bounded long-horizon many-agent command and reports completed work,
   reused tokens, agents/GB, and TTFT under concurrency.

If Qwen2.5-7B fails memory or quality gates, evaluate Qwen2.5-Coder-7B (quality alternate),
then Llama-3.2-3B (capacity alternate), before expanding to candidates without an exact
load witness. No real-Mac capacity or performance gain is claimed here: those remain
**not yet** until the named device artifacts land.
