# Astra-class recurrent-depth support: scoped source audit

Observed 2026-09-05. Tracker: #11700. Research only; no native model execution,
weight download, parity result, or performance claim is part of this study.

## Decision

Separate hosted Astra compatibility from native recurrent-depth execution.
The reviewed Astra launch material does not establish a reproducible native
architecture or checkpoint contract. Nanbeige and Huginn are public reference
architectures, not evidence of Astra's internals. Existing token-recurrent
SSM state is not equivalent to repeated transformer execution within a token.

For: operators running governed models through fak.
Problem: physical layer identity currently carries execution/cache assumptions.
Today: the inspected native path loops over configured physical layers.
Better because: explicit execution depth permits shared weights without aliasing
state and without routing native work to an external runtime.
Witness: reference-matched logits and incremental/full-prefill parity, proposed
below and NOT executed. Tier 2 Serving enables Tier 1 All-in-one; centrality Core.

## Inspected FAK anchors and limits

- `internal/model/config.go`: `NumLayers` reads `num_hidden_layers`.
- `internal/model/forward.go`: physical `l < cfg.NumLayers` execution loop.
- `internal/model/weights.go`: physical layer prefixes identify weights.
- `internal/model/kvcache.go`: layer-indexed K/V and shared token positions.
- `internal/model/qwen35_state_identity.go`: token/layer lineage is useful prior
  art, not proof that depth recurrence is supported.
- `internal/engine/continuous_batcher.go`: active decode/yielded-I/O state is an
  integration point; this inspection is not an exhaustive scheduler absence proof.

These observations identify requirements in the inspected path, not a repository-wide
claim that every other execution backend lacks recurrence.

## Native requirements, in dependency order

| Priority | Change | Boundary / first witness |
|---|---|---|
| P0 | Admit explicit recurrent architectures and shared block tensor mapping | Reject unsupported variants; test real checkpoint tensor shapes |
| P0 | Separate physical weight block index from execution-depth index | Nanbeige fixed 22 blocks x 2 loops produces reference-matched logits |
| P0 | Allocate KV by execution identity, independently of token position | Reusing weights must not overwrite earlier-loop KV |
| P0 | Preserve exact normalization, RoPE and residual ordering | Full-prefill vs incremental decode parity |
| P1 | Include model/config revision, depth and cache policy in reusable state identity | Prefix restore rejects incompatible depth policy |
| P1 | Preserve recurrence lineage through chunked prefill, reset, eviction and batching | One two-request isolation reproduction |
| P1 | Budget actual execution steps and KV, not parameter count | Admission receipt accounts for expanded execution slots |
| P1 | Expose depth, stop reason, cancellation boundary and engine identity | Real CLI receipt reports fak-native and completed work |
| P2 | Add adaptive latent recurrence and bounded stopping | Huginn fixed-depth reference first, adaptive stopping second |
| P2 | Add heterogeneous-depth scheduling and optional cache optimizations | Quality-constrained end-to-end witness before optimization |

Nanbeige4.2 config explicitly specifies 22 layers, 2 loops, hidden size 3072,
48 attention heads, 8 KV heads and head dimension 128. Deriving head dimension
as 3072/48 would incorrectly give 64. Treating it as ordinary Llama with a
larger layer count is insufficient. Its default execution-index KV implies
44 execution slots despite 22 shared weight blocks; this is a slot-count
inference, not a measured byte footprint. The inspected Python implementation
selects physical cache indices only when its optional shared-cache setting is
active. Do not apply dormant newer-model features to the 4.2 configuration.

Huginn additionally needs prelude/core/coda, recurrent latent initialization,
input injection and a stopping policy. Its cache is indexed by recurrent step
and sequence position; specific compressed/reused-cache policies are alternatives,
not universal semantics. Preserve the initial-state/seed policy in reproducibility.
Branching, rollback and speculative decoding require a separate lineage witness
before advertising compatibility.

## Hosted Astra: independent workstream

The OpenAI reasoning guide specifies Responses-only function calling for Astra
and rejects reasoning effort `none`. Preserve original response items, opaque
reasoning payloads and message phase where required, rather than reconstructing
reasoning from its readable summary. Keep sensitive opaque data out of public logs.

Local audit: `internal/modelroute/account.go` defaults generic OpenAI routing to
Chat Completions while Codex uses Responses. `internal/agent/adapters.go` reconstructs
summary text; this is not preservation of the original encrypted reasoning item.
Original tool `call_id` values are already preserved in inspected adapter paths;
async correlation is NOT a proven defect here.

Reconcile rather than duplicate existing trackers: #11516 (model schema adaptation),
#11531 (reasoning propagation), #9550 (separate reasoning metadata capture).
One meaningful hosted witness should exercise a real adapter tool continuation
with supported effort, opaque item round-trip and unchanged call ID.

## Candidate decisions and implementation sequence

- Adapt Nanbeige fixed-loop semantics first: smallest inspectable native spine.
- Borrow Huginn execution/cache separation after fixed-depth parity; do not adopt
  adaptive scheduling before semantic correctness.
- Use Nanbeige's vLLM fork as an implementation reference, not an external engine
  fallback and not proof that generic upstream vLLM supports the model.
- Keep #11700 open as the requirements tracker until research is delivered;
  native implementation needs its own bounded issue and on-device witness.
- First implementation unit: fixed 22x2 Nanbeige loader/executor/cache parity.
  Later independent units: cache lifecycle, admission/observability, Huginn depth.
- No performance estimate or speedup is justified by this source inspection.

## Source ledger

Immutable revisions examined (upstream code; no remote Python executed):

- Nanbeige model/config/code: `https://huggingface.co/Nanbeige/Nanbeige4.2-3B/tree/3384e426066d1a49c3aea90a7190b81260a6533f`
- Huginn model/code: `https://huggingface.co/tomg-group-umd/huginn-0125/tree/bb6621b65e90b6a4b9b29ef88dc83866d450470c`
- Recurrent pretraining: `https://github.com/seal-rg/recurrent-pretraining/tree/1ea7220ec7eb42d13e89db0663df254d0bcdc28e`
- Nanbeige vLLM fork: `https://github.com/Nanbeige/vllm/tree/62f6de733d7ae63b759329993bc209e67afdf431`

Huginn model card and recurrent-pretraining LICENSE identify Apache-2.0.
Recheck licenses for each artifact before porting code or redistributing weights.
Source anchors: Nanbeige `modeling_nanbeige.py` loop at 2217, cache index selection
at 2254-2256 / 2275-2276, loop normalization at 2305; Huginn
`raven_modeling_minimal.py` cache at 152-157 and prelude/core/coda at 553-565.

Mutable API/announcement sources, checked 2026-09-05:

- `https://openai.com/index/gpt-6-astra/`
- `https://developers.openai.com/api/docs/guides/reasoning`

Coverage excludes exhaustive PR queues, accelerator kernels, tokenizer parity,
quantization, model weight execution and unverified claims in third-party papers.
The missing architecture specification prevents claiming native Astra support.
