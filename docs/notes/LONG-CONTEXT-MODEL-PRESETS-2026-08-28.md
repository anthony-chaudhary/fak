# Dated long-context model presets — 2026-08-28

**Verdict:** `internal/modelperfobs` now has source-pinned analytical presets for the released identities **Qwen3.8-Flash-Next** and **GLM-5.3-Flash**. They feed the existing generic estimator; they do not duplicate it, claim benchmark measurements, or introduce an external-engine fallback.

## Scope and witness

- **For:** fak-native model and runtime engineers choosing long-context test envelopes.
- **Problem:** the generic estimator deliberately accepted model-neutral ranges, leaving the released model identities and architecture uncertainty unversioned.
- **Today:** use `DatedLongContextPresets` and `LongContextScenarios` to generate the deterministic 35K, 64K, 128K, and 200K contexts at 200:1 and 300:1 prefill:decode demand.
- **Better because:** facts, analytical assumptions, and unknowns are separate fields, so a missing cache-layout fact cannot silently become a point estimate.
- **Witness:** `long_context_presets_test.go` checks exact identities, pinned facts, non-collapsed KV intervals, analytical traffic labels, deterministic scenario vectors, and execution through `EstimateLongContextEnvelope`.

Centrality is **Enabling**: the presets make the already-landed estimator usable for current released models but do not themselves execute or optimize inference. P1–P4 checks: the preset layer preserves the fak-native engine boundary, does not alter policy/security behavior, refuses unsupported context or schema inputs, and exposes rather than hides architecture uncertainty.

## Official sources consulted

Sources were read on 2026-08-28. Hugging Face configuration facts are pinned to the listed official-repository revisions so later upstream edits do not silently rewrite this note's basis.

### Qwen3.8-Flash-Next

- Qwen release blog: <https://qwen.ai/blog?id=qwen3.8-flash-next>
- Official Qwen model card: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next>
- Official config at revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`: <https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json>

The card reports 125B parameters, 6B activated parameters, and a separately reported 51B n-gram embedding parameter bank. The config reports BF16, 262,144 maximum positions, 48 layers, and a repeating three-linear-attention/one-full-attention pattern (12 full-attention layers). The preset uses the released identity exactly; it does not create a generic “Qwen3.8-Next.”

### GLM-5.3-Flash

- Z.ai release blog: <https://z.ai/blog/glm-5.3-flash>
- Official Z.ai BF16 model card: <https://huggingface.co/zai-org/GLM-5.3-Flash-BF16>
- Official BF16 config at revision `f12e0fe1f6b2ea274c11a569582edfd99d993c5e`: <https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/blob/f12e0fe1f6b2ea274c11a569582edfd99d993c5e/config.json>

The release blog reports 320B total and 18B activated parameters. The config reports BF16, 1,048,576 maximum positions, 45 layers, and 11 `deepseek_sparse_attention` layers interleaved with linear-attention layers. The preset names **GLM-5.3-Flash**, not a fabricated “GLM-5.3-Next.” No GLM-5.3 preset is added because this change only found enough public official configuration detail to bound the Flash artifact without filling architecture gaps by analogy.

## Facts, assumptions, and unknowns

| Model | Source-backed facts | Bounded analytical assumptions | Still unknown |
|---|---|---|---|
| Qwen3.8-Flash-Next | 125B parameters, 6B activated, a separately reported 51B n-gram embedding bank, BF16, 262,144 positions, 48 layers, 12 full-attention layers | The estimator conservatively charges 176B resident BF16 parameters (125B plus the separately reported 51B n-gram bank) while retaining 6B as the published activated count. Metadata overhead is 2–8%. KV bytes/token range from 24,576 (documented full-attention layers only) to 98,304 (every layer charged as the same full-attention KV shape). | whether all 51B n-gram parameters are resident and how lookup cost maps to activated FLOPs; fak-native linear-attention recurrent state/workspaces; actual fused/paged/reused traffic; exact artifact overhead. |
| GLM-5.3-Flash | 320B total, 18B activated, BF16, 1,048,576 positions, 45 layers, 11 sparse-attention layers, `kv_lora_rank=512` | Weight storage is BF16. Metadata overhead is 2–8%. KV bytes/token range from 11,264 (one BF16 compressed rank vector for each sparse layer) to 2,949,120 (conventional expanded BF16 K+V for every head and layer). | fak-native MLA/sparse-index/linear state and workspaces; compressed versus expanded cache materialization; actual fused/paged/reused traffic; exact artifact overhead. |

The KV endpoints are deliberately broad architecture interpretations, not observations. The estimator's memory roofline still charges its supplied KV range as a full-resident scan per token. For these hybrid models that is an **analytical traffic lower/upper exercise**, not measured serving behavior. A future fak-native receipt may narrow the interval only if it identifies the engine, artifact, cache representation, kernel path, context, batch/concurrency, hardware, and measurement method.

## Deterministic scenarios

Each preset generates these eight vectors in stable context-major, ratio-minor order:

| Resident context | Prefill:decode |
|---:|---:|
| 35,000 | 200:1, 300:1 |
| 64,000 | 200:1, 300:1 |
| 128,000 | 200:1, 300:1 |
| 200,000 | 200:1, 300:1 |

Integer division selects decode tokens as `context / (ratio + 1)` and prefill tokens as `ratio * decode`; an unused remainder is permitted. Hardware, concurrency, efficiency, bandwidth, and cache-hit inputs remain caller-owned runtime fields. Model totals, activated parameters, BF16 weight range, metadata range, KV interval, identity, and maximum context come from the preset.

## Execution boundary

These are fak-native planning inputs. `LongContextScenarios` invokes the generic deterministic estimator only; it does not run a model or assert achieved latency/throughput. External engines may be used later for explicit parity, reference, or interoperability work, but are not a recovery or convenience fallback. Acceptance performance evidence must name a fak-native engine in its receipt.

## Residual uncertainty and next evidence

The largest uncertainty is cache materialization for hybrid linear, sparse, MLA, and full attention. The deliberately wide intervals prevent false precision but can make fit and bandwidth conclusions inconclusive. Narrowing them belongs in a separate, hardware-backed fak-native measurement campaign with per-layer allocation/traffic witnesses; downstream outcome grading and provider trace ingestion remain outside `internal/modelperfobs`.
