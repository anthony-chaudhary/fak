# Qwen3.8 upstream runtime, tool, context, and cache support map

**Issue:** [#8130](https://github.com/anthony-chaudhary/fak/issues/8130)  
**Observed:** 2026-08-26  
**Stale after:** 2026-09-09, or immediately when Qwen revises the model/config/template, a mapped runtime publishes a Qwen3.8-specific release or compatibility fix, or fak changes its Qwen3.8 loader, tool protocol, context policy, or cache contract.

## Decision frame

For maintainers selecting the first-class Qwen3.8-27B serving path, this map separates upstream claims from fak evidence. It does not treat `qwen3_5` dispatch compatibility, a model card benchmark, or a runtime's generic cache feature as proof that fak supports the same behavior. Source revisions are immutable commit pins observed on 2026-08-26; runtime rows identify what must be rechecked on the stale trigger.

## Pinned primary-source groups

| Group | Revision | Decision-relevant primary sources |
|---|---|---|
| Qwen model/config/template/license | [`QwenLM/Qwen3.8@2ea10dc725823bf7c3e21ce8557cbe15245132ae`](https://github.com/QwenLM/Qwen3.8/tree/2ea10dc725823bf7c3e21ce8557cbe15245132ae) | Official release repository, model links, deployment guidance, disclosed evaluations, and Apache-2.0 license. Model artifacts/config/tokenizer/chat template remain separately revisioned on Hugging Face and must be digest-pinned for a run. |
| Transformers | [`huggingface/transformers@dabae5fcb924a8eece0e727b627ca5f050b40d40`](https://github.com/huggingface/transformers/tree/dabae5fcb924a8eece0e727b627ca5f050b40d40) | `qwen3_5` configuration/modeling plus generation and cache implementations; compatibility is architecture dispatch, not a Qwen3.8 serving receipt. |
| vLLM | [`vllm-project/vllm@8d301f075b970427ae2486194f3694cdc04fde71`](https://github.com/vllm-project/vllm/tree/8d301f075b970427ae2486194f3694cdc04fde71) | Model registry, OpenAI tool parsing, paged/prefix cache, quantization, tensor parallelism, speculative/MTP, metrics. |
| llama.cpp | [`ggml-org/llama.cpp@5e6a37cb115dc1074e274ac004373f5661909695`](https://github.com/ggml-org/llama.cpp/tree/5e6a37cb115dc1074e274ac004373f5661909695) | GGUF conversion/model dispatch, chat templates, tool calls, KV cache/quantization, split modes, server metrics. Reference/interoperability only under fak's native-inference invariant. |
| SGLang | [`sgl-project/sglang@f8cc1f9525c3a0bf3b14480cc76eccb79db1b4ea`](https://github.com/sgl-project/sglang/tree/f8cc1f9525c3a0bf3b14480cc76eccb79db1b4ea) | Model registry, tool parsers, RadixAttention/prefix cache, quantization, TP, speculative decoding, observability. |
| MLX-LM | [`ml-explore/mlx-lm@74e7cf931e84ef7c2f63e875adf414e20decc1c5`](https://github.com/ml-explore/mlx-lm/tree/74e7cf931e84ef7c2f63e875adf414e20decc1c5) | Apple-Silicon model loading/conversion, quantization, prompt cache, distributed generation, chat-template use. |

**Source limits.** The official repository's benchmark disclosures are vendor-reported comparisons, not matched fak evidence. A moving default branch proves only the inspected code existed at this pin. Open/closed issue claims require reproduction. Exact Qwen3.8-27B and FP8 model revisions, shard hashes, tokenizer/config/template digests, runtime version, flags, accelerator, quality result, and engine receipt belong in each experiment ledger.

## Architecture-sensitive facts to preserve

| Fact | Upstream implication | Required fak check |
|---|---|---|
| Hybrid GDN/full-attention cadence | A `qwen3_5`-family architecture label can load while layer cadence, recurrent state, or attention semantics are wrong. | Receipt must name the fak-native engine and loader must validate config-derived layer kinds/cadence. |
| NextN/MTP heads | Presence in config does not prove speculative acceptance, quality, or net gain. | Keep disabled until target/draft token accounting, quality, and end-to-end gain are witnessed. |
| Multimodal projector fields | Text-only 27B serving must not silently claim vision support or consume projector weights. | Reject/declare unsupported projector inputs unless a native projector path is witnessed. |
| Tool-call chat template | Tool syntax is tokenizer-template data plus parser behavior, not a generic OpenAI flag. | Pin template digest and test encode, generated call parse, tool result, and resumed generation. |
| Long context | Advertised maximum does not establish memory fit, RoPE policy, GDN state correctness, or quality at length. | Publish tested context envelope and explicit extension controls; no inferred maximum. |
| FP8/GGUF quantization | Artifact compatibility, kernel support, and output quality are independent. | Record artifact digest, native kernel receipt, matched quality, residency, and net throughput. |
| Prefix/KV cache | Full-attention KV reuse and recurrent GDN state have different identity/lifetime semantics. | Key on model/template/context/policy and account separately for KV, GDN state, admission, eviction, and verification. |

## Runtime capability matrix

`Native` means a runtime-owned path at the pinned revision, not that fak adopted it. `Verify` means the generic mechanism exists but exact Qwen3.8 behavior needs a pinned run.

| Runtime | Exact model/architecture | Tools/template | Context | Cache semantics | Quant / multi-GPU / MTP | Observability | fak disposition |
|---|---|---|---|---|---|---|---|
| Transformers | `qwen3_5` family loading; exact artifact must be tested | Applies tokenizer chat template; parser/application remains caller-owned | Config/RoPE and generation cache controls | Dynamic/static/offloaded/quantized cache APIs; no fak cache identity | Quant integration is backend-dependent; device maps are not a serving TP witness; MTP requires exact support check | Generation outputs/profilers, not an operations receipt | RECIPE reference loader only |
| vLLM | Registry path must be verified against exact 27B and FP8 configs | OpenAI server and model-specific tool parser/template controls | Runtime max length and memory admission | Paged KV and prefix caching; hybrid/GDN state behavior needs exact-model witness | FP8/other quant, TP and speculative facilities exist; Qwen3.8 MTP acceptance needs measurement | Prometheus/OpenTelemetry and server metrics | OPTIONAL-MODULE benchmark/runtime comparator |
| llama.cpp | GGUF conversion/architecture path must be verified | Template-driven chat and server tool parsing | `n_ctx`/RoPE controls subject to conversion and memory envelope | Reusable/quantized KV facilities; recurrent-state correctness requires witness | GGUF quants and device split; no assumption of native Qwen MTP | Server timings/metrics | EXCLUDE from automatic native fallback; RECIPE for parity/interoperability |
| SGLang | Registry path must be verified against exact config | OpenAI-compatible tool parsers and templates | Serving length/admission controls | RadixAttention prefix reuse; hybrid state needs exact-model witness | Quant, TP and speculative facilities; exact FP8/MTP path must be pinned | Metrics/tracing facilities | OPTIONAL-MODULE benchmark/runtime comparator |
| MLX-LM | Conversion/loading on Apple Silicon must be verified for exact config | Uses tokenizer template; structured tool loop is application-owned | Generation and rotating/prompt-cache controls under unified-memory envelope | Prompt cache is runtime-local; exact hybrid-state serialization/reuse needs witness | MLX quantization and distributed generation; official FP8/MTP equivalence not inferred | CLI timing plus profiling hooks, less server-operational coverage | RECIPE for Mac comparison, not fak default |

## fak current-tree audit

| Capability | Status | Tree evidence and bounded claim |
|---|---|---|
| Qwen3.8 campaign/model identity and receipts | PRESENT | `cmd/qwen38campaign/`, `internal/cachevalue/qwen38_campaign.go`, `docs/supported/qwen38-27b.md`, and `docs/benchmarks/QWEN38-27B-LATEST.md` provide a first-class campaign and witnessed native-engine reporting. |
| Runtime registry / loader compatibility | PRESENT | `internal/model/runtime_registry.go` and its Qwen tests explicitly dispatch Qwen3.8 through the compatible `qwen3_5` architecture family; this proves dispatch, not every architecture feature. |
| Native cache accounting and campaign cache witness | PRESENT | `internal/cachevalue/qwen38_runner.go`, `cmd/fak/cachevalue_qwen38_campaign.go`, and issue-specific witnesses report cache/value behavior with engine identity. |
| Tool-call round trip pinned to Qwen3.8 template | ABSENT | Current Qwen3.8 support doc has no exact-artifact template digest plus encode/parse/tool-result/resume witness. Generic tool support cannot fill this gap. |
| Long-context quality envelope | PARTIAL | Context machinery and campaign controls exist, but `docs/supported/qwen38-27b.md` does not promote an exact long-context quality/residency envelope as supported. |
| GDN/full-attention cache-state identity | PARTIAL | Native GDN and KV work is documented and tested, but the support surface does not yet promise cross-request prefix reuse with separately keyed recurrent state and eviction accounting. |
| NextN/MTP decode | ABSENT | No promoted Qwen3.8 NextN/MTP acceptance-and-quality receipt appears in the support contract; architecture compatibility is insufficient. |
| Multimodal projector execution | ABSENT | The current 27B path is text-focused and has no native projector receipt; inputs must not be advertised as supported. |
| Quantized native execution | PARTIAL | BF16/FP8/GGUF readiness and experiment witnesses exist under `docs/_witnesses/qwen38-27b-*` and quant docs, but support remains artifact/hardware/envelope-specific rather than universal. |
| Multi-GPU | PARTIAL | sanctioned multi-accelerator campaign evidence and readiness surfaces exist, but each topology/artifact still requires a receipt and cannot be inferred from runtime support. |
| Operational observability | PRESENT | Campaign JSON, engine receipts, readiness/acceptance reports, and benchmark ledgers expose model/engine/envelope; upstream runtime metrics are not silently equated with these. |

## Decisions and dispatchable gaps

| # | Class | Decision now | Independent next witness / linkage |
|---:|---|---|---|
| 1 | DEFAULT | Keep Qwen3.8 as the preferred new native-performance target, and require a fak-native engine receipt. | Existing Qwen3.8 campaign and `docs/native-inference-goal.md`; parent #8011. |
| 2 | DEFAULT | Pin model revision plus config, tokenizer, and chat-template digests in every run. | Add a manifest assertion to the campaign receipt; link to the default-model work under #8011. |
| 3 | DEFAULT | Treat GDN state separately from full-attention KV in cache identity and accounting. | File/attach an exact two-request prefix-reuse witness with KV/GDN hit and eviction counters to the cache program. |
| 4 | RECIPE | Use Transformers as an artifact/config/template oracle, not performance evidence. | Exact prompt-token parity fixture against the pinned artifact. |
| 5 | OPTIONAL-MODULE | Keep vLLM and SGLang as explicit benchmark comparators with exact flags and revisions. | Matched fleet run: same artifact, prompts, quality gate, context, accelerator, and net accounting. |
| 6 | EXCLUDE | Never auto-fallback native/performance work to llama.cpp. | Retain explicit parity, conversion, migration, and interoperability recipes only. |
| 7 | RECIPE | Use MLX-LM only for Apple-Silicon comparison until exact model/cache behavior is witnessed. | Exact config load, template parity, cache reuse, quality, memory and tokens/s receipt. |
| 8 | WATCH | Keep NextN/MTP off by default. | Enable only after exact Qwen3.8 target/draft acceptance, quality, memory, and net-throughput evidence. |
| 9 | WATCH | Do not advertise multimodal support from projector-shaped config alone. | Native projector load/input/output witness or explicit text-only rejection test. |
| 10 | RECIPE | Long context is an explicit tested envelope, never the advertised maximum. | Length ladder with quality, peak residency, TTFT, decode, GDN/KV accounting and failure boundary. |
| 11 | DEFAULT | Tool support requires the pinned template and a complete tool round trip. | Fail-before/pass-after fixture: encode call, parse arguments, inject tool result, resume generation. |
| 12 | WATCH | Refresh this map on the stale trigger instead of copying release claims into defaults. | Re-run the deterministic witness plus inspect release notes/issues and attach new pinned evidence to #8130 follow-ons. |

## Reproduction

The captured witness in `docs/_witnesses/issue-8130/` checks the map's pinned source count, five-runtime envelope, architecture facts, decision classes/count, all three fak status classes, stale policy, and issue link. It deliberately validates documentary claims, not hardware behavior. Hardware-dependent follow-ons must run through the sanctioned fleet and publish their own engine/artifact receipts.

