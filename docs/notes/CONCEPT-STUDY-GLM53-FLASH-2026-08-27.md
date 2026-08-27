---
title: "GLM-5.3-Flash: hosted contract and fak-native architecture study"
description: "Pinned study and phased implementation epic for Z.ai GLM-5.3-Flash."
---

# GLM-5.3-Flash: hosted contract and fak-native architecture study

- **Observed:** 2026-08-27
- **Tracker:** [#9433](https://github.com/anthony-chaudhary/fak/issues/9433)
- **Durable receipt:** `study_c821e957a046f539ebbf2876517477318651faf282329ee891aebe1b06f4acff`
- **FP8 artifact:** [`zai-org/GLM-5.3-Flash@04c4e9e`](https://huggingface.co/zai-org/GLM-5.3-Flash/tree/04c4e9e95c5da8862dced7e5056455116f83a7e0)
- **BF16 artifact:** [`zai-org/GLM-5.3-Flash-BF16@f12e0fe`](https://huggingface.co/zai-org/GLM-5.3-Flash-BF16/tree/f12e0fe1f6b2ea274c11a569582edfd99d993c5e)
- **Reference code:** [`huggingface/transformers@1987631`](https://github.com/huggingface/transformers/tree/19876312341f42cf49467bb24d67271cf28cb599/src/transformers/models/glm5_next)

## Verdict

GLM-5.3-Flash needs **two separate support tracks**. Its hosted Z.ai API is a bounded provider-contract change: fak already has generic OpenAI-compatible routing, but its dedicated Z.ai client defaults to GLM-5.2 and sends `thinking.type="disabled"`, which the new model's direct API rejects. Its native checkpoint is a much larger architecture addition: `glm5_next` is neither GLM-5.2 nor Qwen3.5, despite sharing sparse-attention, MoE, and linear-attention ideas with both.

The first native packet is therefore a safety spine, not a kernel: recognize the exact identity, pin the config/tensor contract, and refuse typed before forward selection. On the current tree, any config containing a `linear_attention` layer satisfies `IsQwen35Hybrid` ([`internal/model/qwen35.go:28`](../../internal/model/qwen35.go#L28)), and `ClassifyForwardPath` can select the Qwen GDN path ([`internal/model/arch_support.go:121`](../../internal/model/arch_support.go#L121)). That is a concrete false-compatibility risk for GLM5Next.

The epic is [#9433](https://github.com/anthony-chaudhary/fak/issues/9433), with seven independently shippable packets from admission safety through a sanctioned-CUDA operating-envelope witness. External runtimes remain explicit parity or benchmark arms. Product execution, kernels, memory/state, cache, scheduling, and receipts stay fak-native throughout.

“Epic” is interpreted as the requested work tracker. No official Z.ai catalog, documentation, release, repository, or Hugging Face source inspected in this pass names a second GLM model “Epic.” The full `glm-5.3` model is a separate model and is not in this scope.

## What was released

Z.ai released `glm-5.3-flash` on 2026-08-26. The pinned checkpoint declares:

| Axis | Pinned contract | Native consequence |
|---|---|---|
| Identity | `glm5_next` / `Glm5NextForConditionalGeneration` | Require a new exact family gate; no aliasing. |
| Scale | 320B total / 18B active; hidden size 4,096 | Weight residency and movement cannot be inferred from active parameters. |
| Decoder | 45 layers; 34 KDA linear-attention and 11 DSA layers in a 3:1 cadence | Hybrid recurrent state and searchable sparse history are both required. |
| Residual topology | mHC with four residual streams in the current reference | Existing single-stream decoder composition is not sufficient. |
| Sparse attention | no-PE MLA plus KPool/index top-k selection | GLM-5.2 DSA is useful substrate, not parity. Pool/tail causality is correctness-critical. |
| MoE | 288 routed experts, top-8, plus one shared expert; first three layers dense | Loader, router, expert packing, and residency need exact witnesses. |
| Context | 1,048,576 positions | Config parsing is not a 1M-context support claim. Stage capacity and correctness. |
| Multimodal | native 24-layer vision encoder; image/video token contract | Splicing external embeddings is not native vision support. |
| Formats | block-FP8 primary checkpoint; separate BF16 checkpoint | Validate each artifact and scale layout independently. |
| Interaction | mandatory reasoning; tools and image/video/file inputs | Pin the chat/processor template and preserve reasoning history semantics. |

The model repository changed several times during its first 24 hours, including chat-template behavior. Every implementation phase must recheck config, generation config, processor config, template, API schema, reference implementation, and open runtime work; it must not silently move the study pins.

## Hosted API contract

The official [model guide](https://docs.z.ai/guides/vlm/glm-5.3-flash), [API introduction](https://docs.z.ai/api-reference/introduction), and [parameter guide](https://docs.z.ai/guides/overview/concept-param) establish the following observed contract:

- General OpenAI-compatible base: `https://api.z.ai/api/paas/v4`; chat at `/chat/completions`; bearer `ZAI_API_KEY`.
- Model ID: `glm-5.3-flash`. The coding-plan compatibility layer also documents a `glm-5.3-flash[1m]` alias for Claude Code.
- Context is 1M; default output limit is 65,536 and documented maximum is 131,072 tokens.
- Reasoning is mandatory on the direct endpoint: `thinking.type="enabled"`. `reasoning_effort` accepts `low`, `high`, or `max`, defaulting to `max`.
- Reasoning arrives as `message.reasoning_content` or streaming `delta.reasoning_content`; SSE ends with `[DONE]`.
- OpenAI-style function tools are supported, up to 128; `tool_choice` is `auto`; tool arguments are JSON strings.
- Usage includes cached prompt tokens. Documented finish reasons include normal stop/tool/length and provider safety/context/network outcomes.
- Image URL or base64 input is documented; the Flash guide also shows video and uploaded-file input. The model produces text/reasoning/tool calls, not generated images.
- When `clear_thinking=false`, the client must forward complete, ordered, unmodified historical reasoning content.

Three surfaces are **probe-gated**, not accepted from prose alone: structured JSON output, streamed tool arguments via `tool_stream`, and direct file/video URL variants. The current model guide and OpenAPI request schema do not expose these consistently.

The immediate fak incompatibility is captured at [`internal/zaitask/zaitask.go:98`](../../internal/zaitask/zaitask.go#L98): the request has only a small thinking shape, and `Run` hardcodes `disabled` at [line 140](../../internal/zaitask/zaitask.go#L140). Hosted packet [#9439](https://github.com/anthony-chaudhary/fak/issues/9439) owns the correction and conformance matrix without implying native execution.

### Mutable price snapshot

The official [pricing page](https://docs.z.ai/guides/overview/pricing) showed, per one million tokens, list prices of **$0.15 input / $0.03 cached input / $0.50 output**. A launch promotion showed **$0.075 / $0.015 / $0.25** through 2026-09-09 24:00 UTC+8. These are dated provider observations, not stable product constants or fak savings evidence.

## Upstream implementation state

| Source | Observed state | What it contributes | Limit / refresh trigger |
|---|---|---|---|
| [Transformers `1987631`](https://github.com/huggingface/transformers/tree/19876312341f42cf49467bb24d67271cf28cb599/src/transformers/models/glm5_next) | Merged | Architecture/config validation; KDA, KPool DSA, mHC and MoE reference math; unit tests | Real-weight integration test is still skipped; reference explicitly excludes MTP. Refresh on GLM5Next code/test changes. |
| [vLLM PR #53906](https://github.com/vllm-project/vllm/pull/53906) at `142062f` | Open, review-required, merge-dirty | Proposed NVIDIA cache layouts, KDA, sparse selection, MTP and FP8/BF16 operating constraints | Not shipped; multiple device/correctness defects remain open. Refresh on merge/rebase or issue closure. |
| [SGLang PR #36507](https://github.com/sgl-project/sglang/pull/36507) at `033446b` | Open, review-required, merge-dirty | Proposed KDA/DSA runtime, KPool cache, MTP and parser/serve design | Cookbook is on main, runtime is not. Refresh on PR state and image/runtime release. |
| [KTransformers `0635007`](https://github.com/kvcache-ai/ktransformers/tree/063500785b93dd64ab30ac7d0aa1c9d8f9951f77) | Support merged on main after latest inspected v0.7.0 release | Heterogeneous CPU/GPU expert-streaming and large-host-memory operational evidence | Later optimization input; submodule licenses/provenance need separate audit. Refresh on release. |

Transformers load-bearing anchors at the inspected pin are:

- config/defaults/validation: [`configuration_glm5_next.py:27-228`](https://github.com/huggingface/transformers/blob/19876312341f42cf49467bb24d67271cf28cb599/src/transformers/models/glm5_next/configuration_glm5_next.py#L27)
- MoE, mHC, KDA and FP32 state: [`modular_glm5_next.py:321-746`](https://github.com/huggingface/transformers/blob/19876312341f42cf49467bb24d67271cf28cb599/src/transformers/models/glm5_next/modular_glm5_next.py#L321)
- KPool DSA and hybrid decoder: [`modular_glm5_next.py:749-1370`](https://github.com/huggingface/transformers/blob/19876312341f42cf49467bb24d67271cf28cb599/src/transformers/models/glm5_next/modular_glm5_next.py#L749)
- cache/mHC unit coverage and skipped real-weight test: [`test_modeling_glm5_next.py`](https://github.com/huggingface/transformers/blob/19876312341f42cf49467bb24d67271cf28cb599/tests/models/glm5_next/test_modeling_glm5_next.py)

The canonical Transformers source is `modular_glm5_next.py`; `modeling_glm5_next.py` is generated. Use the former when checking or adapting math.

## What fak has today

The inward query used four layers before raw grep:

1. `fak capabilities "GLM 5.3 Flash model support" --json`
2. `fak capabilities "Z.ai OpenAI-compatible provider" --json`
3. `go run ./cmd/fak-dev index docs|leaves|verbs|claims ... --json`
4. exact issue searches and `rg -n 'GLM-5.3|glm5_next|Glm5Next'`

Capability lookup returned only generic routing/context/security cards. The self-index found older GLM-5.2 and general provider/model infrastructure, but no GLM-5.3-Flash or `glm5_next` capability or claim. Exact open/closed issue searches found no duplicate epic; [#9204](https://github.com/anthony-chaudhary/fak/issues/9204) is the neighboring Qwen4-Exp contract pattern, while [#1476](https://github.com/anthony-chaudhary/fak/issues/1476) remains a GLM-5.2 sibling rather than a duplicate.

| Capability | Status | Current evidence | Decision |
|---|---|---|---|
| Generic hosted routing | **PARTIAL** | OpenAI-compatible gateway exists; Z.ai client is GLM-5.2/reasoning-disabled | Model-scoped optional module, #9439. |
| Exact GLM5Next admission | **ABSENT** | Broad Qwen predicate at `qwen35.go:28-38`; forward selection at `arch_support.go:121-136` | Default fail-closed spine, #9435. |
| Nested config and sharded checkpoint load | **PARTIAL** | Nested `text_config` parsing at `config.go:426-458`; sharded load at `safetensors_quant.go:126-168` | Adapt exact axes/formats, #9435/#9440. |
| KDA + recurrent state | **PARTIAL substrate, ABSENT parity** | Qwen GDN implementation has incompatible projections/gates/state | Dedicated GLM oracle and state, #9434. |
| DSA + KPool + MoE | **PARTIAL substrate, ABSENT parity** | GLM-5.2 DSA at `forward.go:277-315`; no KPool/hybrid contract | Exact selection/router implementation, #9441. |
| Native real-artifact text session | **ABSENT** | No GLM5Next loader/forward/receipt | #9440. |
| Native image/video | **PARTIAL substrate, ABSENT parity** | `forwardHiddenRows` accepts external embeddings; no GLM5Next encoder | #9442. |
| Native 1M/performance claim | **ABSENT** | No engine-named GLM5Next context/performance receipt | Watch until #9438. |

## Ordered support epic

| Order | Issue | Ship-alone witness |
|---:|---|---|
| 1 | [#9435 — exact identity and typed refusal](https://github.com/anthony-chaudhary/fak/issues/9435) | Exact/near-miss config and tensor fixtures; no incompatible forward executes. |
| 2 | [#9439 — hosted contract](https://github.com/anthony-chaudhary/fak/issues/9439) | Captured text/reasoning/SSE/tool/usage/media conformance; provider engine named. |
| 3 | [#9434 — KDA + mHC oracle](https://github.com/anthony-chaudhary/fak/issues/9434) | Prefill/decode/reset/snapshot traces for one 3:1 cadence. |
| 4 | [#9441 — KPool DSA + MoE](https://github.com/anthony-chaudhary/fak/issues/9441) | Selected-index, dense-reference attention, router and four-layer hidden-state parity. |
| 5 | [#9440 — real artifact native text](https://github.com/anthony-chaudhary/fak/issues/9440) | FP8/BF16 manifests and a sanctioned-node, engine-named text session. |
| 6 | [#9442 — native image input](https://github.com/anthony-chaudhary/fak/issues/9442) | Preprocess, encoder/projector and end-to-end image-task parity; video gated separately. |
| 7 | [#9438 — long-context/performance envelope](https://github.com/anthony-chaudhary/fak/issues/9438) | Staged contexts, quality, memory and net-true comparison on sanctioned CUDA. |

MTP/NextN, P/D disaggregation, heterogeneous CPU/GPU expert streaming, and device-specialized optimized kernels stay outside the first spine. The current merged reference itself defines baseline execution without MTP; base text correctness should not wait for optional speculation or productization breadth.

## Problem frame

- For: fak operators who need a fast, long-context, reasoning/tool/multimodal GLM route
- Problem: generic provider compatibility and similar older kernels cannot prove the new API or native architecture
- Today: call Z.ai directly or explicitly select an external reference runtime
- Better because: hosted use becomes witnessed quickly while a separate, fail-closed native sequence preserves fak's kernel and operating ownership
- Witness: config/tensor/chat fixtures, conformance captures, numerical oracles, real artifact decode, then a quality-constrained sanctioned-hardware envelope
- Centrality: **Core** — the final route changes the user-visible kernel path; correctness packets are enabling work for that named outcome
- P1 Context: **advanced** — recurrent state, sparse history, reasoning history and staged 1M context are explicit
- P2 Net value: **advanced** — hosted/native/reference envelopes remain separate and all overhead counts
- P3 Adaptation: **advanced** — exact identity and independently reversible phases bound change
- P4 Operations: **advanced** — discovery, refusal, engine identity, usage, errors, capacity, and recovery are acceptance criteria

## License and provenance

- Official FP8/BF16 artifacts are MIT: config, template and test-vector adaptation is allowed with the required notice.
- The official GLM repository, Transformers, vLLM, SGLang, and KTransformers roots inspected here are Apache-2.0. Preserve notices for direct reuse.
- Prefer **adaptation** for recurrence/cache, KPool/tail semantics, mHC, parser state and serving designs so they fit fak-owned Go/kernel/state contracts.
- Treat unmerged optimization glue, custom runtime images/backend dispatch, and KTransformers heterogeneous transport as **inspiration only** until stabilized, benchmarked, and audited file-by-file.
- Root license does not automatically clear KTransformers submodules or every new file in large open PRs.
- Hosted API documentation is behavioral evidence; absent a separately verified documentation reuse license, paraphrase rather than copy.

## Completeness boundary

The official checkpoint's non-weight artifacts—README, license, config, generation config, processor config and chat template—were read at the FP8 pin. Weight shards were inventoried but not downloaded or executed. Official release/model/API/pricing sources, the merged Transformers config/model/tests/docs, the targeted vLLM/SGLang runtime PRs and related open defects, and KTransformers' GLM support/transport anchors were inspected. No upstream GPU tests or claimed benchmarks were run.

Not exhaustively inspected: raw safetensor bytes, tokenizer vocabulary bytes, every file/review/CI artifact in the large vLLM and SGLang PRs, every downstream runtime issue/discussion/fork, or KTransformers submodules. Those limits block any runtime, performance, quality, release-inclusion, or copied-code claim beyond the pinned evidence above.

Refresh triggers: an official checkpoint/template revision; Z.ai API or price change; Transformers GLM5Next code/test change; vLLM/SGLang PR merge/rebase; KTransformers release; or any implementation packet starting from a different pin.

## Companions

- [Qwen3.8-Flash-Next study](CONCEPT-STUDY-QWEN38-FLASH-NEXT-2026-08-26.md) — neighboring fail-closed hybrid-architecture spine.
- [GLM-5.2 native-performance goal](../native-inference-goal.md) — matched-envelope and engine-identity rules.
- [GLM-5.2 support epic #1476](https://github.com/anthony-chaudhary/fak/issues/1476) — useful older-family substrate, explicitly not an alias.
