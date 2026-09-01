# QWEN38-FLASH-NEXT execution index

**Epic:** [#9204](https://github.com/anthony-chaudhary/fak/issues/9204)  
**Source study:** [#9122](https://github.com/anthony-chaudhary/fak/issues/9122) · [`CONCEPT-STUDY-QWEN38-FLASH-NEXT-2026-08-26.md`](CONCEPT-STUDY-QWEN38-FLASH-NEXT-2026-08-26.md)  
**Study receipt:** `study_207f3c56d6e23d2ccfb0d0881fde3a3a8ca1f81d7952897d1a87f61a61a4d383`  
**Pinned sources:** `QwenLM/Qwen3.8-Flash-Next@513aa6e18a335296fc13e538232a8735b230877d`; `Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`

This is the durable repository index for the epic. GitHub owns mutable issue state; this file owns the dependency map, evidence contract, and source/update pointers. Qwen3.8-Flash-Next is `qwen4_exp`, not dense Qwen3.8-27B (#8011).

## Phase map

| Phase / milestone | Exit evidence | Packets |
|---|---|---|
| [F0 — Contract & admission](https://github.com/anthony-chaudhary/fak/milestone/18) | Immutable artifact/config/tensor/template identity; incompatible execution fails before allocation | [#9124](https://github.com/anthony-chaudhary/fak/issues/9124), [#9125](https://github.com/anthony-chaudhary/fak/issues/9125), [#9205](https://github.com/anthony-chaudhary/fak/issues/9205), [#9206](https://github.com/anthony-chaudhary/fak/issues/9206) |
| [F1 — Fak-native correctness](https://github.com/anthony-chaudhary/fak/milestone/19) | Full pinned checkpoint produces deterministic text/JSON/tool output and restorable state with engine=`fak-native` | [#9123](https://github.com/anthony-chaudhary/fak/issues/9123), [#9198](https://github.com/anthony-chaudhary/fak/issues/9198), [#9207](https://github.com/anthony-chaudhary/fak/issues/9207), [#9208](https://github.com/anthony-chaudhary/fak/issues/9208) |
| [F2 — Hardware & net-true performance](https://github.com/anthony-chaudhary/fak/milestone/20) | Reproducible CUDA and Metal envelopes; explicit PLE/index placement; matched quality-constrained frontier | [#9209](https://github.com/anthony-chaudhary/fak/issues/9209), [#9210](https://github.com/anthony-chaudhary/fak/issues/9210), [#9211](https://github.com/anthony-chaudhary/fak/issues/9211), [#9126](https://github.com/anthony-chaudhary/fak/issues/9126), [#9212](https://github.com/anthony-chaudhary/fak/issues/9212) |
| [F3 — Product readiness](https://github.com/anthony-chaudhary/fak/milestone/21) | Exact registry/readiness identity, operations receipts, support matrix, rollback, and stale-source watch | [#9213](https://github.com/anthony-chaudhary/fak/issues/9213), [#9214](https://github.com/anthony-chaudhary/fak/issues/9214) |

## Critical path

```text
#9122 source study
  ├─ #9124 chat/tool/stop fixtures ─┐
  └─ #9123 + #9198 tiny oracles ───┼─ #9125 exact config/tensor admission
                                   ├─ #9205 immutable artifact manifest
                                   └─ #9206 upstream regression fixtures
                                              │
                                           #9207 full-checkpoint base decode
                                              │
                                           #9208 state/prefix/restore
                                              │
                         ┌────────────────────┼────────────────────┐
                      #9209 CUDA           #9210 Metal        #9211 placement
                         └────────────────────┼────────────────────┘
                                     #9126 optional MTP gate
                                              │
                                           #9212 frontier
                                              │
                                           #9213 readiness
                                              │
                                           #9214 docs/watch/rollback
```

The MTP gate is optional and default-off. It cannot become a dependency of ordinary autoregressive correctness. CUDA and Metal implementation can fan out only after the exact tensor/layout fixtures are frozen.

## Evidence contract

Every phase-exit receipt carries:

1. exact model, checkpoint, shard, config, tokenizer, template, and generation-config identity;
2. engine identity (`fak-native`) and explicit fallback=`none`;
3. hardware/software/runtime identity and declared dtype/quant/placement;
4. quality result plus text, structured JSON, correlated tool call, and state/session evidence as applicable;
5. end-to-end load, memory, traffic, TTFT, prefill, decode, throughput, synchronization, recovery, and optional MTP costs;
6. source pins and a stale trigger rather than mutable “latest” claims.

Config parsing, tiny-model tests, vendor benchmark numbers, or a reference runtime's support badge cannot alone satisfy a full-checkpoint, hardware, performance, or release gate. Context evidence is staged at 8K, 32K, 128K, and 262K; parsing the 262144 limit is not a 262K runtime claim.

## Latest upstream update ledger (observed 2026-08-26)

| Surface | Current evidence | Planning consequence |
|---|---|---|
| Model repository/checkpoint | GitHub `513aa6e`; HF `f5d0827` | These are the initial immutable pins; refresh at phase exit without rewriting prior receipts. |
| Transformers | Qwen4Exp merged in [PR #48337](https://github.com/huggingface/transformers/pull/48337); open [#48349](https://github.com/huggingface/transformers/issues/48349), [#48350](https://github.com/huggingface/transformers/issues/48350), and [#48351](https://github.com/huggingface/transformers/issues/48351) cover silent FP8 exclusion, PLE dequantization, and absent TP-plan hazards | #9206 turns these into negative fixtures; generic loader success is not numerical proof. |
| vLLM | Support/fusion/offload work remains open in [#53896](https://github.com/vllm-project/vllm/pull/53896), [#53899](https://github.com/vllm-project/vllm/pull/53899), [#53908](https://github.com/vllm-project/vllm/issues/53908), [#53909](https://github.com/vllm-project/vllm/pull/53909) | Track as source-pinned comparator/borrow evidence only; never add automatic fallback. |
| SGLang | Cookbook merged, with open PLE offload [#36514](https://github.com/sgl-project/sglang/issues/36514), QSA fallback [#36531](https://github.com/sgl-project/sglang/issues/36531), GDN dtype [#36532](https://github.com/sgl-project/sglang/issues/36532), and thinking/tool loop [#36537](https://github.com/sgl-project/sglang/issues/36537) | Add parser/kernel refusal fixtures and recheck before matched campaigns. |
| llama.cpp / MLX | No exact support claim was established by the source study | Include only when a pinned exact-artifact arm exists; otherwise report unsupported rather than substituting dense Qwen3.8. |

### Tool/runtime observation pins

These are observation inputs, not support claims or dependencies:

| Project | Observed release | Observed HEAD |
|---|---|---|
| Transformers | `v5.16.1` | `dabae5fcb924a8eece0e727b627ca5f050b40d40` |
| vLLM | `v0.28.0` | `17da48596c98946d3e3e6896e2ebd341e809f3bd` |
| SGLang | `v0.5.18` | `5263568bcbf9da7b463e25c136f6aa92eedc3c08` |
| llama.cpp | `v0.3.0` | `5e6a37cb115dc1074e274ac004373f5661909695` |
| MLX-LM | `v0.31.3` | `74e7cf931e84ef7c2f63e875adf414e20decc1c5` |

Issue #9214 owns refreshing this ledger and deciding whether changed source evidence affects any phase. Historical campaign receipts retain their original revisions.
The source study contains the architecture/tensor inventory and what-not-to-borrow decisions. The older [`qwen38-upstream-support-map-2026-08-26.md`](qwen38-upstream-support-map-2026-08-26.md) covers dense Qwen3.8-27B and is not evidence for this architecture.

## Operator queries

```powershell
gh issue view 9204
gh issue list --milestone "QWEN38-FLASH-NEXT F0 — Contract & admission"
gh issue list --milestone "QWEN38-FLASH-NEXT F1 — Fak-native correctness"
gh issue list --milestone "QWEN38-FLASH-NEXT F2 — Hardware & net-true performance"
gh issue list --milestone "QWEN38-FLASH-NEXT F3 — Product readiness"
```



