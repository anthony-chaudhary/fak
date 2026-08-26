---
title: "Qwen4-Exp support, rollback, and upstream watch"
description: "Operational index for the source-pinned Qwen3.8-Flash-Next (qwen4_exp_text) fak-native program."
---

# Qwen4-Exp support, rollback, and upstream watch

**Observed:** 2026-08-26
**Tracker:** [#9214](https://github.com/anthony-chaudhary/fak/issues/9214) · **Epic:** [#9204](https://github.com/anthony-chaudhary/fak/issues/9204) · **Source study:** [#9122](https://github.com/anthony-chaudhary/fak/issues/9122)
**Study receipt:** `study_207f3c56d6e23d2ccfb0d0881fde3a3a8ca1f81d7952897d1a87f61a61a4d383`
**Pinned upstream:** [`QwenLM/Qwen3.8-Flash-Next@513aa6e18a335296fc13e538232a8735b230877d`](https://github.com/QwenLM/Qwen3.8-Flash-Next/tree/513aa6e18a335296fc13e538232a8735b230877d) · [`Qwen/Qwen3.8-Flash-Next@f5d08274bafd880402bd16f5e3e6c514136ec06c`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/tree/f5d08274bafd880402bd16f5e3e6c514136ec06c)

## Verdict and naming

This page is the phase-exit index, not a support announcement. The source checkpoint calls the architecture `qwen4_exp_text`; fak calls its bounded implementation program **Qwen4-Exp**. The public artifact is still named **Qwen3.8-Flash-Next**. These names identify one pinned program here, but must not be generalized to later upstream revisions.

Status is **HOLD** for production inference. The committed fak-native config, four-layer oracle, Flash-Next spine, and MTP admission gate are correctness evidence only. There is no immutable end-to-end receipt yet for full-weight loading, quality-constrained output, production serving, or a matched performance envelope.

## Milestones and phase exits

| Milestone | State | Required exit receipt |
|---|---|---|
| M0 source and checkpoint contract | PASS | The source study receipt and the two immutable pins above. |
| M1 deterministic architecture oracle | PASS | `go test ./internal/qwen4exp -run 'Test(TinyPromptTrace|TensorLayout)$' -count=1` at the tested fak commit. |
| M2 config and Flash-Next loading spine | PASS | `go test ./internal/model -run 'TestQwen4Exp' -count=1` at the tested fak commit. |
| M3 native full-checkpoint load | HOLD | Receipt naming checkpoint revision, tensor manifest digest, engine `fak-native/qwen4exp`, device, peak memory, and zero fallback. |
| M4 tokenizer, chat, tools, and quality parity | HOLD | Captured prompt bytes, parsed tool calls, semantic stops, corpus digest, comparator revision, and acceptance thresholds. |
| M5 optimized prefill/decode | HOLD | Net-true matched-envelope receipt including setup, recovery, verification, quality, memory, latency, and throughput. |
| M6 production promotion | HOLD | Soak, restart/state-restore, observability, rollback drill, and a fresh upstream-watch PASS. |

A phase cannot exit on a vendor recipe, compilation alone, or a comparator run. The receipt must identify the fak-native engine and immutable inputs.

## Support matrix

| Surface | Current state | Evidence class | Operational meaning |
|---|---|---|---|
| Architecture/config `qwen4_exp_text` | PARTIAL | fak-authored | Exact pinned config is recognized and unsupported variants fail closed; this is not full inference. |
| Four-layer GDN/MoE/sparse-attention oracle | TEST-ONLY | fak-authored | Deterministic tiny-shape correctness witness, not checkpoint-scale quality or speed. |
| Flash-Next tensor/load spine | PARTIAL | fak-authored | Pinned layout gates exist; full checkpoint loading remains M3 HOLD. |
| Hybrid MTP n-gram-3 admission | GATED | fak-authored | Candidate activation requires matching model revision and fak-native execution identity. |
| Transformers | REFERENCE-ONLY | upstream/vendor | The pinned model card gives a recipe; fak has not witnessed it in this operating envelope. |
| vLLM | REFERENCE-ONLY | upstream/vendor | Comparator/parity use only; never a silent native fallback. |
| SGLang | REFERENCE-ONLY | upstream/vendor | Comparator/parity use only; never a silent native fallback. |
| llama.cpp / GGUF | UNVERIFIED | upstream/vendor | Converter vocabulary uses `qwen4exp`; exact checkpoint-to-GGUF identity is unresolved. |
| MLX | UNVERIFIED | upstream/vendor | No immutable compatible runtime receipt was found in the source study. |
| FP8, AWQ, GPTQ, GGUF, Unsloth | UNVERIFIED | upstream/vendor | Ecosystem availability claims do not prove fak loading or execution. |

## Exact launch and comparator recipes

Run from repository root at the commit being evaluated. These commands are intentionally fail-closed and do not imply that M3–M6 have passed.

```bash
# Deterministic fak-native correctness and admission witnesses (no model download).
go test ./internal/qwen4exp -run 'Test(TinyPromptTrace|TensorLayout|MTP)' -count=1
go test ./internal/model -run 'TestQwen4Exp' -count=1

# This document and simulated upstream-pin stale transition.
go test ./internal/qwen4exp -run TestQwen4ExpSupportWatchDocument -count=1
```

For full-model work, first materialize the checkpoint by immutable revision and record its resolved snapshot path and manifest digest. Launch only through the explicit fak-native Qwen4-Exp command introduced by M3; until that command and receipt exist, the launch verdict is **HOLD**, not permission to substitute another engine.

Comparator commands are templates that must be filled with immutable runtime revisions and executed in isolated environments. They are parity/reference probes, not fak launch commands:

```bash
# Transformers reference; install <transformers-commit>, never an unpinned moving branch.
python -m pip install 'git+https://github.com/huggingface/transformers.git@<transformers-commit>'
python <pinned-transformers-runner.py> --model Qwen/Qwen3.8-Flash-Next --revision f5d08274bafd880402bd16f5e3e6c514136ec06c

# vLLM and SGLang references; replace image digests and preserve the checkpoint revision in the receipt.
docker run --rm --gpus all <vllm-image@sha256:digest> --model Qwen/Qwen3.8-Flash-Next --revision f5d08274bafd880402bd16f5e3e6c514136ec06c
docker run --rm --gpus all <sglang-image@sha256:digest> python -m sglang.launch_server --model-path Qwen/Qwen3.8-Flash-Next --revision f5d08274bafd880402bd16f5e3e6c514136ec06c

# llama.cpp and MLX remain HOLD until a converter/model artifact digest is recorded.
# Do not translate these HOLD lines into convenience fallbacks.
```

Every comparator receipt must include runtime commit/image digest, checkpoint revision, prompt bytes, sampling and stop configuration, device, and output digest. Results from unlike parameterizations or the conflicting 125B/14B vendor description are not a matched envelope.

## Observability and immutable receipt contract

A promotable run emits one immutable receipt containing:

- fak commit and module revision, engine `fak-native/qwen4exp`, forward path, checkpoint revision, tensor-manifest digest, tokenizer/template digest, and runtime/config digest;
- device and software envelope, quantization/dtype, context and batch shape, seeds, sampling, semantic stops, and tool parser;
- setup/download/load/warmup/recovery/verification time, peak memory, prefill/decode latency, throughput, quality result, and all rejected/fallback attempts;
- upstream-watch baseline digest and PASS/HOLD verdict; and
- rollback target plus the result of restoring it.

Logs must expose stable rejection reason, model revision, engine identity, phase, and receipt ID without credentials, private hosts, prompts containing secrets, or raw internal logs.

## Known limits and rejection reasons

| Rejection reason | Trigger | Required action |
|---|---|---|
| `QWEN4EXP_UPSTREAM_STALE` | Any watched upstream pin differs from this page's baseline. | Mark every dependent row HOLD; re-study and issue a new receipt without rewriting history. |
| `QWEN4EXP_REVISION_MISMATCH` | Model, tokenizer, converter, or MTP binding differs from the admitted revision. | Refuse execution and rebuild evidence for the exact tuple. |
| `QWEN4EXP_NATIVE_PATH_MISSING` | Full-model fak-native command or engine identity is absent. | Remain at M3 HOLD; do not fall back. |
| `QWEN4EXP_LAYOUT_UNSUPPORTED` | Config/tensor inventory differs from the pinned contract. | Fail before allocation and extend the bounded loader deliberately. |
| `QWEN4EXP_COMPARATOR_UNPINNED` | Comparator package, image, checkpoint, or converted artifact is moving/unresolved. | Reject the comparison until every input is immutable. |
| `QWEN4EXP_ENVELOPE_UNMATCHED` | Quality, sampling, prompt, context, device, accounting, or parameter identity differs. | Label the result diagnostic-only; do not claim a gain. |
| `QWEN4EXP_RECEIPT_INCOMPLETE` | Required identity, cost, quality, fallback, or rollback fields are absent. | Keep the phase HOLD. |

Known gaps are full-weight fak-native loading, exact tokenizer/tool-loop parity, sparse-indexer scale behavior, checkpoint-scale MTP quality, production memory bounds, and net-true performance. The source study also records unresolved upstream parameter-count and recipe drift.

## Rollback

1. Stop admission of new Qwen4-Exp runs and retain the failing receipt.
2. Disable the Qwen4-Exp model binding or MTP admission; do not redirect it to llama.cpp, vLLM, SGLang, Transformers, or MLX.
3. Restore the last witnessed non-Qwen4-Exp fak-native model/config tuple by immutable revision.
4. Run that target's captured smoke, quality, state-restore, and policy witnesses; record the rollback receipt.
5. Keep the rejected Qwen4-Exp tuple HOLD until its rejection reason is retired by a new immutable receipt.

Rollback changes the active binding, never historical receipts or this baseline.

## Upstream watch and stale transition

Before every phase exit, compare these six watched identities with the immutable baseline: upstream repository/model checkpoint, Transformers, vLLM, SGLang, llama.cpp, and MLX. Runtime entries are deliberately `UNVERIFIED` until a concrete compatible commit or image digest is witnessed; discovering support is itself a change requiring review, not an automatic promotion.

The deterministic Go witness reads this page and `internal/qwen4exp/testdata/support_watch_changed.json`. The fixture simulates changed upstream-model and Transformers pins. It must mark the architecture, loading, tokenizer/parity, Transformers, and production-promotion rows stale while preserving baseline text and historical receipts byte-for-byte. A real refresh follows the same rule: open a new study/receipt, update dependent statuses only after review, and never rewrite old evidence.

## Evidence ledger

| Claim group | Provenance | Immutable locator or label |
|---|---|---|
| Architecture, report, model-card recipes | upstream/vendor | GitHub `513aa6e18a335296fc13e538232a8735b230877d`; checkpoint `f5d08274bafd880402bd16f5e3e6c514136ec06c`; study receipt above. |
| fak oracle, loader spine, and MTP gate | fak-authored | Repository commit under test plus focused commands above. |
| Transformers, vLLM, and SGLang compatibility | upstream/vendor | REFERENCE-ONLY; compatible runtime revisions are UNVERIFIED pending comparator receipts. |
| llama.cpp/GGUF and MLX compatibility | upstream/vendor | UNVERIFIED; no checkpoint-to-artifact immutable receipt. |
| Full-model quality, performance, and production support | unverified | HOLD until M3–M6 receipts exist. |
