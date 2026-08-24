---
title: "Native raw-model performance hill climb"
description: "The typed Qwen3.8-27B M3 Pro P32/T64 optimization graph, its evidence fences, and its update procedure."
---

# Native raw-model performance hill climb

`fak native-performance` is the authoritative, deterministic map from the current
fak-native Qwen3.8 baseline to the next measured optimization. It is a committed
planning and evidence graph, not a benchmark runner and not a performance claim by
itself.

```bash
fak native-performance
fak native-performance --json
```

The command always validates the graph before rendering it. Duplicate rung IDs,
unknown or cyclic dependencies, invalid expected ranges, and values placed in the
wrong evidence class fail closed.

## Active envelope

The graph is deliberately specific:

| Field | Value |
|---|---|
| Model | `unsloth/Qwen3.8-27B-GGUF` at `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` |
| Artifact | Q4_K_M, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` |
| Machine | Apple M3 Pro, 18 GPU cores, 36 GiB unified memory |
| Execution | `engine=fak-native`, `backend=metal`, `forward_path=metal/qwen35-hybrid-session-v1` |
| Workload | temperature zero, P=32/T=64, three repetitions |
| Metric | end-to-end decode tokens/second, including submission, synchronization, and readback |

Changing any identity, prompt/decode length, sampling setting, repetition count, or
execution marker creates another envelope; it does not silently update this one.

## Committed rungs

| Rung | Depends on | Enabled / status | Expected floor..roof | Witnessed | Gap / next issue |
|---|---|---|---:|---:|---|
| `resident-q4k-baseline` | — | yes / present | 2.3..3.3 tok/s, **hypothesis** for a retest | **3.3 tok/s fak-native**, witnessed 2026-08-23, approximate | Native used P=31/T=64, and repeated synchronous Q4_K submissions remain; [#8324](https://github.com/anthony-chaudhary/fak/issues/8324) |
| `coarse-resident-hybrid-graph` | baseline | no / partial | 5..6.966061 tok/s, **hypothesis** | pending | Keep activations and hybrid state device-visible under one coarse token submission; [#8324](https://github.com/anthony-chaudhary/fak/issues/8324) |
| `matched-native-parity` | coarse graph | no / absent | 6.62..6.966061 tok/s, **hypothesis** | pending | Capture an exact joint P32/T64 quality-constrained receipt; [#8697](https://github.com/anthony-chaudhary/fak/issues/8697) |

The 3.3 native value and 6.966061 llama.cpp b9828 value come from
[#8697](https://github.com/anthony-chaudhary/fak/issues/8697). The native run decoded
64 tokens after 31 prompt tokens, while the comparison is P32/T64; therefore the graph
labels the cross-engine relationship **approximate**, even though machine and artifact
match. The older accepted native range remains 2.3–2.9 tok/s in
[`QWEN38-27B-LATEST.md`](QWEN38-27B-LATEST.md).

The expected values are planning hypotheses:

- 5 tok/s is #8324's stated promotion target.
- 6.62 tok/s is the rounded 95%-of-comparison threshold required by #8697.
- 6.966061 tok/s is reused as a diagnostic planning bound. It is a witnessed
  llama.cpp comparison, not a measured fak-native roof and not a hardware roof.

Only the `witnessed` field can carry a fak-native measurement. A projected value must
remain under `expected` with `classification=hypothesis`; graph validation rejects a
`witnessed` classification there and rejects a `hypothesis` classification in the
measurement field. A pending rung serializes `"witnessed": null` rather than implying
zero throughput.

## How to climb and update the graph

1. Pick one disabled rung whose dependency IDs resolve to completed lower rungs. Keep
   every other feature state fixed so the arm attributes one change.
2. Profile at the hierarchy that is actually limiting the run, then implement only
   that rung. For the current profile, #8324 owns the coarse resident hybrid-token
   submission; llama.cpp remains a comparison/reference path, never an automatic
   execution fallback.
3. Run baseline and candidate with the exact envelope above. Capture each command,
   revision, model hash, per-repetition result, quality result, memory/resource data,
   and the fak-native execution identity.
4. Put a result in `witnessed` only after that captured end-to-end receipt exists.
   Update `enabled`, `status`, `gap`, and `next_issue` from source state. Keep an
   unmeasured floor or roof classified as `hypothesis`, even when an issue supplies it.
5. Run `go test ./internal/nativeperf ./cmd/fak`, then inspect both CLI forms. JSON is
   the machine interface; the human table is the operator checklist.

Kernel-only or stage-only improvements may explain a hypothesis, but they do not become
decode tokens/second. Setup, submission, synchronization, fallback, and verification
overhead remain in the end-to-end metric.

## Design inspirations and disposition

Observed 2026-08-24. These sources shaped the graph discipline only; no external code,
tests, comments, or measured values were copied. Disposition: **INSPIRE-ONLY**.

- NVIDIA's 2020-11-18
  [Nsight Compute roofline analysis](https://developer.nvidia.com/blog/accelerating-hpc-applications-with-nsight-compute-roofline-analysis/)
  explains hierarchical L1/L2/DRAM ceilings. The transferable rule is to localize the
  active bottleneck before optimizing and to keep a subsystem ceiling distinct from an
  end-to-end witness. The NVIDIA/CUDA numbers are not projections for Apple Metal.
- MLCommons' inference policy at
  [`d3eba2f` (2026-08-20), scenarios](https://github.com/mlcommons/inference_policies/blob/d3eba2f21026d868ad65cdcad2bb81e4a17ce3d3/inference_rules.adoc#L132-L143)
  and
  [configuration/reproducibility](https://github.com/mlcommons/inference_policies/blob/d3eba2f21026d868ad65cdcad2bb81e4a17ce3d3/inference_rules.adoc#L539-L546),
  keeps scenario metrics and system configuration explicit. The graph adapts that
  discipline by pinning one P/T, artifact, machine, sampling, repetition, and execution
  identity envelope.
- vLLM's pinned
  [prefix-caching off/on example at `7797b60` (2026-08-24)](https://github.com/vllm-project/vllm/blob/7797b6022c129b862e45ae6aed08822e65d1bccb/examples/features/automatic_prefix_caching/prefix_caching_offline.py#L37-L75)
  and SGLang's independently controllable
  [`chunked_prefill_size` and `disable_radix_cache` arguments at `586211b` (2026-08-25)](https://github.com/sgl-project/sglang/blob/586211bc461c4dbd8df9932bf709aa3d018945d1/python/sglang/srt/server_args.py#L832-L836)
  ([radix switch](https://github.com/sgl-project/sglang/blob/586211bc461c4dbd8df9932bf709aa3d018945d1/python/sglang/srt/server_args.py#L973-L980))
  reinforce independent feature arms. The graph applies the principle through explicit
  `enabled` state and dependencies instead of importing either serving framework.

Refresh these source observations when their pinned revisions change or when the active
hardware/model envelope changes. External feature presence never proves a gain in fak.

Execution boundary: `fak native-performance` is read-only metadata rendering. llama.cpp is permitted only as an explicitly selected parity/reference benchmark comparison; it never executes the fak-native rungs and is never an automatic or fallback engine.
