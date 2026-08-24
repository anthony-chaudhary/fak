---
title: "Native raw-model performance hill climb"
description: "Separate typed Metal and CUDA Qwen3.8-27B optimization envelopes, independently attributable levers, and deterministic next witnesses."
---

# Native raw-model performance hill climb

`fak native-performance` is the authoritative, deterministic map from current
fak-native Qwen3.8 baselines to the next measured optimization. It is a committed
planning and evidence graph, not a benchmark runner or performance claim.

```bash
fak native-performance
fak native-performance --json
fak native-performance --next
fak native-performance --dot
```

The original primary Metal envelope, aggregate `rungs`, `features`, and comparison
remain in human and JSON output for backward readers. Schema v2 adds `envelopes` and
independently addressable `levers`. `--next` returns the first graph-order,
dependency-ready, disabled lever without a witness. `--dot` emits deterministic
Graphviz DOT with one cluster per envelope and explicit dependency/conflict edges.
None of these commands executes a model or llama.cpp.

Validation fails closed on duplicate IDs, unknown references, dependency cycles,
asymmetric or simultaneously enabled conflicts, cross-envelope edges, invalid
platform/backend applicability, invalid enablement/state, and any expected-versus-
witnessed evidence conflation.

## Pinned envelopes: separate curves

Metal and CUDA throughput are not combined, ranked, or drawn as one curve. Each
receipt applies only to its complete envelope.

| Field | Metal envelope | CUDA/A100 envelope |
|---|---|---|
| Stable ID | `qwen38-27b-q4km-m3pro-p32-t64` | `qwen38-27b-q4k-a100-p1-decode` |
| Model | `unsloth/Qwen3.8-27B-GGUF` at `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe` | Qwen3.8-27B GGUF identity pinned by #8635 |
| Artifact / quant | Q4_K_M, SHA-256 `7e78da5d7e3ae28d178121f58646953305f3e5bd3cb46f4a75584e8b6c6fe169` | Q4_K artifact pinned by #8635; the exact receipt must carry its hash |
| Hardware | Apple M3 Pro, 18 GPU cores, 36 GiB | NVIDIA A100; memory variant must be carried by the receipt |
| Applicability | `darwin/arm64`, Metal | `linux/amd64+nvidia-a100`, CUDA |
| Execution | `engine=fak-native`, `backend=metal`, `forward_path=metal/qwen35-hybrid-session-v1` | `engine=fak-native`, `backend=cuda`, `forward_path=cuda/q4k-p1-decode` |
| Workload | temperature zero, P=32/T=64, three repetitions | temperature zero, P=1 decode, three repetitions |
| Metric | end-to-end decode tok/s including submission, synchronization, and readback | repeated same-artifact end-to-end decode tok/s plus strict quality gates |

The CUDA metadata intentionally does not invent an artifact revision, hash, prompt
shape beyond P=1, memory size beyond the A100 class, or a measured result. The
required #8635 receipt must resolve and preserve those identities before a witness
can be populated. Changing any envelope field creates a new envelope.

## Backward-readable Metal rungs

| Rung | Depends on | Enabled / status | Expected floor..roof | Witnessed | Gap / next issue |
|---|---|---|---:|---:|---|
| `resident-q4k-baseline` | — | yes / present | 2.3..3.3 tok/s, **hypothesis** for a retest | **3.3 tok/s fak-native**, 2026-08-23, approximate | Native used P=31/T=64 and synchronous Q4_K submissions remain; #8324 |
| `coarse-resident-hybrid-graph` | baseline | no / partial | 5..6.966061 tok/s, **hypothesis** | pending | Keep activations and hybrid state device-visible under a coarse token submission; #8324 |
| `matched-native-parity` | coarse graph | no / absent | 6.62..6.966061 tok/s, **hypothesis** | pending | Capture an exact joint P32/T64 quality-constrained receipt; #8697 |

The 3.3 native value and 6.966061 llama.cpp b9828 comparison come from #8697.
The native run decoded 64 tokens after 31 prompt tokens; the comparison is P32/T64.
Their relationship remains **approximate** until a joint matched receipt captures both
engines. The older accepted native range remains 2.3–2.9 tok/s in
[`QWEN38-27B-LATEST.md`](QWEN38-27B-LATEST.md).

Expected values remain planning hypotheses: 5 tok/s is #8324's promotion target;
6.62 tok/s is the rounded 95%-of-comparison threshold from #8697; and 6.966061 is
a separately witnessed llama.cpp comparison reused as a diagnostic planning bound,
not a measured fak-native roof or hardware roof.

## Independently attributable levers

Every lever has a stable ID, one envelope applicability, enabled flag and source
state, dependency/conflict edges, a provenance-labelled expected effect, a separate
receipt-backed witnessed effect when one exists, an owning issue, and an exact next
witness requirement.

### Metal raw-decode and serving levers

| Stable ID | State | Dependencies | Expected planning effect / provenance | Witnessed effect | Owner and exact next witness |
|---|---|---|---|---|---|
| `metal.resident-q4k-weights` | enabled / present | — | Preserve the resident baseline; QWEN38 latest + #8697 | 3.3 tok/s after P31/T64, approximate to P32 | #8324: exact P32/T64 three-repetition native baseline with hash and identity |
| `metal.command-buffer-amortization` | disabled / partial | resident weights | Reduce repeated synchronous projection submissions; #8324/#8697 profile, **no gain assumed** | pending | #8324: one-lever OFF/ON P32/T64 A/B with commands, revisions, hash, each latency/tok/s, quality, identity, and profiles |
| `metal.fused-hybrid-graph-coverage` | disabled / partial | command-buffer amortization | Keep activations and recurrent/KV state visible to a coarse token graph; #8324, **no gain assumed** | pending | #8324: coverage OFF/ON with prior lever fixed ON and matched receipt/profile |
| `metal.paged-kv` | disabled / absent | resident weights | Bound KV allocation and expose occupancy; #8395, not raw-decode evidence | pending | #8395 isolated arm with quality, TTFT/ITL p50/p95, aggregate tok/s, peak memory, prefix-hit rate, fallback count |
| `metal.prefix-reuse` | disabled / absent | paged KV | Reuse exact prefix blocks; #8395 | pending | #8395 isolated prefix arm with paged KV fixed ON and complete serving receipt |
| `metal.chunked-prefill` | disabled / absent | resident weights | Bound prefill scheduling; #8395 | pending | #8395 isolated arm on identical prompts/arrival trace with complete serving receipt |
| `metal.continuous-batching` | disabled / absent | paged KV | Improve concurrent serving occupancy; #8395, not a single-request decode claim | pending | #8395 isolated batching arm with paged KV fixed ON and complete serving receipt |
| `metal.matched-parity-receipt` | disabled / absent | fused graph coverage | Plan for >=95% of separately classified llama.cpp comparison; #8697 | pending | #8697 matched native/llama.cpp b9828 P32/T64 campaign with commands, revisions, hash, each result, deterministic quality, identities, profiles |

The serving levers are independent knobs rather than one throughput rung. Their
campaign may combine them only after each isolated arm has a receipt. They remain in
the Metal envelope because #8395 owns the current native serving decomposition; a
new platform campaign must create its own envelope and lever IDs.

### CUDA/A100 P=1 Q4_K levers

| Stable ID | State | Dependencies / conflicts | Expected planning effect / provenance | Witnessed effect | Owner and exact next witness |
|---|---|---|---|---|---|
| `cuda.scalar-f32-activation-baseline` | enabled / present | conflicts with Q8_1 | Preserve the current OFF arm; #8635 reports ~11 tok/s but this graph does **not** classify it as a receipt | pending | #8635 repeated same-artifact A100 baseline with strict quality and identity |
| `cuda.q8_1-activation-quant` | disabled / absent | conflicts with scalar f32 arm | Quantize current activation vector under cosine >=0.995, exact argmax, maxAbs <=0.02; #8635, **no throughput gain assumed** | pending | #8635 strict numerical OFF/ON receipt with all gate values, artifact identity, raw output |
| `cuda.dp4a-q4k-mmvq` | disabled / absent | depends on Q8_1 | Target >=47.36 tok/s; #8635 target, **not witnessed throughput** | pending | #8635 DP4A OFF/ON repeated same-artifact A100 decode A/B with per-run end-to-end tok/s, gates, logs, identity |
| `cuda.default-decode-routing` | disabled / absent | depends on DP4A MMVQ | Promote only after correctness and end-to-end gain; #8635 | pending | #8635 full-model text/JSON/tool correctness without fallback, repeated gain, and default-path inspection |

Q8_1 and scalar-f32 activation products conflict because they are the candidate and
baseline arms at the same decode seam. `--next` can select Q8_1 once the baseline
receipt exists; its experiment explicitly toggles the conflicting baseline OFF.
The >=47.36 tok/s value is the #8635 target envelope. It is not copied into a
`witnessed` field and is never compared numerically to the Metal curve.

## Deterministic next selection and DOT

Selection walks committed lever order after validation and chooses the first lever
that is disabled, unwitnessed, and has all dependencies enabled. Current output
selects `metal.command-buffer-amortization`, owned by #8324, and prints its exact
one-lever receipt requirement. Updating enabled/witnessed state advances selection;
reordering maps or timestamps cannot change it.

DOT emits `depends` edges, dashed red bidirectional `conflicts` edges, and separate
Metal/CUDA clusters. Graphviz is not invoked by fak:

```bash
fak native-performance --dot > native-performance.dot
dot -Tsvg native-performance.dot > native-performance.svg
```

## Evidence fence and update procedure

1. Freeze the full envelope and capture a baseline artifact.
2. Toggle exactly one declared lever; keep dependencies fixed and conflicts OFF.
3. Capture the candidate using that lever's exact `next_witness` requirement.
4. Preserve raw logs, commands, revisions, model/artifact identity, quality gates,
   per-run timings, memory, and fak-native execution identity.
5. Put planning targets only in `expected`; populate `witnessed` only from the
   accepted end-to-end receipt and cite its provenance/receipt.
6. Mark the lever enabled only when the default fak-native path uses it. Add the
   accepted receipt to this document and the owning benchmark document.
7. Run focused nativeperf and CLI tests; inspect human, JSON, `--next`, and `--dot`.

A microbenchmark, proxy model, CUDA result, llama.cpp run, or different machine may
explain a hypothesis. None can populate a Metal fak-native witness. The same fence
applies in reverse for CUDA/A100.

## Pinned external design sources

These sources shape graph structure rather than supply fak measurements:

- NVIDIA hierarchical roofline analysis motivates separating launch, compute,
  bandwidth, and end-to-end bottlenecks.
- MLPerf configuration identity motivates immutable envelopes.
- vLLM and SGLang motivate independently switchable paged KV, prefix reuse,
  chunked prefill, and batching mechanisms.
- llama.cpp Q8_1/DP4A MMVQ motivates #8635's CUDA lever, while execution remains
  fak-native; llama.cpp is explicitly selected only for benchmark or parity/reference diagnosis and is never an implicit fallback.
