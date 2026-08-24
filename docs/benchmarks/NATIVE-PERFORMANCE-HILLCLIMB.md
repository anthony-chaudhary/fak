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

## What this map covers — and what it does not yet cover

The current table is one **decode-throughput envelope**, not the whole native-performance
program. Breadth is the set of distinct envelopes we can compare honestly; depth is the
number of independently attributable levers and phase witnesses inside one envelope.
Never create breadth by joining unlike machines into one curve, and never claim depth by
stacking several changes into one unnamed rung.

| Dimension | Current coverage | Required next coverage | Owner |
|---|---|---|---|
| Backend / hardware | Metal, M3 Pro | A separate CUDA/A100 graph; later hardware stays separate | [#8752](https://github.com/anthony-chaudhary/fak/issues/8752) |
| Model / quant | Qwen3.8-27B Q4_K_M | Add an envelope only after artifact hash and quant path are pinned | [#8752](https://github.com/anthony-chaudhary/fak/issues/8752) |
| Workload phase | Steady decode, approximate P/T match | Prefill, first-token, steady decode, verification, and teardown phase boundaries | [#8760](https://github.com/anthony-chaudhary/fak/issues/8760) |
| Optimization depth | Baseline plus one aggregate coarse-graph hypothesis | Independent resident-state, submission, fusion, kernel, cache, scheduler, and serving arms | [#8752](https://github.com/anthony-chaudhary/fak/issues/8752) |
| Resource evidence | Throughput and a profile diagnosis | Per-repetition latency, memory, launch/sync counts, and backend-native counters | [#8757](https://github.com/anthony-chaudhary/fak/issues/8757), [#8760](https://github.com/anthony-chaudhary/fak/issues/8760) |
| Durability | Manual witnessed point | Envelope-scoped promotion floor and guarded regression/bisect packet | [#8759](https://github.com/anthony-chaudhary/fak/issues/8759) |
| Quality / identity | Deterministic output and native path named in parity issue | Fail-closed receipt fields on every experiment | [#8757](https://github.com/anthony-chaudhary/fak/issues/8757) |

This ordering matters. First make one lever attributable, then deepen the active envelope;
only then add another envelope. A larger model/hardware matrix without comparable receipts
multiplies ambiguity rather than knowledge.

## Shift-left experiment contract

Performance evidence starts **before implementation**, not after a promising kernel diff.
Every optimization issue and worker packet should carry this chain:

1. **Declare the envelope and one lever.** Pin artifact hash, backend/device class, quant,
   P/T/batch/context, sampling, warmup/repetitions, execution identity, and comparison
   semantics. Name one changed lever plus unchanged controls. If the envelope or lever is
   absent from the graph, add it before editing the kernel.
2. **Capture the baseline.** Produce the versioned baseline receipt and raw profile bundle
   from the unmodified revision. A remembered number or benchmark prose is not a baseline.
3. **Classify the bottleneck.** Split load/setup, prefill, first token, steady decode,
   verification, and teardown. Use backend-native counters to choose among launch,
   bandwidth, compute, synchronization, residency, and host-orchestration limits. Counters
   explain a hypothesis; they do not substitute for end-to-end throughput.
4. **Predict the effect and falsifier.** Record the expected direction/range as a
   hypothesis, the counter or phase that should move, and the observation that would reject
   the lever. Do not turn a roofline ceiling or another engine's result into a fak witness.
5. **Run the smallest A/B.** Keep controls fixed, capture every repetition, quality result,
   memory high-water mark, native engine/forward path, and fallback counters. Stop and split
   the experiment if more than one undeclared axis changes.
6. **Promote or reject.** A rung becomes witnessed only when the complete candidate receipt
   beats the accepted baseline inside the pinned envelope without quality, memory, or
   fallback regression. Negative results stay attached to the lever so another worker does
   not repeat them.
7. **Make the gain durable.** Land the focused regression witness, update graph/docs and the
   owning issue in the same change, and register the accepted receipt as the envelope floor.
   Scheduled hardware runs compare against that floor and emit a bounded bisect packet.

The implementation path is tracked in [#8757](https://github.com/anthony-chaudhary/fak/issues/8757)
(receipts), [#8760](https://github.com/anthony-chaudhary/fak/issues/8760) (phase/profile
classification), and [#8759](https://github.com/anthony-chaudhary/fak/issues/8759)
(regression gate). Until those automate the contract, issue bodies and captured artifacts
must provide the same fields manually.

## Applying the same process in other contexts

The shift-left shape is intentionally reusable; only the envelope and witness change.

- **Correctness:** pin input/seed/configuration, capture the failing output first, name one
  behavioral change, then land the failing-before/passing-after test.
- **TUI/visual work:** pin terminal geometry/theme/platform, capture the broken render bytes
  and screenshot first, change one rendering lever, then retain the render witness and
  before/after image.
- **Security/policy:** pin manifest, principal, tool, arguments, and expected decision;
  capture the deny/allow trace before editing and retain both the exploit rejection and a
  nearby legitimate allow case.
- **Reliability/operations:** pin topology, load, failure injection, timeout/retry budget,
  and recovery objective; capture the failure timeline first and compare candidate recovery
  without hiding setup, retries, or cleanup.
- **Cost/token work:** pin provider/model, prompt hash, cache state, tool trace, and quality
  criterion; compare billed end-to-end usage rather than a partial token bucket.

In every context the invariant is the same: **envelope → baseline artifact → one declared
lever → candidate artifact → promotion gate → durable regression witness**. The hill-climb
schema should stay native-performance-specific; shared process language belongs in issue
and worker templates rather than a false universal metric schema.

[#8764](https://github.com/anthony-chaudhary/fak/issues/8764) owns that shared issue/worker-packet
shift-left adapter. It must reuse each context's existing witness primitive and preserve
read-only, trivial-change, urgent-response, and unavailable-system exceptions rather than
turning native-performance fields into universal ceremony.

## Priority path and ticket map

| Order | Ticket | Why it is next | Exit evidence |
|---:|---|---|---|
| 1 | [#8752](https://github.com/anthony-chaudhary/fak/issues/8752) | Split the aggregate middle rung and add deterministic next-arm selection | Typed Metal/CUDA envelopes, dependencies/conflicts, `--next`, and DOT witnesses |
| 2 | [#8757](https://github.com/anthony-chaudhary/fak/issues/8757) | Move comparability and native identity to experiment start | Validated baseline/candidate receipts for Metal and CUDA |
| 3 | [#8760](https://github.com/anthony-chaudhary/fak/issues/8760) | Select work from measured phase/counter evidence | Scrubbed phase/profile bundles and deterministic bottleneck classification |
| 4 | [#8324](https://github.com/anthony-chaudhary/fak/issues/8324) | Implement the measured coarse resident Metal path | Native end-to-end decode clears its 5 tok/s promotion target |
| 5 | [#8697](https://github.com/anthony-chaudhary/fak/issues/8697) | Close exact matched Metal parity after the path is optimized | Joint P32/T64 campaign reaches at least 95% of pinned comparison |
| 6 | [#8759](https://github.com/anthony-chaudhary/fak/issues/8759) | Preserve won rungs on shared trunk | Envelope floor, scheduled verdict, and guarded bisect packet |
| Parallel envelope | [#8635](https://github.com/anthony-chaudhary/fak/issues/8635) | Advance CUDA Q8_1/MMVQ without mixing its curve with Metal | CUDA-native kernel and end-to-end receipt inside its own envelope |

`priority/P1` means product-critical, not safe to run concurrently. #8752 defines the data
model consumed by #8757 and #8760; #8759 consumes their evidence. Kernel work can proceed
only when its baseline/profile already satisfies the manual contract above. GitHub is the
durable tracker; this document explains sequencing and evidence semantics, while
`fak native-performance --json` is the machine-readable current state.

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

- llama.cpp's pinned [`llama-bench` README at `f280b26` (2026-08-24)](https://github.com/ggml-org/llama.cpp/blob/f280b26983ad0fdb705a0d9ebf0503e76f2899b0/tools/llama-bench/README.md) exposes prompt/decode sizes, repetitions, JSON/JSONL output, and device details as benchmark inputs and results. The transferable rule is to make repeated, machine-readable experiment identity routine; llama.cpp remains an explicitly selected reference only and never executes a fak-native rung.
- Apple's MLX repository at [`43d2f06` (2026-08-24), benchmarks](https://github.com/ml-explore/mlx/tree/43d2f06cb87e76895bf9a152bade4fee83408643/benchmarks) separates C++, Python, and NumPy benchmark surfaces. The transferable rule is to preserve layer boundaries when diagnosing overhead instead of treating a kernel result as application throughput; its values are not fak witnesses.
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


## Borrowed measurement discipline

The graph borrows established measurement structure, not third-party execution. Every receipt
must still name `fak-native`; llama.cpp remains a diagnostic reference only.

| Source | Mechanism borrowed | Boundary here |
|---|---|---|
| [Roofline: an insightful visual performance model](https://doi.org/10.1145/1498765.1498785) | Separate attainable ceilings from the measured point and identify the limiting resource before optimizing. | A rung's expected floor/roof is a planning hypothesis until a matched fak-native receipt exists; it is not a hardware roofline measurement. |
| [NVIDIA Nsight Compute Roofline analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html#roofline-charts) | Record arithmetic intensity and achieved throughput together when a kernel roofline is captured. | NVIDIA counters do not describe the active Apple Metal envelope; add backend-specific counters only as a separate witnessed artifact. |
| [MLPerf Inference rules](https://github.com/mlcommons/inference_policies) and [results](https://mlcommons.org/benchmarks/inference-datacenter/) | Pin scenario, system, model, accuracy/quality, and run rules so results remain comparable. | fak's P32/T64 developer envelope is not an MLPerf submission and must not be labeled one. |
| [llama.cpp `llama-bench`](https://github.com/ggml-org/llama.cpp/tree/master/tools/llama-bench) | Emit repeatable machine-readable prompt/decode measurements with explicit model and runtime parameters. | Use only for matched comparison/reference diagnosis; never execute a native rung through llama.cpp. |
| [vLLM benchmark suite](https://docs.vllm.ai/en/latest/contributing/benchmarks.html) and [profiling](https://docs.vllm.ai/en/latest/contributing/profiling.html) | Keep end-to-end serving benchmarks distinct from focused profiler traces and preserve commands/metadata. | vLLM-specific scheduling and CUDA traces are prior art, not evidence for fak-native Metal throughput. |

## Updating the graph

1. Change one feature at a time and update its `enabled`, `status`, owning rung, and
   `observable` receipt contract in `internal/nativeperf/graph.go`.
2. Capture the exact envelope, quality gate, execution identity, repetitions, and all setup,
   synchronization, and verification overhead. Put measured throughput only in `Witnessed`.
3. Keep expected floor/roof values classified `hypothesis` with provenance. A comparison or
   projection never becomes a measurement by appearing in the graph.
4. Run `go test ./internal/nativeperf ./cmd/fak`, inspect `fak native-performance`, and consume
   `fak native-performance --json` for deterministic agent scheduling.
5. Advance a rung only when its feature observables and matched receipt are present; otherwise
   leave it partial/absent and point `next_issue` at the blocker.
