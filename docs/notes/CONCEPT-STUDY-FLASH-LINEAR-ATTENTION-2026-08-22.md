<!-- fak:doc-class: research -->
<!-- fak:doc-owner: compute -->
<!-- fak:doc-state: current -->
<!-- fak:doc-reviewed: 2026-08-22 -->
# Flash Linear Attention: recurrent-state kernels are an oracle, not a new runtime

## Verdict

Study **fla-org/flash-linear-attention** as a high-quality, permissively licensed
reference for linear-attention and gated-recurrence semantics. Do **not** add its Python,
PyTorch, or Triton runtime to fak, and do not make its large architecture catalog a new
product surface.

The useful borrow is narrower:

1. use FLA's naive, chunked-prefill, and fused-recurrent implementations as an independent
   differential oracle for fak's resident Qwen GDN sequence work under existing epic
   [#8344](https://github.com/anthony-chaudhary/fak/issues/8344);
2. preserve the explicit state contract already present in fak: input state, output state,
   packed-sequence boundaries, and decode-after-prefill parity are one correctness surface;
3. borrow FLA's benchmark discipline—correctness before timing, poisoned unwritten memory,
   forward/backward separation, quantiles, and revision-labelled baselines—only where a
   measured fak kernel bottleneck needs it; and
4. register any directly adapted kernel under existing provenance issue
   [#8391](https://github.com/anthony-chaudhary/fak/issues/8391).

No new issue is justified by this study. The core implementation and provenance gaps are
already named by #8344 and #8391. Packed multi-request GDN and graph capture remain WATCH
items until the single-request native CUDA prefill path is complete and measured.

## Observation identity

- **Observed:** 2026-08-22.
- **Repository:** <https://github.com/fla-org/flash-linear-attention>.
- **Pinned revision:**
  [`bc3b101dcb713ddc5bd8924b66754eb68b5ccf89`](https://github.com/fla-org/flash-linear-attention/commit/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89),
  committed 2026-08-22 (`[KDA] Make A_log optional in lowerbound gate for KDA and GDN-2`).
- **Latest release at observation:** `v0.5.2`, published 2026-07-27; the pin is 47 commits
  newer (`v0.5.2-47-gbc3b101`).
- **Repository snapshot:** 5,600 stars, 670 forks, 91 open issue/PR records, 2,084 commits,
  and 100 contributors returned on the first GitHub contributors page.
- **License:** MIT (`LICENSE`; source headers retain 2023-2026 Songlin Yang, Yu Zhang,
  Zhiyuan Li). Direct copying or adaptation is legally possible with attribution and the
  license notice; fak's own provenance contract still applies.
- **Acquisition:** filtered clone in allocated `_scratch/study-flash-linear-attention/`.
  Scratch is not evidence and is reaped after this note lands; links below pin durable
  upstream objects.

The repository is active research infrastructure, not a frozen library. The pin contains
39 operator families under `fla/ops`, 40 layer modules, and 38 model directories. Recent
commits include Ascend kernels, context-parallel validation, final-state-gradient fixes,
and partial-chunk fixes. Any exact code borrow must therefore pin a file and revision, not
say only “from FLA.”

## Feynman-simple value frame

- **For:** fak maintainers making native Qwen3.8 CUDA prefill fast without corrupting GDN
  recurrent state.
- **Problem:** a fast sequence kernel can match one output panel yet leave the conv or
  recurrent state wrong, so the next decode token diverges.
- **Today:** fak has a resident GDN sequence primitive and parity test, while #8344 still
  owns the full 64-layer prefill path and its same-A100 acceptance envelope.
- **Better because:** FLA supplies a second MIT implementation with naive, chunked, and
  recurrent forms plus state/gradient/packed-sequence tests; agreement across independent
  implementations is stronger than agreement with a translation of the same code.
- **Witness:** compare output panel, final conv state, final recurrent state, and the next
  decode output on the same small fixture; then retain #8344's real A100 throughput,
  transfer, memory, text, JSON, and tool-call gates.

The real next-best alternative is fak's existing llama.cpp-derived CPU/reference fixture.
Keep it. FLA adds implementation independence and a wider operating envelope; it does not
replace the current witness or provide a FAK-observed performance claim.

## Problem centrality and P1-P4

**Centrality: Enabling.** The borrow lowers correctness risk on the Core native-Qwen prefill
path. Shipping FLA itself would be Peripheral to fak's agent-kernel mission.

| Check | Effect |
|---|---|
| P1 managed context | A correct recurrent final state carries prompt context into decode without replaying the prompt. |
| P2 net-true efficiency | Differential checks prevent a fast-but-wrong kernel; FLA's published timings are not imported as fak savings. |
| P3 bounded adaptation | The oracle is pinned, fixture-sized, and outside production. Unsupported layout, dtype, or state geometry must fail closed. |
| P4 integrated operations | The result feeds #8344's production prefill and #8391's provenance gate rather than creating a parallel kernel program. |

## What FLA actually provides

### One API, multiple execution schedules

The Gated DeltaNet layer selects `chunk_gated_delta_rule` for chunked sequence work and
`fused_recurrent_gated_delta_rule` for recurrent work. Both receive an `initial_state`, may
return a final state, and accept packed-sequence boundaries through `cu_seqlens`. The layer
then updates its cache with recurrent state, optional convolution state, and token offset.

Source anchors at the pin:

- [`fla/ops/gated_delta_rule/__init__.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/ops/gated_delta_rule/__init__.py)
  exports naive recurrent, naive chunk, optimized chunk, and fused recurrent forms.
- [`fla/layers/gated_deltanet.py#L307-L353`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/layers/gated_deltanet.py#L307-L353)
  makes chunk/recurrent dispatch and cache update explicit.
- [`fla/models/utils.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/models/utils.py)
  carries per-layer conv, recurrent, and attention state rather than pretending every model
  has a conventional KV cache.

This is the most transferable design idea: **algorithm identity and execution schedule are
separate**. Prefill and decode may use different schedules only if they share a typed state
boundary and a parity witness.

### A correctness ladder around optimized kernels

FLA keeps intentionally slow reference implementations beside optimized kernels. Its GDN
tests compare naive recurrent, chunked, fused recurrent, and backend-specific variants;
exercise initial/final state, packed variable-length sequences, layouts, dtypes, and
backward gradients; and use tolerance appropriate to accumulated low-precision arithmetic.
See
[`tests/ops/test_gdn.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/tests/ops/test_gdn.py).

The operator benchmark tooling also separates jobs that are often conflated:

- [`benchmarks/ops/verify.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/benchmarks/ops/verify.py)
  says correctness belongs in pytest and timing belongs in the benchmark runner; its
  correctness path covers forward and backward under NaN memory poisoning.
- [`benchmarks/ops/registry.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/benchmarks/ops/registry.py)
  declares shape providers, setup, callable, output selection, tolerances, and whether
  backward is meaningful.
- [`benchmarks/ops/run.py`](https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/benchmarks/ops/run.py)
  reports median and 20th/80th-percentile latency and can compare a revision in an isolated
  worktree.

For inference-only fak kernels, backward tests are not relevant. The transferable pieces
are unwritten-output detection, explicit shape envelopes, revision-labelled baselines, and
separating correctness from timing.

### Broad integrations, with a heavy runtime cost

FLA offers Hugging Face-compatible layers/models and a large research catalog: linear
attention, GLA, DeltaNet/Gated DeltaNet, KDA, RWKV, retention, Mamba variants, NSA, MoBA,
TTT, and others. The project metadata requires Python >=3.10 plus PyTorch, Triton,
Transformers, NumPy, einops, and related packages. The README recommends recent NVIDIA
GPUs; the observed tree also has active Ascend work.

That breadth is valuable for research comparison but conflicts with fak's one-Go-binary,
zero-external-dependency runtime. FLA belongs in an offline oracle or delegated recipe, not
inside the shipped kernel.

## History, roadmap, and negative knowledge

Open project history clarifies where the attractive abstraction still leaks:

- [FLA #1155](https://github.com/fla-org/flash-linear-attention/issues/1155) proposes CUDA
  graph capture for GDN/KDA chunk and convolution paths; it is not shipped evidence.
- [FLA #1119](https://github.com/fla-org/flash-linear-attention/issues/1119) reports unknown
  `**kwargs` being silently swallowed and changing Kimi K3 numerics. Fak should retain typed
  request fields and fail closed on unsupported flags.
- [FLA #1110](https://github.com/fla-org/flash-linear-attention/issues/1110) reports backend
  dispatch being lost through decorator interaction. A surface that advertises a native
  path must prove which backend ran; marker-only dispatch is insufficient.
- [FLA #1029](https://github.com/fla-org/flash-linear-attention/issues/1029) reports
  non-divisible GQA head counts that can leave outputs unwritten or read invalid heads.
  Geometry validation and output poisoning are correctness mechanisms, not test polish.
- [FLA #872](https://github.com/fla-org/flash-linear-attention/issues/872) requests caller-
  supplied final-state buffers. It may reduce allocations, but is not a default until a fak
  profile identifies state allocation as material.
- [FLA #659](https://github.com/fla-org/flash-linear-attention/issues/659) proposes merging
  similar KDA/GLA/DPLR kernels. Fak should not generalize from visual similarity before its
  one needed GDN path is complete.
- [FLA #942](https://github.com/fla-org/flash-linear-attention/issues/942) is an NPU roadmap.
  It does not prove a stable cross-vendor kernel contract.

Recent fixes at the pinned tip reinforce the same lesson: partial last chunks, short-chunk
cache gradients, masked loads, invalid context-parallel partitions, and final-state
gradients have all needed explicit repair. “Linear-time” does not mean “simple state.”

## FAK inward witness

The required three-way self-query was run against the current checkout on 2026-08-22.

1. `fak capabilities 'linear attention recurrent state kernel modes chunk recurrent variable length backend dispatch correctness oracle' --json`
   returned only `turn-savings` and `portable-session`; neither implements linear attention
   or a GDN oracle. **Lexical capability result: ABSENT.**
2. `fak-dev index docs|leaves|verbs|claims ...` found the broader compute, benchmark, and
   backend surfaces, but no FLA integration or generic linear-attention capability.
   **Generic product capability: ABSENT, correctly so.**
3. Git and issue inspection found the concrete native seam:
   `internal/compute/cuda_qwen35_gdn_sequence.go` and
   `internal/compute/cuda_qwen35_gdn_sequence_test.go`, originally shipped by
   `internal/compute@r2+g8ce4659e59` for closed issue #8345. The test covers output/state
   parity, decode-after-prefill parity, and resident transfer accounting. Open #8344 owns
   the full sequence-prefill envelope. **Qwen GDN sequence spine: PRESENT; production
   sequence prefill: PARTIAL.**

The exact implementation seam is
`internal/compute/cuda_qwen35_gdn_sequence_test.go:TestCUDAQwen35GDNSequenceMatchesDecodeAndStaysResident`.
An FLA-derived fixture would extend that test surface; production runtime code should stay
Go/CUDA and backend-resident.

## Candidate dispositions

| Candidate | FAK state | Placement | Disposition |
|---|---|---|---|
| Independent FLA naive/chunk/recurrent GDN oracle for output and final state | PARTIAL: fak has a llama.cpp-derived parity fixture, not an FLA cross-check | DEFAULT evidence for #8344 | Use when implementing the remaining full prefill path; no duplicate issue. |
| Explicit chunk-prefill vs fused-recurrent decode schedule | PRESENT at the narrow GDN seam; PARTIAL end to end | DEFAULT | Preserve typed state and fail-closed dispatch under #8344. |
| Decode-after-prefill state parity | PRESENT in `internal/compute` | DEFAULT | Keep; this is stronger than output-only parity. |
| Packed variable-length multi-request state boundaries | ABSENT from the narrow fak spine | OPTIONAL-MODULE / WATCH | #8344 explicitly excludes batching; revisit only after single-request prefill meets its envelope. |
| NaN-poisoned outputs and revision-labelled kernel baselines | PARTIAL across fak benchmark/test machinery | RECIPE | Apply to a measured kernel optimization; do not create a general framework now. |
| Caller-owned final-state output buffer | ABSENT | WATCH | Profile first; allocation savings are unproven in fak. |
| CUDA graph capture for GDN/KDA prefill | ABSENT; upstream itself is RFC | WATCH | Wait for shipped upstream evidence and a fak launch-overhead profile. |
| Python/PyTorch/Triton runtime dependency | ABSENT | EXCLUDE | Violates the native one-binary runtime and adds a second serving stack. |
| Full architecture/model catalog | ABSENT | EXCLUDE | Research breadth is not a user outcome for the agent kernel. |
| Directly adapted FLA kernel code | ABSENT by design | OPTIONAL-MODULE | Allowed by MIT only after a hot-path witness; register under #8391 with source path, SHA, notice, destination, and parity test. |
| Import upstream throughput claims | ABSENT evidence | EXCLUDE | Keep upstream claims source-labelled; #8344 requires same-A100 FAK-observed trials. |

## Adoption rule

Borrow FLA code only when all of these are true:

1. #8344's profile names the exact GDN operation as a material remaining bottleneck;
2. a pinned FLA function is shorter or safer than adapting the already-used llama.cpp path;
3. the imported/adapted file is registered through #8391 with MIT notice retention;
4. the fixture proves output, final conv state, final recurrent state, and next decode output;
5. unsupported shapes, flags, and backend selection fail closed; and
6. the real A100 benchmark still counts setup, transfers, memory, and quality parity.

Until then, use FLA as a test oracle and design reference only.

## Source ledger

Primary code and project sources, all observed 2026-08-22:

- pinned tree and license: <https://github.com/fla-org/flash-linear-attention/tree/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89>;
- project overview and supported model families: <https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/README.md>;
- package/runtime contract: <https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/pyproject.toml>;
- GDN layer dispatch/cache seam: <https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/layers/gated_deltanet.py#L307-L353>;
- GDN operator exports and naive references: <https://github.com/fla-org/flash-linear-attention/tree/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/fla/ops/gated_delta_rule>;
- GDN operating-envelope tests: <https://github.com/fla-org/flash-linear-attention/blob/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/tests/ops/test_gdn.py>;
- benchmark registry, runner, and verifier: <https://github.com/fla-org/flash-linear-attention/tree/bc3b101dcb713ddc5bd8924b66754eb68b5ccf89/benchmarks/ops>;
- release `v0.5.2`: <https://github.com/fla-org/flash-linear-attention/releases>;
- open issues and RFCs: #1155, #1119, #1110, #1029, #872, #659, and #942 linked above.

FAK sources and backlog were read from the current `main` checkout and GitHub state on
2026-08-22. Module evidence is cited as `internal/compute@r2+g8ce4659e59`; #8344 and #8391
remain the authoritative open work items.

## Exhaustive inventory refresh (2026-08-25)

Issue [#9001](https://github.com/anthony-chaudhary/fak/issues/9001) refreshes the
study denominator at upstream revision
`8f787def80f6b6862f4f8b84581810d3db537c2d` (FLA 0.6.0). The deterministic map
is [`docs/research/inventory/fla-org-flash-linear-attention.json`](../research/inventory/fla-org-flash-linear-attention.json):
739 tracked files in 153 directories, 7,744,131 bytes and 193,783 text lines,
partitioned into 12 top-level subsystems. Its completeness critic reports
739/739 Git paths indexed, with zero missing and zero untracked-at-revision paths.

### Source-class audit

| Source class | Pinned evidence and result |
|---|---|
| README/docs | `README.md`, `INSTALL.md`, `CONTRIBUTING.md`, operation READMEs, and the 37 documentation-class files in the map describe the supported architecture/backend surface. |
| Architecture/design | `.github/ISSUE_TEMPLATE/rfc.yml`, the operation split under `fla/ops/`, model integrations under `fla/models/`, and the naive/chunked/fused-recurrent schedule boundary were inspected. There is no separate architecture document. |
| Runtime source | All 574 runtime-class files are in the map. The refresh re-read GDN/GDN2 recurrent and chunked paths, Qwen3-Next integration, backend dispatch, cache/state handling, and the post-study Ascend additions. |
| Tests/fixtures | All 121 test-class files are mapped. `tests/ops/test_gdn.py`, `tests/ops/test_gdn2.py`, context-parallel tests, and benchmark verification preserve the useful correctness ladder. |
| History/changelog/releases | Git history from the former pin `bc3b101d...` to the new pin has seven commits (1,142 insertions/513 deletions across 31 files); GitHub exposes 21 releases, headed by 0.6.0 on 2026-08-25. The delta is chiefly Ascend, GDN2 GVA, DPLR, docs, and release metadata; it does not overturn the prior GDN decision. |
| Issues/PRs/discussions | The paginated GitHub audit covered 1,161 issue/PR records: 42 open and 343 closed issues, 48 open and 728 closed PRs. It also covered all six discussions. Recent merged work includes backend lookup caching (#996), dense no-cache short-convolution fusion (#972), and a correctness-gated kernel optimization loop (#959); open #890 records a TileLang race rather than evidence for importing that stack. |
| Roadmap/TODOs | GitHub has three open milestones: “FLA v1.0.0 release” (4 open/9 closed), “Native varlen support” (2/2), and “Enhanced testing” (0/1, overdue). Tree TODOs and open issues reinforce that varlen, backend coverage, and kernel hardening remain active rather than stable contracts. |
| License/provenance | `LICENSE` remains MIT; `pyproject.toml` declares MIT and Python >=3.10. FAK may study or adapt algorithms with attribution, but no upstream source is copied by this refresh. |
| FAK self-query | `fak capabilities` plus `fak-dev index docs|leaves` were queried for GDN schedules/state, packed varlen multi-request state and graph capture, and Triton/PyTorch/TileLang/ROCm backends. Results confirm FAK's resident GDN spine and the existing #8344/#8391 gaps; they do not justify a second runtime. |
| Candidate matrix | The five dispositions below cover every mechanism that survived the source audit. |
| Completeness critic | The generated critic is exact for the pinned tree (739/739); this section supplies the non-tree denominator. No source class remains only asserted. |
| Issue tracking | This denominator refresh is #9001. Actionable packed-state and graph-capture work remains owned by #8344 and #8391; no duplicate follow-on is filed. |

### Refreshed candidate matrix and FAK decisions

| Candidate | FAK decision | Evidence-backed reason | Follow-on |
|---|---|---|---|
| Naive -> chunked-prefill -> fused-recurrent GDN correctness ladder | **ADOPT as oracle** | The three schedules and tests remain the clearest independent cross-check for FAK-native resident GDN semantics. | Keep using pinned equations/tests as diagnosis evidence; copy no Python/Triton runtime. |
| Explicit `initial_state` / `output_final_state` recurrent-state contract | **ADAPT, FAK-native** | It matches FAK's resident sequence-state boundary while leaving allocation, scheduling, and ownership with FAK. | Existing packed multi-request/state-layout work stays in #8344. |
| Packed variable-length execution (`cu_seqlens`) and native-varlen roadmap | **MONITOR then adapt** | FLA's open milestone shows the contract is still moving; FAK needs request isolation and deterministic offsets, not API parity. | #8344 owns the witnessed FAK-native design. |
| Capture-safe fixed buffers and backend-dispatch lookup caching | **MONITOR** | Recent history makes dispatch/capture discipline concrete, but the transferable invariant is stable allocation/selection, not FLA's Python dispatch implementation. | #8391 owns graph-capture proof; reconsider only with a native receipt. |
| PyTorch/Triton/TileLang/ROCm/Ascend runtime and broad model catalog | **REJECT** | Importing it would violate the fak-native execution invariant, duplicate runtime ownership, and expand the dependency/validation envelope far beyond the GDN problem. | None; benchmark/reference use must stay explicitly selected. |

### Completeness-critic verdict

The refresh found no hidden candidate that changes the original verdict. The upstream
0.6.0 delta expands and hardens accelerator backends, while FAK's useful borrowing seam
remains the schedule/state **contract** and its correctness witnesses. All actionable gaps
are already represented by #8344 or #8391, so #9001 creates no new follow-on issue.
