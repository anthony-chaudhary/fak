---
title: "FlashInfer study — reusable execution plans, workload tuning, and traceable kernels"
description: "FlashInfer is not a serving system for FAK to embed. It is a broad, fast-moving GPU kernel library whose strongest transferable lesson is the control plane..."
---
# FlashInfer study — reusable execution plans, workload tuning, and traceable kernels

**Studied:** 2026-08-22
**Source:** [`flashinfer-ai/flashinfer`](https://github.com/flashinfer-ai/flashinfer)
**Pinned revision:** [`fb28d7242b3506a2348265962041acc1fb56cca4`](https://github.com/flashinfer-ai/flashinfer/tree/fb28d7242b3506a2348265962041acc1fb56cca4) (`nightly-v0.6.18-20260819-27-gfb28d724`)
**Revision timestamp:** 2026-08-22T04:34:52-07:00
**Repository state observed:** 6,209 stars, 1,319 forks, 804 GitHub open issue/PR items, 243 commits in the preceding 31 days; 2,741 tracked files at the pin (1,487 Python, 800 C/C++/CUDA-header/source, 603 under `tests/`). These are dated discovery signals, not quality scores.
**License:** Apache-2.0 ([root `LICENSE`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/LICENSE)).

## Verdict first

FlashInfer is not a serving system for FAK to embed. It is a broad, fast-moving GPU kernel library whose strongest transferable lesson is **the control plane around kernels**: prepare shape-sensitive work before the hot path, own/reuse workspace explicitly, tune only against representative profiles with correctness gates, and capture enough execution evidence to replay and explain a choice.

Three witnessed FAK gaps are now filed under the integrated compute/cache/serve parent #6042:

- [#8607](https://github.com/anthony-chaudhary/fak/issues/8607) — reusable plan/run contracts with caller-owned workspace;
- [#8608](https://github.com/anthony-chaudhary/fak/issues/8608) — offline correctness-gated tuning from replayable workload profiles;
- [#8609](https://github.com/anthony-chaudhary/fak/issues/8609) — bounded device-event traces feeding replay and tuning.

The best default frontier is #8607 → #8609 → #8608: make execution preparation explicit, make it observable, then let evidence select among candidates. FlashInfer itself remains an **optional integration/oracle**, not FAK's default runtime dependency.

## Value frame and problem checklist

- **For:** operators and kernel authors trying to make FAK's native inference path faster without turning one benchmark win into an unauditable permanent default.
- **Problem:** FAK has a useful compute HAL, kernel implementations, provenance, and benchmark witnesses, but no common shape-aware plan, device-event trace, or closed workload-to-kernel-choice artifact.
- **Today:** `internal/compute/compute.go:338-385` invokes whole backend operations directly; `internal/compute/kernel_provenance.go:28-58` records where kernels came from and whether code was reused, not when a shape/device-specific selection is valid.
- **Better because:** preparation, execution evidence, and selection become separate inspectable artifacts; a stale or incompatible choice can fail closed rather than silently running.
- **Witness:** the three filed issues each define a minimal captured spine and bounded operating envelope.

Problem centrality is **Enabling** for #6042's Core integrated operation outcome. P1 managed context is preserved; P2 net-true efficiency advances only with setup, workspace, timing, and amortization evidence; P3 bounded adaptation advances through immutable compatibility tuples and refusal; P4 integrated operations advances through one HAL-level trace and manifest rather than benchmark-local scripts.

## What FlashInfer is at the pinned revision

FlashInfer exposes PyTorch-facing APIs backed by prebuilt and JIT-compiled CUDA/C++/CuTe-DSL kernels. Its surface is much broader than attention: the package exports attention/decode/prefill/cascade, GEMM/grouped GEMM, MoE, quantization, sampling, communication, Mamba/GDN/KDA, normalization, sparse operations, profiling, tracing, and tuning. It targets NVIDIA GPU generations and integrates with serving frameworks rather than owning scheduling, admission, model lifecycle, or an agent/tool security boundary.

The repository architecture is layered:

1. **Python operator and wrapper APIs** under `flashinfer/`, including long-lived attention wrappers.
2. **JIT and artifact machinery** under `flashinfer/jit/`, which resolves prebuilt operators or compiles generated sources and loads them into PyTorch.
3. **Kernel implementations** across `include/`, `csrc/`, CuTe DSL, CUTLASS/TRT-LLM-derived paths, and generated/template sources.
4. **Validation and performance surface** under `tests/`, `benchmarks/`, tuning configurations, logging/profile hooks, and trace/replay tooling.
5. **Integration surface** for vLLM, SGLang, JAX/TVM FFI, TensorRT-LLM-generation kernels, and communication/MoE paths.

That breadth is a warning as well as an asset: FAK should borrow modular contracts and use FlashInfer as a backend/oracle where appropriate, not duplicate a rapidly changing NVIDIA kernel catalogue in Go.

## Evidence coverage checklist

| Surface required by the study workflow | Evidence read | What changed in the conclusion |
|---|---|---|
| Code and architecture | `flashinfer/decode.py`, `cascade.py`, `jit/core.py`, `jit/env.py`, `autotuner/autotuner.py`, `trace/`, `trace_apply/`, package exports | The durable borrow is the control-plane contract around kernels, not an attention implementation. |
| Tests and examples | attention/autotuner/trace test searches; tutorial and API examples | Plan/run, profile-key, trace, and graph-compatibility behavior are exercised as operating contracts, not documentation-only ideas. |
| Build/package/install | `pyproject.toml`, `docs/installation.rst`, artifact/JIT code | Prebuilt artifacts plus local JIT are central to compatibility; importing this machinery would violate FAK's dependency and portability posture. |
| History | 243 commits in the preceding 31 days; file histories for tuning, JIT cache, tracing, recursive attention | The API and backend matrix are moving too quickly to mirror wholesale; pin every comparison and keep adapters narrow. |
| Releases | GitHub releases through the latest observed release/nightly lineage | Release cadence reinforces the optional-backend boundary and the need for explicit version/provenance tuples. |
| Open/closed issues and PRs | Current issue/PR searches; merged PRs #3172 and #3984; recent routing/MoE work at the pin | Cache-key normalization and CUDA-graph-compatible instrumentation reveal real failure modes worth encoding in FAK's contracts. |
| Proposals/design docs | CuTe-DSL kernel cache and MoE API/EP design docs | Separate shipped mechanisms from proposed distributed cache/registry ideas; keep fleet artifact distribution WATCH-only. |
| Discussions/roadmap signals | GitHub discussions and active design/integration work sampled on 2026-08-22 | Direction is expansion into full inference primitives and integrations, not a stable narrow attention-only library. |
| License/provenance | Apache-2.0 root license; source comments/docs identify borrowed backend lineages | Direct adaptation is legally possible with notice, but this pass intentionally files clean Go contracts rather than copying Python/CUDA. |

## Dated source ledger

All source assertions below are anchored to the pinned revision unless a PR/issue date is explicit.

| Source | Status on 2026-08-22 | Observation | Effect on this study |
|---|---|---|---|
| [`README.md`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/README.md) and [`flashinfer/__init__.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/__init__.py) | shipped | Broad PyTorch kernel library, not only paged attention. | Excluded wholesale reimplementation and narrowed borrows to reusable mechanisms. |
| [`decode.py` wrapper](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/decode.py#L640-L759) and [`plan`/`run` path](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/decode.py#L1200-L1319) | shipped | Caller-owned workspace and a plan/run lifecycle move auxiliary setup out of repeated execution and support graph-sensitive use. | Produced #8607. |
| [`jit/core.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/jit/core.py) and [`jit/env.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/jit/env.py) | shipped | Generated sources, compilation flags, architecture, Torch/CUDA environment, and artifact paths determine module identity/load behavior. | Informed compatibility tuples; excluded an in-process JIT port. |
| [`docs/design_docs/cute_dsl_kernel_cache.md`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/docs/design_docs/cute_dsl_kernel_cache.md) | mixed: local cache shipped/proposed evolution | Defines in-memory/local content-addressed cache direction and discusses more distributable artifact handling. | Local immutable artifacts are a candidate; distributed registry remains WATCH until FAK has real repeated compile pain. |
| [`autotuner/autotuner.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/autotuner/autotuner.py) and [`docs/autotuning.rst`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/docs/autotuning.rst) | shipped | Candidate parameters and selection keys are registered; cache misses are profiled; candidate validity precedes timing; results are cached/reused. | Produced #8608, with correctness-first and offline-only fences. |
| [PR #3984](https://github.com/flashinfer-ai/flashinfer/pull/3984), merged 2026-08-06 | shipped fix | Normalized nearest-profile cache keys after inconsistent key forms caused misses/incorrect reuse behavior. | Made canonical keys and explicit fallback semantics acceptance criteria, not implementation detail. |
| [`docs/fi_trace.rst`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/docs/fi_trace.rst), [`trace/`](https://github.com/flashinfer-ai/flashinfer/tree/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/trace), and [`trace_apply/`](https://github.com/flashinfer-ai/flashinfer/tree/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/trace_apply) | shipped/evolving | Separates workload recording, generated templates/solutions, validation, and application/replay. | Produced #8609 and the trace→tune dependency direction. |
| [PR #3172](https://github.com/flashinfer-ai/flashinfer/pull/3172), merged 2026-05-15 | shipped | Made high-detail logging/trace behavior CUDA-graph compatible and fixed templates. | Added bounded/no-op-default/capture-compatible instrumentation constraints to #8609. |
| [`cascade.py`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/flashinfer/cascade.py) and [`recursive_attention.rst`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/docs/tutorials/recursive_attention.rst) | shipped | Shared-prefix attention states can be computed separately and merged, reducing repeated shared-prefix work for suitable batches. | Classified as PRESENT-adjacent/WATCH: FAK already centers prefix/KV reuse; kernel-level state merging needs a model/backend-specific witness before another issue. |
| [`docs/design_docs/flashinfer_moe_api.md`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/docs/design_docs/flashinfer_moe_api.md) and EP docs | proposed + partially shipped family | Push toward unified MoE routing/expert-parallel APIs across multiple kernel backends. | No new issue: FAK already has active MoE coalescing/residency work (#5251 and companions); use FlashInfer as an oracle/integration option there. |
| [`LICENSE`](https://github.com/flashinfer-ai/flashinfer/blob/fb28d7242b3506a2348265962041acc1fb56cca4/LICENSE) | current at pin | Apache-2.0. | Allows adaptation with attribution/notice; no GPL-style boundary. |

## Candidate matrix

`Fact` means observed in pinned source or dated merged history. `Inference` is the FAK-specific conclusion and is labeled as such.

| Candidate technique | Source status | FAK witness | Disposition | Smallest proof / ticket |
|---|---|---|---|---|
| Immutable plan/run lifecycle with caller-owned workspace | **Fact, shipped** | **PARTIAL:** optional HAL capability seam exists, reusable plan does not | **DEFAULT** | #8607; one op, prepare once/run twice, exact workspace, mismatch refusal, break-even measurement |
| Workload-profile autotuning with candidate validation and canonical lookup keys | **Fact, shipped; key bug fixed** | **PARTIAL:** candidates/oracles/provenance exist, trace→choice manifest does not | **OPTIONAL-MODULE** initially; promote only after positive net result | #8608; two profiles/two candidates, wrong-result rejection, deterministic manifest, real GPU witness |
| Bounded record→transform→replay device trace | **Fact, shipped/evolving** | **PARTIAL:** generic telemetry exists, HAL device-event schema does not | **DEFAULT** when disabled/no-op; capture opt-in | #8609; one op, one accelerator timer, bounded drops, overhead and reconciliation |
| Content-addressed local compiled-kernel artifact | **Fact/mixed evolution** | **PARTIAL:** FAK has provenance manifests and general artifacts, not compile products | **WATCH**, then optional module if native/JIT compile repetition becomes measured pain | Do not file separately before #8607/#8608 establish the key/manifest consumer |
| Distributed kernel artifact registry | **Proposal/inference from design direction** | **ABSENT**, but no demonstrated FAK need | **WATCH** | Require compile-frequency, artifact-size, trust/signature, and fleet-reuse evidence first |
| Recursive/shared-prefix attention-state merging | **Fact, shipped** | **PRESENT-adjacent:** FAK already manages prefix/KV reuse; compute-level merge is model-specific | **RECIPE/WATCH** | Evaluate inside a concrete attention-backend issue; avoid a generic duplicate |
| FlashInfer as a selectable NVIDIA backend/oracle | **Fact: integration-oriented library** | **PARTIAL:** FAK has backend HAL and GPU routes, no direct FlashInfer adapter | **OPTIONAL-MODULE** | File only when a supported model/op has a parity/performance target and deployment owner |
| Unified MoE routing/EP abstraction | **Mixed shipped/proposed** | **PRESENT/PARTIAL:** #5251 and existing MoE/expert-residency work own the gap | **WATCH / feed existing work** | No duplicate issue; cite pinned kernels/design in the next MoE witness |
| Copy broad Python/JIT/CUDA stack into FAK | **Possible under license, operationally poor** | Conflicts with zero-dependency Go kernel and portability | **EXCLUDE** | Use adapters, subprocess/backends, or oracle comparisons instead |
| Runtime online tuning in request path | FlashInfer supports dynamic profiling patterns, but operational tradeoffs vary | FAK has no bounded safety/latency envelope for it | **EXCLUDE as default** | Offline deterministic tuning first; online exploration needs a separate safety case |

## Self-query and seam witnesses

The field-borrow workflow requires asking FAK rather than assuming absence. Queries were run on 2026-08-22 against the current checkout.

### Plan/run and workspace

```text
$ fak capabilities "attention plan run reusable workspace CUDA graph"
Reuse stable prompt and context work
Carry a session forward without transcript replay
Avoid unnecessary model turns
Attribute cache and token savings
Enforce the supporting capability floor
```

No card represented compute planning or caller-owned device workspace. `fak-dev index leaves "attention workspace planner"` and raw grep found the HAL but no execution-plan type. Classification: **PARTIAL**, grounded at `internal/compute/compute.go:338-385` and the optional capability doctrine at `:457-460`. Filed #8607.

### Autotuning and lookup

```text
$ fak capabilities "kernel autotuning trace replay lookup table"
no matching capability
```

`fak-dev index docs "autotuning trace replay"` and raw grep found benchmark/oracle/provenance neighbors, not a workload profile or selected-kernel manifest. Classification: **PARTIAL**, grounded at `internal/compute/kernel_provenance.go:28-58` plus the direct backend methods in `internal/compute/compute.go:338-385`. Filed #8608.

### Device-event trace

The same capability query returned no compute-trace card. Raw grep found endpoint/model telemetry and replay fixtures but no HAL-level operation/phase/timer-domain schema. Classification: **PARTIAL**, grounded at the direct operation seam in `internal/compute/compute.go:338-385`. Filed #8609.

### Shared-prefix/cascade attention

```text
$ fak capabilities "recursive cascade attention shared prefix KV cache"
Reuse stable prompt and context work
Attribute cache and token savings
```

Raw grep confirms extensive prefix/KV reuse machinery in FAK. Classification: **PRESENT-adjacent**, not ABSENT. FlashInfer's state merge is lower-level and backend/model-specific, so this pass does not file a duplicate generic cache issue.

## What to copy, adapt, integrate, or reject

### Adapt now

- The **semantic split** between plan and run, including explicit workspace ownership and compatibility checks (#8607).
- The **correctness-before-timing** and canonical profile-key discipline of tuning (#8608).
- The **record → stable artifact → replay/summary** trace pipeline with graph/capture-aware instrumentation (#8609).

These should be clean Go designs at FAK's existing seams. No upstream source file needs to be copied.

### Integrate optionally

- Treat FlashInfer as an accelerator backend or performance/correctness oracle for concrete NVIDIA model operations.
- Pin the exact FlashInfer release/revision, CUDA/Torch ABI, GPU architecture, generated-source digest, compile flags, and operator parameters in any resulting artifact.
- Keep failure fallback to an existing FAK backend explicit; do not make package/JIT availability part of the default control-plane proof.

### Reject as defaults

- A Python/PyTorch dependency in the Go kernel.
- Unbounded JIT compilation on first user request.
- "Auto" kernel selection without a visible compatibility key, correctness result, and fallback.
- A fastest-sample benchmark used as a permanent winner without setup/amortization and representative workload weights.
- Detailed tracing always enabled or silently synchronized, because observation can dominate the operation being measured.

## Negative knowledge and failure lessons

1. **Cache keys are API semantics.** PR #3984 exists because semantically equivalent profile keys did not normalize consistently. FAK's profile and plan keys must have one canonical encoding and tests for exact/nearest/refusal behavior.
2. **Instrumentation changes execution.** PR #3172's CUDA-graph work shows logging/tracing must respect capture constraints. FAK must label timer domain, bound buffers, count drops, and measure disabled/enabled overhead.
3. **Workspace reuse has lifecycle hazards.** A caller-owned buffer is only safe when size, device, stream/concurrency, and plan compatibility are explicit. #8607 therefore refuses mismatch rather than resizing/replanning invisibly in the hot path.
4. **JIT breadth creates deployment coupling.** Torch ABI, CUDA architecture, compiler/toolchain, generated sources, and flags all affect loadability. A FAK artifact key that omits any effective dimension can produce false reuse.
5. **Nearest-profile tuning is adaptation, not truth.** Fallback can be useful, but must be bounded and observable; exact matches remain the safe first spine.
6. **Library breadth is not integration evidence.** A shipped FlashInfer kernel does not prove it wins for FAK's model, shapes, quantization, hardware, or end-to-end serving path. Every adoption needs an oracle and a real node witness.

## Portfolio frontier and bounded superset

### Best-default frontier

1. **#8607 plan/run:** smallest structural prerequisite; no optimizer is useful if setup and compatibility remain implicit.
2. **#8609 trace:** makes launch/setup/device phases and observability cost visible through one stable artifact.
3. **#8608 tune:** consumes representative traces and emits bounded immutable selections.
4. **Optional FlashInfer backend/oracle:** only for concrete NVIDIA operations after parity and end-to-end benefit are witnessed.

### Bounded-superset coverage

- **CPU/Metal/simple backends:** keep direct HAL operations; planning remains optional and can be zero-work.
- **Static/precompiled NVIDIA kernels:** use plan and trace without JIT.
- **JIT/generated kernels:** add content-addressed local artifacts behind the same compatibility/provenance seam only when measured compile reuse warrants it.
- **Fleet artifact distribution:** later optional module with signatures/trust and exact environment compatibility; not a local default.
- **Specialized shared-prefix attention:** backend recipe selected by workload shape, not a global KV-cache replacement.

This portfolio lets FAK cover stronger cohort-specific options without displacing the dependency-free and portable default.

## Filed work and deduplication

Before filing, GitHub search covered `FlashInfer`, `kernel cache`, `artifact cache`, `autotun`, `trace replay`, `CUDA graph`, `attention kernel`, `plan/run`, and `recursive attention`. Existing relevant work was #6042 (parent coordination), #905 (closed production-kernel field scan), #5251 (MoE trace witness), #34 (paged/block KV work), and multiple general cache/trace issues. None supplied the three exact HAL contracts above.

- #8607 — plan/run/workspace; 3/100 points toward #6042.
- #8608 — offline workload tuning; 5/100 points toward #6042.
- #8609 — bounded device-event tracing; 3/100 points toward #6042.

All three were created with `compute`, `priority/P1`, and `gen/now`, assigned to **Generation G0 - Now / Immediate**, and linked back to #6042. They are deliberately small, independently provable spines rather than one "adopt FlashInfer" monolith.

## Honest limits

- This was a source/history/API study, not a live FlashInfer benchmark; no performance claim is made.
- The pin is a fast-moving nightly-adjacent main revision, not a stability promise. Re-check before implementation.
- GitHub's 804 `open_issues_count` combines issue and PR state; it is reported only as a dated activity signal.
- Source sampling covered the major architecture, tests, docs, history, releases, issues/PRs, discussions, and design surfaces, but did not line-review every one of 2,741 files.
- FAK self-query is lexical and can miss capabilities; every ABSENT/PARTIAL conclusion was cross-checked with index search and raw grep.
- At study time, `internal/compute/kernel_provenance.go` was peer WIP rather than committed trunk; it is current-state evidence only, not a claim about the pinned FAK commit.

## Companions

- Parent epic: [#6042](https://github.com/anthony-chaudhary/fak/issues/6042)
- Earlier production-kernel scan: [#905](https://github.com/anthony-chaudhary/fak/issues/905)
- Filed plan/run spine: [#8607](https://github.com/anthony-chaudhary/fak/issues/8607)
- Filed tuning spine: [#8608](https://github.com/anthony-chaudhary/fak/issues/8608)
- Filed trace spine: [#8609](https://github.com/anthony-chaudhary/fak/issues/8609)
- Active MoE trace witness: [#5251](https://github.com/anthony-chaudhary/fak/issues/5251)
- Upstream repository: [`flashinfer-ai/flashinfer@fb28d724`](https://github.com/flashinfer-ai/flashinfer/tree/fb28d7242b3506a2348265962041acc1fb56cca4)


## Exhaustive inventory refresh (2026-08-25)

Issue [#9002](https://github.com/anthony-chaudhary/fak/issues/9002) refreshes the denominator without moving the studied revision. The machine-readable map is
[`inventory/flashinfer-ai-flashinfer.json`](../research/inventory/flashinfer-ai-flashinfer.json), generated by `fak study-inventory` from a detached checkout of
`fb28d7242b3506a2348265962041acc1fb56cca4` and then augmented with pinned non-tree evidence.

### Pinned denominator and source ledger

- **Tree:** 2,737 files, 323 directories, 56,793,190 bytes, and 1,126,050 text lines across all 20 top-level subsystems. The map walked every regular file except `.git`; in particular it covers README/docs, architecture/build metadata, runtime sources, tests/fixtures, benchmarks, examples, CI, profiler, JIT/cubin packaging, licenses, and vendored-patch surfaces.
- **History and releases:** full local history through the pin contains 2,850 commits and 311 merged tags. The releases API was paged to exhaustion: 359 releases existed by the pin timestamp, newest `nightly-v0.6.18-20260819`. There is no checked-in changelog, so commits, tags, and releases are the history denominator.
- **Issues and pull requests:** the REST issues endpoint was paged to exhaustion and split by `pull_request` presence. Keeping records created no later than `2026-08-22T11:34:52Z` yields 1,211 issues and 3,434 pull requests. Their open/closed counts (303/908 issues and 477/2,957 PRs) are observed state on 2026-08-25, not historical state at the pin.
- **Discussions:** GraphQL returned all 24 discussions (21 open, 3 closed, 4 answered) in one page; all were created before the pin.
- **Roadmap and unfinished-work markers:** no dedicated roadmap file exists. The tree contains 408 unfinished-work-marker matches across 170 runtime files; these and the complete tracker ledger are distributed planning evidence, not promises.
- **License/provenance:** root `LICENSE` is Apache-2.0; `NOTICE` attributes NVIDIA, and `licenses/` carries CUTLASS, FlashAttention-3, fmt, and spdlog notices. This study copies no source. Any future code-level borrow still requires file-level provenance review.

### Completeness critic

The local inventory opened every non-skipped regular file and classified all immediate subsystems; `.git` is the only skipped control directory. The separate audit closes the classes a tree walk cannot prove: full Git/tag/release history, paged open and closed issues/PRs, every discussion, distributed roadmap and unfinished-work-marker evidence, license provenance, candidate-specific `fak capabilities` self-queries, candidate adjudication, and independent GitHub read-back of issue tracking. No material directory or non-tree source class remains unopened at the checked revision.

Honest limit: this is exhaustive **inventory and candidate triage**, not a claim that every one of 1.1 million lines was semantically reviewed. Deep code reading remains concentrated on the candidate-bearing seams cited above; operator-specific kernels continue through per-op `fak sota` review.

### Refreshed field-borrow decisions

| Candidate | FAK decision | Durable follow-on / reason |
|---|---|---|
| Reusable plan/run contracts with caller-owned workspace | **Borrow the contract, clean-room** | [#8607](https://github.com/anthony-chaudhary/fak/issues/8607) remains open. It is the smallest enabling spine and preserves fak ownership of kernels, memory, scheduling, cache, and receipts. |
| Offline correctness-gated tuning from replayable profiles | **Borrow after the spine** | [#8608](https://github.com/anthony-chaudhary/fak/issues/8608) remains open; quality and operating-envelope gates are mandatory. |
| Bounded device-event trace capture and replay | **Borrow as diagnostics** | [#8609](https://github.com/anthony-chaudhary/fak/issues/8609) remains open; traces must be bounded and scrubbed. |
| FlashInfer JIT/cubin cache as dependency or fallback engine | **Reject direct binding** | FAK may learn artifact-key and cache-validation design, but native inference may not silently cede kernel/artifact ownership. |
| Attention, GEMM, MoE, norm, and sampling kernels | **Route per operator** | These are not one repository-level unit. Each requires a matched-envelope, provenance-aware `fak sota` borrow/bind/stay-minimal decision. |

Candidate-specific `fak capabilities` queries found context/cache reuse surfaces but no existing compute-HAL substitute. Independent `gh issue view` read-back confirmed #8607, #8608, and #8609 remain open and already express the surviving candidates, so this refresh creates no duplicate follow-on. The machine map records fact versus inference, alternatives, dispositions, source paths, and issue URLs.
