# llama.cpp study: the Metal residency pattern behind native Qwen3.8

**Observed:** 2026-08-20  
**Primary source:** [`ggml-org/llama.cpp`](https://github.com/ggml-org/llama.cpp)  
**Pinned revision:** `0e1d9185c5fe82e905d1f5ae6b2e5dcd607a8dfd`  
**License:** MIT. This pass copied no source bytes; dispositions are **ADAPT** or **WATCH** against FAK's existing Go/Metal seams.

## Verdict

The useful lesson is not “replace FAK with llama.cpp.” It is a narrower execution contract: keep immutable quantized weights mapped for model lifetime, bind them to Metal once, encode a large graph slice before waiting, and synchronize only where host-visible output is required. FAK's streamed Q4_K path violated the first half of that contract by rereading and reallocating weights during decode. `internal/model@r448+g8145dc0bea` now materializes each lazy Q4_K tensor once into aligned model-owned backing and caches its Metal handle; the exact Qwen3.8 27B Q4_K_M Mac campaign improved from about 0.03 tok/s to 0.4-2.9 tok/s while preserving text, JSON, and forced-tool behavior.

That is a usable native spine, not engine parity. On the same machine llama.cpp reached 7.29 tok/s. The next dominant FAK gap is the whole-token graph boundary: its conventional dense decoder already has a resident Metal path, but the Qwen3.5-family hybrid route explicitly declines it. That work is [#8324](https://github.com/anthony-chaudhary/fak/issues/8324). File-backed tensor ownership [#8325](https://github.com/anthony-chaudhary/fak/issues/8325) and 4-8-vector Q4_K tile reuse [#8326](https://github.com/anthony-chaudhary/fak/issues/8326) are measured follow-ons, not prerequisites for the working spine.

## For / Problem / Today / Better because / Witness

- **For:** a local Apple-Silicon operator who wants Qwen3.8 inside FAK's native policy and agent loop.
- **Problem:** the exact 27B Q4_K_M model was functionally wired but decode was unusably slow because streamed weights were repeatedly promoted and submitted.
- **Today:** llama.cpp is the faster local serving alternative; before this pass, FAK's native route was roughly 0.03 tok/s and did not report its Metal identity clearly.
- **Better because:** the smallest borrowed mechanism removes repeated checkpoint work while retaining FAK's native execution identity, capability gate, and exact tool-call witness.
- **Witness:** the checked-in acceptance fold and raw run inputs under `docs/_witnesses/qwen38-27b-2026-08-20/`, plus the focused model/agent tests named there.

Problem centrality is **Core**. P1 managed context is preserved because token/KV semantics do not change. P2 net-true efficiency advances through measured eliminated rereads and same-device latency. P3 bounded adaptation is an explicit Q4_K Metal opt-in with reference fallbacks. P4 integrated operations is covered by backend identity, readiness, resource, concurrency, and tool-admission evidence.

## What was studied

| Source class | Pinned coverage |
|---|---|
| Model and mapping lifecycle | `src/llama-model-loader.cpp:1554-1583`, `src/llama-model.cpp:1026-1047,1693-1704` |
| Metal allocation and command submission | `ggml/src/ggml-metal/ggml-metal-device.m:1701-1793`; `ggml-metal-context.m:438-458,509-556,663-721` |
| Synchronization boundary | `src/llama-context.cpp:2022-2025,2475-2497` |
| Quantized operation routing | `ggml-metal-ops.cpp:2334-2445,2433-2537`; `ggml-metal.metal:4086-4190,8497-8616` |
| Qwen architecture | pinned `qwen35` metadata/model registration, converter paths, and hybrid operator routing; no separate `qwen38` architecture identifier exists at this revision |
| Tests and history | backend operator tests, Qwen architecture fixtures, relevant path history, root license, and current repository/release metadata |

The completeness critic found three limits. First, this was not a line-by-line audit of the whole repository; coverage followed the measured residency, submission, Q4_K, and Qwen seams. Second, FAK's Mac witness used request and OS resource telemetry, not Metal performance counters, so kernel-vs-submit attribution remains #8324's first deliverable. Third, the exact-model campaign covered short text, strict JSON, forced tools, warmup overlap, and two concurrent requests—not long-context quality, sustained multi-user throughput, or every Qwen3.8 variant. Those boundaries prevent a general “Qwen3.8 parity” claim.

## Worldview comparison

llama.cpp treats the model graph and mapped checkpoint as the product center: broad architecture coverage feeds a graph runtime whose backend owns buffers, kernels, and asynchronous evaluation. FAK treats the admitted agent turn as the product center: native model execution sits behind explicit identity, policy, cache, lifecycle, and witnessed-completion seams. The projects therefore meet cleanly at mechanism boundaries. FAK should adapt proven Metal ownership and submission patterns while retaining its simpler static-binary/fail-closed operational contract; it should not recreate llama.cpp's entire graph runtime inside the kernel.

Qwen routing is metadata-led in both systems. The pinned llama.cpp tree calls the family `qwen35`; FAK likewise accepts the exact Qwen3.8 checkpoint because its metadata conforms to the existing hybrid geometry. Converter-side value-head permutation remains load-bearing upstream, but it is not a new FAK runtime feature and no copy was proposed.

## Candidate ledger

| Candidate | FAK axis and witness | Disposition |
|---|---|---|
| Model-lifetime Q4_K ownership and cached Metal handle | Repeated checkpoint rereads were present in the exact native route; tests plus the Mac campaign prove the new owner survives repeated GEMV and teardown | **SHIPPED / ADAPT** — [#8306](https://github.com/anthony-chaudhary/fak/issues/8306), `internal/model@r448+g8145dc0bea` |
| Resident Qwen hybrid token graph with late synchronization | `internal/metalgemm/decode.m` has the conventional dense path; `internal/model/metal_decode.go` declines Qwen3.5-family hybrids | **FILED / ADAPT** — [#8324](https://github.com/anthony-chaudhary/fak/issues/8324) |
| File-backed, page-aligned Q4_K Metal views | Current owner is aligned heap backing; cold readiness remains 103.858 s versus 34.897 s with OS file cache | **FILED / ADAPT IF MEASURED** — [#8325](https://github.com/anthony-chaudhary/fak/issues/8325) |
| Q4_K P=4..8 kernel that dequantizes once per tile | FAK batches encoder submissions but repeats single-vector dequantization | **FILED / ADAPT WHEN CONSUMED** — [#8326](https://github.com/anthony-chaudhary/fak/issues/8326) |
| Indirect-command-buffer replay | Longer-run submission lever already owned by the existing Metal portfolio | **FOLD** — [#3416](https://github.com/anthony-chaudhary/fak/issues/3416) |
| Persistent Metal megakernel | Larger fusion lever already owned and deliberately follows the working resident spine | **FOLD** — [#3417](https://github.com/anthony-chaudhary/fak/issues/3417) |
| OS residency-set control | The captured run swapped zero bytes; current bottleneck evidence points at execution boundaries | **WATCH** — revisit only when same-checkpoint pressure telemetry proves eviction/page-in churn |
| Copy llama.cpp's graph runtime or Qwen converter | FAK already has the native hybrid model and a smaller execution seam; wholesale import would duplicate product scope | **REJECT** |

The candidate pass spans model ownership, runtime graph boundaries, Q4_K kernel shape, submission reuse, OS residency, and architecture conversion. Every surviving gap is either shipped, folded into an existing exact owner, or filed as a contract-ready issue; nothing remains as prose-only work.

## License and provenance

llama.cpp is MIT at the pinned root. The shipped FAK implementation was written independently against existing FAK abstractions; it adapts the ownership pattern, not source text. Future work must keep the upstream URL, revision, and source anchors in its issue/commit witness and retain any notices if source is ever copied.

## Companions

- Native acceptance and launch contract: [`docs/supported/qwen38-27b.md`](../supported/qwen38-27b.md)
- Captured exact-model fold: [`docs/_witnesses/qwen38-27b-2026-08-20/README.md`](../_witnesses/qwen38-27b-2026-08-20/README.md)
- Parent model-support epic: [#8011](https://github.com/anthony-chaudhary/fak/issues/8011)
- Capacity-preflight recalibration: [#8101](https://github.com/anthony-chaudhary/fak/issues/8101)
- Borrowing workflow: `field-borrow` + `study-repo` + `sota-check`
