---
title: "llama.cpp exhaustive upstream index and FAK ticket priority — 2026-08-26"
description: "Pinned llama.cpp tree and open-forge inventory, deduplicated against FAK tickets, with a ranked native-inference queue."
---

# llama.cpp exhaustive upstream index and FAK ticket priority — 2026-08-26

## Verdict

The complete obtainable **open** llama.cpp forge corpus at the cutoff contains
**2,226 records: 830 issues and 1,396 pull requests** across 23 REST pages. The
pinned source tree at `925e1179947ea0c0ebfb0032df18af3a729822be` contains
**3,871 entries** and was not truncated. Exact FAK full-text searches for
`"llama.cpp"`, `"llama cpp"`, and `llamacpp` produce **357 unique local issues**,
**87 open** and **270 closed**.

The index found three dispatchable gaps and created them:

1. **#9292 P1** — adaptive draft depth plus DFlash2 selector evaluation.
2. **#9293 P1** — native grammar-sampling performance.
3. **#9294 P1** — token lineage on native KV cells for exact reuse witnesses.

The active Qwen3.8-Flash-Next spine remains **#9204 P0**. Existing tickets cover
disaggregated prefill/decode (#50), Metal hot-path borrowing (#8394), NUMA weight
replication (#5127), and ternary GPU work (#4867); creating duplicates would make
the backlog worse.

## Observation identity

| Field | Value |
|---|---|
| Parent | #9270 |
| Cutoff | 2026-08-26T22:20:00Z |
| Repository | `ggml-org/llama.cpp` |
| Default branch | `master` |
| Pinned commit | `925e1179947ea0c0ebfb0032df18af3a729822be` |
| Commit time | 2026-08-26T21:34:28Z |
| Tree | 3,871 entries: 3,498 blobs, 373 trees; `truncated=false` |
| Open forge | 2,226 unique records: 830 issues, 1,396 PRs |
| Open range | created 2023-03-17 through 2026-08-26 |
| Recent commits checked | 119 commits from 2026-08-20 through the pinned commit |
| Releases checked | 6,958; latest observed `b10642`, published 2026-08-26T22:00:33Z |
| Local search union | 357 FAK issues: 87 open, 270 closed |
| Machine map | [`docs/research/inventory/ggml-org-llama-cpp.json`](../research/inventory/ggml-org-llama-cpp.json) |

The repository metadata count moved from 2,226 to 2,227 while the audit was in
progress. The committed corpus is deliberately pinned to the earlier, fully
paginated 2,226-record snapshot and names that boundary instead of pretending a
moving forge is timeless.

## What “exhaustive” means here

This pass exhaustively retains every record from the GitHub REST
`issues?state=open` pagination at the cutoff, then separates issues from PRs by
the API's `pull_request` field. It also captures the complete recursive tree at
the pinned commit, all releases obtainable by pagination, the 119 commits since
the August 20 study, and the exact union of three local issue searches.

It does **not** claim that all historical closed upstream issues and PRs received
semantic classification. Closed upstream work enters through the pinned tree,
recent commits, releases, dated studies, and named references such as merged PR
#27762. That boundary is explicit in the machine map.

Categories are deterministic, non-exclusive search tags. They are an inventory
aid, not an automatic product decision. Every promoted candidate received a
manual centrality, duplicate, dependency, witness, and fak-native audit.

## Source-shape map

The pinned tree is dominated by the actual implementation and operational
surfaces, not only top-level prose:

| Top-level surface | Entries |
|---|---:|
| `ggml/` | 1,369 |
| `tools/` | 1,159 |
| `examples/` | 302 |
| `src/` | 217 |
| `models/` | 121 |
| `conversion/` | 90 |
| `.github/` | 84 |
| `tests/` | 83 |
| `scripts/` | 80 |
| `docs/` | 75 |
| `common/` | 75 |
| `benches/` | 11 |
| `grammars/` | 10 |

The backend tree includes CPU, CUDA, Metal, Vulkan, HIP, SYCL, OpenCL, CANN,
MUSA, OpenVINO, WebGPU, Hexagon, RPC, zDNN, ZenDNN, and other adapters. FAK does
not need parity with every backend. The useful denominator is that no backend,
model, server, test, example, conversion, or build surface was invisible to the
index before prioritization.

## Ranked queue

| Rank | Disposition | FAK ticket | Upstream evidence | Reason |
|---:|---|---|---|---|
| 1 | Execute | **#9204 P0** | llama.cpp #27742 | Qwen3.8-Flash-Next is the active native model spine and is already decomposed through #9205–#9213. |
| 2 | Execute | **#9292 P1** | #27210, #27342 | Fixed speculative depth is an uncovered performance policy gap; DFlash2's trained pieces remain WATCH-gated. |
| 3 | Execute | **#9293 P1** | #4218 | FAK has constrained-decoding correctness seams but no open grammar-sampling performance ticket. |
| 4 | Execute | **#9294 P1** | merged #27762 | Token lineage strengthens native cache rollback/reuse evidence and builds on #8464/#8468. |
| 5 | Existing | #8394 P1 | #4085 | The existing profiled Metal hot-path borrow ticket covers compile-time kernel specialization. |
| 6 | Existing | #50 P0 | #21266 | The disaggregated-serving epic already owns prefill/decode separation. |
| 7 | Existing | #5127 | #16000 | Per-NUMA weight replication already has a FAK ticket. |
| 8 | Existing | #4867 | #11183 | Ternary Qwen/GGUF support is the matching product lane for TQ2_0 kernels. |
| 9 | Watch | none | #27560 | Windows/Vulkan context-checkpoint crash lacks a current matching FAK-native Vulkan envelope. |
| 10 | Exclude | none | #27725 | The generic memory-leak report has no reproducible FAK-native failure or matched envelope. |

## New ticket construction

### #9292 — adaptive native speculative depth

**For:** fak-native Qwen3.8 speculative decode.  
**Problem:** #23 and #4202 cover the broad spine and sampled correctness, not an
acceptance-driven depth policy.  
**Better because:** measure accepted tokens, verifier work, latency, bytes, and
joules, then adapt depth with bounded hysteresis.  
**Witness:** quality-complete matched receipts against fixed depths; deterministic
fallback under adversarial acceptance.  
**Constraint:** llama.cpp is a reference/benchmark only. Target verification
remains the exactness boundary.

This reconciles the earlier DFlash2 study: retain the serial dynamic-programming
selector and trained convolution as WATCH candidates, while implementing the
smaller adaptive-depth spine first.

### #9293 — grammar-sampling performance

**For:** native schema-constrained tool calls.  
**Problem:** closed #469 and #4548 cover feasibility and semantic quality, while
upstream #4218 identifies a continuing sampling hot path.  
**Better because:** attribute compile, transition, allowed-token, mask, and sync
costs and optimize only the measured dominant seam.  
**Witness:** identical allowed-token sets and outputs, plus per-token latency and
tokens/s across nested JSON, Unicode, impossible schemas, and an unconstrained
control.

### #9294 — KV token lineage

**For:** exact prefix reuse, rollback, and cache diagnostics.  
**Problem:** FAK has segment address and lifecycle work in #8464/#8468 but no
single explicit token-lineage witness attached to resident native KV state.  
**Better because:** compact token identity or a collision-safe digest can expose
wrong-prefix reuse and rollback corruption without making llama.cpp the engine.  
**Witness:** corruption/shift/reuse/rollback tests plus metadata-byte and update-
overhead receipts. The first version is diagnostic, not admission-critical.

## Dedup and closure decisions

- **Qwen3.8-Flash-Next:** #27742 maps to #9204 and its children, not a new model
  ticket.
- **DFlash2:** the August 20 study remains authoritative negative knowledge;
  #9292 narrows work to the untrained adaptive policy and measurement seam.
- **Disaggregated serving:** upstream #21266 maps to #50 and #4302.
- **NUMA mirroring:** upstream #16000 maps to #5127.
- **Metal specialization:** upstream #4085 maps to #8394 and must still be chosen
  by a FAK profile, not copied speculatively.
- **Ternary kernels:** upstream #11183 maps to #4867; a separate kernel ticket is
  premature until that model lane selects the exact representation/envelope.
- **OpenCL MoE batch heuristic #27637:** useful prior art, but current critical
  paths are Metal/CUDA and already have Qwen/MoE kernel tickets. Retain indexed;
  do not create an OpenCL product lane by accident.
- **Qwen3.6 tool-choice bug #27767:** historical Qwen3.6 compatibility only. The
  Qwen3.8 tool-parser regression surface is already #9206.

## Priority rubric

1. **P0:** blocks the current Qwen3.8 native execution spine or its correctness.
2. **P1:** measured native performance/correctness gap with an existing execution
   path and a falsifiable witness.
3. **P2:** useful after a named dependency or operating envelope exists.
4. **Watch:** upstream signal without a current FAK-native envelope.
5. **Exclude:** no reproducible FAK problem, non-core product surface, duplicate,
   or a path that would surrender native ownership.

Within a band, prefer work that improves a shared kernel/cache/scheduler seam,
has a matched receipt, and compounds across models. Backend breadth and upstream
novelty do not outrank the active native path.

## Validation contract

The machine map is valid when:

- its schema is `fak-llamacpp-index/1`;
- upstream item identities are unique and counts reconcile to 2,226 / 830 / 1,396;
- the tree is not truncated and its counts reconcile;
- local search identities are unique and counts reconcile to 357 / 87 / 270;
- every priority row has a disposition, reason, upstream record, and an existing
  FAK issue when `fak_issue` is non-null;
- all newly constructed tickets are open, labeled, linked to #9270, and state a
  witness plus the fak-native/no-silent-fallback constraint where applicable.

## Completeness critic

- The open forge moved during collection. The timestamp, SHA, counts, and drift
  are therefore part of the result.
- GitHub discussions were not obtainable through the REST issue corpus and are
  not claimed covered.
- The full historical closed forge is too large to semantically classify in this
  spine; the source tree, releases, recent commits, prior studies, and exact
  local issue history cover landed and locally relevant history.
- Regex categories can over-tag broad words such as “model” or “cache.” They are
  denominator metadata only; the ranked queue is manually adjudicated.
- The launched independent ultracode audit exceeded its parent token budget and
  returned an invalid orchestration verdict. Its output is not used as evidence.

## Next refresh

Refresh when llama.cpp lands a material Qwen3.8, cache, speculative, grammar,
scheduler, quantization, CUDA, or Metal change, or after 30 days. Reuse the
forge-corpus spine tracked by #9272 rather than committing raw API pages.
