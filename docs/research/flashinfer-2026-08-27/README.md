# FlashInfer exhaustive upstream study — 2026-08-27

**Verdict:** the pinned corpus is complete and valid at `39b484f1ce2fff086c66f9a899a0a58ba7f0ec3e`: **5,218 forge records** (3,519 pulls, 1,227 issues, 359 releases, 87 labels, 24 discussions, 2 milestones) plus a non-truncated **3,384-path** tree. The 22 decision-changing candidates all deduplicate to existing FAK work or an explicit dependency-boundary rejection. **No genuine new issue remained.**

## Boundary

FAK remains native end to end: it owns kernels, memory, scheduling, cache, adaptation, and operations. FlashInfer is Apache-2.0 prior art and failure evidence, not an automatic backend or fallback. Performance candidates require a FAK-native receipt naming exact device, model, shape, quality oracle, memory/setup overhead, and engine.

## Receipt

- Repository: [`flashinfer-ai/flashinfer@39b484f1ce2fff086c66f9a899a0a58ba7f0ec3e`](https://github.com/flashinfer-ai/flashinfer/tree/39b484f1ce2fff086c66f9a899a0a58ba7f0ec3e)
- Synchronized cutoff: `2026-08-27T10:00:00Z`
- Corpus: [`../inventory/flashinfer-ai-flashinfer-corpus.jsonl`](../inventory/flashinfer-ai-flashinfer-corpus.jsonl), 17,309,960 bytes, SHA-256 `d8099c9e03fe55d2cc39ec55b2ce8499d00e7ef607def5e65a9dd178f698fafa`
- Validation: `go run ./cmd/fak study-forge validate --receipt docs/research/inventory/flashinfer-ai-flashinfer-corpus.jsonl`
- Atomicity: complete but sequential across forge endpoints; the receipt records the non-atomic delta.

## Decisions

| Candidates | Mechanism family | FAK disposition |
|---|---|---|
| C01–C04 | paged, ragged, cascade, and block-sparse attention | Extend #3254/#3088 and graph safety in #1479. |
| C05, C08, C20 | prefill/decode planning and stable caller-owned workspace | One canonical contract: #8607. |
| C06–C07 | sampling and speculative helpers | Extend #4202/#8657; retain request-local RNG and empty-query oracles. |
| C09–C12, C21 | JIT/AOT/prebuilt lifecycle, generators, architecture gates | Extend #4184/#4180/#9139; no silent artifact or engine fallback. |
| C13 | vLLM/SGLang adapters | Reject runtime adoption; borrow contracts only under #6042. |
| C14, C18–C19 | benchmarks, tracing, offline tuning | Reuse shipped #8608/#8609 and shape tuning #9139; upstream numbers are not FAK claims. |
| C15–C17 | failure history, MLA, Qwen GDN | Feed regression oracles to #969/#4356/#9209. |
| C22 | persistent decode specialization | Extend #9137 with matched operating-envelope evidence. |

Detailed evidence, hardware/model envelopes, quality constraints, lifecycle costs, and one disposition per candidate are in [`candidate-ledger.json`](candidate-ledger.json). Required dedup coverage for #6042, #8607–#8609, #3254, #9137, #9139, and #9238 is explicit in [`dedup-map.json`](dedup-map.json). Source-class accounting and limitations are in [`source-completeness.json`](source-completeness.json).

## Completeness critic

The study does not call sequential API pages atomic, infer FAK gains from upstream benchmarks, or treat integration availability as permission to surrender native ownership. PR review comments and external dashboards were not independent endpoints; the complete issues/PRs/releases corpus, pinned tree, docs, code, tests, CI, benchmarks, roadmap labels, and failure history were all accounted for. No candidate was silently omitted or filed twice.
