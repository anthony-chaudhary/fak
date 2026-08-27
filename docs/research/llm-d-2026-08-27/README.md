# llm-d exhaustive upstream study â€” 2026-08-27

**Verdict:** the fixed-cutoff forge corpus is valid and complete for every declared endpoint, but it is explicitly a sequential, non-atomic crawl. Twenty candidate mechanisms are accounted for: 10 `INTEGRATE`, 3 `ADAPT`, 4 `WATCH`, and 3 `REJECT`; only C11 and C12 were genuine unduplicated implementation gaps; they were filed as #9385 and #9386.

## Pin and receipt

- Repository: `llm-d/llm-d@bc20f73bd344b5a0faad5afca93831088aeee957`
- Inclusive cutoff: `2026-08-27T10:00:00Z`
- Corpus: `../inventory/llm-d-llm-d-corpus.jsonl` â€” 5273559 bytes, 2,474 normalized records, SHA-256 `8df8bb3eb0ead297ce143ce2a21bda4a4cfc486f7e71617b9dbab632e5c8ff7f`.
- Validation: `go run ./cmd/fak study-forge validate --receipt docs/research/inventory/llm-d-llm-d-corpus.jsonl` â†’ `valid study forge receipt`.
- Forge pages: issues 24/2,386 raw rows; pulls 19/1,850; discussions 1/0; releases 1/11; labels 1/66; milestones 1/11. Issue normalization removes pull rows, yielding 536 issue + 1,850 pull + 88 release/label/milestone records.
- Atomicity: **not claimed**. `source-completeness.json` records the sequential crawl window and the receipt's exact non-atomic delta.

The 5.27 MB corpus is retained because the repository already versions a larger 13.29 MB Dynamo corpus in the same inventory directory; replacing this evidence with a lossy summary would reduce reproducibility.

## Artifacts

- `candidate-ledger.json` â€” one row per extracted mechanism, with source state, exact evidence, FAK seam, ownership boundary, deployment assumptions, alternative, disposition, and dedup.
- `source-completeness.json` â€” every declared tree and forge source class, checksums, counts, limits, and completeness-critic verdict.
- `dedup-map.json` â€” every candidate mapped to an existing ticket, explicit watch/reject decision, or an independently searched gap.

## Ownership boundary

FAK remains native end-to-end: model execution, cache/KV ownership, routing and scheduling authority, policy/security gating, and evidence receipts stay inside FAK. The study borrows portable contracts and negative knowledge. It rejects Kubernetes CRDs, Gateway API product surface, ModelService, and any topology that would silently put llm-d or an underlying runtime in charge of FAK execution.

## Candidate outcome summary

| IDs | Outcome | Rationale |
|---|---|---|
| C01â€“C05, C08â€“C10, C13, C15 | `INTEGRATE` | Existing FAK tickets already own the portable seam. |
| C06, C11, C12 | `ADAPT` | Portable serving contracts fit, while Kubernetes/operator mechanics do not. |
| C14, C17, C20 | `WATCH` | Connector, proposal-stage, or workload-specific evidence lacks a stronger current native gap. |
| C16, C18, C19 | `REJECT` | Double-routing, surrendered ownership, or superseded product API. |

## Completeness critic

No declared class is absent. The pinned tree walk covered 1,139 paths, including docs, architecture, proposals, guides, workflows/tests, release/history, and license. Forge pagination covered issues, PRs, discussions, releases, labels, and milestones. No dedicated roadmap/TODO/backlog file exists; proposals, milestones, releases, issues, and PRs are the observed roadmap surfaces. Limits are explicit in `source-completeness.json`.

## Filed gaps

- #9385 - typed queued/admitted/in-flight overload control with retry and eviction receipts (`ADAPT`).
- #9386 - quarantine unhealthy native workers with fenced readiness recovery (`ADAPT/WATCH`; upstream proposal-stage).

No other candidate justified a new ticket at this cutoff.
