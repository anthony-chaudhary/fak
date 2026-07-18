---
title: "KernelWiki borrow study: 5 sub-epics + 28 leaves onto existing fak seams, 47 candidates witnessed"
description: >
  A deep /study-repo pass over mit-han-lab/KernelWiki @ 76d27b56 (a curated GPU-kernel
  knowledge base + CI-invariant ingestion pipeline), scanned for the corpus-hygiene
  disciplines fak's knowledge/claim/ingest surfaces lack. 47 grounded candidates, one
  capability-witness each (dogfooding fak's own self-index): 43 PARTIAL, 3 ABSENT, 1
  PRESENT. The signal is not "fak lacks a feature" but "fak has the seam (freshness.go
  DriftKinds, CLAIMS.md/honesty.go one-tag lint, idea_scout firehose, benchauthority)
  without the invariant (closed vocab, bidirectional registry, reproducibility floor,
  committed decision ledger, byte-checkable provenance)." Organized the fileable set as
  5 single-axis sub-epics under umbrella epic #3946, each fully decomposed into
  independently-shippable child leaves (28 leaves total, #4072-#4099); deduped 2 into
  already-open #3925/#3923; captured the 8 Blackwell decode-kernel borrows as a
  file-after-measurement shortlist. All borrows inspire (Python->Go clean-room); no
  bytes vendored.
metadata:
  type: project
---

# KernelWiki borrow study (2026-07-10)

## What was studied

- **Repo:** [mit-han-lab/KernelWiki](https://github.com/mit-han-lab/KernelWiki)
- **Pinned SHA:** `76d27b56f804e7e7295d4c570e1e5d7eef4b0a75` (`76d27b56`, "docs: mark knowledge cutoff 2026-04-27"). All `path:line@76d27b56` citations below resolve against this pin.
- **What it is:** a curated knowledge base + ingestion pipeline for GPU-kernel technique. It harvests the vLLM/DeepGEMM/CUTLASS/Triton PR-blog-doc firehose, triages it into a committed candidate ledger with a closed decision enum, distills each accepted item into one immutable content-addressed source page, synthesizes wiki pages that cite sources by id, and guards the whole corpus with a large `scripts/validate.py` of CI invariants (closed-vocab tags, reproducibility floors, bidirectional registries, checksummed derived indices).

## Method

8 parallel subsystem readers over the pinned tree → 47 grounded candidate borrows (each ablated to the one axis it optimizes, each anchored at a real `path:line@76d27b56`) → one capability-witness per candidate that **dogfooded fak's own self-index** (`fak_feature_query` / `fak_index_*`) to grade fak on that axis and name the fak seam → completeness critic. 56 agents, 0 errors. Workflow journal: `…/subagents/workflows/wf_ab242ddc-86d/journal.jsonl`.

## License gate

README §License: tooling (`scripts/`, `references/`, `data/`) is "MIT-style" (no formal `LICENSE` file in the tree); wiki syntheses are derivative works citing upstream. Every borrow here is a Python→Go **clean-room reimplementation of a discipline**, not a byte copy ⇒ all **`inspire`**, **no bytes vendored**. Same posture as the sibling APEX / ktransformers / lmcache passes.

## Decisive finding

fak already has the *seams* but not the *invariants*. Witness tally across 47 candidates: **43 PARTIAL, 3 ABSENT, 1 PRESENT**. The high fileable rate is real (fak's knowledge/claim/ingest surfaces genuinely lack these disciplines). The candidates organize onto **five existing fak seams**, filed as **five single-axis sub-epics** under umbrella epic **#3946** — and each sub-epic is **fully decomposed into independently-shippable child leaves** (28 leaves, #4072–#4099), since every folded rung is a separable PR against a distinct seam. Two candidates were **deduped into already-open issues**; eight object-level kernel borrows were **captured but not filed** (they need a fak-side measurement first).

## Filed this pass

Structure: one top umbrella epic (**#3946**) → **5 sub-epics** (one per fak seam) → **28 leaves** (each folded rung filed as an independently-shippable ticket, #4072–#4099). The 35 fileable candidates in these five clusters collapse to 28 leaves (a handful of near-identical candidates merged into one leaf; noted per-cluster below).

| Sub-epic | Axis | Seam | Leaves |
|---|---|---|---|
| **#3947** `epic(idea_scout)` | no accepted triage item silently vanishes (committed decision-ledger + skip-audit closure) | `tools/idea_scout.py:445` | #4072–#4076 (5) |
| **#3948** `epic(devindex)` | cross-ref index = pure byte-reproducible projection of frontmatter, no dangling internal cite | `internal/devindex/freshness.go` | #4077–#4083 (7) |
| **#3949** `epic(claims)` | top claim tier must cite ≥2 independent evidence classes + machine-witnessed repro floor | `internal/marketing/honesty.go:30`, `internal/benchauthority` | #4084–#4089 (6) |
| **#3950** `epic(devfresh)` | version-pinned claim resolves bidirectionally to a central registry + decays offline | `internal/devindex/freshness.go`, `internal/docfreshrsi` | #4090–#4095 (6) |
| **#3951** `epic(field-borrow)` | borrowed excerpt carries durable byte-checkable provenance separate from synthesis | `internal/swebenchsota/swebenchsota.go:79` | #4096–#4099 (4) |

**Deduped into already-open issues (no new issue):**
- **#3925** (`feat(devindex): synonym + fuzzy fallback`) ← alias-aware ranking mechanism (canonical→alias map, best-variant MAX scoring, alias-normalized facet filters).
- **#3923** (`feat(antipattern): SOLUTION_GAMES_CHECKER`) ← object-reuse-cache eval exploit as a new named rung.

## Full candidate ledger (47) → filed home

Every candidate maps to a filed home so nothing is lost. `[V]` = witness verdict. Level: `sys` = system/tooling discipline, `kno` = object-level knowledge.

### → sub-epic #3947 · idea_scout committed decision-ledger + skip-audit closure (Cluster C, 6 candidates → 5 leaves #4072–#4076; inclusion-reason folded into the closure-gate leaf #4073)
| [V] | candidate | KernelWiki source @76d27b56 |
|---|---|---|
| PARTIAL | ledger-decision-enum-selfchecking-rollup | `scripts/validate.py:594-619` |
| PARTIAL | decision-to-artifact-closure-gate | `scripts/validate.py:764-808` + `generate-pr-pages.py:279-307` |
| PARTIAL | monotone-refresh-append-defer-plus-subset-invariant | `scripts/refresh_candidate_ledger.py:103-143` + `validate.py:847-874` |
| PARTIAL | inclusion-reason-plus-skip-audit-ledger | `scripts/generate-pr-pages.py:279-307` |
| PARTIAL (kno) | two-stage-cheap-then-file-level-retriage | `scripts/generate-pr-pages.py:102-142,380-388` |
| PARTIAL (kno) | dedup-as-structured-supersession-exclude | `candidates/deepgemm.yaml:40-48` |

### → sub-epic #3948 · devindex regenerated indices + internal citation-graph gate (Cluster D, 8 candidates → 7 leaves #4077–#4083; the two regenerated-index candidates merged into #4077)
| [V] | candidate | KernelWiki source @76d27b56 |
|---|---|---|
| PARTIAL | derived-index-layer-regenerated-from-frontmatter | `scripts/generate-indices.py:42-65` |
| PARTIAL | frontmatter-derived-indices-regenerated | `scripts/generate-indices.py:42-65,322-342` |
| PARTIAL | checksum-verified-reproducible-derived-index | `scripts/compute_core_prs.py:505-537` + `verify_core_prs.py:2-22` |
| PARTIAL | generated-symptom-to-fix-reverse-index | `queries/by-problem.md:5-13` |
| PARTIAL | wiki-cites-source-by-id-with-dangling-citation-gate | `scripts/validate.py:373-377` |
| PARTIAL | bidirectional-refgraph-integrity | `scripts/validate.py:374-377,661-686` |
| PARTIAL | closed-vocab-tag-registry-validated | `data/tags.yaml` + `scripts/validate.py:257-264,241-249` |
| PARTIAL | out-of-scope-retention-justification | `scripts/validate.py:335-345` |

### → sub-epic #3949 · claims evidence-tier gate + reproducibility floor (Cluster A, 10 candidates → 6 leaves #4084–#4089; the 3 reproducibility-floor and 2 dual-evidence candidates merged into #4085 and #4084)
| [V] | candidate | KernelWiki source @76d27b56 |
|---|---|---|
| PARTIAL | evidence-basis-quorum-for-verified | `scripts/validate.py:379-407` |
| PARTIAL | reproducibility-ladder-machine-witnessed | `scripts/validate.py:150-168,104-107,409-413` |
| PARTIAL | snippet-compilability-repro-floor | `scripts/validate.py:150-168,410-413` |
| PARTIAL | reproducibility-floor-on-harvested-knowledge | `schema.md:82-99` + `plan.md:95-101` |
| PARTIAL (kno) | confidence-evidentiary-ladder | `data/tags.yaml:86-90` + `validate.py:282-285` |
| PARTIAL (kno) | tiered-confidence-dual-evidence-gate | `internal/devindex/devindex.go:289` (fak seam) |
| PARTIAL (kno) | two-evidence-class-verified-gate | `Makefile:200` (fak seam) |
| PARTIAL (kno) | six-field-perf-claim-schema | `scripts/validate.py:347-371` + `data/schemas.yaml:186` |
| PARTIAL (kno) | submission-truth-provenance-tier-enum | `data/schemas.yaml:89-104` |
| PARTIAL (kno) | granular-pathway-evidence-memo | `data/triton-3.6-evidence.md:8-33` |

### → sub-epic #3950 · version-sensitive-claim registry + cutoff SoT + tool-version decay (Cluster B, 7 candidates → 6 leaves #4090–#4095; the two bidirectional-registry candidates merged into #4090)
| [V] | candidate | KernelWiki source @76d27b56 |
|---|---|---|
| PARTIAL | bidirectional-claim-registry | `scripts/validate.py:661-686,698-711` |
| PARTIAL | version-sensitive-bidirectional-registry | `scripts/validate.py:643-686` + `data/schemas.yaml:318-358` |
| PARTIAL | cutoff-date-sot-equality | `scripts/validate.py:814-838` + `data/refresh-cutoff.yaml:12` |
| PARTIAL | claim-signature-missing-pointer-guard | `scripts/validate.py:965-1008` |
| PARTIAL | tool-version-snapshot-time-decay | `scripts/check_version_freshness.py:66-111` |
| PARTIAL | upstream-amendment-detection | `scripts/verify_verbatim.py:122-164` |
| PARTIAL | dev-branch-pin-confidence-gate | `scripts/validate.py:1018-1075` + `data/schemas.yaml:358` |

### → sub-epic #3951 · immutable source-page layer + provenance taxonomy (Cluster F, 4 candidates → 4 leaves #4096–#4099)
| [V] | candidate | KernelWiki source @76d27b56 |
|---|---|---|
| PARTIAL | immutable-per-source-page-schema | `scripts/generate-pr-pages.py:228-249` |
| PARTIAL | provenance-taxonomy-sha-pin | `scripts/validate.py:1272-1296,1348-1351` |
| PARTIAL | durable-provenance-pin-with-drift-recheck | `SKILL.md:100-103` |
| PARTIAL | follow-sources-provenance-expansion | `scripts/get_page.py:160-175` |

### → #3925 (dedup) · alias-aware retrieval (Cluster E, 2)
| [V] | candidate | KernelWiki source @76d27b56 |
|---|---|---|
| PARTIAL | alias-aware-ranking-best-variant | `scripts/query.py:57-64,114-142` |
| PARTIAL | alias-normalized-facet-filters | `scripts/query.py:159-165,178-183` |

### → #3923 (dedup) · grader-gaming rung (Cluster H, 1)
| [V] | candidate | KernelWiki source @76d27b56 |
|---|---|---|
| PARTIAL (kno) | eval-object-reuse-reward-hack-mitigation | `wiki/patterns/moe-load-imbalance.md:36-42` |

## Noted, not filed — Blackwell decode-kernel shortlist (Cluster G, 8)

Object-level kernel knowledge from a *wiki*, not from fak-side profiling. Each witness itself said "measure the fak-side occupancy/traffic gap first," so filing kernel tickets here would be speculative. **File-after-measurement**: land a decode-shaped occupancy/traffic witness in `internal/compute` first, then promote the ones that show a real gap.

**The gate landed (#4188)** — the witness is `internal/compute/decode_occupancy.go`
(`DecodeGapReport`, commit `84a8034ac`), printed by `make cuda-occupancy`, with the A100
Nsight-Compute corroboration harness `tools/dgx_decode_occupancy_ncu.sh`. Per-candidate verdicts
and the committed baseline live in
[`2026-07-11-decode-occupancy-witness-measurement.md`](2026-07-11-decode-occupancy-witness-measurement.md):
4 FILE (promoted as #4289–#4292 under epic #3946), 1 measured-no-gap
(tmem-accumulator-migration — grid-bound, not per-SM-bound), 3 defer (not A100-measurable:
nvfp4, PDL, CLC try-cancel). The device ncu percentages remain pending (bridge-gated), recorded
as not-yet rather than fabricated.

| [V] | candidate | fak seam |
|---|---|---|
| ABSENT | tmem-accumulator-migration | `internal/compute/cuda_kernels.cu:719` |
| ABSENT | l1-cache-hints-decode (`__ldcs` stream-once loads) | `internal/compute/cuda_kernels.cu:726,742` |
| ABSENT | clc-decode-tile-scheduling | `internal/compute/cuda_kernels.cu:455` |
| PARTIAL | clc-try-cancel-speculative | `internal/compute/discard_admit.go:44` |
| PARTIAL | nvfp4-two-level-block-scale | `internal/ggufload/gguf_dequant.go:422` |
| PARTIAL | pdl-moe-kernel-overlap | `internal/compute/cuda_kernels.cu:454` |
| PARTIAL | moe-launch-fusion-ladder | `internal/compute/fusion_traffic.go:141` |
| PARTIAL | persistent-kernel-work-stealing-tail-fix | `internal/compute/cuda_kernels.cu:1084` |

## Dropped as PRESENT (1)

- **bidirectional-manifest-drift-closure** — fak's `internal/devindex/freshness.go` orphan/dead-link folds (`OrphanNotes`, `DeadDocLinks`, `UndeclaredLeaves`) already cover the manifest↔tree closure axis both directions.

## Coverage disclosure (completeness critic: gaps-remain)

The pass did **not** open these areas; deferred to a deeper read (no silent cap):
- The live `data/pr-page-skipped.yaml` (1747-row skip-audit ledger) and `data/refresh-search-results.yaml` — the runtime witnesses for #3947.
- `scripts/check_dod_fixtures.py` + `data/phase3-dod-fixtures.yaml` — a DoD-fixture asset-threshold gate (canonical capability question ⇒ on-disk assets meeting min_lines + per-file provenance_mode). Distinct un-candidated gate.
- `scripts/repo_size_check.py` + `data/phase3-size-budget.yaml` — pilot-projected corpus-growth budget (measure 20-item pilot → MiB projection → active ceiling + hard 6000-file budget, pre-commit enforced).
- `tests/fixtures/ledger-shape-{ok,bad}.yaml` — validator self-test via paired golden fixtures (the gate is itself tested with a conformant + a deliberately-defective input documenting each defect). A borrow-worthy testing discipline for fak's own guards.
- `scripts/fetch_pr_diff.py` — size-cap truncation-with-flag (over-budget assets replaced by a stub carrying `size_cap_truncated:true`) + capture-time back-pointer write.
- `scripts/install_precommit_hook.sh` — conditional/phase-gated enforcement (a gate fires only when its driving data file exists).
- 10/15 `wiki/techniques/` + 9/12 `wiki/kernels/` pages (software-exp, ping-pong-scheduling, grouped-gemm, flashmla, nsa, sparse-mla…) — the knowledge-borrow harvest is <40% swept.

## Sibling passes

Epics: #3921 (apex), #3900 (ktransformers), #3366 (lmcache), #3922 (apexagents). Parent effort: #3357 (scout-loop). Related fak epics touched: #2618 (wikimem), #1287 (devindex), #1278 (docs-freshness), #3807 (market-provenance).
