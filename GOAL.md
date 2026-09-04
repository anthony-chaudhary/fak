---
loop: goal
witness: python tools/check_index_sync.py --audit-tree && python tools/gen_llms_full.py --check && python tools/gen_structured_data.py --check
budget: { max_iters: 20 }
lane: docs
---
# Objective
Refresh repository indexes and query navigation concepts: reconcile `INDEX.md` with unindexed dated notes under `docs/notes/` so the reciprocal index-sync gate passes, regenerate `llms-full.txt` from `llms.txt`, regenerate structured data in `docs/index.md` and `docs/_includes/head-custom.html`, and refresh concept disambiguation index/scorecard artifacts.

# Non-Goals
- Do not touch or commit peer-dirty WIP files outside the index and navigation documentation surfaces.
- Do not modify kernel code or introduce unsolicited abstractions.

# Plan
- [x] 1. Inspect unindexed notes and determine titles/summaries for `INDEX.md`.
- [x] 2. Add missing dated notes to `INDEX.md` and verify `python tools/check_index_sync.py --audit-tree` passes.
- [x] 3. Regenerate `llms-full.txt` using `python tools/gen_llms_full.py` and verify `--check`.
- [x] 4. Regenerate structured data using `python tools/gen_structured_data.py` and verify `--check`.
- [x] 5. Refresh concept disambiguation scorecard/index artifacts (`python tools/concept_disambiguation_scorecard.py --markdown-dir docs/concept-disambiguation-scorecard`).
- [x] 6. Execute full witness exit-gate command and seal results.

# Scratch / last-refusal
- Witness exit-gate command succeeded (all 3 gates passed):
  1. `python tools/check_index_sync.py --audit-tree` -> `index-sync: clean (no dangling links, no unlisted dated notes).`
  2. `python tools/gen_llms_full.py --check` -> `llms-full.txt up to date (57 documents, 1,522,405 bytes)`
  3. `python tools/gen_structured_data.py --check` -> `structured data up to date (no drift, no answer artifacts)`
- Devindex freshness:
  `go run ./cmd/fak-dev index freshness` -> 0 orphan-note findings reported.
- Disambiguation docs verification:
  `go run ./cmd/fak disambiguation docs --check` -> clean (all pages unchanged).
