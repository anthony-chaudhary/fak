# Related-system inventory: make deep study measurable before borrowing

- **Date:** 2026-08-25
- **Status:** validator and first map generator spine shipped locally; root #8930 and child issues #8931-#8937 filed live
- **Spine witness:** `go test ./internal/studymonitor ./cmd/fak -run 'StudyMonitor|StudyInventory|InventoryMap|InventoryReport'`, `go run ./cmd/fak study-monitor --inventory-check --json`, and `go run ./cmd/fak study-inventory --root _scratch/study-inventory/ruflo --repository ruvnet/ruflo --revision 4dcff483482cee316f47552a961bcbaadc89f378 --json --out docs/research/inventory/ruvnet-ruflo.json`
- **Tracking marker:** `<!-- fak-related-system-inventory-key: related-system-inventory-2026-08-25 -->`

## Verdict

The old monitor made source freshness visible but let a study stop after a shallow note. The new spine adds a separate inventory gate: `fak study-monitor --inventory-check` treats candidate and studied rows as exhaustive by default and fails until each carries a machine-readable map path, matching indexed revision, positive subsystem count, completeness-critic result, and the full required source-class set.

The second spine adds the missing producer: `fak study-inventory` walks a local checkout at a pinned revision and writes a deterministic map of subsystems, language/file counts, representative paths, skipped dependency/control dirs, and required source-class status rows. This does not claim the study is complete; it creates the denominator a deep pass must finish.

The monitor now reads a registry row's JSON `map_path` and refuses unreadable, wrong-schema, wrong-repo, wrong-revision, or mismatched subsystem-count maps, so inventory metadata cannot pass by assertion alone.

The next hardening pass makes the denominator harder to fake: a map file must carry positive totals, a non-empty completeness critic, and one source-class status row for every required class. Local tree classes can clear through `covered` or `checked_absent` map rows; partial or external classes need explicit `source_evidence` rows. A registry row can no longer clear full forge/process coverage by listing `open_closed_issues_prs_discussions`, `fak_selfquery_witness`, `candidate_matrix`, or `issue_tracking` without evidence.

The first live readout is intentionally red: `fak-study-inventory-report/1` now reports `ok=false` with 39 blockers across the current registry. That gives the backlog a concrete denominator instead of an instruction to "study harder."

The first real map was created for `ruvnet/ruflo` at pinned revision `4dcff483482cee316f47552a961bcbaadc89f378`: [`docs/research/inventory/ruvnet-ruflo.json`](../research/inventory/ruvnet-ruflo.json), with [`docs/research/inventory/ruvnet-ruflo.md`](../research/inventory/ruvnet-ruflo.md) as the human rendering. It inventories 5,517 files across 18 immediate subsystems. The registry now points at that map but keeps the row red until full forge/API issue and PR read-back, fak self-query witness, candidate matrix, and issue-tracking artifacts exist.

## GitHub tracking state

The requested first step was GitHub tracking. Early attempts failed with `401 Bad credentials`, but a later `gh auth status` read-back succeeded for `anthony-chaudhary`, and exact marker searches found no duplicate issue for `related-system-inventory-2026-08-25`.

Live tracker:

- Root epic: [#8930](https://github.com/anthony-chaudhary/fak/issues/8930)
- Ruflo non-tree completion: [#8931](https://github.com/anthony-chaudhary/fak/issues/8931)
- Candidate map backfills: [#8932](https://github.com/anthony-chaudhary/fak/issues/8932), [#8933](https://github.com/anthony-chaudhary/fak/issues/8933), [#8934](https://github.com/anthony-chaudhary/fak/issues/8934), [#8935](https://github.com/anthony-chaudhary/fak/issues/8935)
- Studied-row backfill: [#8936](https://github.com/anthony-chaudhary/fak/issues/8936)
- Indexing-backend decision: [#8937](https://github.com/anthony-chaudhary/fak/issues/8937)

The root issue also has a child-link comment: <https://github.com/anthony-chaudhary/fak/issues/8930#issuecomment-5416600699>.

## Root issue body

Filed as [#8930](https://github.com/anthony-chaudhary/fak/issues/8930). The body below is retained because it is the reproducible issue text.

**Title:** `epic: exhaustive related-system inventory and repo indexing for study-repo`

**Milestone:** `Generation G0 - Now / Immediate`
**Labels:** `epic`, `research`, `gen/now`, `priority/P1`

```markdown
<!-- fak-related-system-inventory-key: related-system-inventory-2026-08-25 -->

## Value

- For: maintainers and agents using `study-repo`, `field-borrow`, and `scout-loop` to decide what external systems teach FAK.
- Problem: study passes can stop after a shallow README/code skim, then either file a monolith or file nothing.
- Today: `docs/research/monitored-repositories.json` tracks freshness, but not whether a repo has an exhaustive indexed map.
- Better because: every candidate/studied source has a checkable inventory denominator before any borrow/adopt/dismiss decision.
- Centrality: Enabling - it improves the quality of Core backlog generated from external systems.
- P1: advanced - agents can load a bounded map instead of a transcript.
- P2: advanced - study effort becomes reusable and dedupeable.
- P3: preserved - inventory is evidence, not a commitment to copy or adopt.
- P4: advanced - GitHub issues and dispatch lanes get concrete source-class gaps.

## Core through-line

Add a repo-native inventory contract to `study-monitor` plus a local map generator -> run them over `docs/research/monitored-repositories.json` and a pinned scratch clone -> file child leaves for missing maps, non-tree source classes, and indexing integration -> dispatch workers only after issue IDs exist.

## Gold-plating boundary

Do not build a full Sourcegraph/Glean clone in this issue. The root spine is the failing/passing inventory gate plus issue-ready backlog. Full text indexes, semantic indexes, CodeQL databases, and per-repo study backfills are child leaves.

## Scope / tree

`internal/studymonitor/**`, `cmd/fak/study_monitor.go`, `cmd/fak/study_monitor_test.go`, `cmd/fak/study_inventory.go`, `cmd/fak/study_inventory_test.go`, `docs/research/monitored-repositories.*`, `docs/research/inventory/**`, and the dated research note.

## Acceptance

`fak study-monitor --inventory-check --json` emits `fak-study-inventory-report/1`, defaults candidate/studied rows to exhaustive, exits nonzero on missing maps, and names the exact missing source classes. `fak study-inventory` emits `fak-study-inventory-map/1` from a pinned local checkout and explicitly distinguishes local tree coverage from source classes that require forge/API or study-pass artifacts. The map file itself is now part of the proof: missing source-class status rows, empty totals, empty completeness critics, or bare registry claims for partial/external classes are blockers.

## Witness / proof

`go test ./internal/studymonitor ./cmd/fak -run 'StudyMonitor|StudyInventory|InventoryMap|InventoryReport'` passes. A live run against the registry reports 39 blockers rather than silently accepting shallow studies. A pinned Ruflo scratch clone rendered `docs/research/inventory/ruvnet-ruflo.json` plus the Markdown companion.

## Placement

`gen/now`, `priority/P1`, `studymonitor` lane.
```

## Generated follow-on plan

Plan command:

```bash
go run ./cmd/fak-dev issue fanout \
  --title "study inventory completeness gate" \
  --leaf studymonitor \
  --spine "go test ./internal/studymonitor ./cmd/fak -run 'StudyMonitor|StudyInventory|InventoryMap|InventoryReport' && go run ./cmd/fak study-monitor --inventory-check --json && go run ./cmd/fak study-inventory --root _scratch/study-inventory/ruflo --repository ruvnet/ruflo --revision 4dcff483482cee316f47552a961bcbaadc89f378 --json --out docs/research/inventory/ruvnet-ruflo.json" \
  --paths internal/studymonitor \
  --areas qa,dogfood,product,integration,docs \
  --max 8 \
  --json
```

Contract readback:

```text
schema=fak.issue-contract-result.v1
ok=true dispatchable=8 triage_only=0 refused=0 step_budget=25
```

| Key | Title | Labels | Generation | Steps |
|---|---|---|---|---:|
| `fanout-studymonitor-spine-d6082a942fe2a0c4-qa-edge-sweep` | qa: adversarial + edge-case sweep for study inventory completeness gate | `fanout`, `qa` | `gen/now` | 3 |
| `fanout-studymonitor-spine-d6082a942fe2a0c4-qa-failure-paths` | qa: failure-path + refusal coverage for study inventory completeness gate | `fanout`, `qa` | `gen/now` | 3 |
| `fanout-studymonitor-spine-d6082a942fe2a0c4-qa-determinism` | qa: determinism + race witness for study inventory completeness gate | `fanout`, `qa` | `gen/now` | 2 |
| `fanout-studymonitor-spine-d6082a942fe2a0c4-dogfood-self-run` | dogfood: run study inventory completeness gate on this repo's own live work | `fanout`, `dogfood` | `gen/now` | 3 |
| `fanout-studymonitor-spine-d6082a942fe2a0c4-dogfood-usage-ledger` | dogfood: usage ledger so study inventory completeness gate adoption is measured, not claimed | `fanout`, `dogfood` | `gen/next` | 4 |
| `fanout-studymonitor-spine-d6082a942fe2a0c4-product-cli-reference` | product: docs/cli-reference.md entry + usage parity for study inventory completeness gate | `fanout`, `product` | `gen/next` | 2 |
| `fanout-studymonitor-spine-d6082a942fe2a0c4-product-lcd-demo` | product: LCD demo/example for study inventory completeness gate meeting the run-the-demos bar | `fanout`, `product` | `gen/next` | 5 |
| `fanout-studymonitor-spine-d6082a942fe2a0c4-product-error-ux` | product: refusal/error-message quality pass for study inventory completeness gate | `fanout`, `product` | `gen/next` | 3 |

Live fan-out filing with `fak-dev issue fanout --live` exited 0 but skipped all eight generated rows because the root issue body already contains their marker keys. That is an anti-duplication behavior, not evidence that the eight rows have independent issue numbers. The concrete repo-scope children were filed manually as #8931-#8937.

## Filed scope issues

1. [#8931](https://github.com/anthony-chaudhary/fak/issues/8931) completes Ruflo's non-tree classes: full forge/API issue and PR read-back, fak self-query witness, candidate matrix, and issue-tracking artifacts.
2. [#8932](https://github.com/anthony-chaudhary/fak/issues/8932) maps `Untrivial-ai/agent-orchestrator` at `ee2c58a577317d4480a2b2ee5ff77f25e5307af9`.
3. [#8933](https://github.com/anthony-chaudhary/fak/issues/8933) maps `langchain-ai/open-swe` at `a6c360047186cc5b8afe3a74012a12bfc94ae7c7`.
4. [#8934](https://github.com/anthony-chaudhary/fak/issues/8934) maps `EveryInc/compound-engineering-plugin` at `345e2bea42cabea2f32d25a68fd1739e8e92cd03`.
5. [#8935](https://github.com/anthony-chaudhary/fak/issues/8935) maps `obra/superpowers` at `b36e0829c6d0140e93cfef2ca599b1b07d4a7797`.
6. [#8936](https://github.com/anthony-chaudhary/fak/issues/8936) backfills or re-issues the already-studied registry rows.
7. [#8937](https://github.com/anthony-chaudhary/fak/issues/8937) decides which repo-indexing layers stay native versus optional backend recipes.

## Prior-art indexing lessons checked

- Sourcegraph separates search-based navigation from precise compile-time navigation. FAK should keep that split: a broad lexical map is useful immediately, while semantic precision needs explicit build/index evidence. Source: <https://sourcegraph.com/docs/code-navigation>.
- Zoekt is the practical local text-search candidate: it indexes local git repos, can sync repository roots, supports search through CLI/API, and is built around fast code-oriented text search. Source: <https://github.com/sourcegraph/zoekt>.
- SCIP is a language-agnostic index format for definitions/references and already has a tooling ecosystem. It is a better interchange shape for semantic facts than inventing a FAK-only symbol schema first. Source: <https://scip-code.org/>.
- Kythe's extractor/indexer split is the useful discipline for deep inventories: capture the program/dependencies/build arguments before claiming semantic completeness. Source: <https://kythe.io/docs/schema/writing-an-indexer.html>.
- OpenGrok highlights a source-class FAK already cares about: code search plus cross-reference plus version-control history. Source: <https://github.com/oracle/opengrok>.
- Glean stores typed schema-defined facts and queries them with a Datalog-style language; that points to a future semantic inventory layer, not a requirement for the first spine. Source: <https://glean.software/>.
- CodeQL database creation requires checkout state, dependencies/environment, and usually build commands for compiled languages. FAK should record those prerequisites before treating a semantic database as reproducible evidence. Source: <https://docs.github.com/en/code-security/tutorials/customize-code-scanning/prepare-code-for-analysis>.
- GitHub's code search writeup reinforces commit-stable, content-addressed indexing: ngram indexes make substring search fast, blob IDs avoid duplicate content, and commit-level consistency matters. Source: <https://github.blog/engineering/architecture-optimization/the-technology-behind-githubs-new-code-search/>.

## Agent delegation status

No live FAK dispatch wave was launched in this pass. `python3 tools/dispatch_status.py --fast` reported `READY_TO_GROW`, `SPAWN_OK`, and `0/4 live`. A Codex wave dry-run for #8932 was held by `MODEL_UNSUPPORTED` for `gpt-5.6-sol` under the current credential class, but a Claude backend dry-run returned `ok=true`, `verdict=WOULD_WAVE`, selected `docs#8932`, lane `docs`, lease `resolve-docs-8932`, and tree `docs/research/inventory/` plus `docs/research/monitored-repositories.json`.

Per the super-loop contract, the live worker was not launched without explicit post-plan approval. The staged launch command is recorded on [#8932](https://github.com/anthony-chaudhary/fak/issues/8932#issuecomment-5416689770).

## Current blockers

- The live registry has 39 inventory blockers by the new gate.
- The gate proves map absence, not map quality after a row is backfilled; the follow-on QA and dogfood issues cover adversarial maps, determinism, and adoption/usage evidence.
