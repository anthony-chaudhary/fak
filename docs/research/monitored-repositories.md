---
title: "Monitored related repositories"
description: "Checked: 2026-08-28 Machine source: monitored-repositories.json Refresh view: fak study-monitor --due-days 14 (add --json for automation) Inventory gate:"
---
# Monitored related repositories

**Checked:** 2026-08-29
**Machine source:** [`monitored-repositories.json`](monitored-repositories.json)
**Refresh view:** `fak study-monitor --due-days 14` (add `--json` for automation)
**Inventory gate:** `fak study-monitor --inventory-check --json`
**Inventory map:** `fak study-inventory --root <scratch-clone> --repository owner/name --revision <sha> --json --out docs/research/inventory/owner-name.json`
**Forge census:** `fak study-forge capture --repository owner/name --cutoff <RFC3339> --out <allocated-artifact>.json`

This is the durable discovery queue for external harness, model-serving, and agentic-performance repositories. It complements dated deep-study notes: the registry records candidates before a full study, pins what was actually checked, and keeps `last_checked` visible so a scout can distinguish a fresh lead from a stale one. Exact `owner/name` is the deduplication key.

## Current ranking

| Rank | Repository | Why it is hot and helpful now | Discovery evidence |
|---:|---|---|---|
| 1 | [`ruvnet/ruflo`](https://github.com/ruvnet/ruflo) | The most direct meta-harness overlap: coding-agent swarms, memory, skills, MCP, hooks, and Codex/Claude integrations. | 67,866 stars; 47 commits from 2026-08-07 through check time; pushed 2026-08-14. |
| 2 | [`Untrivial-ai/agent-orchestrator`](https://github.com/Untrivial-ai/agent-orchestrator) | The closest fleet-control analogue: planning, worktree-isolated agents, CI repair, review, and merge-conflict handling. | 9,518 stars; at least 100 commits in the seven-day window; pushed 2026-08-14. |
| 3 | [`langchain-ai/open-swe`](https://github.com/langchain-ai/open-swe) | An asynchronous coding-agent harness with sandbox, task lifecycle, desktop, evaluation, and session seams. | 10,553 stars; 80 commits in the seven-day window; pushed 2026-08-14. |
| 4 | [`EveryInc/compound-engineering-plugin`](https://github.com/EveryInc/compound-engineering-plugin) | A cross-harness Codex/Claude/Cursor plugin whose plan-review-work-compound loop is directly comparable to fak's skill and evidence workflows. | 24,259 stars; 20 commits in the seven-day window; pushed 2026-08-14. |
| 5 | [`obra/superpowers`](https://github.com/obra/superpowers) | The strongest adoption signal for a portable skills, hooks, planning, testing, and subagent-development methodology not already tracked by exact repository name. | 272,168 stars; release activity at pinned revision; pushed 2026-08-13. |
| 6 | [`local/tensor-build`](../notes/CONCEPT-STUDY-TENSOR-BUILD-2026-08-29.md) | Whole-tree recheck of evidence, measurement, and native-runtime contracts. It retained seven exact fak owners, excluded or watched four TensorRT-shaped mechanisms, and filed 13 bounded leaves (#10268-#10271, #10278-#10286). | Snapshot SHA-256 `64986cf8ff942cdcd6178491d3d9af0199c354ce41f998b89ced2b0286f6772d`; 6,012 raw / 5,972 indexed files; 32/32 source hashes read back; source history and license unavailable, so concepts only. |

The GitHub API's 100-item page cap means “at least 100” is deliberate. Stars are only a discovery signal. A deep study still has to pin source, inspect implementation seams, check the license, query fak first, and independently witness any claimed gap.

## Maintenance contract

1. Run `fak study-monitor --due-days 14` before selecting a scout lead.
2. Search this registry and tracked notes by exact `owner/name`; do not rediscover an existing source under a nickname.
3. On every meaningful check, update `last_checked`, `checked_revision`, `stars_at_check`, and `last_push_at_check` together.
4. Use `candidate` for an unstudied lead, `studied` only with `study_note`, `watch` when no present borrow survives, and `dismissed` only with the reason retained in `why`.
5. A completed `study-repo` pass updates this row in the same commit as its dated study note. A `scout-loop` pass selects due candidates before opening a fresh outward search.
6. Candidate and studied rows default to **exhaustive inventory** for `fak study-monitor --inventory-check`. Start each pass by generating the local map with `fak study-inventory`, then complete the non-tree classes. To clear the gate, add an `inventory` block with `map_path`, `indexed_revision` equal to `checked_revision`, positive `subsystem_count`, a `completeness_critic`, and every required source class: `readme_docs`, `architecture_design`, `runtime_source`, `tests_fixtures`, `history_changelog_releases`, `open_closed_issues_prs_discussions`, `roadmap_todos`, `license_provenance`, `fak_selfquery_witness`, `candidate_matrix`, `completeness_critic`, and `issue_tracking`.
7. Do not satisfy a class by naming it alone when the machine map says the class is `partial` or `external_required`. Prefer `inventory.forge_receipt_path` for the compound forge class: it points to a complete `study-forge` corpus or receipt whose repository, revision, cutoff, six source receipts, counts, and checksums the inventory gate validates. This external receipt does not change the local map's status. Add `source_evidence` rows with durable commands, exported artifacts, note anchors, or issue URLs for fak self-query witnesses, candidate matrices, issue tracking, or legacy forge evidence. Legacy forge evidence must name issue, pull-request, and discussion coverage. The check does not replay commands, re-fetch URLs, or mistake the top-level forge census for comment/review/event enrichment.

This list intentionally does not claim that these repositories contain a borrowable gap. It proves only that they were fresh, relevant, deduped by exact source identity, and worth a disciplined study pass or refresh. The 2026-08-25 update added current model-serving and agentic-performance candidates for the Qwen3.8/native-performance, discovery, and queue/supervision lanes.

## Newly studied source

| Repository | Result | Pinned evidence |
|---|---|---|
| [`modular/modular`](https://github.com/modular/modular) | Full-tree, full-history, compiler/runtime/serving/model/kernel study produced 87 witnessed candidates: 68 contract-clean issues (#9910-#9977) and 19 recorded-only dispositions under #9900. FAK retains native execution ownership; no Modular runtime fallback was proposed. | `1c9fd2e03331f77d3a1034127cb3700b7fa43c02`; [study note](../notes/CONCEPT-STUDY-MODULAR-MONOREPO-2026-08-28.md), [inventory](inventory/modular-modular.json), and [forge receipt](inventory/modular-modular-forge-receipt.json). |
| [`SemiAnalysisAI/InferenceX`](https://github.com/SemiAnalysisAI/InferenceX) | AgentX lifecycle/interactivity, scope-versioned measured-power, and reuse integrity yielded #8773-#8775; lineage reuse otherwise converged with fak. | `0b0138fd7de0a6f927f9769b19d594d01f586107`; [study note](../notes/BORROW-BENCHMARK-SERVING-METRICS-INFERENCEX-STUDY-2026-07-13.md). |
| [`strukto-ai/mirage`](https://github.com/strukto-ai/mirage) | Deep study retained one borrow: a strict pre-effect drift contract for mutable external inputs on resume/replay. Typed backend capabilities and transactional workspaces map to stronger existing fak seams/issues; a universal VFS is a bounded optional integration, not the default. | [`mirage-study-2026-08-17.md`](../notes/mirage-study-2026-08-17.md), revision `e0a4f51109cbe6b8a239700d8348f0cbebd70b26`; 3,499 stars and last push `2026-08-17T23:41:51Z` at check time. |
| [`agent0ai/agent-zero`](https://github.com/agent0ai/agent-zero) | Deep study found lifecycle legibility and effect-centered Time Travel worth retaining as design guidance. FAK already covers the central kernel mechanisms with stricter typed policy, isolated workers, managed context, and witnessed scheduling; full-computer and semantic-memory breadth remains modular. | [`agent-zero-study-2026-08-18.md`](../notes/agent-zero-study-2026-08-18.md), revision `baadd0dd0b09fa769a1027c183b964be85d5c8cc` (`v2.9`), forward edge `add781d3b3e5b3972fbd7cef54657b7bfb274ae9`; 18,903 stars and last push `2026-08-16T17:45:50Z` at check time. |
