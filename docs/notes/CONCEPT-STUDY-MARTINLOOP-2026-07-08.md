---
title: "Study-repo: MartinLoop → fak (2026-07-08)"
description: "A study-repo pass over MartinLoop finding fak/DOS already ships or exceeds most of its agent-trust core, and filing three issues for the genuine gaps."
---

# Study-repo: MartinLoop → fak (2026-07-08)

A `/study-repo` → `/field-borrow` pass over **MartinLoop**
(`github.com/Keesan12/martin-loop`), pinned at `@b06882f450610db8dc9788d81dbacb5ae1019e23`.
Apache-2.0 — permissive, compatible; every surviving borrow is **INSPIRE**
(clean-room TS→Go), none `integrate` (no foreign bytes vendored).

## What it is

MartinLoop is a direct conceptual sibling of fak/DOS: it governs AI-coding-agent
runs with budgets, stop conditions, rollback rules, verifier gates, a 13-class
failure taxonomy, HMAC-signed hash-chained receipts, an `EVIDENCE_BOUNDARY` proof
state, and an MCP surface. So the study's value is mostly *witnessing that fak
already ships (and usually exceeds) the trust core* — and isolating the narrow
residue fak genuinely lacks.

## What I read (not the pitch)

Cloned into scratch (`--filter=blob:none`), read the load-bearing `packages/core`
modules + their tests + the CLI proof surface — not the README:

- `packages/core/src/leash.ts` — typed safety-surface evaluator (command/fs/secret/
  network/dependency), obfuscation-resistant destructive-command detection,
  execution-profile network gating.
- `packages/core/src/test-integrity.ts` — circular/trivial/empty new-test detection.
- `packages/core/src/grounding.ts` — repo symbol index + `scanPatchForGroundingViolations`
  (hallucinated file/symbol/import refs in a diff; content-only-diff detection).
- `packages/core/src/red-blue/red-phase.ts` + `risk-tiers.ts` — adversarial patch
  trap taxonomy (assertion-deletion, evasion-pragma, self-context write, budget
  self-report, scope creep, silent revert), tiered by risk.
- `packages/core/src/context-integrity.ts` — prompt-injection pre-gate over
  user/tool/history/retrieved channels.
- `packages/core/src/persistence/integrity.ts` — hash-chained + HMAC-signed receipts.
- `packages/core/src/rollback.ts`, `packages/cli/src/proof-card.ts`,
  `packages/cli/src/reliability-score.ts`, `packages/contracts/src/{governance,execution-policy}.ts`.

## Candidate table — borrow · source `path:line@sha` · witness · route · filed

| Borrow | Source `@b06882f` | Witness vs fak | Route | Outcome |
|---|---|---|---|---|
| Hash-chained tamper-evident receipt + HMAC sig | `persistence/integrity.ts` | **PRESENT — fak ahead**: `internal/agent/receipt.go:3-19` (chain-bound, re-folded on verify; fak deliberately chose chain-commitment over PKI/HMAC) | inspire | dropped |
| Cost/token provenance `actual\|estimated\|unavailable` | `adapters/src/runtime-support.ts:21` | **PRESENT — fak ahead**: `cachewitness.Provenance` WITNESSED/OBSERVED, "observed-only, silent-when-untimed" | inspire | dropped |
| `EVIDENCE_BOUNDARY` verdict (neither pass nor fail) | `cli/src/proof-card.ts:293` | **PRESENT (dos)**: `dos_verify source:"none"` / `dos_commit_audit ABSTAIN` | inspire | dropped |
| Injection pre-gate on recalled-memory channel | `context-integrity.ts` | **PRESENT (live path)**: `memq.RenderNotesDigest → ctxmmu.ScreenBytes` (`internal/ctxmmu/mmu.go:510`); only dead-code `memoryread.RenderDigest` unscreened | inspire | dropped (adversarial verify caught a false-absent) |
| Rollback boundary + rollback-outcome artifact | `rollback.ts` | **PARTIAL**: fak uses git-worktree isolation; no distinct rollback-receipt artifact | inspire | deferred |
| Config provenance `{field, source:default\|config\|request}` | `contracts/src/governance.ts:19` | **PARTIAL**: fak has flag-over-config + IFC trust-provenance, no per-field config-origin ledger | inspire | deferred |
| Pre-work / coordination budget cap | `contracts/src/execution-policy.ts` `RoutingPolicy` | **ABSENT** (no seam firm enough to file this pass) | inspire | deferred |
| **Circular/trivial test-integrity** | `test-integrity.ts` | **PARTIAL**: `tools/code_slop_scorecard.py:986 kpi_vacuous_tests` exists but counts `t.Log`/`t.Skip` as assertions (`:1021`), has no circular/only-new-symbols rung, and is not wired into the keep-bit `internal/dispatchtick/witness.go:271 CommitWitnessed` | inspire | **#3364** |
| **Patch-honesty probes** (evasion-pragma + self-`.claude/`-write) | `red-blue/red-phase.ts:52-114` | **ABSENT**: `decide.go:459` lint rung checks parseability only; `SelfModifyGlobs` (`decide.go:1225-1234`) omit `.claude/`/`CLAUDE.md` | inspire | **#3362** |
| **Diff-grounding hallucination scan** | `grounding.ts:231` | **ABSENT**: `selfquery.go:180 Query` ranks, never checks reference existence; `internal/hooks/parse.go` parses diffs only for text gates | inspire | **#3363** |

## Filed (milestone 6 · epic #2871 · `class:dev`)

- **#3362** — `feat(adjudicator)`: two deterministic patch-honesty rungs (grounding-evasion-pragma scan + self-context-write guard).
- **#3363** — `feat(selfquery)`: pre-apply diff-grounding scan for hallucinated references.
- **#3364** — `feat(dispatchtick)`: fold a witnessed test-integrity rung into `CommitWitnessed` + close the `t.Log`/circular gaps in `kpi_vacuous_tests`.

Children linked from epic **#2871** (self-improving-agent playbook → WITNESSED,
deny-by-structure primitives).

## Method note (the adversarial verify earned its keep)

Each candidate was ground → drafted → **adversarially re-verified** in a workflow
before filing. The verify stage flipped two "ABSENT" candidates to `DO_NOT_FILE`:
test-integrity (a shipped `tools/code_slop_scorecard.py` detector the initial greps
missed by never scanning `tools/`) and recalled-memory-injection (the live recall
path already screens via `ctxmmu.ScreenBytes`; the cited unscreened function is dead
code). test-integrity was **rescoped** and filed against only its genuine residue;
recalled-memory was **dropped as PRESENT**. A witnessed "we already had this" is a
complete pass, not a failure.

## Honest fences

- Witnesses are lexical + a snapshot as of 2026-07-08; re-witness before acting.
- The clone was `--depth 1` (no history) — a "they removed this" read would need a
  deeper clone.
- License read (Apache-2.0) is a good-faith compatibility check, not legal advice;
  all borrows are clean-room INSPIRE, so no entanglement arises.

## Companions

- Skills: [`/study-repo`](../../.claude/skills/study-repo/SKILL.md),
  [`/field-borrow`](../../.claude/skills/field-borrow/SKILL.md).
- Sibling 2026-07-08 study passes: [Dynamo](CONCEPT-STUDY-DYNAMO-2026-07-08.md),
  [MinIO MemKV](CONCEPT-STUDY-MINIO-MEMKV-2026-07-08.md),
  [headroom](CONCEPT-STUDY-HEADROOM-2026-07-08.md).
- Parent epic: [#2871](https://github.com/anthony-chaudhary/fak/issues/2871).
