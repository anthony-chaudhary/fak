---
title: "fak repo-hygiene scorecard — the hygiene-debt measuring stick"
description: "fak's deterministic repo-hygiene scorecard: eleven KPIs across verbosity, organization, indexing, and accessibility, folded into a composite score and the headline hygiene-debt metric, re-derived from the git-tracked tree."
---

# Repo-hygiene scorecard

This is the measuring stick for the repo-3x program — the structural counterpart of the docs and code scorecards. Every number below is re-derived from the git-tracked tree by `tools/repo_hygiene_scorecard.py` — no hand-entry. The headline metric is **hygiene-debt**: the count of concrete, mechanical structural defects you fix by *deleting, consolidating, moving, or indexing* — a duplicate doc, an oversized doc, root clutter, a misplaced dated note, an orphaned doc no index links, an AI-tell phrase. Driving hygiene-debt toward zero is what keeps the repo lean and findable as it grows.

> Regenerate: `python tools/repo_hygiene_scorecard.py --markdown --stamp DATE > docs/REPO-HYGIENE-SCORECARD.md`

## Headline

| Metric | Value |
|---|---|
| **Hygiene-debt (total HARD defects)** | **111** |
| **a11y-debt (accessibility HARD defects)** | **1** |
| Composite score | 67.4/100 (grade D) |
| Advisory (soft) signals | 577 |
| Debt by group | verbosity:25 · organization:33 · indexing:52 · accessibility:1 |

## Per-KPI

Twelve KPIs, each 0–100, in four groups. `debt` = units of HARD hygiene-debt. The accessibility group's HARD KPIs (`alt_text`, `ai_tells`) sum to **a11y-debt**. `jargon` and `plain_language` are advisory (they score but emit no hard debt — gaming a gloss is not clarity).

| Group | KPI | Score | Debt | Detail |
|---|---|---:|:--:|---|
| indexing | `orphans` | 93 | 52 | 727/779 reader-facing docs reachable from an index (93.3%) |
| organization | `placement` | 0 | 30 | 30 misplaced dated doc(s) |
| verbosity | `redundancy` | 0 | 16 | 16 near-duplicate pair(s), 19 candidate(s) |
| verbosity | `bloat` | 0 | 9 | 9 oversized, 14 long |
| organization | `dir_discipline` | 64 | 3 | 3 near-duplicate dir group(s) |
| accessibility | `ai_tells` | 82 | 1 | 1 AI-tell phrase(s) across 483 doc(s) |
| accessibility | `plain_language` | 67 | 0 | 187 dense doc(s), 429 doc(s) with undefined acronyms, 23 literal-reader idiom(s) |
| accessibility | `jargon` | 86 | 0 | 284 naked first-screen jargon term(s) (0.3/doc) |
| organization | `root_hygiene` | 100 | 0 | root holds only front-door / meta files |
| indexing | `index_presence` | 100 | 0 | all expected index surfaces present |
| indexing | `index_integrity` | 100 | 0 | every index entry resolves |
| accessibility | `alt_text` | 100 | 0 | every doc image carries alt-text |

## Hygiene-debt work-list

### `orphans` (indexing) — 52 defect(s), score 93
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/abi-refusal-reason-vocabulary-abi-reason.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/account-seat-dispatch-account-seat.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/activation-posture-disambiguation-activation.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/adjudication-verdict-policy-decision.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/agent-kernel-product-fak.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/agent-session-runtime-internal-session.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/capability-floor-policy-authority.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/capability-maturity-rung-capability-maturity.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/compute-fleet-dispatch-fleet.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/compute-kernel-computing-processor.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/context-compaction-runtime-codex-context.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/disambiguation-package-package-internal-disambiguation.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/dispatch-lane-dispatch-lane.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/dispatch-loop-dispatch-loop.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/dispatch-ownership-lane-ownership-lane.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/dispatch-wave-dispatch-wave.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/dispatch-worker-dispatch-worker.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/dos-decision-kind-vocabulary-dos-decision-kind.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/exported-go-symbol-candidate-disambiguation-go-symbol-candidate.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/fak-cli-kernel-cli-fak.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/fak-commit-stamp-ownership-commit-stamp.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/fak-measurement-arm-claims-fak-arm.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/fleet-supervisor-dispatch-supervisor.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/hook-gate-class-vocabulary-hook-gate-class.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/index-lifecycle-class-disambiguation-entry-authority.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/kernel-cli-fak.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/kernel-package-internal-disambiguation.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/lane-lease-dispatch-lease.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/leaf-identity-ownership-leaf.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/model-kv-cache-cache-model-attention.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/model-mediated-check-policy-model-mediated.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/module-revision-identity-ownership-module-revision.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/naive-baseline-claims-naive-baseline.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/net-true-claim-claims-net-true.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/package-capability-token-disambiguation-capability-token.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/policy-declaration-policy-declaration.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/policy-posture-verdict-vocabulary-policy-verdict.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/provider-prompt-cache-cache-provider-prompt-prefix.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/radix-prefix-cache-cache-radix-prefix-snapshots.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/recovery-checkpoint-runtime-internal-session.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/runtime-runtime-agent-application.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/runtime-runtime-gateway-serving.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/runtime-runtime-guard-enforcement.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/runtime-runtime-model-serving.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/runtime-runtime-worker-execution.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/session-recovery-runtime-internal-session.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/session-resume-runtime-internal-session.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/simulated-evidence-claims-simulated.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/structural-preflight-policy-structural-preflight.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/tool-result-cache-cache-tool-results.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/tuned-baseline-claims-tuned-baseline.md — index it or delete it
- orphan (reachable from no index/hub): docs/generated/disambiguation/identities/witness-provenance-claims-provenance.md — index it or delete it

### `placement` (organization) — 30 defect(s), score 0
- dated/research doc outside docs/notes/: docs/_witnesses/agent-default-code-tools-non-fak-2026-08-19.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/defaults-selfcheck-non-fak-2026-08-19.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/dispatch-thread-pressure-2026-08-14.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/end-to-end-value-chain-selfcheck-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/fleet-res-rollup-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/guard-codex-default-profiles-non-fak-2026-08-18.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/DOGFOOD-LIVE-WORK-2026-08-22.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/EDGE-ADVERSARIAL-2026-08-20.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/LIVE-CODING-GPT-5.6-SOL-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/LIVE-FAK-L4-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/LIVE-NATIVE-PROBE-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/OPERATOR-HOME-2026-08-20.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/harness-web-demo/SHIPPED-FAK-LAUNCH-2026-08-15.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/issue-8494/trajectory-audit-2026-08-21.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/launch-posture-cross-wire-non-fak-2026-08-19.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/launch-posture-missing-vcache-calibration-2026-08-18.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/manage-parity-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/manage-parity-hooks-2026-08-14.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/managed-tool-search-compat-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/micro-paired-value-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/microagent-real-kernel-2026-08-12.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/openai-decoded-context-view-default-non-fak-2026-08-18.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/openai-stale-read-elision-non-fak-2026-08-18.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/vcache-calibrated-steering-non-fak-2026-08-18.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/vcache-calibrated-ttl-tier-non-fak-2026-08-18.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/vcache-calibrated-write-pricing-2026-08-19.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/vcache-forced-mismatch-demotion-2026-08-19.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/vcache-provider-calibration-live-2026-08-18.md → move it and index it
- dated/research doc outside docs/notes/: docs/_witnesses/wip-ownership-seam-2026-08-13.md → move it and index it
- dated/research doc outside docs/notes/: docs/research/microagents-to-harnesses-2026-08-18.md → move it and index it

### `redundancy` (verbosity) — 16 defect(s), score 0
- near-duplicate (96%): docs/concepts/disambiguation-attention.md ≈ docs/concepts/disambiguation-decision.md — consolidate to one
- near-duplicate (96%): docs/concepts/disambiguation-attention.md ≈ docs/concepts/disambiguation-evict.md — consolidate to one
- near-duplicate (81%): docs/concepts/disambiguation-attention.md ≈ docs/concepts/disambiguation-layout.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-context-ctx.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-gateway-engine.md — consolidate to one
- near-duplicate (96%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-loop.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-cache.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one
- near-duplicate (92%): docs/concepts/disambiguation-context-ctx.md ≈ docs/concepts/disambiguation-gateway-engine.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-context-ctx.md ≈ docs/concepts/disambiguation-loop.md — consolidate to one
- near-duplicate (92%): docs/concepts/disambiguation-context-ctx.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one
- near-duplicate (96%): docs/concepts/disambiguation-decision.md ≈ docs/concepts/disambiguation-evict.md — consolidate to one
- near-duplicate (81%): docs/concepts/disambiguation-evict.md ≈ docs/concepts/disambiguation-layout.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-gateway-engine.md ≈ docs/concepts/disambiguation-loop.md — consolidate to one
- near-duplicate (92%): docs/concepts/disambiguation-gateway-engine.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one
- near-duplicate (93%): docs/concepts/disambiguation-layout.md ≈ docs/concepts/disambiguation-score-debt.md — consolidate to one
- near-duplicate (94%): docs/concepts/disambiguation-loop.md ≈ docs/concepts/disambiguation-policy-capability.md — consolidate to one

### `bloat` (verbosity) — 9 defect(s), score 0
- oversized doc LEARNING-PATH.md (2763 lines > 1000) — split into sections or trim
- oversized doc docs/FAQ.md (2947 lines > 1000) — split into sections or trim
- oversized doc docs/_witnesses/issue-8494/trajectory-audit-2026-08-21.md (1031 lines > 1000) — split into sections or trim
- oversized doc docs/cli-reference.md (1423 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/INDEX.md (3654 lines > 1000) — split into sections or trim
- oversized doc docs/concept-disambiguation-scorecard/README.md (2793 lines > 1000) — split into sections or trim
- oversized doc docs/fak/concept-glossary.md (2389 lines > 1000) — split into sections or trim
- oversized doc docs/generated/disambiguation/canonical-terms.md (1019 lines > 1000) — split into sections or trim
- oversized doc docs/generated/verb-surface.md (1102 lines > 1000) — split into sections or trim

### `dir_discipline` (organization) — 3 defect(s), score 64
- near-duplicate sibling dirs: ['docs/benchmark', 'docs/benchmarking', 'docs/benchmarks'] — merge into one
- near-duplicate sibling dirs: ['internal/ctxplan', 'internal/ctxplans'] — merge into one
- near-duplicate sibling dirs: ['internal/market', 'internal/marketing'] — merge into one

### `ai_tells` (accessibility) — 1 defect(s), score 82
- AI-tell phrase in docs/concept-disambiguation-scorecard/README.md: “bespoke” — say it plainly

