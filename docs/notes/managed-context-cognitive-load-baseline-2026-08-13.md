---
title: "Managed-context journey cognitive-load baseline"
description: "A reproducible deterministic-proxy baseline for restore, switch, conflict, team share, publish, provenance, and interrupted-apply recovery, with a candidate vocabulary, mental model, and usability budgets."
---

# Managed-context journey cognitive-load baseline

**Issue:** [#6597](https://github.com/anthony-chaudhary/fak/issues/6597)
**Study date:** 2026-08-13
**Repository pin:** `883662e2514288b0198052b8ec90ef7870e8dcee`
**Maturity:** research / modeled baseline, not observed usability evidence
**Machine witness:** [`managed-context-cognitive-load-baseline-2026-08-13.json`](../research/managed-context-cognitive-load-baseline-2026-08-13.json)

## Verdict

The current best complete procedure changes by journey. In this frozen proxy,
Codex's native surfaces are the quickest complete route for project switching,
chezmoi plus Git is the quickest complete route for five of seven novice journeys,
and a maintained manual manifest is the quickest complete novice route for
provenance inspection. There is no current single mental model or transaction
spanning all seven.

This is useful design evidence, not a product win: all durations and confidence
values are **MODELED** fixed weights. The deterministic witness proves that the
scripts, scoring, coverage, and conclusions are reproducible; it cannot prove how
fast or accurately a person will perform. Human evidence remains assigned to
[#6606](https://github.com/anthony-chaudhary/fak/issues/6606).

## Value frame and centrality

- **For:** a person moving agent context between machines/projects, a team curator,
  and a public publisher.
- **Problem:** users currently reconstruct identity, scope, precedence, privacy,
  provenance, and rollback from several tool-specific models.
- **Today:** a tuned native harness, a tuned dotfile/VCS workflow, and a careful
  manual procedure each solve different subsets; none owns the whole outcome.
- **Better because:** a fak spine can expose task verbs over one portable object
  and transaction model, while retaining the controls experts use today.
- **Witness:** the checked-in scorer independently folds every action in a frozen
  7 journeys × 2 personas × 3 alternatives corpus into machine-readable results.

This is **Core** work. It exercises all four problem checks: managed context (P1),
net-true comparison to tuned alternatives (P2), preview/recovery boundaries (P3),
and one operational model spanning personal, team, and public flows (P4).

## Reproduce the proxy

From the repository root:

```text
go run ./cmd/uxjourneyproxy \
  -corpus cmd/uxjourneyproxy/testdata/corpus.json \
  -check docs/research/managed-context-cognitive-load-baseline-2026-08-13.json

go test ./cmd/uxjourneyproxy -count=1
```

The first command must report `42` scripts and `18/18` vocabulary cases. The
scorer refuses a missing persona/alternative cell, an unknown action, a non-modeled
provenance label, an incomplete privacy contract, or a malformed budget. Its output
has no clock, host path, or map-order dependency. The result binds the exact input
bytes with SHA-256
`a29df08aa32b5e31429cef6b138e4c9ed38d13af4eeb4dc4cbf1256264a657c9`.

### Frozen environment and alternatives

The local control point reported Codex CLI `0.147.0`, Git
`2.51.0.windows.1`, and Go `1.26.5`. chezmoi was not installed, so its arm is
modeled from its current primary documentation rather than presented as a local
execution result.

| Arm | Tuned setup, not a strawman | Honest boundary |
|---|---|---|
| `codex_native` | Version-controlled `AGENTS.md`, project config and repo skills; separately backed-up non-auth user config/profiles; named profiles, directory-scoped resume, plugin packaging, and JSON output | Native scopes are real, but no documented portable identity/transaction spans config, skills, instructions, sessions, and plugins. |
| `chezmoi_git` | Private source repo; templates and machine data already separated; password-manager references or age encryption; diff before apply; Git review, signing, and secret scanning for shared flows | Strong file desired-state and history; typed agent semantics, activation authority, cross-object atomicity, and receipts remain conventions. |
| `manual_checklist` | A maintained inventory, encrypted staging archive, checksums, change log, rollback copy, second-person review, and non-executing inspection directory | Can preserve control, but the human maintains the inventory, ordering, trust, and recovery model. |

The Codex pin is grounded in official documentation: user/project config and
profile layering are separate ([configuration reference](https://learn.chatgpt.com/docs/config-file/config-reference));
skills load from repository, user, admin, and system locations
([skills](https://learn.chatgpt.com/docs/build-skills)); `AGENTS.md` merges by
global/project directory scope
([AGENTS.md](https://learn.chatgpt.com/docs/agent-configuration/agents-md)); and
resume is session- and directory-aware
([developer commands](https://learn.chatgpt.com/docs/developer-commands?surface=cli)).
The absence of one cross-surface portability transaction is an inference from
those separately documented surfaces, not an OpenAI claim.

The dotfile/VCS arm uses chezmoi's documented common-source/local-config split and
new-machine `init` → `diff` → `apply` flow
([setup](https://www.chezmoi.io/user-guide/setup/)), dry-run/apply controls
([apply](https://www.chezmoi.io/reference/commands/apply/)), and explicit
destination/source/target reconciliation
([merge](https://www.chezmoi.io/user-guide/tools/merge/)). Git supplies the
versioned conflict/abort path
([git-merge 2.55](https://git-scm.com/docs/git-merge)); GitHub documents repository
visibility and the need for secret/security controls before public exposure
([repository security](https://docs.github.com/en/enterprise-cloud@latest/repositories/creating-and-managing-repositories/about-repositories),
[secret scanning](https://docs.github.com/en/code-security/concepts/secret-security/secret-scanning)).
All web sources were checked on 2026-08-13.

## Protocol and fixtures

The corpus uses only synthetic managed objects: a harmless skill, a deny-only
policy, a named model/profile preference, an instruction file, a fake session
receipt, and two concurrent edits with reserved `example.invalid` values. It never
reads a participant home, secret, transcript, command history, private repository,
or corporate policy. The proxy executes action records, not real file mutations.

For each journey:

1. Start from the same declared setup and desired outcome.
2. The **novice script** may use the documented common path but assumes no recall of
   hidden roots, precedence, three-way states, or signing vocabulary.
3. The **expert script** may add selectors, raw manifests, merge/signature controls,
   policy decisions, and machine output; its fixed action times are lower because
   tool operation is assumed familiar.
4. Inject the corpus's deterministic failure, if any: an unavailable old local
   session, a concurrent-edit conflict, or interruption after first mutation.
5. Fold action effects against an outcome oracle. Missing any required effect makes
   the script incorrect even if it is fast.
6. Emit every row and aggregate as JSON. No incomplete script may win the
   `best_current` selection.

The complete novice and expert scripts live in
[`corpus.json`](../../cmd/uxjourneyproxy/testdata/corpus.json). Each action names its
operator-visible label, persona-specific fixed time, decisions, concepts, commands,
controls, effects, errors, and whether it belongs to recovery.

### Scoring rubric

| Measure | Deterministic rule |
|---|---|
| Task time | Sum of persona-specific action weights. **MODELED seconds**, never observed time. |
| Required decisions | Sum of choices that can change scope, content, trust, conflict resolution, activation, or recovery. Routine command syntax is not a decision. |
| Concepts recalled | Count of unique tool/product concepts named by actions; duplicates fold once. |
| Step count | Number of goal-directed macro-actions. Compound commands inside a macro-action are counted separately as `command_count`. |
| Errors | Count of injected or documented failed attempts. A surfaced conflict is one error event, not silently treated as success. |
| Recovery time | Sum of action weights marked `recovery_phase`, from failure recognition to a verified safe/continued state. |
| Correctness | Set inclusion: every journey-required effect must be produced. Missing effects are emitted. |
| Confidence calibration | Absolute gap `abs(confidence - correctness)` where correctness is 0 or 1. Confidence is a frozen proxy value, not participant self-report. |
| Expert-control coverage | Required control axes exercised by the script divided by all axes required for that journey. |

## Baseline results

### Cross-journey aggregate

| Persona | Alternative | Complete | Mean modeled time | Mean decisions | Mean concepts | Mean steps | Mean calibration gap | Mean expert-control coverage |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Novice | Codex native | 5/7 | 469.29 s | 6.86 | 7.86 | 3.57 | 0.350 | 0.789 |
| Novice | chezmoi + Git | 7/7 | 427.14 s | 6.14 | 7.14 | 3.29 | 0.206 | 0.906 |
| Novice | Manual checklist | 7/7 | 622.14 s | 9.29 | 8.29 | 3.29 | 0.319 | 0.853 |
| Expert | Codex native | 6/7 | 179.29 s | 7.43 | 9.86 | 4.14 | 0.157 | 0.894 |
| Expert | chezmoi + Git | 7/7 | 160.71 s | 6.71 | 8.71 | 3.86 | 0.070 | 0.954 |
| Expert | Manual checklist | 7/7 | 250.71 s | 9.86 | 10.00 | 3.86 | 0.131 | 0.934 |

The expert rows sometimes contain more steps and recalled concepts than novice rows
because they deliberately exercise extra provenance, receipt, and rollback controls.
That is retained control, not automatically a UX regression.

### Fastest complete current script per journey

| Journey | Novice arm | Time / decisions / steps | Expert arm | Time / decisions / steps |
|---|---|---:|---|---:|
| New-machine restore | chezmoi + Git | 425 s / 6 / 4 | chezmoi + Git | 155 s / 6 / 4 |
| Switch project context | Codex native | 190 s / 3 / 3 | Codex native | 85 s / 4 / 4 |
| Reconcile sync conflict | chezmoi + Git | 520 s / 7 / 4 | chezmoi + Git | 170 s / 7 / 4 |
| Share governed team collection | chezmoi + Git | 520 s / 8 / 3 | chezmoi + Git | 215 s / 9 / 4 |
| Publish safe public subset | chezmoi + Git | 510 s / 10 / 2 | chezmoi + Git | 200 s / 11 / 3 |
| Inspect provenance | Manual checklist | 260 s / 4 / 1 | Codex native | 115 s / 5 / 3 |
| Recover interrupted apply | chezmoi + Git | 440 s / 4 / 5 | chezmoi + Git | 150 s / 4 / 5 |

The biggest proxy attention sinks are not raw command count. They are the decisions
around public egress (10 novice decisions), team governance (8), semantic conflict
(7), and restore provenance/privacy (6). Recovery is the sharpest power gap:
Codex can resume the conversation but does not prove the prior file state or a
cross-object apply receipt in this fixture; the complete chezmoi + Git recovery
takes 260 modeled novice recovery seconds, and the manual path takes 500.

The wrong-mental-model warning is visible rather than averaged away. The incomplete
novice native restore is assigned 0.65 proxy confidence while missing preserved
provenance. Native interrupted-apply recovery is assigned 0.62 novice / 0.55 expert
confidence while missing prior-state and/or receipt effects. A native “resume” is
not the same claim as a recovered apply transaction.

## Candidate vocabulary and one-page mental model

Use task verbs in the front door—**backup, restore, switch, share, publish, inspect,
recover**—and reveal these six nouns progressively:

```text
Object       one typed managed thing (skill, policy, profile, workflow, ...)
    │
    ├── grouped intentionally, with dependencies ───────────────► Collection
    │                                                               │
    │                                      selected + precedence ───► Context
    │                                                               │ active now
    └── sealed, redacted, content-addressed ─────────────────────► Package
                                                                    │
                                      moved through a trust route ──► Channel

Preview ──► apply / switch / merge ──► receipt ──► undo or recover
                         one bounded Transaction
```

Plain-language version: **objects live in collections; a context says which
collections are active; a package is the safe portable form; a channel moves it;
a transaction previews and changes active state with a receipt and recovery path.**
Inspecting a collection or package never activates it. Moving a package through a
different channel never changes its content identity. Resuming a conversation does
not by itself recover an interrupted transaction.

The vocabulary proxy contains 18 positive and contrast prompts—three per noun—and
scores 18/18. It distinguishes Package from Channel, Collection from Context, and
Package from Transaction. That proves the proposed definitions are mechanically
separable in this corpus only. The human gate below must still test recall,
paraphrase, and safe action choice.

## Budgets for the implementation spine

These are ceilings, not claimed results. Recovery budgets are measured at p90 from
failure recognition to independently verified safe state.

| Journey | Novice decisions ≤ | Novice macro-steps ≤ | Recovery p90 ≤ |
|---|---:|---:|---:|
| New-machine restore | 3 | 5 | 120 s |
| Switch project context | 1 | 3 | 60 s |
| Reconcile sync conflict | 4 | 6 | 180 s |
| Share governed team collection | 4 | 6 | 120 s |
| Publish safe public subset | 5 | 7 | 120 s |
| Inspect provenance | 1 | 2 | 30 s |
| Recover interrupted apply | 2 | 4 | 90 s |

Every common path must also have zero secret/private-data egress, no ambiguous active
state, and one obvious next action on refusal. Expert mode must retain 100% of the
journey's declared control axes through selectors/policy/JSON even if the novice path
hides them.

## Later usability gate and evidence-collection plan

[#6606](https://github.com/anthony-chaudhary/fak/issues/6606) should replace modeled
times/confidence with an observed, privacy-safe lab:

1. Recruit at least 12 novice and 12 expert participants. Run a three-arm
   counterbalanced crossover in three sessions so every participant completes all
   seven tasks in every arm without one-session fatigue determining the result.
2. Give every arm the tuned setup above and the same two isolated temporary homes,
   synthetic organization, public consumer, failure schedule, and outcome oracle.
   Do not use a participant's real home, history, credentials, repositories, or
   employer data.
3. Log only pseudonymous participant ID, persona, randomized arm order, monotonic
   timestamps, task/step IDs, decisions, errors/retries, controls exercised,
   confidence before correctness reveal, oracle result, recovery-safe timestamp,
   and leak-scan result. Delete raw screen/audio capture after adjudication; commit
   only redacted aggregates and failing synthetic fixtures.
4. Blind correctness grading to arm. A second adjudicator resolves disagreements.
   Report median and p90 with bootstrap intervals; retain individual rows outside
   Git only under the consent/retention plan.
5. Gate the later UX on every journey, versus its fastest **correct** tuned arm:
   median decisions at least 30% lower, median time at least 20% lower, all table
   ceilings met, correctness no worse, confidence-calibration gap ≤0.15, p90 recovery
   within budget, zero unsafe egress/ambiguous active state, and 100% expert-control
   coverage.
6. After the tasks, test the one-page model without the page: at least 80% first-pass
   noun selection and 90% correct safety choices on inspect-vs-activate,
   Package-vs-Channel, and resume-vs-recover contrasts. Report each contrast; do not
   hide a critical conflation inside an average.

## Findings, fak seams, and deduplication

No new follow-up was filed. Every supported finding is already owned by an open
child or adjacent epic; filing another would duplicate its done condition.

| Finding | Existing repository seam | Existing owner; dedupe result |
|---|---|---|
| F1 — identity, scope, and provenance are reconstructed from roots/tool concepts during restore | `internal/registrations`, `cmd/fak/session_inventory.go`, `internal/disambiguation` | [#6595](https://github.com/anthony-chaudhary/fak/issues/6595) inventories types; [#6598](https://github.com/anthony-chaudhary/fak/issues/6598) owns the portable contract; [#6272](https://github.com/anthony-chaudhary/fak/issues/6272) owns canonical-term contrasts. |
| F2 — current restore/switch paths require separate inventory, privacy, apply, behavior, and provenance reasoning | `cmd/fak/session_checkpoint.go`, `internal/contextq`, `internal/contextq/diff.go` | [#6599](https://github.com/anthony-chaudhary/fak/issues/6599) owns the personal-continuity spine. |
| F3 — privacy classification is a repeated high-stakes decision in backup/share/publish | `internal/normgate`, `internal/secretgate`, `internal/policy` | [#6600](https://github.com/anthony-chaudhary/fak/issues/6600) owns typed sensitivity/redaction and egress denial. |
| F4 — file merge exposes base/ours/theirs or destination/source/target, but not typed object semantics or one precedence explanation | `internal/contextq/diff.go`, policy explain surfaces | [#6601](https://github.com/anthony-chaudhary/fak/issues/6601) owns sync reconciliation; [#6605](https://github.com/anthony-chaudhary/fak/issues/6605) owns the common conflict/explain/recovery UX. |
| F5 — team share spends attention on owner, approval, policy, rollout, audit, and activation authority | `internal/policy`, `internal/agent/receipt.go`, `internal/witness` | [#6602](https://github.com/anthony-chaudhary/fak/issues/6602) already owns exactly these organization controls. |
| F6 — publication requires an egress/license/compatibility/signature/identity chain, not just upload | secret gates, `internal/witness`, Git provenance seams | [#6603](https://github.com/anthony-chaudhary/fak/issues/6603) owns safe inspect/publish/install/update/revoke. |
| F7 — conversation resume can be mistaken for transaction recovery; prior active state and receipt need independent proof | `internal/agent/receipt.go`, `cmd/fak/session_receipt.go`, checkpoint/recovery seams | [#6606](https://github.com/anthony-chaudhary/fak/issues/6606) owns the lab; [#6607](https://github.com/anthony-chaudhary/fak/issues/6607) and [#6432](https://github.com/anthony-chaudhary/fak/issues/6432) own coordinated lifecycle recovery. |

## Promotion boundary

Promote the budgets and vocabulary into maintained product contracts only after the
working spine in #6599 emits real transaction receipts and #6606 records observed
human results. Until then, this note and its JSON are a reproducible baseline and
design input, not evidence that fak is easier to use.
