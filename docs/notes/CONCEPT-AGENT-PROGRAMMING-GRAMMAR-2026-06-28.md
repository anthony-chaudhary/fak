---
title: "The agent programming grammar — lifting fak's invariants into a domain-free substrate other agents build on"
description: "fak's contribution is not any one primitive (0/29 novel) but one invariant carried at every scale. This note extracts that invariant as a reusable grammar — nouns, verbs, and a closed refusal vocabulary — names what is already lifted into the domain-free DOS form, and scopes the under-expressed concepts that should become the next grammar verbs."
---

# The agent programming grammar

> Design note. Snapshot for `/goal` 2026-06-28. The shipped surface is cited by
> package / doc / `dos_*` verb; every *proposed* verb is labelled `not yet` and
> mapped to its fak home. This note is the "express + generalize" half of the
> survey; the durable catalog is [`docs/INNOVATIONS-INDEX.md`](../INNOVATIONS-INDEX.md),
> and the actioning epic is the grammar epic it links.

## The thesis

fak leads with an unusual honesty: a 29-claim prior-art audit scored **0/29 novel**
([`CLAIMS.md`](../../CLAIMS.md)). Every primitive — reference monitor, capability
floor, content-addressed store, taint label, witness — is established. The
contribution is the **assembly**: one in-process kernel where the tool call is a
syscall, fused so the same boundary is safe *and* fast, carrying **one invariant at
every scale** ([`engineering-is-building-loops.md`](../explainers/engineering-is-building-loops.md)).

That invariant has a one-sentence form:

> **A decision no participant can move by narrating a number.** Evidence the
> claimant did not author is the only admissible truth; a claim is untrusted until
> witnessed; a refusal carries a token from a closed, checkable set.

The next contribution is to **lift that invariant out of fak's packages into a
domain-free grammar** — a small set of nouns, verbs, and a closed vocabulary — that
any agent fleet can adopt by configuration, without forking the kernel. That grammar
already has a seed: **DOS**, the trust substrate fak dogfoods on its own repo
([`dos.toml`](../../dos.toml), the `dos_*` MCP verbs, [`docs/dos-kernel-transfer-playbook.md`](../dos-kernel-transfer-playbook.md)).
This note maps what is already lifted, the shape that makes a lift correct, and the
under-expressed concepts that should become the next verbs.

## The grammar as it stands

The de-facto grammar an agent already has, today, across `dos.toml` + the `dos_*`
verbs + the frozen ABI:

**Nouns.** `lane` (a named file-tree scope) · `lease` (a live lock on a lane,
exclusive or shared) · `reason token` (a closed-vocabulary refusal, `[reasons.*]`) ·
`witness` (the forgeability rung of a claim: `diff-witnessed` vs `subject-only`) ·
`verdict` (`Allow/Deny/Quarantine`; `RECALL_FRESH/STALE`; `OK/CLAIM_UNWITNESSED/ABSTAIN`) ·
`claim` (a worker self-report, untrusted until witnessed) · `ladder rung` (a closed
maturity level promoted only by third-party evidence) · `scope` (`ScopeAgent/ScopeTenant/ScopeFleet`,
the share boundary on any `abi.Ref`).

**Verbs.** `arbitrate` (may this worker take this lane given the live leases? —
`dos_arbitrate`) · `verify` (did a plan/phase ship, from git, not the worker? —
`dos_verify`) · `audit` (did a commit's diff do what its subject claims? —
`dos_commit_audit`) · `review` (fold a commit range into residual vs cleared
attention bands — `dos_review`) · `refuse` / `check_reason` (emit / validate a token
from the closed set — `dos_refuse_reasons`, `dos_check_reason`) · `recall` (re-check a
saved memory's named artifacts against the live tree — `dos_recall`) · `resolve`
(does a citation exist in a third-party reporter? — `dos_citation_resolve`) ·
`status` (fold a run's liveness + verified progress + region into one digest with
**no `claimed` field** by construction — `dos_status`) · `doctor` / `answer`
(introspect the workspace; score a question against the corpus).

The thread through every verb: a claim graded `subject-only` / `CLAIM_UNWITNESSED`
is surfaced for human residual review, never silently passed.

## Two structural shapes the grammar already proves

Newcomers should see *why* this is a grammar and not a pile of checks. Two shapes
recur and are worth naming as first-class patterns:

1. **The verification ladder** — graduated, cost-ordered rungs (vDSO re-output →
   in-process structural adjudication → posture/complain → require-witness → CI →
   git-evidence → isolated-worktree keep-bit → human ESCALATE), where the discipline
   is *start at the smallest rung that can conclusively decide the property, climb
   only on `INDETERMINATE` or warranted risk*
   ([`verification-ladder-doctrine.md`](verification-ladder-doctrine.md)). It is the
   agent-kernel restatement of seccomp's restrictiveness fold, LSM stacking +
   capabilities, AppArmor complain→enforce, the IMA integrity-granularity ladder, and
   the eBPF prove-before-admit verifier.

2. **The two lenses (Rosetta)** — the same primitive reads as a *security control* to
   one audience and a *systems optimization* to the other, because it is the same code
   path ([`EXPLAINER-trust-floor-two-lenses-2026-06-17.md`](EXPLAINER-trust-floor-two-lenses-2026-06-17.md)).
   A reference monitor *is* the syscall boundary; taint analysis *is* the page-fault
   handler; a durable-taint re-check *is* demand paging from a core dump. This is the
   structural reason "win-win-win" (charter #9) is achievable at all: a grammar verb
   earns its keep when it is safe and fast *for the same reason*, not as a tax.

A correct grammar verb obeys both shapes: it sits at a definite rung, and it pays for
itself on the optimization lens, not only the safety lens.

## Already lifted into the domain-free form

These fak concepts have already crossed from package-specific code into a domain-free
DOS verb, contract, or vocabulary — proof the lift is real, not aspirational:

| fak concept | domain-free form | where |
|---|---|---|
| structured refusal | closed `[reasons.*]` token set + `UNCLASSIFIED` fail-closed | `dos_refuse_reasons` / `dos_check_reason`, `dos.toml` |
| ship verification | "did it land, from git, not self-report" | `dos_verify` (binds the `(fak <leaf>)` stamp grammar) |
| commit-claim witness | `diff-witnessed` vs `subject-only` (forgeability rung) | `dos_commit_audit`, `dos_review` |
| disjoint-lease admission | lane taxonomy + lock-mode tree rule | `dos_arbitrate`, `dos.toml [lanes]` |
| recall freshness | re-verify a memory's named artifacts at read time | `dos_recall` |
| run-status digest | liveness + verified progress + region, no `claimed` field | `dos_status` |
| claim-salience partition | `[SHIPPED]`→LIVE vs `[SIMULATED]/[STUB]`→PARKED, no-loss | `dos.salience.partition` ([claims-salience-register](../claims-salience-register.md)) |
| net-true-value | 6-question gain rubric (real baseline / net / scope / provenance / witness / default) | [`docs/standards/net-true-value.md`](../standards/net-true-value.md) |
| shared-state ladder | 5-rung vocabulary for shared/durable/disaggregated state | [`docs/shared-state-ladder.md`](../shared-state-ladder.md) |
| coordination invariant | every coordination act is an adjudicated synthetic tool call | [`multi-agent-coordination-protocol.md`](../multi-agent-coordination-protocol.md) |

## The under-expressed concepts — the next grammar verbs

These are mechanisms fak has *built and proven* but left locked inside its own
packages or prose. Each has a clean domain-free shape and no DOS verb yet. They are
the actioning backlog for the grammar epic; ordered by leverage.

| # | concept (fak home) | the general primitive | proposed grammar shape | status |
|---|---|---|---|---|
| G1 | **readiness / surface-ceiling ladder** (`tools/product_scorecard.py`) | a closed maturity ladder where each rung is gated by evidence the promoter didn't author + a surface cap that stops a benchmark posing as a product | `dos readiness` verb + `READINESS_OVERCLAIM` reason | concept captured ([CONCEPT-DOS-READINESS-VERDICT-LADDER](CONCEPT-DOS-READINESS-VERDICT-LADDER-2026-06-26.md), [#582](https://github.com/anthony-chaudhary/fak/issues/582)); verb `not yet` |
| G2 | **verification ladder** (`internal/adjudicator`, `internal/shipgate`) | smallest-sufficient-rung adjudication with a first-class `INDETERMINATE` that forces escalation | a declarable rung spec + `dos verify --ladder` / an `INDETERMINATE` verdict in the vocabulary | doctrine only ([verification-ladder-doctrine](verification-ladder-doctrine.md)); `not yet` |
| G3 | **durability-class promotion gate** ("context is not memory", `internal/ctxmmu`, `internal/recall`) | a memory write must pass a truth-duration class gate before it advances to a longer-lived tier | `dos promote` verb + a promotion predicate over a `durability` field | fak-specific; `not yet` |
| G4 | **materialized-view-over-lossless-history** (`internal/ctxplan`, `internal/vdso`, vToolcall design) | what a reader sees is a scope-redacted view of an append-only log; a miss is a demand-page fault, never a lost fact | `dos context-contract` — declare the view + its closed invalidation contract | fak-specific; `not yet` |
| G5 | **prediction-vs-reality calibration** (`internal/dojo`, `internal/resume` Backtest) | back-test every projection against real telemetry before defaulting it on; name the conservative bias | `dos calibrate` taking `(prediction, measurement, eval-fn)` → calibration verdict | partially ([#1021](https://github.com/anthony-chaudhary/fak/issues/1021) for the dojo's own loop); generic verb `not yet` |
| G6 | **one-sided screen + witnessed-loss polarity** (`internal/wirescreen`, `internal/ctxmmu`) | an additive screen may only *tighten* a floor (Allow→Quarantine); a wrong proposal costs one fault, never a lost fact | a correctness predicate any additive safety component declares + checks | fak naming only; `not yet` |
| G7 | **taint / IFC sink-gating** (`internal/ifc`, `abi.Ref.Taint`) | does this value's taint forbid it crossing this boundary into a sink? | `dos taint-check` — a standalone admission check other runtimes call | fak ABI only; `not yet` |
| G8 | **per-aspect routing + ensembles** (`internal/modelroute`) | the routed unit is an *aspect* of a request, not the whole request; an ensemble + reduction is a first-class plan | a portable routing schema + `dos route` over an aspect→worker policy | fak manifest only; live dispatch is `[STUB]`; `not yet` |
| G9 | **net-true-value claim-check** ([net-true-value](../standards/net-true-value.md)) | run any incoming efficiency claim through the 6-question rubric mechanically | `fak claim-check` / `dos claim-check` verb | standard written; verb named, `not yet` |

## What makes a lift correct (the recipe, not a wishlist)

A new grammar verb is only worth adding if it keeps the invariant. The five rules,
read off the verbs that already shipped:

1. **Closed vocabulary.** A verdict / reason comes from a finite, checkable set; an
   out-of-set token is `UNCLASSIFIED` and refused conservatively — never silently
   more-permissive (`dos_check_reason`).
2. **Evidence-bound, no `claimed` field.** The verb folds evidence the claimant did
   not author; a self-report is an input to witness, never an output (the `dos_status`
   digest has no `claimed` field *by construction*).
3. **Fail-closed.** Absence of an affirmative allow is a deny; an `INDETERMINATE`
   escalates, it does not pass.
4. **Data, not code.** A binding (a lane tree, a stamp grammar, a reason's
   summary/fix) is `dos.toml` data that introduces *no spontaneous refusal* — it fires
   only at an opt-in surface or its named floor. The mechanism stays in the installed
   `dos` package; only policy crosses into the tree.
5. **Both lenses.** It must pay on the optimization lens (a saved call, a smaller
   resident set, a skipped rung), not only the safety lens — or it is a tax, not a
   primitive.

## Honest fences

- This is a **design note**, not a shipped feature. Every G-row verb is `not yet`;
  the value here is the extraction and the recipe, not code.
- G1 (readiness) already has an issue ([#582](https://github.com/anthony-chaudhary/fak/issues/582))
  and an explicit decision to lift it; this note does not re-file it, it sequences it.
- The grammar's home is **DOS**, a substrate that ships in the installed `dos`
  package; some verbs may instead land as `fak` subcommands when they are fak-shaped
  (e.g. G9 `claim-check`). The boundary is: domain-free trust logic → DOS; fak-tree
  policy/measurement → `fak`. Per [AGENTS.md](../../AGENTS.md), a new verb is Go in a
  leaf, never a new `tools/*.py`.
- The grammar does **not** replace the token engine, the model, or the harness. It is
  the governance band — the same scope fak already owns.

## Read next

- [`docs/INNOVATIONS-INDEX.md`](../INNOVATIONS-INDEX.md) — the durable catalog this note generalizes from.
- [`engineering-is-building-loops.md`](../explainers/engineering-is-building-loops.md) — the loop×invariant grid this grammar is the cross-cut of.
- [`verification-ladder-doctrine.md`](verification-ladder-doctrine.md) · [`EXPLAINER-trust-floor-two-lenses-2026-06-17.md`](EXPLAINER-trust-floor-two-lenses-2026-06-17.md) — the two structural shapes.
- [`CONCEPT-DOS-READINESS-VERDICT-LADDER-2026-06-26.md`](CONCEPT-DOS-READINESS-VERDICT-LADDER-2026-06-26.md) — the worked example of a lift (G1).
