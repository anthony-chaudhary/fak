---
title: "The agent programming grammar — the normative trust grammar a second implementation conforms to"
description: "The normative standard for fak's domain-free trust grammar: the closed nouns (lane, lease, reason token, witness, verdict, claim, ladder rung, scope), the shipped verbs each with an input -> verdict signature and the closed vocabulary it draws from, the lift recipe as MUST clauses (closed vocabulary, evidence-bound with no `claimed` field, fail-closed, data-not-code, both-lenses), the G6 one-sided-screen + witnessed-loss polarity predicate as a checkable MUST, and a conformance checklist a `dos`-compatible host answers per verb. The contract role `internal/abi`'s golden freeze plays for the ABI, played for the agent-coordination grammar — promoted from the design note CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md (#1209)."
---

# The agent programming grammar

This is a **normative standard**. It fixes the grammar an agent fleet coordinates by — the
nouns, the verbs, and the closed vocabularies — so a *second* implementation can be
conformance-checked against it, the same role [`internal/abi`](../../internal/abi)'s golden
freeze plays for the ABI. The companion design note
([`CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md`](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md))
explains *why* this grammar exists and catalogs the under-expressed concepts (`G1`–`G9`) that
become the next verbs; this page states *what* a conforming host MUST do. It sits in
`docs/standards/` beside [net-true-value](net-true-value.md), [the observer-effect
contract](observer-effect.md), and the per-verb schemas ([verification-ladder](verification-ladder-spec.md),
[context-contract](context-contract-schema.md), [taint-check](taint-check-schema.md),
[agent-routing](agent-routing-schema.md), [prediction-calibration](prediction-calibration.md)).

The keywords **MUST**, **MUST NOT**, **SHOULD**, and **MAY** are used as in RFC 2119.

The grammar carries one invariant at every scale:

> **A decision no participant can move by narrating a number.** Evidence the claimant did
> not author is the only admissible truth; a claim is untrusted until witnessed; a refusal
> carries a token from a closed, checkable set.

A conforming host is a substrate (the reference is **DOS** — [`dos.toml`](../../dos.toml)
plus the `dos_*` MCP verbs) that an agent fleet adopts by *configuration*, without forking
the kernel. Every clause below is read off a verb that already ships.

## The nouns (closed)

The grammar's nouns are a closed set. A conforming host MUST model each one and MUST NOT
admit a synonym that erases its invariant.

| Noun | What it is | Reference home |
|---|---|---|
| `lane` | a named file-tree scope work is admitted against | [`dos.toml [lanes]`](../../dos.toml) |
| `lease` | a live lock on a lane — `exclusive` or `shared` | `dos_arbitrate`, `dos.toml` lock-mode tree rule |
| `reason token` | a closed-vocabulary refusal (`[reasons.*]`); out-of-set ⇒ `UNCLASSIFIED` | `dos.toml [reasons]`, `dos_check_reason` |
| `witness` | the forgeability rung of a claim: `diff-witnessed` (non-forgeable) vs `subject-only` | `dos_commit_audit`, `internal/witness` |
| `verdict` | a value from a CLOSED set: `Allow`/`Deny`/`Quarantine`; `allow`/`deny`/`defer`/`indeterminate`; `OK`/`CLAIM_UNWITNESSED`/`ABSTAIN`; `RECALL_FRESH`/`RECALL_STALE` | `internal/abi`, the `dos_*` verbs |
| `claim` | a worker self-report — an INPUT to a witness, never an admissible output | every verb |
| `ladder rung` | a closed maturity/cost level promoted only by evidence the promoter did not author | [verification-ladder-spec](verification-ladder-spec.md) |
| `scope` | the share boundary on any `abi.Ref`: `ScopeAgent` < `ScopeTenant` < `ScopeFleet` | `internal/abi`, `internal/gateway` L3 share check |

## The verbs (shipped) — input → verdict signature

Every verb below maps to a `dos_*` MCP verb or a `dos.toml` surface that exists **today**.
The signature is `input → verdict`; the rightmost column names the closed vocabulary the
verdict is drawn from. The thread through every row: a claim graded `subject-only` /
`CLAIM_UNWITNESSED` is surfaced for human residual review, **never silently passed**.

| Verb | `dos_*` surface | Input → verdict | Closed vocabulary drawn from |
|---|---|---|---|
| `arbitrate` | `dos_arbitrate` | `(lane, kind, mode, tree, live_leases)` → `acquire` \| refuse(`COLLISION_RISK`) | lane taxonomy + lock-mode tree rule; reason tokens |
| `verify` | `dos_verify` | `(plan, phase, workspace)` → `{shipped: bool, source ∈ registry\|grep\|none}` | the `(fak <leaf>)` ship-commit grammar (`dos.toml [stamp]`) |
| `audit` | `dos_commit_audit` | `(ref, workspace)` → `verdict ∈ {OK, CLAIM_UNWITNESSED, ABSTAIN}`, `witness ∈ {diff-witnessed, subject-only}` | the witness forgeability rung |
| `review` | `dos_review` | `(commit range)` → residual-vs-cleared attention bands | the witness rung folded over a range |
| `refuse` / `check_reason` | `dos_refuse_reasons` / `dos_check_reason` | `(token)` → `{known: bool, summary, fix}`; out-of-set ⇒ `UNCLASSIFIED` | the closed `[reasons.*]` set |
| `recall` | `dos_recall` | `(saved memory's named artifacts)` → `RECALL_FRESH` \| `RECALL_STALE` (`STALE_RECALL`) | the recall-freshness vocabulary |
| `resolve` | `dos_citation_resolve` | `(citation)` → `{exists_in_reporter: bool}` | third-party-reporter existence |
| `status` | `dos_status` | `(run id)` → digest `{liveness, verified progress, region}` — **no `claimed` field by construction** | the witnessed-status contract (`RUN_STATUS_CLAIMED_FIELD` floor) |
| `doctor` | `dos_doctor` | `(workspace)` → workspace introspection | the lane/reason/stamp surfaces it reads back |
| `answer` | `dos_answer` | `(question)` → score against the corpus | the orientation-doc corpus |

The closed vocabularies the verbs draw from, in one place: the **reason tokens**
(`dos.toml [reasons.*]`, with `UNCLASSIFIED` the fail-closed catch-all); the **verdict**
enums above; the **witness rung** (`diff-witnessed` / `subject-only`); the **scope** lattice
(`ScopeAgent`/`ScopeTenant`/`ScopeFleet`); the **cost** and **risk-class** enums of a ladder
([verification-ladder-spec](verification-ladder-spec.md)); and the **provenance labels**
`WITNESSED`/`OBSERVED`/`MODELED`/`SIMULATED` ([observer-effect](observer-effect.md)). A token
outside its set is rejected at the boundary, never silently treated as more-permissive.

## The lift recipe (normative MUST clauses)

A component is a conforming grammar verb only if it keeps the invariant. These five rules,
read off the verbs that already shipped, are **MUST** clauses — a verb that breaks any one is
non-conformant.

1. **Closed vocabulary.** A verdict or reason MUST come from a finite, checkable set. An
   out-of-set token MUST be treated as `UNCLASSIFIED` and refused conservatively — never
   silently more-permissive. (`dos_check_reason` is the validator; `dos.toml [reasons]` is the
   set.)

2. **Evidence-bound, no `claimed` field.** A verb MUST fold evidence the claimant did not
   author. A self-report MUST be an INPUT to a witness, never an output. The status digest
   MUST NOT carry a `claimed` field — the floor refuses one (`RUN_STATUS_CLAIMED_FIELD`).

3. **Fail-closed.** Absence of an affirmative allow MUST be a deny. An `INDETERMINATE`
   verdict MUST escalate to a costlier rung before commit; it MUST NOT pass and MUST NOT fold
   to an allow. (The kernel ships this as a non-committable `VerdictIndeterminate`;
   [verification-ladder-spec](verification-ladder-spec.md) is the declarable form.)

4. **Data, not code.** A binding (a lane tree, a stamp grammar, a reason's summary/fix) MUST
   be `dos.toml` data that introduces **no spontaneous refusal** — it fires only at an opt-in
   surface or its named floor. The mechanism stays in the installed `dos` package; only policy
   crosses into the tree.

5. **Both lenses.** A verb MUST pay on the optimization lens (a saved call, a smaller
   resident set, a skipped rung), not only the safety lens. A component that is only a tax is
   not a grammar primitive.

## G6 — the one-sided-screen + witnessed-loss polarity predicate (a checkable MUST)

Any *additive* safety component — a screen, a proposer, a triage model bolted onto the floor
(home: [`internal/wirescreen`](../../internal/wirescreen),
[`internal/ctxmmu`](../../internal/ctxmmu)) — MUST declare and satisfy this predicate. It is
the polarity rule that lets fak add a lossy local model to the wire *without* widening the
trust surface. Each clause is checkable, not aspirational:

- **G6.1 — Monotone polarity (one-sided screen).** The component MUST only move a verdict
  toward MORE careful on the closed verdict lattice (`Allow` → `Quarantine`/`Deny`); it MUST
  NOT move any verdict toward more-permissive. **Check:** for every input, the component's
  output verdict rank ≥ the deterministic floor's verdict rank. This is the
  [`internal/wirescreen`](../../internal/wirescreen) contract verbatim — *"a proposer may only
  make the system MORE careful (quarantine, demote, redact), never weaker than a deterministic
  floor."*

- **G6.2 — Witnessed loss (a wrong proposal costs a fault, never a fact).** A wrong proposal
  MUST cost at most one demand-page fault, never a lost fact. The original bytes MUST stay
  pinned in the content-addressed store, and a gated `PageIn` after a witness `Clear` MUST
  restore them **byte-exact**. **Check:** the original is recoverable byte-for-byte after any
  proposal. This is `ctxmmu`'s quarantine + `PageIn` witness — *"a wrong proposal costs one
  demand-page fault, never a lost fact."*

- **G6.3 — Default-inert.** The component MUST be gated off (build tag or env) until its
  end-to-end latency is measured, introducing no spontaneous refusal until an operator opts in
  — the data-not-code MUST (recipe rule 4) applied to an additive screen.

A component that tightens a floor and keeps its loss witnessed is safe to add by
construction: the worst case is a recoverable page fault, and the verdict can only get more
careful. A component that fails G6.1 widens the attack surface; one that fails G6.2 can
silently destroy a fact. Either is non-conformant.

## Conformance checklist — what a `dos`-compatible host MUST answer per verb

A host claiming conformance answers each row affirmatively, with evidence the host did not
author:

| Verb | The host MUST be able to answer | Floor / surface |
|---|---|---|
| `arbitrate` | "Given these live leases, may this lane be taken without two agents mutating the same tree?" — and refuse `COLLISION_RISK` when not | `dos_arbitrate`, `dos.toml [lanes]` |
| `verify` | "Did this (plan, phase) ship, from git evidence, not the worker's word?" — naming the source (`registry`/`grep`/`none`) | `dos_verify`, `dos.toml [stamp]` |
| `audit` | "Does this commit's diff do the KIND of thing its subject claims?" — `diff-witnessed` vs `subject-only` | `dos_commit_audit` |
| `review` | "Across this range, what residual attention is uncleared?" | `dos_review` |
| `refuse` / `check_reason` | "Is this refusal token in the closed set, and what is its summary + fix?" — `UNCLASSIFIED` if not | `dos_refuse_reasons` / `dos_check_reason` |
| `recall` | "Are this saved memory's named artifacts still present in the live tree?" — `RECALL_FRESH`/`RECALL_STALE` | `dos_recall` |
| `resolve` | "Does this cited authority actually exist in a third-party reporter?" | `dos_citation_resolve` |
| `status` | "What is this run's liveness + ledger-verified progress + lease region?" — with **no `claimed` field** | `dos_status` |
| `doctor` | "What lanes, reasons, and stamp grammar does this workspace declare?" | `dos_doctor` |
| `answer` | "How does this question score against the orientation corpus?" | `dos_answer` |

Plus the recipe and polarity gates: every verb MUST satisfy the five MUST clauses, and every
*additive* safety component MUST satisfy G6.1–G6.3.

## Honest fences

- This is a **standard** that promotes a design note to normative status; it introduces **no
  code and no spontaneous refusal**. The shipped verbs (`arbitrate`/`verify`/`audit`/`review`/
  `refuse`/`check_reason`/`recall`/`resolve`/`status`/`doctor`/`answer`) are real today; the
  next verbs (`readiness`/`promote`/`context-contract`/`calibrate`/`taint-check`/`route`/
  `claim-check`/`verify --ladder`) are tracked as `G1`–`G9` in the
  [design note](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md) and are **not yet**.
- **G6 is a predicate fak's additive screens already satisfy, not a new gate.** The
  [`internal/wirescreen`](../../internal/wirescreen) proposer spine and `ctxmmu`'s
  quarantine + `PageIn` witness are the reference implementation; this page lifts their
  contract into a checkable MUST any additive component declares. It adds no rung and changes
  no fold.
- Per [`AGENTS.md`](../../AGENTS.md), any future verb implementing this spec is **Go in a
  leaf**, never a new `tools/*.py`. The grammar's home is DOS (domain-free trust logic);
  fak-tree policy/measurement lands as a `fak` subcommand.
- This standard does **not** replace the token engine, the model, or the harness. It is the
  governance band — the same scope fak already owns.

## Cross-references

- [The agent-programming-grammar design note](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md) — the *why* and the `G1`–`G9` backlog this standard is the normative head of.
- [Net-true-value](net-true-value.md) · [The observer-effect contract](observer-effect.md) · [The support-maturity honesty fence](support-maturity-honesty-fence.md) — the sibling prose standards in `docs/standards/`.
- [The verification-ladder spec](verification-ladder-spec.md) (`G2`) · [the context-contract schema](context-contract-schema.md) (`G4`) · [the taint-check schema](taint-check-schema.md) (`G7`) · [the agent-routing schema](agent-routing-schema.md) (`G8`) · [the prediction-calibration contract](prediction-calibration.md) (`G5`) — the per-verb schemas that conform to this grammar.
- [`dos.toml`](../../dos.toml) — the live lane taxonomy, reason vocabulary, and stamp grammar a conforming host declares.
- [`docs/INNOVATIONS-INDEX.md`](../INNOVATIONS-INDEX.md) Part 4 — the durable catalog this standard heads.
- [Claims ledger](../../CLAIMS.md) — shipped vs stub, claim by claim.
