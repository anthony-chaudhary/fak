# Context-safety DOS refusal tokens (#1231, C13 of epic #1217)

_Research / design only. This is the **C13 structured-refusal spec** for the
context-safety epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217).
It specs the three new `dos.toml [reasons]` tokens a context-safety panel emits
to **refuse green with a verifiable reason** instead of rendering an unearned
OK: `SAFETY_UNRECOVERED`, `VALUE_UNRECONCILED`, `SAFETY_UNWITNESSED`. No code
ships here — the deliverable is this committed spec: each token's category,
summary, fix, and the oracle condition that decides it. It is the refusal
vocabulary the C9 re-derivation checker
([#1227](https://github.com/anthony-chaudhary/fak/issues/1227),
[`context-safety-rederivation-checker-1217.md`](context-safety-rederivation-checker-1217.md))
and the C4 gap
([#1221](https://github.com/anthony-chaudhary/fak/issues/1221)) emit when a
number cannot be earned._

---

## Why a panel needs a refusal vocabulary

> The whole epic exists to stop fak rendering "value preserved" when it can't
> prove it. The structural enforcement of that is: when a number can't be
> witnessed or reconciled, the panel does not go green — it **refuses with a
> reason from a closed, verifiable vocabulary.** A free-text "couldn't compute"
> is drift; a closed token an oracle can check is a first-class refusal. This is
> the same discipline DOS already uses for lane admission (`LANE_DRAINED`,
> `OFF_TRUNK`, `L3_PAGE_DIGEST_MISMATCH`, …) — a refusal is *emittable,
> verifiable, and refusable* only if it is in the closed set.

**Witnessed gap (the anchor).** All three proposed tokens were checked with
`dos_check_reason` against this workspace: each returns
`known=false, category=UNCLASSIFIED` today. That is the precise, verifiable gap
this note closes — a context-safety panel **cannot** emit any of these tokens
now, because emitting an UNCLASSIFIED token is exactly the prose-drift the kernel
refuses. They must be **declared** (in `dos.toml [reasons]`) before a panel may
use them. This note specs that declaration.

---

## The DOS reason schema (from the live vocabulary)

Each reason in the closed set carries a fixed shape (observed via
`dos_refuse_reasons` over this workspace): a `token`, a `category` (the coarse
class it rolls up to), a `refusal` bool (does carrying it block, vs advisory),
a one-line `summary`, a `fix` (how to clear it), and `see_also` pointers. The
coarse categories in use are `TRUE_DRAIN` / `OPERATOR_GATE` / `STALE_CLAIM` /
`MISROUTE`. fak's own **verify-rung** workspace refusals — `L3_PAGE_DIGEST_MISMATCH`
("the bytes do not hash to the digest the page CLAIMS") and
`L3_CROSS_TENANT_SCOPE_DENIED` — are declared under `OPERATOR_GATE`, the closest
existing precedent for "the system declines to assert something it cannot
prove." The three tokens below follow that precedent.

---

## The three tokens

Each is **`refusal: true`** (carrying it blocks the panel going green) and each
binds to an **oracle condition** the C9 checker evaluates — verifiable, not
asserted.

### SAFETY_UNRECOVERED

- **category:** `OPERATOR_GATE`
- **summary:** A context-removal event happened but **no tamper-evident witness
  records it** — the event is real, the evidence is absent (e.g. an un-recovered
  evict panic killed the handler before a journal row was written, so the ribbon
  has a hole exactly where a shed occurred).
- **oracle condition:** there exists an observed shed/liveness contradiction (a
  decode-rate drop, a `/healthz`-vs-throughput disagreement, a metric delta)
  with **no corresponding journal `CAP_EVICT` / `CAP_FAULT` row** to witness it.
  Decided by the C9 checker over the journal + `/metrics`.
- **fix:** ensure the shed path writes its journal row *before* the risky
  operation (the recovery landed for #1143; the **witness** row is what
  `SAFETY_UNRECOVERED` demands). Until a row exists, the panel stays red — an
  invisible shed is the worst class.
- **see_also:** `internal/journal/journal.go`, `internal/gateway/gateway.go`
  (`recoverRecurrentEvictUnsupported`), the C2 catalog F1.

### VALUE_UNRECONCILED

- **category:** `STALE_CLAIM`
- **summary:** Two independent derivations that must agree **disagree** — the
  realized-reuse *count* (C4) credits value the *correctness* witness (C14
  `EvaluatePrefixCoherence` / `StabilityReport`) refutes. A high reuse count over
  a broken prefix is value that looks real and isn't.
- **oracle condition:** `RealizedReused > 0` **and** **not** `(Coherent ∧
  Stable)` — the C14 reconciliation rule fails. Decided by re-deriving both the
  count and the coherence/stability verdicts and comparing.
- **fix:** do not credit the reuse value — re-establish coherence (the reused
  prefix must be the KV the current prefix would produce) or treat the reuse as
  cold. The gap chart shows the count, but the panel refuses green until the two
  derivations reconcile.
- **see_also:** `internal/cachemeta/prefix_coherence.go`,
  `internal/cachemeta/prefix_stability.go`, the C4 + C14 specs.

### SAFETY_UNWITNESSED

- **category:** `STALE_CLAIM`
- **summary:** A rendered number has **no tamper-evident source at all** — it
  cannot be re-derived from the journal, `/metrics`, git, or a CAS digest, so it
  may not render green (the weakest case: not a disagreement, an absence).
- **oracle condition:** the C9 `NumberCheck` finds **no tamper-evident
  `Source`** for the number (`Status = REFUSED`). Decided by the re-derivation
  checker enumerating sources and finding none.
- **fix:** bind the number to a tamper-evident source (anchor it to a journal
  `Seq`, a WITNESSED metric, a git object, or a CAS digest) before rendering it.
  An un-anchored number is a self-report, and a self-report may not be green.
- **see_also:** `context-safety-rederivation-checker-1217.md` (C9), the doctrine
  C1 fence (c) (un-witnessed = `not yet`, not a result).

### At a glance

| token | category | refusal | oracle condition (what decides it) |
|---|---|---|---|
| `SAFETY_UNRECOVERED` | `OPERATOR_GATE` | true | observed shed/liveness contradiction with **no** journal `CAP_EVICT`/`CAP_FAULT` row |
| `VALUE_UNRECONCILED` | `STALE_CLAIM` | true | `RealizedReused > 0` ∧ ¬(`Coherent` ∧ `Stable`) — C14 reconciliation fails |
| `SAFETY_UNWITNESSED` | `STALE_CLAIM` | true | C9 `NumberCheck` finds no tamper-evident `Source` (`Status = REFUSED`) |

---

## Declaration sketch (`dos.toml [reasons]`)

The tokens are declared in the workspace `dos.toml [reasons]` table (which
`dos_refuse_reasons` shows overlays the built-ins), following the observed
schema. Illustrative shape (final keys per the live `[reasons]` grammar):

```toml
[reasons.SAFETY_UNRECOVERED]
category = "OPERATOR_GATE"
refusal  = true
summary  = "A context-removal event happened but no tamper-evident witness records it."
fix      = "Write the journal CAP_EVICT/CAP_FAULT row before the risky shed; an unwitnessed shed stays red."
see_also = ["internal/journal/journal.go", "internal/gateway/gateway.go", "docs/notes/context-safety-failure-mode-catalog-1217.md"]

[reasons.VALUE_UNRECONCILED]
category = "STALE_CLAIM"
refusal  = true
summary  = "Realized-reuse count credits value the coherence/stability witness refutes."
fix      = "Re-establish prefix coherence, or treat the reuse as cold; do not credit unreconciled value."
see_also = ["internal/cachemeta/prefix_coherence.go", "internal/cachemeta/prefix_stability.go", "docs/notes/context-safety-reuse-correctness-witness-1217.md"]

[reasons.SAFETY_UNWITNESSED]
category = "STALE_CLAIM"
refusal  = true
summary  = "A rendered number has no tamper-evident source and may not render green."
fix      = "Anchor the number to a journal Seq / WITNESSED metric / git object / CAS digest before rendering."
see_also = ["docs/notes/context-safety-rederivation-checker-1217.md", "docs/notes/context-safety-visuals-tracking-1217.md"]
```

(Schema confirmed against the live `dos.toml` `[reasons.*]` table — e.g.
`[reasons.ARCH_LAYER_VIOLATION]` / `[reasons.OFF_TRUNK]` use exactly these
`category` / `refusal` / `summary` / `fix` / `see_also` keys, `dos.toml:424`,
`:445`.)

> **Category-set note (open decision).** The three map cleanly onto the existing
> coarse classes (`OPERATOR_GATE` for the missing-witness gate, `STALE_CLAIM` for
> the unproven/refuted claim) and follow fak's `L3_*` verify-rung precedent. If
> the maintainer prefers a *new* coarse category for the value-preservation floor
> (e.g. a `VALUE_FLOOR` class), that is a one-line kernel-vocabulary decision
> flagged here rather than presumed — the tokens and their oracle conditions are
> unchanged either way.

---

## Honest `not yet` gaps (per fence c)

- **The tokens are UNCLASSIFIED until declared.** Verified now via
  `dos_check_reason` (all three `known=false`). A panel cannot legally emit them
  until `dos.toml [reasons]` carries them and `dos_check_reason` returns
  `known=true`. This note is the spec; the declaration is the build step.
- **The oracle conditions depend on C9 + C14.** `SAFETY_UNWITNESSED` and
  `SAFETY_UNRECOVERED` are decided by the C9 checker (#1227);
  `VALUE_UNRECONCILED` by the C14 reconciliation (#1232). The tokens are the
  *refusal surface*; those children are the *deciding oracles*. Declaring the
  tokens without the oracles would let a panel refuse-green but not *decide* when
  to — both halves ship together.

---

## Acceptance check (against #1231)

The issue's acceptance: *each token gets a category, summary, fix, and the
oracle condition that decides it; verify with `dos_check_reason` before
emitting._

- **Three tokens, each fully specced** with `category` / `refusal` / `summary` /
  `fix` / `see_also` and an **oracle condition** an independent check evaluates
  (no asserted refusals).
- **Verified via `dos_check_reason`:** all three return
  `known=false, UNCLASSIFIED` today — the witnessed gap — and the note states the
  rule that a panel may emit them only once declared and `dos_check_reason`
  returns `known=true`.
- **Grounded in the live vocabulary:** the schema and category choices follow the
  observed `dos_refuse_reasons` shape and fak's `L3_*` verify-rung precedent, with
  the new-coarse-category question flagged as an open maintainer decision rather
  than presumed.
- **Bound to the deciding oracles** (C9 #1227, C14 #1232) so the refusal surface
  and its decision procedure ship together.

---

_Filed as research / planning only under epic #1217. Parent: #1217. The refusal
vocabulary emitted by the C9 checker (#1227), the C4 gap (#1221), and the C14
reconciliation (#1232); grounded in the live DOS `[reasons]` schema. Design-only
— no implementation ships under #1217 until the notes are reviewed._
