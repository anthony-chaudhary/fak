---
title: "The verification-ladder spec — a declarable rung ladder + a first-class INDETERMINATE verdict any runtime can author as data"
description: "The engine-free schema that lifts fak's verification ladder (docs/notes/verification-ladder-doctrine.md) out of the kernel into a domain-free contract any agent runtime can declare as DATA and the `dos verify --ladder` verb walks. A ladder is a cost-ordered set of rungs (cheapest->costliest); a checker picks the SMALLEST rung that can conclusively decide a claim and climbs only when that rung returns INDETERMINATE — the first-class verdict for 'I could not decide this cheaply; a costlier rung MUST be consulted before commit', distinct from both a fail-open allow and a bare defer-to-DEFAULT_DENY. Three contracts make it portable: (1) it round-trips author -> check -> review as data with no model and no fak engine in the loop; (2) the verdict set, the risk-class set, and the cost set are CLOSED, validatable vocabularies — an out-of-set token is rejected at the boundary, INDETERMINATE itself is dos_check_reason-validatable (declared in dos.toml [reasons]); (3) it is fail-closed BY CONSTRUCTION — the schema pins on_exhaustion to deny and forces escalate_on to contain indeterminate, so a fail-open ladder is rejected at the authoring boundary, and a residual INDETERMINATE can never be a committed verdict. The honest fence: the kernel-side VerdictIndeterminate + lazy chain fold is SHIPPED (internal/abi, internal/kernel, epic #657); this page is the declarable spec it implements — it does not re-implement the rungs, and the `dos verify --ladder` verb is not yet."
---

# The verification-ladder spec

Verification should be flexible and granular: by default pick the **smallest rung that
can conclusively establish the property at hand**, and climb to a costlier rung only when
the cheap one comes back **INDETERMINATE** or the risk warrants paying more. That is the
agent-kernel restatement of five things Linux already does — seccomp's most-restrictive
return fold, least-privilege capability sets, AppArmor complain→enforce, the integrity
granularity ladder, and the eBPF prove-before-admit verifier — drawn out in full in
[the verification-ladder doctrine](../notes/verification-ladder-doctrine.md).

fak's adjudicator already **is** that machine: a cost-ordered chain folded by a
restrictiveness lattice, cheapest rung first, fail-closed when nothing affirmatively
allows. And the doctrine's named-missing half is now built — the kernel ships a
first-class `VerdictIndeterminate` and a lazy chain fold that makes it non-committable
(a residual `Indeterminate` yields `Deny`, never `Allow`):
[`internal/abi/types.go`](../../internal/abi/types.go) (`VerdictIndeterminate`, the
closed kind), [`internal/abi/registry.go`](../../internal/abi/registry.go) (`FoldRank`
15, strictly between `Defer` and `Transform`), and
[`internal/kernel/kernel.go`](../../internal/kernel/kernel.go) (the fold:
`sawIndeterminate` → climb, else fail closed).

The grammar gap this page closes: that ladder lives **inside fak's kernel**. There was
no standalone, domain-free form a *different* agent runtime could declare — its own
rungs, its own risk classes, its own fail-closed tail — and hand to a checker that picks
the smallest sufficient rung. This page is that form, written as a contract: a
cost-ordered rung list as data, the closed verdict vocabulary (with `INDETERMINATE`
first-class), and the smallest-sufficient-rung walk. It is `G2` of the
[agent-programming-grammar epic](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md)
(#1210), the trust-floor sibling of [the portable taint-check schema](taint-check-schema.md)
(`G7`, an admission check), [the portable agent-routing schema](agent-routing-schema.md)
(`G8`, a routing decision), and [the prediction-calibration contract](prediction-calibration.md)
(`G5`, a calibration verdict). The verb that walks this schema over a claim is
`dos verify --ladder`; its home is the installed DOS package, and it is **not yet** (see
the [honest fences](#honest-fences)). fak's kernel is the reference implementation — the
offline witness that the rungs below are real, not a wish.

## The schema

A verification ladder is a small set of nouns. None of them mentions a tool, a network,
or a fak package — the schema is the *ladder's shape* and the *decision* a walk over it
produces. The machine-checkable form is
[`verification-ladder-spec.json`](verification-ladder-spec.json) (Draft 2020-12); the
root validates a `Ladder` (the authored policy), and `$defs/Decision` is the verdict a
checker returns after walking it.

### Ladder — the declarable, cost-ordered rung spec (the `dos.toml [ladder]` shape)

```
Ladder {
  rungs         : [ Rung ]    // ordered CHEAPEST -> costliest; >= 1
  on_exhaustion : "deny"      // PINNED: the fail-closed tail (a fail-open ladder is unrepresentable)
  escalate_on   : [ Verdict ] // the verdicts that trigger a climb; MUST contain "indeterminate"
}

Rung {
  id          : int >= 0      // ladder position; 0 is the cheapest, ascending == costlier
  name        : string        // a human label, e.g. "in-process structural"
  cost        : Cost          // CLOSED: reuse < in_process < corroborate < suite < worktree_spawn < human
  max_risk    : RiskClass     // the highest-risk claim this rung can CONCLUSIVELY ALLOW
  establishes : string        // OPTIONAL prose: what this rung proves (and so what it cannot, so you climb)
}
```

A `Ladder` is **data** — the `dos.toml [ladder]` table, here as JSON because TOML and
JSON are isomorphic as data. It introduces no spontaneous refusal: a checker walks it
offline. `max_risk` bounds only the rung's affirmative *allow*: a cheap structural rung
may still conclusively *deny* a higher-risk claim (a `self_modify` glob refusal), but it
cannot *allow* a claim riskier than its `max_risk` — it returns `indeterminate` and the
checker climbs.

### Subject — the claim being classified

```
Subject {
  risk_class : RiskClass   // CLOSED: read < write < self_modify
  label      : string      // OPTIONAL bounded, payload-free note (never the value's bytes)
}
```

The `risk_class` is what selects the smallest sufficient rung: a checker picks the
cheapest rung whose `max_risk` covers the claim's `risk_class`.

### Decision — the reviewable Allow | Deny(reason), and the rungs it walked

```
Decision {
  subject      : Subject
  path         : [ Step ]   // the ordered rungs consulted and the verdict each returned
  rung_reached : int        // the rung id whose verdict committed (== the last step's rung)
  climbed      : bool        // true iff it escalated past a cheaper rung (path length > 1)
  verdict      : "allow" | "deny"   // the FINAL committed verdict — CLOSED to the two committable outcomes
  reason       : DenyReason  // REQUIRED when verdict == "deny"
  witness      : string      // a bounded, payload-free note (the rung + risk labels, never the value)
}

Step { rung : int, verdict : Verdict }
```

Walking a `Ladder` for a `Subject` yields a `Decision`: the rungs it consulted in order,
which rung committed, whether it climbed, and the final `allow`/`deny`. The decision is
data — reviewable, diffable, produced with no model in the loop.

### The closed verdict vocabulary

`Verdict` is **closed and additive** — a new verdict is a new named value plus a fold
arm, never a free-text field. It mirrors the kernel's closed `VerdictKind`
([`internal/abi/types.go`](../../internal/abi/types.go)):

| `verdict` | meaning | committable? |
|---|---|---|
| `allow` | conclusively admitted | yes |
| `deny` | a provable refusal | yes |
| `defer` | a bare abstention — "I have no opinion, ask the next rung" | no (folds to `DEFAULT_DENY` if nothing allows) |
| `indeterminate` | **the C2 verdict** — "I could not CONCLUSIVELY decide this cheaply; a costlier rung MUST be consulted before commit" | **no** — escalates, or fails closed |

`indeterminate` is distinct from both ends it is squeezed between. It is **not** a
fail-open allow (it never commits) and **not** a bare `defer` (a `defer` says "nothing
here," an `indeterminate` says "climb, do not guess"). In the lattice it ranks above
`defer` and below every conclusive kind (`FoldRank` 15, between `Defer` 10 and
`Transform` 20 in [`internal/abi/registry.go`](../../internal/abi/registry.go)) — so a
conclusive `allow`/`deny` from any rung still wins outright, and the fold never gets
stuck at `indeterminate` when something conclusive exists.

`INDETERMINATE` is also a **`dos_check_reason`-validatable** refusal token: when a ladder
exhausts with a residual indeterminate, its fail-closed deny carries `reason:
"INDETERMINATE"`, declared in this workspace's
[`dos.toml [reasons]`](../../dos.toml). It names "I could not decide cheaply; the
costlier rung was unavailable, so fail closed — escalate, never silently allow."

### The cost and risk vocabularies

`Cost` is the cost-ordered class of a rung (the reason to prefer the smallest sufficient
one): `reuse` (a cached/vDSO re-output, ns, in-proc) < `in_process` (a structural
name/arg/lint check, ns–µs) < `corroborate` (a require-witness handback to one
out-of-band resolver) < `suite` (a local build+vet+test or git-evidence read, seconds) <
`worktree_spawn` (an isolated-worktree measure, ms-spawn + suite) < `human` (an operator
ESCALATE). `RiskClass` is the ordered risk of a claim: `read` < `write` < `self_modify`.
Both are CLOSED enums; an out-of-set value is rejected at the authoring boundary.

## Smallest-sufficient-rung selection — domain-free, deterministic, fail-closed

The walk is a pure function of the ladder and the claim's risk class. Read top to bottom:

```
1. Select the SMALLEST rung whose max_risk covers the subject's risk_class.
2. Consult it:
   - a conclusive allow/deny      -> commit it (rung_reached = this rung)
   - an escalate_on verdict        -> CLIMB to the next-costlier rung; repeat from 2
     (a rung whose max_risk does NOT cover the claim returns indeterminate -> climb)
3. If the ladder is exhausted and the residual is still indeterminate:
   -> on_exhaustion == "deny" : fail closed, reason = INDETERMINATE. NEVER allow.
```

A **low-risk read** is conclusively allowed at the cheap in-process structural rung
(its `max_risk` covers `read`), so the checker never climbs: `path` is one step,
`climbed = false`, `rung_reached = 1`. A **write** cannot be conclusively *allowed* by a
rung whose `max_risk` is only `read`, so that rung returns `indeterminate`; `escalate_on`
contains `indeterminate`, so the checker climbs to the require-witness rung (`max_risk
write`), which corroborates the claimed effect and conclusively allows: `path` has two
steps, `climbed = true`, `rung_reached = 3`. Those are the two
[on-disk witnesses](#the-round-trip-as-data) — the same ladder, two risk classes, two
different smallest-sufficient rungs.

The walk is **monotone**: making a claim *more* risky (`read`→`write`→`self_modify`)
never lowers the rung it reaches; it only ever climbs. It is the same property the kernel
fold keeps — the lattice fold is order-independent, and a residual indeterminate is
non-committable by construction.

## Fail-closed, made structural

The fail-closed recipe rule — *a cheap rung that comes back INDETERMINATE must escalate,
never silently allow* — is not left to the checker's good behavior. It is encoded in the
schema so a fail-open ladder is **rejected at the authoring boundary**, never admitted:

1. **`on_exhaustion` is pinned to `const "deny"`.** A ladder literally cannot declare a
   fail-open tail; a ladder that tries `on_exhaustion: "allow"` is refused
   ([`ladder-fails-open.json`](fixtures/verification-ladder-invalid/ladder-fails-open.json)).
2. **`escalate_on` MUST contain `indeterminate`** (`contains: const "indeterminate"`). An
   `indeterminate` rung always has something to trigger a climb; it can never be silently
   dropped ([`ladder-no-escalate.json`](fixtures/verification-ladder-invalid/ladder-no-escalate.json)).
3. **A `Decision` whose `path` contains an `indeterminate` step MUST have `climbed:
   true`.** The ladder cannot decide *in place* on an indeterminate — it must have
   escalated.
4. **The final `verdict` is closed to `{allow, deny}`.** An `indeterminate` can never be
   a committed verdict; a residual one folds to `deny` via `on_exhaustion`
   ([`indeterminate-final-verdict.json`](fixtures/verification-ladder-invalid/indeterminate-final-verdict.json)).

These are the kernel's own guarantees, lifted into the data shape: the kernel's lazy fold
holds an `Indeterminate` as non-committable and resolves a residual one to `Deny`
([`internal/kernel/kernel.go`](../../internal/kernel/kernel.go)); the schema makes a
host's *declared* ladder unable to opt out of that.

## The three contracts (the acceptance, made checkable)

This spec is portable because it holds three properties an external runtime can verify
without fak's kernel:

1. **It round-trips with no engine.** author (declare a `Ladder`) → check (walk it for a
   `Subject` → a `Decision`) → review (read the `Decision` as data) is a pure data
   transform. No model runs; no network is touched. The
   [round-trip below](#the-round-trip-as-data) is the witness, on disk as fixtures — the
   *same* ladder yields a one-step read decision and a two-step climbing write decision,
   the only difference being the claim's risk class.
2. **The vocabularies are closed and validatable.** The verdict set, the risk-class set,
   and the cost set are finite enums; a validator decides membership with a finite
   switch. The schema is published as a machine-checkable JSON Schema
   ([`verification-ladder-spec.json`](verification-ladder-spec.json), Draft 2020-12), so
   any runtime authors and validates a ladder with an off-the-shelf validator, **no fak
   engine present**. And `INDETERMINATE` is `dos_check_reason`-validatable —
   `dos_check_reason INDETERMINATE` returns `known=true` (declared in
   [`dos.toml [reasons]`](../../dos.toml)); an out-of-set token is `UNCLASSIFIED` and
   refused conservatively. The five on-disk
   [negative fixtures](fixtures/verification-ladder-invalid/) — a fail-open tail, a
   missing escalate-on, an out-of-set verdict, a non-committable final verdict, and an
   unknown field — are each rejected at the boundary; the
   [validation recipe](fixtures/verification-ladder-invalid/README.md) runs the whole
   round-trip with a stock validator, so the "validatable" claim is checkable, not
   asserted.
3. **It is fail-closed and evidence-bound.** Absence of an affirmative allow is a deny;
   a residual indeterminate folds to `deny`, never passes (the
   [structural rules above](#fail-closed-made-structural)). And the rung verdicts are
   evidence the claimant did not author — the require-witness rung corroborates a claimed
   git/object effect, the keep-bit rung is a non-forgeable AND of measured signals. A
   self-reported "I passed" is an input to a rung, never a `Decision`'s output.

## The round-trip, as data

The schema's whole claim is that author → check → review is data, not narration. The
fixtures under [`fixtures/`](fixtures/) are the on-disk witness:

- [`verification-ladder.json`](fixtures/verification-ladder.json) — the **AC#2 witness**:
  a five-rung verification ladder expressed as data (the `dos.toml [ladder]` shape).
  `on_exhaustion` is `deny`; `escalate_on` contains `indeterminate`.
- [`verification-ladder-decision-read.json`](fixtures/verification-ladder-decision-read.json)
  — the **AC#3 witness, low-risk**: a `read` stops at rung 1 (`climbed: false`).
- [`verification-ladder-decision-write.json`](fixtures/verification-ladder-decision-write.json)
  — the **AC#3 witness, climbing**: a `write` is `indeterminate` at rung 1 and climbs to
  the require-witness rung 3 (`climbed: true`).

A reviewer reads the fixtures and the walk binding them; no model and no fak engine are
needed to confirm the selection. fak's kernel walks the same lattice and reaches the same
verdicts — offline, witnessed by `go test ./internal/kernel`, the reference
implementation's proof that the rungs below are real.

## Reference implementation and witness

| Schema element | Reference stick (`internal/kernel`, `internal/abi`, `internal/adjudicator`, `internal/shipgate`) | Status |
|---|---|---|
| `indeterminate` verdict (closed, non-committable) | `abi.VerdictIndeterminate` (`internal/abi/types.go`) + `FoldRank` 15 (`internal/abi/registry.go`) | [SHIPPED] |
| Lazy fold: residual `indeterminate` → `Deny`, conclusive wins | `kernel.Fold` (`internal/kernel/kernel.go`) + `FoldExplain` mirror (`internal/kernel/explain.go`) | [SHIPPED] |
| `INDETERMINATE` as a `dos_check_reason`-validatable refusal token | [`dos.toml [reasons.INDETERMINATE]`](../../dos.toml) | [SHIPPED] |
| Cost-ordered chain, cheapest rung first, most-restrictive fold | `kernel.Fold` + the `FoldRank` lattice (`internal/abi/registry.go`) | [SHIPPED] |
| Rung 0 (reuse / vDSO re-output) | `internal/kernel/kernel.go` (FastPath consulted before the fold) | [SHIPPED] |
| Rung 1 (in-process structural: name / self-modify / arg-predicate / lint) | `internal/adjudicator/decide.go` (per-call short-circuit) | [SHIPPED] |
| Rung 2 (posture admit-and-log) | `internal/adjudicator/decide.go` (`PostureAdmitAndLog`, read-prefix-only) | [SHIPPED] (binary/global) |
| Rung 3 (require-witness corroboration) | `internal/kernel/kernel.go` + `internal/shipgate/adjudicate.go` | [SHIPPED] |
| Rung 4 (isolated-worktree keep-bit, non-forgeable) | `internal/shipgate/shipgate.go` (`improvedBit`) | [SHIPPED] |
| Offline determinism witness | `go test ./internal/kernel ./internal/abi` (no model in the loop) | [SHIPPED] |
| Portable `dos verify --ladder` verb (walks a declared ladder) | the installed DOS package | **not yet** |
| Per-claim risk→rung selection *inside the kernel* (data-driven `decide.go`) | `internal/adjudicator` (rung order is hard-coded today) | **not yet** (epic #663) |

## Honest fences

- **This is the declarable spec, not a re-implementation of the rungs.** The rungs
  (0–7) are SHIPPED in fak's kernel; the kernel-side `VerdictIndeterminate` + lazy chain
  fold is SHIPPED ([epic #657](../notes/verification-ladder-epics.md)). This page lifts
  that machine into a domain-free, declarable form. It does not add a rung, change a
  fold, or convert any existing rung to emit `indeterminate`.
- **The `dos verify --ladder` verb is not yet.** The portable verb that walks a declared
  ladder over a claim lives in the installed DOS package (the grammar's home; see the
  [epic](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md)). This page is the
  schema it implements; the verb is a named follow-on, not shipped here. The fixtures are
  the round-trip witness in the meantime.
- **Risk→rung selection is declarable here, but the kernel's own rung order is still
  hard-coded.** This schema lets a *host* declare which rung covers which risk class
  (`max_risk`), and a checker picks the smallest sufficient one. fak's in-process
  `decide.go` still runs a fixed rung sequence on every call (only `LintWrites` is
  opt-in) — making that data-driven inside the kernel is the separate epic #663, named
  in the [doctrine](../notes/verification-ladder-doctrine.md), not built here.
- **`max_risk` bounds the affirmative allow, not the deny.** A cheap rung may still
  conclusively *deny* a higher-risk claim (a structural `self_modify` refusal); it just
  cannot *allow* one above its `max_risk`. The schema models the climb-on-allow side; a
  rung's deny authority is the kernel's, unchanged.
- **The closed verdict set is the floor's, not a superset.** A host may not invent a
  sixth committable verdict by declaring it — the schema's `Verdict` enum is closed, and
  the final `Decision.verdict` is closed to `{allow, deny}`. A new verdict is a kernel
  change (a new `VerdictKind` + a fold arm), not a `dos.toml` knob.

## Cross-references

- [`verification-ladder-spec.json`](verification-ladder-spec.json) — the machine-checkable JSON Schema (Draft 2020-12): validate a ladder + a decision with any off-the-shelf validator, no fak engine present.
- [The verification-ladder doctrine](../notes/verification-ladder-doctrine.md) — the design note this spec lifts (the ladder, the five Linux mirrors, the honesty boundary, the half-done finding).
- [The verification-ladder epic roadmap](../notes/verification-ladder-epics.md) — the eight epics that build the lazy half; epic #657 (the SHIPPED `VerdictIndeterminate` + lazy fold) and epic #663 (data-driven rung selection) are this spec's kernel siblings.
- [The portable taint-check schema](taint-check-schema.md) · [the portable agent-routing schema](agent-routing-schema.md) · [the prediction-calibration contract](prediction-calibration.md) — the `G7`/`G8`/`G5` siblings; same recipe (closed vocabulary, evidence-bound, fail-closed, data-not-code), different verb.
- [The agent-programming grammar](../notes/CONCEPT-AGENT-PROGRAMMING-GRAMMAR-2026-06-28.md) — the epic this schema is `G2` of, and the recipe every lift keeps.
- [Net-true-value](net-true-value.md) · [The observer-effect contract](observer-effect.md) · [The support-maturity honesty fence](support-maturity-honesty-fence.md) — the sibling standards in `docs/standards/`.
- [Claims ledger](../../CLAIMS.md) — shipped vs stub, claim by claim.
