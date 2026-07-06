---
title: "Triage + contract: R8 readout that proves the doctrine — overlay counter + one-ledger join (#2206)"
description: "Generation-horizon classification and mechanism-vs-witness gap for the autoctx R8 rung — the operator readout that renders 'nobody managed context this period and nothing was silently lost' from witnessed records only. Verified at HEAD 2026-07-06: R8 is genuinely blocked — the #1918 base readout verb does not exist and the R2 residency+warmth one-ledger join (#2200, reopened by the owner today) has no live record to re-derive from. This note classifies the rung, records the block, and specifies the exact three-row readout extension contract so the next builder starts warm. Triage + contract only; #2206 stays OPEN."
date: 2026-07-06
---

# Triage + contract: R8 readout that proves the doctrine (#2206)

Status: **triage + contract only — #2206 stays OPEN.** This note records the
generation-horizon decision, the mechanism-vs-witness gap verified at HEAD, and
the precise readout-extension contract R8 must satisfy, so the next dispatcher
(Claude or opencode) starts warm instead of re-deriving the map. It does **not**
implement R8: the rung *extends* a readout verb that does not exist yet (#1918)
and reads a residency+warmth join ledger that does not exist yet (#2200,
reopened 2026-07-06), so its witness — one screen re-deriving the doctrine
sentence from witnessed records — is unreachable today and cannot be honestly
claimed done or faked.

Date: 2026-07-06. Lane routed: `docs` (the issue's file-tree). Labels on the
live issue at triage time: `managed-context`, `generation`, `gen/next`; milestone
`Generation G1 - Next Gen`. Parent epic: [#2198](https://github.com/anthony-chaudhary/fak/issues/2198)
(spine: [`CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md`](CONCEPT-AUTOMATIC-CONTEXT-2026-07-01.md)),
rung R8 of R1–R8.

## The ask (verbatim invariant)

> Extend the readout with: the manual-overlay counter and its trend; the joined
> residency+warmth ledger roll-up (provenance-separated); and the abstentions
> (where the kernel declined to manage automatically, with structured reasons).
> Every number re-derives from journal/metrics/ledger rows — no self-report
> fields.

Done condition: one screen renders "nobody managed context this period and
nothing was silently lost" (or its honest negation) from witnessed records only.
Witness: the C9-style re-derivation check passes on the rendered numbers, and
deleting a ledger row makes the screen refuse green with a structured reason.

## Generation horizon: `gen/next`

Classified from issue evidence, not guessed (the dispatch frame's proof bar), and
matching the owner's own classification comment on #2206:

- The parent epic **#2198** files R1–R8 as a `managed-context` foundation; R8 is
  the terminal rung whose whole job is to *witness* the doctrine, filed as
  `feat(autoctx)`.
- R8's Done condition is a **default-exposure/observability proof** — one screen
  re-deriving the doctrine sentence — which is the `gen/next` shape in
  [`docs/generation.md`](../generation.md) ("near-term foundation … still needs a
  gate, dogfood run, schema, or default-exposure proof"), not a `gen/now` win on
  the current loop.
- It is **not** `gen/now`: two of the three inputs the readout folds do not exist
  as witnessed records at HEAD (see the gap below), so the screen the issue names
  cannot be produced today.

## Mechanism-vs-witness gap (why it stays OPEN, verified at HEAD 2026-07-06)

R8 folds **three** inputs. Exactly one is live; the other two are absent, which is
why the rung is genuinely blocked rather than build-ready.

- **R1 — the manual-overlay counter: SHIPPED.** `internal/ctxknobs/` scans the
  tree for context flags/env/skills and classifies each `operator-debug` vs
  `user-required` (the ratchet); `fak index ctxknobs` renders it
  (`cmd/fak/index.go`, `indexCtxKnobs`). This input is real and queryable — the
  overlay-counter row can re-derive today.
- **R2 — the residency+warmth one-ledger join: ABSENT (blocked input).** #2200 was
  **reopened by the owner on 2026-07-06** with a HEAD verification: the one-ledger
  join does not exist — `internal/cachevalueledger/ledger.go` carries no joined
  record (the WITNESSED/OBSERVED strings there are honesty-fence comments, not a
  join), and no per-turn record joins residency actions with warmth deltas. The
  correlation *fold* exists as a pure function (`internal/vcacheobserve/contextjoin.go`,
  `JoinContext` → `fak.vcache.contextjoin.v1`) but it has **no persisted ledger of
  rows** to roll up. So "the joined residency+warmth ledger roll-up
  (provenance-separated)" has nothing to re-derive from.
- **Base #1918 readout verb: ABSENT (blocked input).** #1918 is still OPEN. The
  readout is fully *specified* — `docs/managed-context-continuous-usage.md`
  (§ Operator Readout) gives the exact seven-row screen — but there is **no `fak`
  verb in `cmd/fak/` that renders it**. R8 "extends" a render surface that does not
  exist yet; a source search for a readout command finds none.

R8 is therefore blocked on **#1918** (the base verb to extend) and **#2200** (the
join ledger to roll up). Its witness is unreachable until both land.

## The R8 readout-extension contract (build this once #1918 + #2200 land)

R8 adds **three rows** to the #1918 screen (spec in
`docs/managed-context-continuous-usage.md` § Operator Readout — seven rows:
`objective / budget / resident_view / assumptions / reset / memory / cache`), plus
one headline verdict line. Every field re-derives from an existing row; no
self-report field is permitted (that is the load-bearing invariant, and the
delete-a-row-refuses-green test is what enforces it).

```text
overlays:    user_required=<n> operator_debug=<n> trend=<±n vs prior period>
one_ledger:  WITNESSED{shed=<n> elide=<n> page_out=<n>} OBSERVED{cache_read=<tok> reuse=<n>} $=<dollars>
abstentions: <reason>=<n> …            # closed-vocabulary reasons only; empty ⇒ none
verdict:     nobody managed context this period and nothing was silently lost
             | <honest negation naming the offending count or missing witness>
```

Row-by-row derivation contract:

1. **`overlays`** re-derives from R1: `ctxknobs.Scan` → `UserRequired` /
   `OperatorDebug` counts; `trend` is this period's `user_required` minus the prior
   period's (the ratchet must only fall — a rising `user_required` is an honest
   negation, not a green screen). No new counter: read `fak index ctxknobs`.
2. **`one_ledger`** re-derives from R2's per-turn join record — **provenance kept
   separate**: WITNESSED (fak's own shed/elide/page-out actions) and OBSERVED
   (provider `cache_read`, reuse) are rendered as distinct groups and **never
   summed** (the vanity-metric failure mode; law L4 / the spine's honesty fence).
   Blocked until #2200 lands the record type.
3. **`abstentions`** re-derives from the structured-reason rows the kernel already
   emits where it *declined to manage automatically* (law L6 "abstain honestly" —
   a closed-vocabulary reason, never a silent degrade). The readout renders those
   reasons as counts; it does not invent reason text.
4. **`verdict`** is a pure fold of rows 1–3 plus the base screen: green **iff**
   `overlays.user_required == 0` for the default path, no `one_ledger` row shows
   silent loss (every removal is a named tier with a witness, law L3), and the
   re-derivation check reconciles. Any failing predicate flips the line to its
   honest negation naming the specific offending count or the missing witness.

Witness the builder must ship with it: (a) a **C9-style re-derivation check** that
recomputes every rendered number from the source rows and asserts equality
(no field is a stored self-report); (b) a test that **deleting one ledger/journal
row makes the screen refuse green with a structured reason** — proving the screen
cannot render continuity it cannot witness (the `not yet` escape the base spec
already names at `docs/managed-context-continuous-usage.md:142`).

## Triage evidence (generation contract)

- **Promotion evidence** (what moves R8 toward `now`): FIRST #2200 lands its
  per-turn residency+warmth join record (WITNESSED/OBSERVED provenance-separated,
  appended beside `cachevalueledger` rows) AND #1918 ships a `fak` verb that
  renders the seven-row base screen. THEN R8 reduces to a small additive rendering
  rung: three rows + a verdict fold + the two witnesses above. This is exactly the
  owner's stated promotion trigger on #2206 ("once both a residency+warmth roll-up
  and a base readout exist, R8 becomes a small additive rendering rung with a real
  C9 re-derivation witness — ready to build").
- **Demotion / retirement evidence** (what pushes it back or closes it): #1918 is
  superseded by a different readout surface (e.g. `fak productscorecard` absorbing
  the per-axis readout), leaving R8 nothing to extend; or the WITNESSED/OBSERVED
  separation contract in #1607/#2200 is reopened, since R8's provenance-separated
  roll-up rides on it.
- **Invalidating assumption** (named, per the frame): "R8 renders *on top of* the
  #1918 verb reading R2's join." If an operator decides to **re-split** #2206 and
  ship a *counter-only* slice — a minimal screen reading only the already-shipped
  R1 `ctxknobs` counter, dropping the R2-dependent `one_ledger` row and the
  abstentions to a later rung — that slice could promote independently and does not
  need #1918/#2200. That re-split is an operator decision, not a dispatch worker's
  to make unilaterally; this note keeps R8 whole and blocked, and flags the slice
  as the available narrowing if an operator wants it.

## Mis-close caution (2026-07-06)

This is a **triage + contract** note; it does not resolve #2206. The sibling rungs
#2200 (R2) and #2205 (R7) were each **mis-closed** by the dispatch loop's
close-resolved arm citing a `docs(...)`-typed triage commit that merely carried
`(#N)` in its subject — the owner reopened #2200 today for exactly this. A
`docs(...)`-typed / triage-verbed commit referencing `(#2206)` must **not** arm
close-resolved for #2206: R8's witness (one screen re-deriving the doctrine
sentence from witnessed records, with the delete-a-row-refuses-green test) does
not exist, and its #1918/#2200 prerequisites are still absent. **#2206 stays
OPEN.** Its body's "Blocked by" set should read #1918 (base verb) + #2200 (join
ledger), not a solved dependency. This commit lands the classification + contract
note only — no build-state change.
