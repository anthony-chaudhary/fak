# Context-safety re-derivation checker (#1227, C9 of epic #1217)

_Research / design only. This is the **C9 keystone spec** for the context-safety
epic [#1217](https://github.com/anthony-chaudhary/fak/issues/1217) — the
genuinely missing piece the doctrine names. It specs the **re-derivation
checker**: the contract under which a rendered context-safety visual is **RED
unless every number on it re-derives from a tamper-evident source**. No code
ships here — the deliverable is this committed contract: what "re-derive" means,
which sources are tamper-evident, the in-tree pattern it generalizes, and the
structured refusal it emits when a number can't be re-derived. It closes
doctrine mechanism **D** of the C1 note
([#1218](https://github.com/anthony-chaudhary/fak/issues/1218),
[`context-safety-visuals-tracking-1217.md`](context-safety-visuals-tracking-1217.md))
and the **G-A honest gap** of the C2 catalog
([#1219](https://github.com/anthony-chaudhary/fak/issues/1219),
[`context-safety-failure-mode-catalog-1217.md`](context-safety-failure-mode-catalog-1217.md)).
Its CI-gate sibling is C10 ([#1228](https://github.com/anthony-chaudhary/fak/issues/1228))._

---

## The missing piece, in one line

> fak has the numbers and a tamper-evident place they come from. What it does
> **not** have is a checker that, given a rendered visual, **re-derives every
> number on it from that source and reds when they disagree.** A visual rendered
> at scrape time and trusted thereafter is a self-report wearing a chart's
> authority. This checker turns the visual into a *witness*.

This is the confirmed gap (#1217 fact 2; C2 catalog gap G-A): rendering today is
downstream (Grafana over `/metrics`, Jekyll over scorecard JSON) and **never
re-validates its numbers from the source** after render. The journal *is*
replayable and the index *does* prove a round-trip — but no checker extends that
discipline to a **rendered visual**. C9 specs that checker.

---

## What "re-derive" means — the rebuild-and-prove-equal contract

A number on a context-safety visual is **green only if** an independent
derivation, run by the checker against a tamper-evident source, produces a value
that **equals** the rendered one. Not "is close to"; equals (within the datum's
declared tolerance — `max|Δ|=0` for bit-identity, exact integer match for token
counts, a named epsilon for ratios). If the re-derivation differs, or cannot be
performed, the number is **RED** — never silently trusted.

This is the **rebuild-and-prove-equal** discipline, already proven in-tree for
the *index* and here generalized to the *rendered visual*:

- **The in-tree precedent (the index).** `internal/ctxplan/image.go` never
  trusts derived state: the loader rebuilds the index from its image
  (`Image()` `:61-63`, `RestoreIndex()` `:71-76`) and proves
  `RestoreIndex(ix.Image()) == ix` (the round-trip equality, witnessed by
  `image_test.go:18` `TestIndexImageRoundTripIsIdentity`, a `reflect.DeepEqual`).
  The derived form is only trusted because it was rebuilt and proven equal to
  the source.
- **C9 generalizes it.** Replace "index" with "rendered visual" and "image"
  with "the rendered numbers": the checker rebuilds every rendered number from
  source and proves it equals what the visual shows. The one analog the tree
  does not yet have.

---

## The tamper-evident sources the checker re-derives from

A re-derivation is only as trustworthy as the source it rebuilds from. The
checker is restricted to sources fak can prove were not edited after the fact:

| Source | Why tamper-evident | What it grounds |
|---|---|---|
| The hash-chained journal `internal/journal/journal.go` | `Seq`/`PrevHash`/`Hash` chain; `Verify(path)` / `VerifyRows(rows)` recompute the chain from the rows themselves — an edited row breaks the hash | the **event** half: every `CAP_EVICT`/`CAP_FAULT`/`CAP_VERSION_BIND` row (ribbon, primitive 1; the gap's D1 anchor) |
| `/metrics` (the WITNESSED `fak_gateway_*` / `fak_engine_*` family) | monotonic counters; a re-scrape re-reads the same accumulator, not a cached render | the **realized** half: `fak_gateway_kv_prefix_reused_tokens_total` (gap D2), the decode-rate accumulator (liveness light) |
| git (commit DAG) | content-addressed; `dos commit-audit` already re-derives claim-vs-diff from it | provenance and any number sourced from committed scorecard JSON |
| disk CAS bodies | digest-addressed; re-hashing the body re-derives the digest (`recall.PageSyndromeFor`) | the G4 cell-integrity half |

The checker **may not** re-derive from a rendered artifact, a cached snapshot, or
any number a producer could have edited after emission. If the only available
source for a number is non-tamper-evident, that number is **not re-derivable**
and the panel refuses green (below).

---

## The checker contract

```
RederivationVerdict {
  Visual      string            // which primitive (1..6) / panel id
  Numbers     []NumberCheck     // one per rendered number on the visual
  AllGreen    bool              // ∀ n: n.Status == GREEN
  Refusal     string            // "" | SAFETY_UNRECOVERED | VALUE_UNRECONCILED | SAFETY_UNWITNESSED
}

NumberCheck {
  Label       string   // what the number is ("ClaimedSaved", "max|Δ|", "ReuseRatio", ...)
  Rendered    float64  // the value the visual shows
  Rederived   float64  // the value the checker rebuilt from source
  Source      string   // "journal" | "metrics" | "git" | "cas" — must be tamper-evident
  Tolerance   float64  // 0 for bit-identity / integer counts; a named epsilon for ratios
  Status      string   // GREEN (|Rendered-Rederived| <= Tolerance) | RED (mismatch) | REFUSED (not re-derivable)
}
```

**The contract, stated as the acceptance property:**

1. **Every rendered number has a `NumberCheck`.** A number with no re-derivation
   path is `REFUSED`, not absent — an un-checked number may not render green.
2. **GREEN requires equality within tolerance** against a re-derivation from a
   **tamper-evident** `Source`. `RED` on any mismatch.
3. **The visual is GREEN iff every number is GREEN.** One RED or REFUSED number
   reds the whole visual — a visual is only as trustworthy as its weakest
   number (the conservation discipline: a roll-up whose parts don't re-derive is
   visibly broken).
4. **A number that cannot be re-derived refuses green with a structured DOS
   reason** (C13, #1231), not an unearned OK:
   - `SAFETY_UNRECOVERED` — the event happened but no tamper-evident witness
     records it (e.g. an un-recovered panic left no journal row).
   - `VALUE_UNRECONCILED` — two derivations that must reconcile disagree (the
     gap's reuse *count* vs the C14 coherence witness).
   - `SAFETY_UNWITNESSED` — the number has no tamper-evident source at all.
   Each token is verified with `dos_check_reason` before emission (C13 owns the
   token definitions; C9 names which condition fires which).

---

## In-tree precedents this contract generalizes

C9 is not a greenfield invention — it lifts a discipline fak already practices
in four places into a single Go contract over rendered visuals:

- **`ctxplan/image.go` `RestoreIndex(ix.Image()) == ix`** — rebuild-and-prove-equal
  for the index (the direct structural ancestor; C9 extends it to visuals).
- **`internal/journal/journal.go` `Verify` / `VerifyRows`** — recompute the hash
  chain from the rows; the event half's tamper-evidence.
- **`.claude/skills/memory-compact/check_memory.py`** — re-derives the memory
  index from disk (byte/line counts of `MEMORY.md`, the `](*.md)` link set vs the
  on-disk `*.md` glob, the orphan bijection) and compares (the
  re-derive-from-source pattern, no Go analog for visuals yet).
- **`tools/dispatch_status.py` `closure_rate`** — strict **diff-witnessed**: an
  issue counts as resolved only when a DOS-verified commit witnesses it
  (`TRUE_RESOLVED / (TRUE_RESOLVED + CLAIMED_CLOSED)`), never the self-report.
- **`internal/gateway/metrics.go` `writeVCacheMetrics`** — the `proven`
  break-even gate that **refuses to claim value until break-even is witnessed**
  (the refuse-green-rather-than-render-unearned discipline, already shipped for
  one metric; C9 makes it the rule for every rendered number).

`fak audit verify` already recomputes the journal chain on demand — C9's checker
is the visual-facing analog: where `audit verify` proves the chain is intact,
the re-derivation checker proves the *picture* still equals the chain.

---

## Honest `not yet` gaps (per fence c)

- **No Go checker re-derives a rendered visual today.** This is the gap itself
  (C2 catalog G-A). This note is the *contract*; the implementation is the
  follow-on build epic, gated on review of the #1217 design notes.
- **D1 (claimed-saved) is not yet pinned to the journal row** (named in the C4
  spec). Until it is, the gap's claimed half re-derives from the metric stream,
  which is WITNESSED but not chain-anchored — a weaker tamper-evidence than the
  `Hash`-chained row. The checker treats a metric-only source as GREEN-eligible
  but flags it as lower-assurance than a journal-anchored number.
- **The CI gate that reds `make ci` on a persistent mismatch is C10 (#1228),
  not this note.** C9 defines the per-visual verdict; C10 defines when a RED
  verdict fails the build (with the false-red fence: change-point / persistence,
  not a single tick).

---

## Acceptance check (against #1227)

The issue's acceptance: *a contract for "the visual is RED unless every number
re-derives from a tamper-evident source."*

- **The contract is stated** as `RederivationVerdict` / `NumberCheck` with the
  four-point acceptance property: every number checked, GREEN requires equality
  against a tamper-evident re-derivation, one RED/REFUSED number reds the
  visual, an un-re-derivable number refuses green with a DOS reason.
- **The tamper-evident sources are enumerated** (journal `Verify`/`VerifyRows`,
  WITNESSED `/metrics`, git, CAS digests) with why each resists post-hoc edits,
  and the rule that a non-tamper-evident source is not a valid re-derivation
  source.
- **The generalized pattern is named** —
  `RestoreIndex(ix.Image()) == ix` extended from the index to the rendered
  visual — with the four in-tree precedents it lifts (`image.go`, journal
  `Verify`, `check_memory.py`, `dispatch_status.py` diff-witnessed, the
  `proven` vCache gate).
- **The structured refusal** (`SAFETY_UNRECOVERED` / `VALUE_UNRECONCILED` /
  `SAFETY_UNWITNESSED`, C13) is bound to the condition that fires it, verified
  via `dos_check_reason`.

---

_Filed as research / planning only under epic #1217. Parent: #1217. Closes
doctrine mechanism D of the C1 note (#1218) and the G-A gap of the C2 catalog
(#1219); consumes the C4 saved-vs-realized gap (#1221) and the C14 coherence
reconciliation (#1232) as numbers it re-derives; emits the C13 DOS refusal
tokens (#1231); its CI-gate sibling is C10 (#1228). Design-only — no
implementation ships under #1217 until the notes are reviewed._
