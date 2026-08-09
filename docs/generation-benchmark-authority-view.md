---
title: "Generation-Aware Benchmark Authority View"
description: "A derived view over BENCHMARK-AUTHORITY.md that slices every committed number by horizon (which generation the number speaks for), witness type (the rung its evidence clears), and promotion relevance (whether the number moves a claim toward now). Answers one question the authority table cannot: does this number prove current value, or future potential?"
---

# Generation-Aware Benchmark Authority View

**Issue:** #1669.
**Parent:** #1625.
**Stream:** `gen/future`.
**Milestone:** Generation G3 - Future.
**Status:** design memo — a view *specification* over existing rows. It defines
no new numbers, no new provenance vocabulary, and no runtime gate. A later
`gen/next` stream may render it as a `fak` verb or a lint; today it is doc-only
and inert by design.

This memo is the handoff a future agent can use without rereading the whole
generation epic. It answers one question [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md)
raises but does not resolve: the authority table proves each number is **true**;
it does not say what each number **entitles you to claim**. A reader cannot tell,
from the table alone, whether a row demonstrates value the product delivers today
or potential a future rung would unlock. Both are honest rows. They are not
interchangeable in a sentence.

The canonical stream taxonomy lives in [`docs/generation.md`](generation.md). The
per-generation minimum-evidence rungs live in
[`docs/generation-witness-ladders.md`](generation-witness-ladders.md) — this memo
consumes that ladder's **benchmark column** rather than restating it. The process
that mints and verifies a row is [`BENCHMARK-GOVERNANCE.md`](../BENCHMARK-GOVERNANCE.md).

## The gap

The authority table's columns are `Claim | Number | Model | Baseline | Commit |
Artifact`. Every column answers *is this number real?* None answers *what horizon
does this number speak for?* Two rows that sit adjacent today:

- **fak CPU Q8 single-stream vs llama.cpp** — decode `0.55–0.73×`. Measured,
  reproducible, artifact-backed. It is a **current-value** number, and it records
  a **loss**.
- **Speculative decode** — `E(K=4) = 1.00× → 5.00×` (full-accept ceiling),
  `lossless=true`, bit-exact. Also real, also witnessed. But the row states
  plainly: *"NOT wall-clock tokens/sec: there is no GPU here, so the on-hardware
  2–3× tokens/sec headline stays HW-gated (a measured number needs the #535 bench
  harness on a GPU)."* It is a **future-potential** number.

Both rows are honest. The second is, by some measures, the *stronger* witness —
it is deterministic and bit-exact, where the first is a wall-clock ratio. So
"future potential" cannot mean "weaker evidence." That conflation is the thing
this view exists to kill.

## The three slice axes

### 1. Horizon — but derived, not asserted

The load-bearing move: a row's horizon is **computed from its witness**, not
declared by its prose. Two distinct quantities:

- **Entitled horizon** — the strongest generation the row's evidence supports,
  read straight off the benchmark column of the witness ladder
  ([`docs/generation-witness-ladders.md`](generation-witness-ladders.md)).
- **Claimed horizon** — the horizon the row's prose (and anything citing it)
  actually asserts.

The single invariant this view enforces:

> **`claimed ≤ entitled`.** A row may always under-claim. It may never
> over-claim.

`claimed > entitled` is **horizon laundering** — the failure mode
[`docs/generation-public-narrative.md`](generation-public-narrative.md) bans in
public copy, here given a mechanically checkable form on the authority rows
themselves. Because `entitled` is a function of the witness rung, the check is
derivable rather than editorial: it needs no new metadata beyond what the
governance status block and the provenance labels already carry.

This is also why generation is **not** a quality axis. A `gen/future` row can be
`VERIFIED` (spec-decode's bit-exactness is independently reproducible) while a
`gen/now` row is merely `MEASURED`. Horizon says *which question the number
answers*; the witness rung says *how strongly it answers it*.

### 2. Witness type — the existing vocabulary, unchanged

This view introduces **no new provenance words**. It reads what is already there:

- Governance status: `THEORETICAL` | `MEASURED` | `VERIFIED`
  ([`BENCHMARK-GOVERNANCE.md`](../BENCHMARK-GOVERNANCE.md), the Measurement
  Status Policy).
- Finer provenance: `MODELED`, `SIMULATED`, `OBSERVED`, `WITNESSED`.
- Enforcement: [`tools/check_provenance_labels.py`](../tools/check_provenance_labels.py)
  — a MODELED number labeled "measured" is not a low rung, it is a **broken**
  witness, and it fails CI at every generation.

Mapping witness → entitled horizon (the benchmark column of the ladder, restated
only as a lookup, not redefined):

| Witness the row carries | Entitled horizon |
|---|---|
| `THEORETICAL` / `MODELED`, inputs stated, labeled projected | `gen/future` |
| Simulation or fixture micro-measurement, counterfactual named, still projected | `gen/second-next` |
| `MEASURED` on a real (if small) workload, reproducible from a named command, baseline named | `gen/next` |
| `MEASURED`/`VERIFIED` + baseline + N + command, provenance-labeled `WITNESSED`/`OBSERVED`, passing the provenance guard; regression-gated if it claims a default | `gen/now` |

### 3. Promotion relevance — a closed four-token set

Not every true number is *doing work*. Promotion relevance says whether the row
moves a claim toward `now`, defends one already there, is waiting on a named
gate, or is a tombstone:

| Token | Meaning | Must name |
|---|---|---|
| `PROMOTING` | The number is the witness that would retire a named blocker and move a claim one generation toward `now`. | The blocker it retires. |
| `HOLDING` | The number substantiates a claim already at its entitled horizon. It defends the row; it does not advance it. | The default or regression gate it defends. |
| `GATED` | The row exists, deliberately, with its number **withheld** pending a named gate. | The exact gate, by issue number. |
| `RETIRED` | Stale, superseded, or retracted. Zero promotion relevance; kept as provenance/tombstone only. | The row that superseded it, or the retraction reason. |

`GATED` is the highest-integrity future-potential row, not the weakest: it
publishes the *absence* of a number together with the exact condition that would
produce one.

## The view, against real rows

A derived read — **it claims no new numbers**, the same discipline the hero
comparison follows. Every cell below traces to a row already in
[`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md) or a page it indexes.

| Authority row | Witness | Entitled | Promotion relevance | Proves |
|---|---|---|---|---|
| RadixAttention live speedup `4.58× → 6.95×` | `MEASURED`, artifact + commit, reproducible | `gen/now` | `HOLDING` — defends the reuse default | **current value** |
| fak CPU Q8 decode `0.55–0.73×` vs llama.cpp | `MEASURED`, artifact-backed, both engines at best thread config | `gen/now` | `HOLDING` — a defended **loss** | **current value** |
| Speculative decode `E(K=4) → 5.00×` ceiling | `VERIFIED` (bit-exact, deterministic) but **no GPU**; wall-clock explicitly not claimed | `gen/future` | `PROMOTING` — blocker named: the #535 GPU bench harness | **future potential** |
| Qwen3.6-27B Metal Q4_K decode `1.2 tok/s` | `MEASURED`; row states *"Still trails SOTA; no pass claimed"* | `gen/now` | `PROMOTING` — blockers named (#1113/#69, weight upload + round-trip) | **current value** (a measured deficit) |
| [FrontierSWE time-to-solution](benchmarks/FRONTIERSWE-RESULTS.md) | none — page created **empty and gated** | `gen/future` | `GATED` — official grader (#1719) + score parity (#1717) | **future potential** |
| ~~Session value-add `11.2–14.5×`~~ | ❌ `STALE`, superseded by a re-measured row | — | `RETIRED` | nothing |
| Compaction per-session share `~15%→~75%` | **RETRACTED** — double-counting artifact | — | `RETIRED` | nothing |

Read the last two rows as the view working as intended: the authority already
carries its own demotions. This view names them as a *slice* rather than leaving
them as strikethrough prose a citing agent can miss.

## The rule this view exists to state

> A row proves **current value** if and only if its **entitled horizon is
> `gen/now`** — a captured, reproducible measurement with baseline, N, and
> command, provenance-labeled, passing the provenance guard. Every other row
> proves **future potential**, and every citation of it must carry the fence the
> row already states.

Note what this does *not* say. It does not say a current-value row is good news
(`0.55–0.73×` is a loss). It does not say a future-potential row is weak
(spec-decode's witness is bit-exact). And it does not rank the two. It says only:
**they answer different questions, and a sentence may not silently swap one for
the other.** That fence is the same one
[`docs/standards/net-true-value.md`](standards/net-true-value.md) draws around a
gain claim; this memo draws it around a *horizon* claim.

## Orthogonality (the generation invariants this artifact must restate)

The view is metadata and a reading discipline — not a branch, a priority, or a
runtime switch.

- **Orthogonal to priority.** A slice is not a value judgment. A `GATED`
  `gen/future` row (FrontierSWE) can be the single most important number the
  project does not yet have, while a `HOLDING` `gen/now` row defends a settled
  default nobody thinks about. Horizon sets *which claim the number licenses*,
  never *how much the work matters*. `gen/future` is a horizon label, never
  "lower priority" — matching #1669's non-goal.
- **Orthogonal to shared trunk.** The view is a derived read of rows that all
  land on `main`, by explicit path, under one DCO and ship-stamp rule. There is
  no per-generation benchmark file, no `gen/future` results branch, and no
  side worktree for long-horizon numbers — #1669's other non-goal. Slicing the
  authority never forks it.
- **Orthogonal to runtime feature gates.** The entitled horizon decides *what a
  number licenses you to say*; a feature gate decides *whether the code is
  reachable at runtime*. Spec-decode is gated off (`FAK_POLYMODEL`, default-off)
  **and** its row is `gen/future` — but these are two independent facts. Landing
  a measured GPU number would promote the row's horizon without touching the
  gate; flipping the gate on would not promote the row by one rung. Neither
  substitutes for the other.
- **Orthogonal to completion percentage.** A slice is not a progress bar. The
  reuse and kernel programs are ongoing optimization programs: they report a
  frontier at whatever rung their rows currently clear. Counting `gen/now` rows
  as a "percent done" would re-import exactly the category error
  [`docs/generation.md`](generation.md) forbids.

## Promotion evidence (future → second-next → next → now)

This memo promotes when the view is **computed**, not merely described:

- **future → second-next:** classify every row currently in
  [`BENCHMARK-AUTHORITY.md`](../BENCHMARK-AUTHORITY.md) by hand into the three
  axes, and find at least one row where `claimed > entitled`. A single real
  laundering hit is the promotion witness — it proves the view detects something
  prose review missed. Finding *zero* is also a result, and it demotes (below).
- **second-next → next:** a read-only `fak` verb (e.g. `fak bench authority-view
  --json`) that parses the authority rows plus their governance status blocks and
  emits `(row, witness, entitled, relevance)`. It reuses
  [`tools/check_provenance_labels.py`](../tools/check_provenance_labels.py)'s
  label extraction as its first wired input — that guard already reads the only
  field `entitled` depends on. Read-only, default-off, no CI wiring.
- **next → now:** that verb runs in `make ci` and **refuses** a commit whose
  authority row asserts a claimed horizon above its entitled one — the
  `claimed ≤ entitled` invariant as a gate, in the same family as the existing
  `FRONTIERSWE_SCORE_PARITY_FAILED` refusal, which is precisely a promotion gate
  that already blocks a time-to-solution win claim until the evidence clears.

## Demotion / retirement evidence

- **Demote** to a recut if hand-classification (the `future → second-next` step)
  finds **no** row where `claimed > entitled`. That would mean the authority's
  existing prose fences and the provenance guard already prevent laundering, and
  the third axis is decoration. The honest response is to narrow the memo to the
  promotion-relevance axis alone, not to wire a gate that can never fire.
- **Demote** if `entitled` proves not to be a function of the witness — i.e. if
  classifying real rows requires a human judgment call the ladder does not
  supply. Then the view is not derivable, and it must return to `gen/future` for
  a redesign around explicit per-row metadata instead of derivation.
- **Retire** the memo if [`BENCHMARK-GOVERNANCE.md`](../BENCHMARK-GOVERNANCE.md)
  absorbs the horizon column directly into its Measurement Status block (a fifth
  status, or a `horizon:` field). Then the slice is a property of every row at
  mint time and there is no separate view left to carry — the correct end state,
  and this doc collapses into a pointer.
- **Retire** if the authority stops mixing horizons — if every row becomes a
  shipped `gen/now` default — which would make the distinction vacuous. (The
  `GATED` FrontierSWE page makes this unlikely soon.)

Retirement of this memo follows the closed sunset-trigger vocabulary and cadence
in [`docs/generation-future-sunset-criteria.md`](generation-future-sunset-criteria.md);
the two `Retire` conditions above map to `SUPERSEDED` and `STRIKE_UNREACHABLE`
respectively. Retirement is by named trigger with a witnessed trail, never by
label movement alone.

## Invalidating assumptions (kill criteria)

State them so a later agent can check them cheaply:

1. **Entitled horizon is a pure function of the witness rung.** The whole design
   — and the entire promotion path to a computed verb — rests on being able to
   derive `entitled` from the governance status + provenance label + presence of
   a baseline/command, with no human judgment. **This is the assumption most
   likely to fail.** Several real rows carry a *mixed* witness: the fleet
   cache-savings row is `WITNESSED` for shed tokens but `MODELED` for the
   percentage share, in one row. If a row's witness is not a scalar, `entitled`
   is not a function, and the view needs per-*claim* rather than per-*row*
   granularity. Check it against that row first; it is the known hard case.
2. **Three axes are enough, and they are independent.** The memo assumes horizon,
   witness type, and promotion relevance do not collapse into each other. They
   plausibly do at the extremes: is a `RETIRED` row's entitled horizon even
   defined? Is `GATED` a promotion-relevance value or actually a *witness* value
   (the witness being "none, deliberately")? If `GATED` belongs on the witness
   axis, the third axis loses its most interesting member and the design needs a
   recut.
3. **Regimes do not cross horizons.** Governance already forbids mixing regimes
   (a live speedup is not a session value-add). This memo assumes horizon slicing
   composes *on top of* regime boundaries rather than cutting across them. If a
   single regime turns out to span horizons in a way that makes two rows look
   comparable when they are not, the view would license exactly the comparison
   governance bans — and the fix is to make the view refuse to render two rows
   side by side unless regime **and** horizon match.

## First classification pass (sourced, mechanical)

Step 1 of the handoff below assumed hand-classification. It does not need one: a
**structured record already exists** — [`docs/benchmarks/registry.jsonl`](benchmarks/registry.jsonl),
52 claims, one JSON object each, carrying two independent enums that are exactly
the witness axis this memo reads:

- `provenance` ∈ `measured` | `functional` | `modeled` | `unknown`
- `status` ∈ `canonical` | `live` | `gated` | `pending` | `stale` | `retracted`

Crossing them (reproduce below) gives the first pass:

| Finding | Count |
|---|---:|
| `provenance: modeled` in a **citable** status (`canonical` 1 + `live` 5) | 6 |
| `canonical × modeled` — top "quote this" tier, witness is a geometry model | **1** |
| `unknown` provenance (`gated` 1, `pending` 1, `retracted` 4) | 6 |
| `stale` / `retracted` — `RETIRED`, zero promotion relevance | 5 |

```bash
python -c "
import json,collections
R=[json.loads(l) for l in open('docs/benchmarks/registry.jsonl',encoding='utf-8') if l.strip()]
print(dict(collections.Counter((r['status'],r['provenance']) for r in R)))
"
```

**Result: one structural candidate, zero confirmed prose hits.** The single
`canonical × modeled` row (`webbench-webvoyager-hero`) is entitled to `gen/future`
by the ladder above (`MODELED` → projected), yet the record files it at
`canonical`, `tier: 1` — the record's own words for *the number to quote*. Its
prose, however, fences itself correctly ("MODELED geometry, no model … not
measured"), and so do the other five modeled rows.

So the hit is **structural, not editorial**: the prose fences hold, but the
`status` field — which a citing agent reads *first*, and which outranks a fence it
can skim past — encodes an authority the witness does not entitle. That is a
weaker result than "prose laundering found," and a stronger one than the null
result that would demote this memo: it relocates the laundering risk from the
prose (already guarded by
[`tools/check_provenance_labels.py`](../tools/check_provenance_labels.py)) to the
**status/provenance pair, which no view reconciles.** A row's citability and its
witness are set independently, and nothing renders them together.

This sharpens, rather than replaces, the promotion path: the derived column is
computable *today* from the committed record, with no new metadata — which is
assumption 1 surviving its first contact with real data on 46 of 52 rows. The 6
`unknown`-provenance rows are where it has not yet been tested, and the
mixed-witness cache-savings row (assumption 1's named hard case) does not appear
in the structured record as a separable claim at all — which is itself the
evidence that per-*claim* rather than per-*row* granularity is the real
requirement. Issue #3431 resolves that design decision in
[`notes/PER-CLAIM-BENCHMARK-WITNESS-GRANULARITY-2026-08-09.md`](notes/PER-CLAIM-BENCHMARK-WITNESS-GRANULARITY-2026-08-09.md):
mixed rows get optional stable assertions with their own status and witness kind,
while the scalar row remains a conservative compatibility envelope.

## Handoff (continue from here without the epic)

A future agent picking this up should, in order:

1. ~~**Hand-classify the authority rows**~~ — **done mechanically above** against
   `registry.jsonl`. Do not redo it by hand. The residual is the 6
   `unknown`-provenance rows and the mixed-witness cache-savings row, which the
   structured record does not decompose.
2. ~~**Decide per-claim versus per-row witness granularity**~~ — **done** in the
   [#3431 decision note](notes/PER-CLAIM-BENCHMARK-WITNESS-GRANULARITY-2026-08-09.md).
   Adopt optional claim-level assertions only for mixed rows, keep a conservative
   scalar roll-up for old consumers, and add `simulated` at claim level to recover
   the currently-collapsed `gen/second-next` rung.
3. **Reconcile `status` against `provenance`.** The one `canonical × modeled` cell
   is the concrete target: either demote its status, or render every row as
   `status (entitled: <horizon>)` so the pair cannot be read apart. A CI check
   that merely *reports* the `canonical × modeled` count is the cheapest first
   gate and would have caught this cell without a human.
4. **If assumption 2 survives**, build the read-only `fak bench authority-view
   --json` verb before wiring any gate. Derive first, refuse later; a gate on an
   underived column would hard-code the very human judgment the design claims to
   have eliminated. Read it from `registry.jsonl`, or from the typed seam already
   in the tree: [`internal/benchauthority/benchauthority.go`](../internal/benchauthority/benchauthority.go)
   defines a `Claim` with both `Status` and `Provenance` fields, and
   [`internal/benchauthority/registry.go`](../internal/benchauthority/registry.go)
   holds `var registry = []Claim{}` — a deliberately empty seed awaiting exactly
   this transcription.

The hub ([`docs/generation.md`](generation.md)) stays the front door. The witness
rungs ([`docs/generation-witness-ladders.md`](generation-witness-ladders.md)) stay
the source of `entitled`; this memo is the *authority-surface projection* of that
ladder's benchmark column — the answer to "which of these numbers proves the
product works today, and which one proves it could."
