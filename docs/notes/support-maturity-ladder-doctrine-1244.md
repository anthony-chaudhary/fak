# Support-maturity ladder M0–M7 — doctrine (#1244)

Part of epic **#1243**. This note is the doctrine for **C1**: the closed, ordered
support-maturity ladder defined as a Go enum in `internal/supportmaturity`. It draws
the line between that one ladder and each sibling "support" vocabulary it unifies, so
the project stops adding an (N+1)th parallel scale every time the question "is this
supported?" comes up.

## The gap

"Supported" is one fuzzy word collapsing a whole ladder — none / loads / runs /
correct / optimized / SOTA-parity / beyond-SOTA. That fuzz misroutes effort: a
10,000-step optimization loop gets pointed at something that actually only *loads*, or
a correct-but-slow path ships as if it were production-ready. The ladder replaces the
fuzz with one totally-ordered vocabulary.

## The ladder

A rung is an **envelope with a witness**, never a promise. It names the most a cell can
honestly claim given the witness it carries; a correct-but-slow cell is honestly M4,
not a failure.

| Rung | Label       | Means                                                        | Bound witness (C2 / #1245) |
|------|-------------|--------------------------------------------------------------|----------------------------|
| M0   | none        | undefined — a silently-reachable wrong-result path, or not even a parseable model | (the debt — nothing) |
| M1   | fenced      | an honest refusal: it refuses out loud rather than diverging | requirePreNorm / arch / capacity fence |
| M2   | loads       | header parsed, arch known, fits — safe to load               | `ggufload` preflight READY |
| M3   | runs        | runs and is correct on the scalar reference proof path; numeric claim asserted, not CI-proven | the cpu proof path |
| M4   | correct     | correctness witnessed by a CI-runnable oracle                | `OracleInCI` / `CorrectnessClass` gate |
| M5   | optimized   | accelerated fast path **and** a committed bench witnesses the speedup | `compute.Caps` + a bench |
| M6   | parity      | matches the SOTA-local baseline                              | turnbench `sota-local-baseline` |
| M7   | beyond-sota | beats the SOTA-local baseline                                | (open top — no current witness) |

The order is total by construction: `Rung` is a `uint8` and `M0 < M1 < … < M7`. The
witness test (`supportmaturity_test.go`) asserts the ladder is exactly eight rungs,
strictly increasing, and a strict total order (irreflexive + trichotomy). It also pins
each lowering's *band* — `TestLoweringBands` fails if `covmatrix.Support` ever maps
outside M0–M4 or a preflight verdict outside M0–M2 — so the prose claims below cannot
silently drift from the code.

## The line drawn against each sibling vocabulary

This package **unifies**, it does not add. Each existing scale lowers onto a band of
the one ladder; none of them is a separate ladder.

**`covmatrix.Support`** (UNDEFINED / FENCED / PROOF-PATH-ONLY / SUPPORTED) is the
*lower-band* scale — "is this (family, backend) cell present, honest, and does it run
on the reference?" It lowers losslessly, order-preserving:

- `UNDEFINED → M0` — the silently-reachable wrong path; this is `growth_debt`.
- `FENCED → M1` — the accelerated path refuses honestly (a fence is honest, not debt).
- `PROOF-PATH-ONLY → M3` — runs and is correct on the cpu reference, but no CI oracle.
- `SUPPORTED → M4` — its definition (`covmatrix.go`) is "runs **and** has a CI-runnable
  witness". `covmatrix.StaleCells` already flags the accelerated-SUPPORTED cells whose
  witness is in fact absent; binding that per-cell witness so such a cell **drops** to
  M3 is C2 (#1245). C1 maps the rung the value *claims*; C2 confirms or drops it.

M2 ("loads") is deliberately not produced by `covmatrix.Support` — it is owned by the
preflight verdict below. That is the unification working: the two scales meet on one
ladder instead of overlapping.

**`ggufload` preflight verdicts** (READY / REFUSE_TOO_BIG / REFUSE_BAD_ARCH /
REFUSE_BAD_HEADER) answer the narrower "can this model even load on this host?" They
lower onto the M0–M2 sub-band:

- `READY → M2` — header parsed, arch known, fits; safe to load.
- `REFUSE_TOO_BIG → M1` — an honest capacity fence (a fine model, a wrong-size device).
- `REFUSE_BAD_ARCH → M1` — an honest refusal (the arch / required keys are absent).
- `REFUSE_BAD_HEADER → M0` — not a parseable model; nothing is defined here.

Verdicts may share a rung (both REFUSE_* fences are M1). The ladder contract over the
preflight scale is *totality* (every verdict maps to exactly one rung), not injectivity
— unlike `covmatrix.Support`, which lowers losslessly.

**`compute.CorrectnessClass`** (Reference / Approx) is **not a rung** — it is the *bar*
for M4. Reference is held to bit-identity plus the HF argmax oracle; Approx to
argmax-exact plus a logit-cosine gate. Either, when its gate passes, witnesses M4. The
class says *how* correctness is judged, never *whether* a higher rung is reached. So
`FromCorrectnessClass` returns M4 for both, by design.

**`compute.Caps`** (FusedAttn, GraphCompile, Async, …) advertises an *optimization
capability*, not a maturity. A capability plus a committed bench that proves the
speedup is what witnesses **M5**; the capability alone is not a rung. Wiring that bench
binding is C2/#1245, not C1.

**`Dtype` / `KVPrecision`** are precision *coordinates*, not rungs. A cell is
(family × backend × precision); precision parameterizes *which* cell you are grading,
it does not move the cell up or down the ladder. The cross-support tensor that adds the
precision axis is C5 (#1248). Drawing this line keeps a precision tier from being
mistaken for a maturity tier.

**`BlockTopology`** (PreNorm / PostNorm / SandwichNorm / ParallelResidual / SparseAttn)
is a structural *family property*: it decides which accelerated fence applies, and so
it feeds the M0–M3 lowering of `covmatrix.Support` (a non-PreNorm topology is what
makes an accelerated cell FENCED rather than SUPPORTED). It is an input to the rung, not
a rung.

**The turnbench parity class** (`sota-local-baseline`) is the witness for **M6**: a cell
that matches the SOTA-local baseline is at parity. Beating it is M7, which no current
vocabulary witnesses — the open top of the ladder.

## Scope of C1, and what comes next

C1 is **vocabulary only**: the enum, the total order, and the lowering of the existing
scales onto it (`FromSupport`, `FromPreflightVerdict`, `FromCorrectnessClass`), with a
witness test that every `covmatrix.Support` value and every preflight verdict maps to
exactly one rung and the order is total. It binds no per-cell witness and promotes
nothing on its own.

The next children stand on this:

- **C2 (#1245)** — bind each rung to a non-author witness, a shipgate-gated promotion
  rule, and drop-on-regression (the Caps+bench → M5, turnbench → M6 bindings live here).
- **C3 (#1246)** — fold the grid into a `support_maturity_debt` scorecard.
- **C7 (#1250)** — the rung → dev-regime router (R0 explore / R1 prototype / R2 optimize
  / R3 production).

The honesty fence (C13/#1256) holds from the start: no rung is self-reported, a rung can
drop when its witness regresses, and M7 has no witness yet — it stays empty rather than
being claimed.
