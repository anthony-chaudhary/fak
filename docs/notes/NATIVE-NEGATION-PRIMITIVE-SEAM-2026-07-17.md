# Native negation primitive seam — 2026-07-17

`internal/model.NegationPrimitive` is the experimental L4 architecture seam for moving a
polarity flip out of deliberate transformer depth. A candidate accepts a representation,
returns the same shape, reports its parameter and operation bounds, and must make double
negation an identity. Nothing is installed by default; this remains a probe, not a serving
claim.

Three candidates are compared by `TestNegationComposition`:

| Candidate | Representation | Parameters (2-wide probe) | Ops / negation | Involution |
|---|---|---:|---:|---|
| learned involution | regularized linear reflection | 4 | 4 | structural for the selected matrix |
| pi phase | complex-coordinate phase offset of pi | 0 | 2 | structural |
| polarity channel | content plus a dedicated sign coordinate | 0 | 1 | structural |

The committed witness is `internal/model/testdata/native_negation_primitives.json`. It uses
an assert / negate / double-negate composition probe and compares each cheap primitive with
a depth-one deliberate baseline that cannot complete a two-step construct-then-suppress
negation. This synthetic result demonstrates the seam and falsifiable cost distinction; it
does not claim quality on a weight-backed language model.

The L4 seam is additive to the residual-hook and L3a affine seams documented in
[`MODEL-ARCH-SEAM.md`](../../MODEL-ARCH-SEAM.md); it does not alter their production routing.
