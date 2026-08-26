# Answer-shape: the consumer-facing degeneration/verbosity witness

[← Claims index](../../CLAIMS.md)


- [SHIPPED] `answershape.Measure` grades the SHAPE of a text — word-n-gram repeat (rep-n), repeated-line-block coverage, and short-period tiling (a graded generalization of the `looksDegenerate`/`dominantPeriod` detector, issue #91), headlined as one `repeat` fraction in [0,1] plus a rune-length budget — and judges it against caller thresholds (`--max-repeat`, `--max-chars`) above a 24-rune floor. It is the GRADED, tunable consumer dual of the context-MMU's conservative write-time repeat-admit rung (`ctxmmu.repeats`): the kernel quarantines only blatant byte-repeat pollution, this catches a loop the kernel's binary gate deliberately admits. Pure, deterministic, stdlib-only (architest tier 1). Witness: `internal/answershape` tests incl. `proofs_witness_test.go` (determinism, threshold-load-bearing, floor-load-bearing, repetition-monotone).
- [SHIPPED] `fak answer-shape` is the consumer WITNESS (reads stdin on `-`, exit 1 when degenerate — a pipeline gate); `fak doctor` wraps it into operator recommendations and cross-checks the real kernel admit verdict on the same bytes (`ctxmmu.ScreenBytes`), the fak analogue of `dos doctor`. Witness: `cmd/fak` `answershape_test.go` / `doctor_test.go` (exit-code + JSON contracts, kernel cross-check).

