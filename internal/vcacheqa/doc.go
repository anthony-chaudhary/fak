// Package vcacheqa is the shared QA harness + witness contract every vCache gate
// (M1 vcachecal, M2 vcachestar, M3 vcachewarm, M4 vcachechain, M5 vcachegov, and the
// attribution/conflation surfaces) must pass before it is allowed to flip default-on.
// Issue #1495 (child of the vCache QA track #1490): this package BUILDS the reusable
// contract; it does not flip any gate's default itself.
//
// The contract has four pillars, each mapped onto an EXISTING mechanism elsewhere in
// the tree rather than a reinvented one:
//
//  1. Honesty test (Law A2, "correctness never depends on warmth" — see
//     internal/vcachegov's doc.go and internal/vcachecal's doc.go, both already
//     carrying the same sentence verbatim). HonestyLint is an architest-style AST scan
//     (go/ast + go/parser, the internal/architest idiom) that FAILS the moment a live
//     (non-test) source file elides re-sending context because "the provider probably
//     has it cached" — the literal violation class this package's planted-violation
//     test proves it catches. ForcedMiss drives the REAL demote+byte-diff mechanism
//     that already exists in internal/vcachestar.FoldTelemetry (Belief/Telemetry/
//     FoldResult, ReasonBelievedWarmZeroRead) so a gate's forced-cache-MISS test never
//     reinvents that reconciliation logic — it constructs a believed-warm Belief, feeds
//     a zero-cache-read Telemetry, and asserts the FoldResult actually demoted.
//
//  2. Non-forgeable witness (the guard-RSI keep-bit shape, internal/guardrsi's
//     Iteration.Kept + CheckIteration). WitnessRow builds a row in the EXACT
//     internal/journal.Row JSON schema (byte-identical field set/order) and chains it
//     with the same sha256/unit-separator algorithm internal/journal's private
//     chainHash uses (documented at the call site — internal/journal exports no public
//     row constructor, only Emit(abi.Event), so a gate outside the abi/kernel wiring
//     must build the row itself; the INDEPENDENT check is the load-bearing half, and
//     that reuses internal/journal.VerifyRows verbatim, unmodified). A caller who
//     tampers with a returned row and re-runs journal.VerifyRows gets exactly the same
//     "tampered row" rejection journal.Verify gives a tampered on-disk journal — no
//     number here is ever trusted from the row producer's own claim.
//
//  3. Provenance fence (OBSERVED vs WITNESSED, internal/cachewitness's Provenance
//     vocabulary, echoed by internal/cachevaluereport and internal/conflationscore).
//     ProvenanceFence folds a gate's reported facts against internal/conflationscore's
//     token tables (via its exported ExtractHelpStrings-shaped surfaces) so an
//     unlabeled OBSERVED number is caught the same way the conflation scorecard already
//     catches one on internal/gateway/metrics.go — this package ADDS a gate's own
//     surface to the check without editing conflationscore's owned files.
//
//  4. Determinism (pure decision layers stay pure). DeterminismCheck re-runs a
//     gate's decision function twice over the identical input and fails if the two
//     verdicts differ — the "same inputs -> same verdict" acceptance criterion, folded
//     generically over any func(In) Out the caller supplies (no reflection over a
//     gate's internals, no clock/IO smuggled into the kernel of the check itself).
//
// GateReport folds all four pillars into one result per gate, so an M1-M5 child
// assembles a report from HonestyLint + ForceCacheMiss + Chain/VerifyWitness +
// ProvenanceFence + CheckDeterminism and calls GateReport.OK() once, plus supplies its
// own three planted-violation tests (mirrored here as the harness's OWN proof it
// catches what it claims to catch — see vcacheqa_test.go's TestPlantedViolation_* trio).
//
// Tier: mechanism (2) — see internal/architest. Imports journal(2), guardrsi(1),
// cachewitness(1), vcachestar(2), conflationscore(1) and stdlib; NOT registered into
// the kernel (an off-path QA/test-support seam, not a request-path component).
package vcacheqa
