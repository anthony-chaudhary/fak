package ablate

// Observational anchor verdict (#2809). AnchorABSweep (anchor_ab.go) is the CLEAN A/B: it prices
// ONE session measured under BOTH anchors and differences them, so its net-of-burst delta isolates
// the anchor choice exactly. But that shape needs a COUNTERFACTUAL — the same traffic replayed under
// FirstBP and under Head — and real traffic does not carry one: a live session runs under whichever
// anchor is configured, never both. So on the recorded cache-savings ledger AnchorABSweep has no
// matched pairs to fold, and forcing real single-arm rows into it would fabricate the missing arm.
//
// This is the HONEST read of what the single-arm ledger DOES witness, and why the #1407
// de-starvation question ("was turning the lever on net positive?") is answerable without the
// counterfactual. The durable cache-savings ledger (docs/nightrun/cache-savings.jsonl) books the two
// sides of a head-anchor fire on SEPARATE rows:
//
//   - mechanism "compaction_shed" rows carry the fak-authored shed valued at the input price
//     (compaction_saved_usd) — the GROSS benefit head-anchoring earned by firing, with NO burst on
//     that row. Summed over the window this is GrossShedUSD, the head arm's realized saving.
//   - mechanism "provider_prompt_cache" rows carry the provider cache-write premium
//     (write_premium_usd) — the cache_creation cost. The burst a fire pays to re-prime its shed
//     suffix lands HERE, but the row does not attribute which write came from which fire, and the
//     sum also includes cache writes no fire caused (initial priming, TTL refresh, non-fire turns).
//
// So the ledger gives the head shed exactly but only an UPPER BOUND on the fire-caused burst: the
// whole window write premium. Charging the head arm that entire premium is a deliberate over-charge
// — FirstBP-idle would itself write cache (and, carrying the LARGER unshed prefix, could re-prime
// MORE on TTL expiry, making the incremental burst zero or negative). Netting the full premium out
// therefore yields a conservative FLOOR on head-anchoring's true net-of-burst benefit, never an
// inflated one: true net-of-burst ≥ GrossShedUSD − WindowWritePremiumUSD.
//
// The verdict is directional only when that floor is unambiguous. A positive floor means the gross
// shed cleared its burst even charged every cache-write dollar in the window — head-anchoring is
// net-beneficial on real traffic, no per-fire attribution required. A non-positive floor does NOT
// prove head-anchoring lost money (the charge is pessimistic); it means the conservative bound cannot
// distinguish it, and separating the fire's real burst from the window total needs the per-fire
// attribution siblings (#1490/#1072) this issue lists out of scope. Zero fires means the lever never
// exercised head behavior on this traffic (FirstBP dormancy, #1407) — nothing to price either way.
//
// Generation posture (gen/now, #2783): a $0 deterministic valuation over caller-supplied ledger
// aggregates — no clock, no I/O, no model. It does not itself read the ledger or run the anchor
// split; the collect-and-aggregate shell and the published verdict artifact live outside this pure
// core, mirroring the compaction_ab.go / cachevaluereport pure-Fold + impure-shell convention.

import "fmt"

// ObservedAnchorArm is the single-arm real-traffic read of head-anchoring against the FirstBP idle
// default, built from the aggregates the durable cache-savings ledger actually witnesses over a
// window (there is no matched-session counterfactual; see the package-level note above). All dollars
// are already-priced sums off the ledger, so this type does no pricing of its own.
type ObservedAnchorArm struct {
	// GrossShedUSD is the head arm's realized benefit: the sum of compaction_saved_usd over the
	// window's "compaction_shed" rows (the fak-authored shed valued at the input price). >= 0.
	GrossShedUSD float64
	// WindowWritePremiumUSD is the ENTIRE window's provider cache-write premium (sum of
	// write_premium_usd over "provider_prompt_cache" rows) — an UPPER BOUND on the fire-caused
	// burst, deliberately over-charging the head arm so the netted result is a conservative floor. >= 0.
	WindowWritePremiumUSD float64
	// Fires is the number of witnessed compaction fires in the window (shed rows with a positive
	// saving) — the N the floor is earned over. Zero means the head lever never exercised.
	Fires int

	// Provenance so a reader can re-derive the aggregates from the same source.
	LedgerPath  string
	WindowStart string // earliest ledger date folded (YYYY-MM-DD)
	WindowEnd   string // latest ledger date folded (YYYY-MM-DD)
}

// ConservativeNetFloorUSD is the LOWER BOUND on head-anchoring's net-of-burst benefit vs the FirstBP
// idle default: the gross shed minus the ENTIRE window write premium. Because the true fire-caused
// burst is a subset of that premium, the real net-of-burst benefit is >= this floor — a positive
// floor is a hard directional win, a non-positive floor is merely undistinguished (not a proven loss).
func (a ObservedAnchorArm) ConservativeNetFloorUSD() float64 {
	return a.GrossShedUSD - a.WindowWritePremiumUSD
}

// Verdict renders the #2809 done condition IN WORDS over real single-arm traffic: whether
// head-anchoring is net-beneficial vs the FirstBP idle default, stated in net dollars, with the
// worded read never out-running the conservative floor behind it. Three exhaustive outcomes:
//
//   - no fires witnessed → head-anchoring UNWITNESSED on this traffic (the lever stayed idle, #1407
//     FirstBP dormancy); no net-dollar claim earned.
//   - floor > 0 → head-anchoring IS net-beneficial: the gross shed cleared its burst even charged
//     the entire window write premium, so the #1407 de-starvation switch was net positive on real
//     traffic, needing no per-fire burst attribution.
//   - floor <= 0 → head-anchoring is GROSS-beneficial but net-of-burst NOT DISTINGUISHABLE under the
//     conservative charge; separating the fire's real burst from the window total needs the per-fire
//     attribution siblings (#1490/#1072, out of scope), so no directional net claim is earned.
func (a ObservedAnchorArm) Verdict() string {
	floor := a.ConservativeNetFloorUSD()
	switch {
	case a.Fires == 0 || a.GrossShedUSD == 0:
		return fmt.Sprintf("head-anchoring is UNWITNESSED on this traffic: 0 compaction fires over %s..%s (FirstBP idle dormancy, #1407) — no net-dollar claim earned",
			a.WindowStart, a.WindowEnd)
	case floor > 0:
		return fmt.Sprintf("head-anchoring IS net-beneficial vs firstBP on real traffic: conservative net floor $%+.4f over N=%d fires (%s..%s) = gross shed $%+.4f minus the entire window write premium $%.4f — clears its burst even charged every window cache-write dollar",
			floor, a.Fires, a.WindowStart, a.WindowEnd, a.GrossShedUSD, a.WindowWritePremiumUSD)
	default:
		return fmt.Sprintf("head-anchoring is GROSS-beneficial ($%+.4f shed over N=%d fires) but net-of-burst is NOT DISTINGUISHABLE under the conservative charge (floor $%+.4f = shed minus the full window write premium $%.4f); a directional net verdict needs per-fire burst attribution (#1490/#1072, out of scope)",
			a.GrossShedUSD, a.Fires, floor, a.WindowWritePremiumUSD)
	}
}

// Caveat names the single-arm limitation so a reader never mistakes this conservative-floor read for
// the clean matched-session A/B (AnchorABSweep): the ledger carries only the live head arm, the burst
// charge is a window-total upper bound rather than a per-fire attribution, and no FirstBP
// counterfactual was replayed.
func (a ObservedAnchorArm) Caveat() string {
	return "single-arm observational read: no FirstBP counterfactual is recorded, and the burst is charged as the entire window write premium (an upper bound), not a per-fire attribution — so a positive floor is a conservative lower bound on the true net-of-burst benefit, and a non-positive floor is undistinguished, not a proven loss."
}
