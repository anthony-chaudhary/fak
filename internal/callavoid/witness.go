package callavoid

// witness.go — the in-lane half of #820: split a deny's amplification into a HARD deny
// (no forward guidance, symmetric, credits nothing) and a WITNESSED productive deny that
// carries the ENUMERATED futile variants it structurally pruned, so that pruning can
// graduate from a speculative upper bound (bare Redirects counts, #816) to REALIZED
// credit folded into the graded amplification.
//
// The distinction is provenance, not size. A bare Redirects entry is a NUMBER — "this
// deny pruned ~N variants" — an assertion no counter backs, so it stays on the
// speculative axis and never moves the grade. A WitnessedRedirect is the SAME pruning
// but with the variants ENUMERATED (e.g. the policy rule's known covered-variant set):
// the count is the size of a concrete, deduplicated set the deny structurally rules out,
// not a guess. Only the witnessed form is credited into the realized headline, and even
// then it is bounded (per-deny cap) and surfaced, never an unbounded counterfactual.

// WitnessedRedirect is one productive deny whose pruned futile sub-tree is ENUMERATED.
// Variants are the concrete futile calls the deny structurally rules out — e.g. the
// covered-variant set of the policy rule that fired. The realized credit is the size of
// the DEDUPLICATED, non-empty variant set (computed by witnessedFanout), bounded by the
// per-deny cap: a witness can only credit what it can name, so an empty or all-blank
// variant set credits zero and is treated as a hard deny.
type WitnessedRedirect struct {
	// Rule is an optional label for the deny rule that produced the witness (the policy
	// id / capability name). It is informational — it never affects the credited count —
	// but it lets a report attribute the realized pruning to a named structural cause.
	Rule string `json:"rule,omitempty"`
	// Variants enumerates the futile calls this deny pruned. The credited fan-out is the
	// number of DISTINCT non-empty entries (a witness that lists the same variant twice,
	// or pads with blanks, credits each real variant once and nothing for the padding).
	Variants []string `json:"variants"`
}

// witnessedFanout returns the realized fan-out a witnessed redirect credits: the count
// of DISTINCT, non-empty variants, clamped to fanoutCap. Deduplication is what makes the
// witness non-gameable on enumeration — listing one variant a thousand times credits one,
// not a thousand — and the clamp keeps even an honest large set bounded. The returned
// `capped` reports whether the clamp bit, so the surface can surface it (never silent).
func witnessedFanout(w WitnessedRedirect, fanoutCap int) (fanout int, capped bool) {
	if fanoutCap < 0 {
		fanoutCap = 0
	}
	seen := make(map[string]struct{}, len(w.Variants))
	for _, v := range w.Variants {
		if v == "" {
			continue // a blank entry names no futile call — it credits nothing.
		}
		if _, dup := seen[v]; dup {
			continue
		}
		seen[v] = struct{}{}
	}
	fanout = len(seen)
	if fanout >= fanoutCap {
		return fanoutCap, true // >= so an exactly-at-cap witness surfaces too, mirroring the speculative path.
	}
	return fanout, false
}

// witnessedTotals folds a Tally's witnessed redirects into the realized fan-out the grade
// is allowed to credit, plus the bookkeeping the report surfaces. Each witness that
// credits a non-zero fan-out is a REALIZED productive deny; a witness whose variant set
// is empty/blank credits nothing and is counted as an effective hard deny (it carried no
// nameable pruning), so an empty witness can never inflate the headline.
//
// The aggregate is saturated at DefaultMaxSpeculativePrunedTurns — the same finite ceiling
// the speculative axis uses — so even an unbounded NUMBER of witnessed denies cannot push
// the realized credit past a fixed bound. realizedDenies counts how many witnesses
// actually credited; emptyWitnesses counts those that named nothing (folded into HardDeny
// by the caller is not required — Account treats them as zero-credit here directly).
func witnessedTotals(witnessed []WitnessedRedirect, fanoutCap int) (totalFanout, realizedDenies, emptyWitnesses, cappedCount int, aggCapped bool) {
	for _, w := range witnessed {
		f, capped := witnessedFanout(w, fanoutCap)
		if capped {
			cappedCount++
		}
		if f <= 0 {
			emptyWitnesses++
			continue
		}
		realizedDenies++
		totalFanout += f
		if totalFanout >= DefaultMaxSpeculativePrunedTurns {
			totalFanout = DefaultMaxSpeculativePrunedTurns
			aggCapped = true
			break
		}
	}
	return totalFanout, realizedDenies, emptyWitnesses, cappedCount, aggCapped
}
