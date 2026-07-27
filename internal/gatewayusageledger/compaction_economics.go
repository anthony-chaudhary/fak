package gatewayusageledger

import "github.com/anthony-chaudhary/fak/internal/cacheprice"

// compaction_economics.go — the per-session compaction-economics TRAILER carried on a
// gateway-usage row (#2792, program #2783).
//
// The exit row (c9a8c26a) already carries every RAW compaction counter: fires, shed
// tokens, the observed post-fire cache_read. What it did not carry is the one line an
// operator actually reads — was this session's compaction worth it? Answering that from
// the raw row means knowing, by heart, that a shed token landing on a WARM prefix is
// worth only the 0.1x read marginal (not full input), and that a fire ALSO buys a
// one-time cold re-write of the suffix it shifted. Two economic facts, applied in the
// right order, per row. That is a join a reader should never have to do by hand, and
// every hand-done version drifts from the live gateway's own attribution.
//
// So the trailer is a PURE PROJECTION of the counters already on the row, priced on the
// canonical cacheprice basis (#2798) — the same ReadMultiplier / Write5mMultiplier the
// fire gate decides on, the live gateway's MechanismSavings splits, and the Track-2
// report values. No new observation, no clock, no I/O: same Counters in, same trailer
// out. A reader that distrusts the projection can recompute it from the same row.
//
// TWO honesty fences, both structural:
//
//   - The shed is valued by cacheprice.ShedTokenEquiv — the PROPORTIONAL warm/cold blend,
//     not a binary flip. min(shed, observed cache_read) prices at the read marginal (those
//     tokens were already being served from cache, so dropping them saves the read, not
//     fresh input); the remainder prices at full input. This is byte-identical to what
//     gateway.MechanismSavings.FakTokenEquiv reports live, so the durable trailer and the
//     in-process /debug/vars attribution cannot disagree about one session.
//
//   - The induced-creation debit is only as real as its witness. Per-fire induced creation
//     is #2785 and is NOT yet plumbed to the gateway, so CompactionInducedCreationTokens is
//     structurally 0 on today's rows. A 0 debit would make Net read like a COMPLETE net when
//     it is really a ceiling, so an unwitnessed row sets NetIsUpperBound — the same
//     refuse-to-let-an-unpopulated-field-look-like-a-measurement discipline the segment fold
//     applies when it quarantines a phantom-100% shed fraction (compaction_segments.go).
//     When #2785 lands and populates the counter, the flag clears with no shape change here.
//
// Sign convention matches rsiloop.ScoreCompactionFire (the per-fire twin): net = shed
// saving − burst cost, SIGNED. A value-destroying session reads NEGATIVE and is never
// floored at zero (#1303).

// CompactionEconomics is the compaction-economics trailer: the five figures #2792 asks the
// exit row to carry (fires, shed, observed cache_read, induced creation, net-of-both), plus
// the two intermediate token-equivalent terms the net is the difference of — so a reader
// sees WHICH side moved a net, not just that it moved.
//
// Every *TokenEquiv figure is in INPUT-TOKEN-EQUIVALENTS on the cacheprice basis (the $ dual
// is the caller's base input price), the same currency as fak_vcache_saved_token_equiv and
// gateway.MechanismSavings.
type CompactionEconomics struct {
	// Fires is the WITNESSED count of history rewrites this session actually shipped
	// (Counters.CompactionFired) — a turn counts only when the prefix it shipped was
	// byte-identical, so this is fak's own witness, not a provider report.
	Fires uint64 `json:"fires"`
	// ShedTokens is the WITNESSED resident tokens those fires removed
	// (Counters.CompactionShedTokens) — the gross saving, before it is priced.
	ShedTokens uint64 `json:"shed_tokens"`
	// ObservedCacheReadTokens is the OBSERVED provider cache_read at this session's fires
	// (Counters.CompactionCacheReadTokens, #2784) — the warm WITNESS the shed is priced
	// against, relayed verbatim from the provider. It is not proof fak preserved the cache
	// (the byte-identity is); it is how much of the shed was already cheap when dropped.
	ObservedCacheReadTokens uint64 `json:"observed_cache_read_tokens"`
	// InducedCacheCreationTokens is the suffix-burst base: cache_creation the fires induced
	// by shifting bytes after the drop point and invalidating the downstream breakpoint
	// (#2785). 0 with NetIsUpperBound set means UNWITNESSED, not measured-zero.
	InducedCacheCreationTokens uint64 `json:"induced_cache_creation_tokens"`

	// ShedSavingTokenEquiv is the shed priced on the warm/cold blend
	// (cacheprice.ShedTokenEquiv) — the credit side.
	ShedSavingTokenEquiv float64 `json:"shed_saving_token_equiv"`
	// BurstCostTokenEquiv is the one-time excess-over-read paid to cold-rewrite the
	// invalidated suffix: induced × (Write5mMultiplier − ReadMultiplier) — the debit side.
	// The 5m tier is the conservative default this repo applies to an unattributed write
	// (see gateway.splitCacheCreationPremiumTokenEquiv); it is the CHEAPER write, so an
	// unattributed 1h burst is under-debited, never over-debited.
	BurstCostTokenEquiv float64 `json:"burst_cost_token_equiv"`
	// NetTokenEquiv is the headline: ShedSavingTokenEquiv − BurstCostTokenEquiv, SIGNED.
	// Negative means this session's compaction destroyed value versus not firing.
	NetTokenEquiv float64 `json:"net_token_equiv"`

	// NetIsUpperBound marks a net computed with NO induced-creation witness (#2785 not yet
	// plumbed): the debit side is structurally absent, so NetTokenEquiv is a CEILING on the
	// true net, not the net. Omitted once a witness exists, so its mere PRESENCE is the
	// caveat — a reader never has to know which upstream issues had landed when the row
	// was written.
	NetIsUpperBound bool `json:"net_is_upper_bound,omitempty"`
}

// CompactionEconomicsOf projects a counter snapshot into the trailer, or nil when the
// session has no compaction story to tell (never fired AND shed nothing AND induced
// nothing). Returning nil for a quiet session is what keeps a non-compacting row
// byte-identical to the pre-trailer schema — the trailer is `omitempty` on Row, so only
// sessions that actually compacted grow the line.
//
// PURE: no clock, no I/O, no package state. Same Counters in, same trailer out.
func CompactionEconomicsOf(c Counters) *CompactionEconomics {
	if c.CompactionFired == 0 && c.CompactionShedTokens == 0 && c.CompactionInducedCreationTokens == 0 {
		return nil
	}
	shedSaving := cacheprice.ShedTokenEquiv(c.CompactionShedTokens, c.CompactionCacheReadTokens)
	burstCost := float64(c.CompactionInducedCreationTokens) * (cacheprice.Write5mMultiplier - cacheprice.ReadMultiplier)
	return &CompactionEconomics{
		Fires:                      c.CompactionFired,
		ShedTokens:                 c.CompactionShedTokens,
		ObservedCacheReadTokens:    c.CompactionCacheReadTokens,
		InducedCacheCreationTokens: c.CompactionInducedCreationTokens,
		ShedSavingTokenEquiv:       shedSaving,
		BurstCostTokenEquiv:        burstCost,
		NetTokenEquiv:              shedSaving - burstCost,
		NetIsUpperBound:            c.CompactionInducedCreationTokens == 0,
	}
}
