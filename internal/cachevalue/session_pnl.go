package cachevalue

// Per-session owner-split P&L (issue #10948, child #1496 of the vCache epic).
//
// The fold reports per-session hit rate and write-amp, and FlagRegressions
// flags the churny ones — but neither answers the question default-on vCache
// enablement actually turns on (#10948): for THIS session's owner, did the
// provider cache net-save tokens, and where is the bloat? Emptily: aggregate
// rollups blend every tenant into one number, so a cache-hostile session can
// hide inside a healthy fleet average forever.
//
// The P&L scores ONE ledger row in the same input-token-equivalent currency
// the rest of the package prices in (warmReadMarginal 0.1x, cold 1x,
// cacheWriteMarginal 1.25x; see compaction_economics.go). Baseline is the
// no-cache counterfactual (every prompt token full price); actual is what the
// provider charged through the cache; NetSavedTokEq is their difference. The
// sign is the owner's cache economics: positive = the cache paid for itself,
// negative = writes cost more than reads saved (cache bloat, the exact risk
// that keeps vCache opt-in today), zero-noise = neutral. OutputTokens are
// deliberately excluded: decode output bills identically with or without the
// prompt cache, so it is P&L-neutral by construction — including it would
// only inflate both sides of the ledger by the same amount.
//
// Ownership is carried by the ledger row itself (schema-additive `owner`
// field, tolerated absent on pre-owner ledgers); a row stamped without one
// is grouped under OwnerUnknown rather than dropped — dropping it would hide
// exactly the unattributed bloat the split exists to surface. Pure, like
// the rest of the package: no I/O, no clock, std only.

import "sort"

// OwnerUnknown is the bucket a ledger row without an `owner` stamp groups
// under. Kept visible so a report renderer can flag unattributed sessions
// instead of silently blending them into a named tenant.
const OwnerUnknown = "unknown"

// Stable reason tokens for a per-session P&L verdict. Fixed strings so a
// report renderer or guard summary matches on them rather than parsing
// prose, mirroring the Reason* / ReasonCompaction* tokens above.
const (
	// ReasonSessionNetSaved: the prompt cache net-saved tokens for this
	// session — the cache paid for itself.
	ReasonSessionNetSaved = "session_net_saved"
	// ReasonSessionNetNegative: cache writes cost more than cache reads
	// saved — the cache-bloat direction. The signal that must fire before
	// vCache is safe to default-on for this owner's workload shape.
	ReasonSessionNetNegative = "session_net_negative"
	// ReasonSessionNeutral: the net move sat inside the dead-band (or the
	// row was all zeros) — no economic claim either way.
	ReasonSessionNeutral = "session_neutral"
)

// SessionPnL is the owner-attributable profit & loss reading for ONE ledger
// row (one guard session). All three token-equivalent figures are carried so
// the "prompt size" story never blends into the "net after cache effects"
// story, mirroring CompactionVerdict's separation of shed value from net.
type SessionPnL struct {
	// GeneratedAt is the session key, copied from the row.
	GeneratedAt string
	// Owner is the row's owner stamp; OwnerUnknown when the row carried none.
	Owner string
	// BaselineTokEq is the no-cache counterfactual: every prompt-side token
	// at full price.
	BaselineTokEq float64
	// ActualTokEq is the charged reading: input at full price, cache reads
	// at the warm marginal, cache creation at the write premium.
	ActualTokEq float64
	// NetSavedTokEq is Baseline - Actual: positive = the cache net-saved.
	NetSavedTokEq float64
	// Reason is the stable verdict token (one of the ReasonSession* constants).
	Reason string
}

// SessionProfitAndLoss scores ONE ledger row's per-session cache P&L. It
// always returns a verdict: an all-zero row is honestly neutral (no prompt
// tokens, no economics to judge), not skipped — the same fail-open
// discipline Fold uses for an unknown hit rate.
func ScoreSessionPnL(row Row) SessionPnL {
	owner := row.Owner
	if owner == "" {
		owner = OwnerUnknown
	}
	prompt := row.InputTokens + row.CacheReadTokens + row.CacheCreationTokens
	p := SessionPnL{
		GeneratedAt:   row.GeneratedAt,
		Owner:         owner,
		BaselineTokEq: coldInputMarginal * float64(prompt),
		ActualTokEq: coldInputMarginal*float64(row.InputTokens) +
			warmReadMarginal*float64(row.CacheReadTokens) +
			cacheWriteMarginal*float64(row.CacheCreationTokens),
	}
	p.NetSavedTokEq = p.BaselineTokEq - p.ActualTokEq
	switch {
	case p.NetSavedTokEq < -compactionNetEps:
		p.Reason = ReasonSessionNetNegative
	case p.NetSavedTokEq > compactionNetEps:
		p.Reason = ReasonSessionNetSaved
	default:
		p.Reason = ReasonSessionNeutral
	}
	return p
}

// OwnerSplit is the owner-aggregated P&L over a set of sessions: the split
// the fleet average hides. NegativeSessions is the bloat surface — the count
// of this owner's sessions whose own verdict was ReasonSessionNetNegative —
// because a net-positive owner can still carry individual bloaty sessions
// inside it.
type OwnerSplit struct {
	// Owner is the bucket key (OwnerUnknown for unattributed rows).
	Owner string
	// Sessions is the number of ledger rows folded into this bucket.
	Sessions int
	// Raw token axes, aggregated so a renderer can re-derive rates without
	// re-reading the ledger.
	InputTokens         int64
	CacheReadTokens     int64
	CacheCreationTokens int64
	// BaselineTokEq / ActualTokEq / NetSavedTokEq are the sums of the
	// members' per-session figures (additive by construction).
	BaselineTokEq float64
	ActualTokEq   float64
	NetSavedTokEq float64
	// NegativeSessions counts members flagged ReasonSessionNetNegative.
	NegativeSessions int
}

// SplitOwnerPnL scores every session and aggregates the P&L by owner, in
// deterministic ascending-owner order (OwnerUnknown participates normally),
// so a report is byte-stable across runs over the same ledger.
func SplitOwnerPnL(ms []Metrics) []OwnerSplit {
	idx := map[string]int{}
	out := make([]OwnerSplit, 0, 4)
	for _, m := range ms {
		p := ScoreSessionPnL(m.Row)
		i, ok := idx[p.Owner]
		if !ok {
			i = len(out)
			idx[p.Owner] = i
			out = append(out, OwnerSplit{Owner: p.Owner})
		}
		s := &out[i]
		s.Sessions++
		s.InputTokens += m.Row.InputTokens
		s.CacheReadTokens += m.Row.CacheReadTokens
		s.CacheCreationTokens += m.Row.CacheCreationTokens
		s.BaselineTokEq += p.BaselineTokEq
		s.ActualTokEq += p.ActualTokEq
		s.NetSavedTokEq += p.NetSavedTokEq
		if p.Reason == ReasonSessionNetNegative {
			s.NegativeSessions++
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Owner < out[j].Owner })
	return out
}
