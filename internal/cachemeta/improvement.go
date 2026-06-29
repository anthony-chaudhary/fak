package cachemeta

// This file is L6 / T13 of the self-tax plane (epic #1147,
// docs/notes/self-tax-performance-assurance-tracking-1147.md): "detect
// improvement — the positive case." The rest of the plane proves fak's
// mediation does not silently get SLOWER than its budget; this surfaces the
// dual — when realized cache reuse made work FASTER — as a net-true POSITIVE,
// per session or per fleet.
//
// The honesty hazard the positive case carries is double counting: a naive
// "tokens we did not re-prefill" sum would fold a remote provider prompt-cache
// hit (cost/latency telemetry about a prefix the PROVIDER kept resident) into
// the LOCAL reuse win (prefill a fak-local cache actually re-served). That would
// manufacture a bigger "improvement" than fak earned. SavingsSplit already
// partitions a trace into the two with no overlap (the #112 guard); this builds
// the net-true verdict on top of it, so the improvement number reported is net
// of the provider tokens a double-counting sink would wrongly credit as local,
// with the provider-vs-local split kept intact.

// Improvement is the net-true POSITIVE verdict for a realized-reuse trace — the
// "even detect if it is increasing performance" read-out. It is computed over a
// trace at either scope: a single session's cache entries, or a fleet's
// aggregated stream. The headline (NetLocalReuseTokens) is the WITNESSED local
// reuse win; the provider side (Split.ProviderReadTokens) is OBSERVED cost
// telemetry, reported separately and never folded into the local win.
type Improvement struct {
	// Scope is a passthrough label for the read-out ("session" | "fleet" | any
	// caller-supplied scope). The detector treats both scopes identically — a
	// fleet trace is just a session trace with more entries — so one code path
	// serves per-session and per-fleet surfacing.
	Scope string

	// Split is the provider-vs-local partition this verdict rests on, surfaced so
	// the split stays INTACT for any downstream reader: a consumer can always
	// recover both sides and see they do not overlap.
	Split SavingsSplit

	// NetLocalReuseTokens is the net-true local improvement: the prefill tokens a
	// fak-local cache re-served (work the kernel did NOT redo), net of the
	// provider tokens a naive sum would have mis-credited as local. It equals
	// Split.LocalReuseTokens by construction — the guard is what makes "net-true"
	// mechanical rather than a convention.
	NetLocalReuseTokens int64

	// NaiveAllAsLocalTokens is what a double-counting sink WOULD have reported as
	// the local win (local + provider). Carried so a witness can assert the guard
	// actually subtracted something: when a provider hit is present this strictly
	// exceeds NetLocalReuseTokens.
	NaiveAllAsLocalTokens int64

	// DoubleCountedTokens is the provider read-tokens the guard kept OUT of the
	// local win (== Split.ProviderReadTokens). NetLocalReuseTokens =
	// NaiveAllAsLocalTokens - DoubleCountedTokens, exactly.
	DoubleCountedTokens int64

	// Positive reports whether the trace realized a local reuse improvement at all
	// (NetLocalReuseTokens > 0). A trace whose only "savings" were provider-side is
	// NOT a local positive — the guard is what stops it reading as one.
	Positive bool
}

// DetectImprovement folds a realized-reuse trace (a session's or a fleet's cache
// entries) into the net-true positive verdict, double-count-guarded by
// SavingsSplit. The provider-vs-local split is surfaced intact on the result.
func DetectImprovement(scope string, entries []Entry) Improvement {
	var split SavingsSplit
	for _, e := range entries {
		split.Add(e)
	}
	return improvementFromSplit(scope, split)
}

// DetectImprovementFromTokens builds the same verdict directly from already-
// summed realized counters — the shape a caller holding the live metrics already
// has: localReuseTokens from the in-kernel KV-prefix reuse total
// (fak_gateway_kv_prefix_reused_tokens_total / cacheobs realized reuse), and
// providerReadTokens from the provider cache-read total. It exists so a
// session/fleet read-out can reach the net-true verdict without an upward import
// of cachemeta from the metrics tier and without re-materializing entries.
// Negative inputs are clamped to zero so a malformed counter cannot fabricate a
// false positive or a negative "win".
func DetectImprovementFromTokens(scope string, localReuseTokens, providerReadTokens int64) Improvement {
	if localReuseTokens < 0 {
		localReuseTokens = 0
	}
	if providerReadTokens < 0 {
		providerReadTokens = 0
	}
	return improvementFromSplit(scope, SavingsSplit{
		LocalReuseTokens:   localReuseTokens,
		ProviderReadTokens: providerReadTokens,
	})
}

func improvementFromSplit(scope string, split SavingsSplit) Improvement {
	return Improvement{
		Scope:                 scope,
		Split:                 split,
		NetLocalReuseTokens:   split.LocalReuseTokens,
		NaiveAllAsLocalTokens: split.LocalReuseTokens + split.ProviderReadTokens,
		DoubleCountedTokens:   split.ProviderReadTokens,
		Positive:              split.LocalReuseTokens > 0,
	}
}

// Provenance labels each side of the verdict by what fak CONTROLS vs only
// OBSERVES — the net-true-value provenance fence (docs/standards/net-true-value.md).
// The local reuse win is WITNESSED (a fak-local cache re-served it); the provider
// read-tokens are OBSERVED (relayed from the remote provider, never a fak action).
func (i Improvement) Provenance() map[string]string {
	return map[string]string{
		"net_local_reuse_tokens": "WITNESSED",
		"provider_read_tokens":   "OBSERVED",
	}
}
