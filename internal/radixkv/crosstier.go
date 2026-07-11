// Cross-tier contiguous-prefix assembly accounting (#3379).
//
// MatchLen (radixkv.go) answers "how many leading tokens are resident in THIS tree" —
// a single-tier probe. A tiered deployment additionally holds evicted-but-retained
// spans in a colder tier (disk / remote KV), addressed by digest rather than by node.
// The reuse a session can actually schedule is the ONE contiguous prefix it can splice
// together across tiers: the hot radix match first, then colder spans while — and only
// while — they remain gap-free. The first hole ends the assembly, because a KV prefix
// with a hole in the middle is not a prefix: nothing past the gap can be reused without
// recomputing the gap, so crediting it would overstate the schedulable savings.
//
// AssembleCrossTierPrefix is the pure fold that does that accounting. Like MatchLen it
// mutates nothing, takes no lock, and touches no KV — it is the measurement seam a
// tiered planner sums over a workload, mirroring the single-tier walk/MatchLen contract.
package radixkv

// Tier identifies which residency tier owns a contiguous range of an assembled
// cross-tier prefix.
type Tier int

const (
	// TierHot is the in-memory radix tree (this package's Tree) — the tier MatchLen probes.
	TierHot Tier = iota
	// TierCold is a colder residency tier (disk / remote) holding digest-addressed spans.
	TierCold
)

// String reports the tier name for logs and test failure messages.
func (t Tier) String() string {
	switch t {
	case TierHot:
		return "hot"
	case TierCold:
		return "cold"
	default:
		return "unknown"
	}
}

// ColdSpan is one colder-tier resident span, given in prefix order. Digest is the
// content address the cold tier stores the span under; Len is its length in tokens.
// A span with an empty Digest or a non-positive Len is a HOLE — the caller's way of
// saying "this stretch of the prefix is not resident" — and ends the assembly.
type ColdSpan struct {
	Digest string
	Len    int
}

// PrefixRange tags one contiguous half-open token range [Start, End) of the assembled
// prefix with the tier that owns it.
type PrefixRange struct {
	Start int
	End   int
	Tier  Tier
}

// AssembleCrossTierPrefix folds a hot-tier match length and an in-prefix-order list of
// colder-tier spans into the single contiguous reusable prefix they assemble, returning
// the per-tier ranges and the total contiguous token count (== the End of the last range).
//
// Assembly starts with the hot match [0, hotMatchLen) — the value MatchLen reports —
// then appends cold spans in order while they stay contiguous. The FIRST gap truncates:
// a missing span (empty Digest), a zero- or negative-length span, or by construction any
// later span (which cannot be contiguous once a hole exists) ends the prefix, and spans
// after the gap are NOT credited. Adjacent cold spans are kept as distinct ranges (one
// per span, in input order) so each range still maps back to the digest that backs it.
//
// Degenerate inputs are pure accounting identities: hotMatchLen <= 0 with no usable cold
// spans yields (nil, 0); a prefix may also assemble entirely from cold spans when
// hotMatchLen is 0 (the cold tier holds the prefix from position 0). No mutation, no
// lock, no KV access — safe to call concurrently with any Tree operation.
func AssembleCrossTierPrefix(hotMatchLen int, coldSpans []ColdSpan) (ranges []PrefixRange, total int) {
	pos := 0
	if hotMatchLen > 0 {
		ranges = append(ranges, PrefixRange{Start: 0, End: hotMatchLen, Tier: TierHot})
		pos = hotMatchLen
	}
	for _, s := range coldSpans {
		if s.Len <= 0 || s.Digest == "" {
			break // first gap: nothing past a hole is a contiguous prefix
		}
		ranges = append(ranges, PrefixRange{Start: pos, End: pos + s.Len, Tier: TierCold})
		pos += s.Len
	}
	return ranges, pos
}
