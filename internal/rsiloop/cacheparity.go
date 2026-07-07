package rsiloop

// cacheparity.go — WITNESS the review-fork prefix-cache parity claim (#2838, part of
// Track A #2834). Its sibling reviewgate.go (#2837) PRICES a forked review turn ASSUMING
// its re-read session prefix is billed at the cache-read marginal (ReviewCost.PrefixTokens
// × cacheprice.ReadMultiplier). This file turns that assumption into a live WITNESS.
//
// THE GAP (the Hermes mechanism this improves on). The Hermes background-review fork
// inherits the parent's cached system prompt for byte-identical tools[] prefix-cache
// parity, claimed at "~26% cost reduction (issue #25322)" in agent/background_review.py.
// It is a ONE-TIME issue-number citation: nothing continuously verifies the fork actually
// hits the cache in production. A fork that silently COLD-WRITES its prefix (a cache miss)
// still reads as parity because the claim was asserted once and never re-measured.
//
// WHAT THIS DOES. fak sees the real per-turn provider usage block and already reports
// cache_read vs cache_creation (metrics.go inferCachedTokens / inferCacheCreationTokens —
// the cache-value surface). This makes the parity claim a live witness: for every forked/
// child turn, measure the fraction of the fork's prefix actually served FROM cache, and a
// regression fence (à la #2819's D2 net-nonregression gate) reds any fork whose prefix
// cache-read fraction drops below a pinned floor — a fork that cold-wrote its prefix. The
// general pattern: a forked or delegated turn should PROVE it reused the parent prefix,
// not assume it.
//
// DISTINGUISHING A REGRESSION FROM AN EXPECTED FIRST-TURN MISS (the confusion-risk fence
// #2838 names). A genuine first turn with no parent prefix yet cold-writes its whole prefix
// and that is EXPECTED, not a regression. ForkTurnUsage.HadParentPrefix carries that fork
// lineage bit, and the fence flags a cold-write ONLY when a parent prefix existed to reuse.
// A raw cache_read/creation ratio without that bit would false-positive on every real
// first turn.
//
// Pure and deterministic: the same usage split + baseline scores identically every time,
// importing nothing but fmt, so the anti-regression witness below is a fixed test (the
// #2819 discipline — "the fence is green today and would have caught the spike"). The live
// wiring — feeding this fence each forked turn's realized usage off the gateway usage seam,
// keyed by fork lineage — is the named follow-on, exactly as reviewgate.go's spawn decision
// landed pure ahead of its own live spawn-path wiring.

import "fmt"

// ForkParitySchema tags a durable per-fork cache-parity witness row so a reader can never
// confuse it for another rsiloop journal (the review-fire ledger, the curator ledger). The
// "/1" is the row-shape version. The row itself is built pure (NewForkParityRow) and
// persisted by the caller that wires this fence onto the live gateway usage seam.
const ForkParitySchema = "fak-fork-cache-parity/1"

// ForkTurnUsage is the per-forked-turn provider usage a fork-parity witness reads — the
// same cache_read / cache_creation split fak's gateway usage seam already reports
// (inferCachedTokens / inferCacheCreationTokens), plus the one fork-lineage bit that keeps
// an expected first-turn miss apart from a real regression. Structured counts only (no
// prompt/result prose), so a corpus of these stays committable — the same discipline
// ReviewTrace and internal/sessionobs hold.
type ForkTurnUsage struct {
	// CacheReadTokens is the provider cache_read_input_tokens for the forked turn: prefix
	// tokens served FROM cache — the parity the fork is supposed to inherit from its parent.
	CacheReadTokens uint64
	// CacheCreationTokens is the provider cache_creation_input_tokens: prefix tokens the
	// fork COLD-WROTE this turn (a cache miss it paid the write premium for instead of
	// reading the parent's cached prefix).
	CacheCreationTokens uint64
	// HadParentPrefix records whether a parent turn had already cached a prefix this fork was
	// expected to reuse. It distinguishes a GENUINE first-turn cache miss (no parent yet —
	// expected, never a regression) from a fork that SHOULD have hit its parent's cache but
	// cold-wrote instead (the actual regression this witness fences). Without this lineage bit
	// a raw cache_read/creation ratio would flag every real first turn.
	HadParentPrefix bool
}

// CacheablePrefixTokens is the fork's total cacheable prefix this turn: the tokens that were
// either read from cache or cold-written. It is the denominator of the parity fraction, so a
// zero here means the fork sent no cacheable prefix at all (nothing to reuse, nothing missed).
func (u ForkTurnUsage) CacheablePrefixTokens() uint64 {
	return u.CacheReadTokens + u.CacheCreationTokens
}

// PrefixCacheReadFraction is the parity METRIC in [0,1]: the fraction of the fork's cacheable
// prefix that was served FROM cache rather than cold-written. 1.0 == full parity (the whole
// prefix was a cache read — the byte-identical-prefix ideal the Hermes claim asserts); 0.0 ==
// a full cold-write (the silent cache miss). A fork with no cacheable prefix at all
// (read+creation == 0) has no miss to observe, so it is defined as 1.0 — the fence then keys
// the regression on HadParentPrefix, never on this degenerate ratio alone.
func (u ForkTurnUsage) PrefixCacheReadFraction() float64 {
	denom := u.CacheablePrefixTokens()
	if denom == 0 {
		return 1
	}
	return float64(u.CacheReadTokens) / float64(denom)
}

// ForkParityBaseline is the pinned "last accepted honest" parity floor the fence ratchets a
// fork against — the minimum acceptable prefix cache-read fraction. It is the per-fork
// analogue of #2819's committed fak_share_net baseline: a fork's realized parity may not
// regress below this without the fence reding.
type ForkParityBaseline struct {
	// MinCacheReadFraction is the floor in [0,1]: a fork with a parent prefix whose realized
	// PrefixCacheReadFraction falls below this cold-wrote its prefix and reds the fence.
	MinCacheReadFraction float64
}

// DefaultForkParityBaseline is the shipped floor: a fork that inherits a byte-identical
// parent prefix should read essentially all of it from cache, so parity should sit near 1.0.
// The 0.9 floor tolerates minor breakpoint/rounding drift (a small cold suffix, a moved
// cache_control marker) while still catching a real cold-write, whose fraction craters toward
// 0. It is a modeling floor the operator tunes; only its position relative to a fork's
// realized fraction decides the fence.
func DefaultForkParityBaseline() ForkParityBaseline {
	return ForkParityBaseline{MinCacheReadFraction: 0.9}
}

func (b ForkParityBaseline) withDefaults() ForkParityBaseline {
	if b.MinCacheReadFraction <= 0 {
		b.MinCacheReadFraction = DefaultForkParityBaseline().MinCacheReadFraction
	}
	return b
}

// parityEps is the tolerance below which a fraction is treated as equal to the floor, so
// float formatting noise (0.9 stored vs re-derived) can never spuriously red or clear the
// fence — the same guard cachevalue.go's D2 gate uses (gateEps).
const parityEps = 1e-9

// ForkParityVerdict is the fence's folded decision for one forked turn: whether it cold-wrote
// its prefix (a regression the caller should flag), the realized fraction and the floor that
// decided it, whether a parent prefix existed, and a structured reason — everything a caller
// needs to journal and audit the verdict.
type ForkParityVerdict struct {
	// ColdWrite is the BLOCK signal: true iff a parent prefix existed and the fork's realized
	// prefix cache-read fraction fell below the pinned floor — a silent cache miss.
	ColdWrite bool
	// Fraction is the fork's realized PrefixCacheReadFraction.
	Fraction float64
	// Floor is the pinned MinCacheReadFraction the fraction was ratcheted against.
	Floor float64
	// HadParentPrefix echoes the fork-lineage bit that gated the verdict.
	HadParentPrefix bool
	// Reason is the human/audit string: the pass note or the cold-write defect.
	Reason string
}

// CheckFork is the PURE fence fold for one forked turn: it measures the fork's prefix
// cache-read fraction and reds (ColdWrite) iff a parent prefix existed AND that fraction fell
// below the pinned floor. A fork with no parent prefix never reds (an expected first-turn
// miss), and a fork at or above the floor passes. It never does I/O — a caller wires it onto
// the live usage seam and persists NewForkParityRow(...) — so the fence stays deterministic
// and unit-testable, the #2819 discipline.
func CheckFork(baseline ForkParityBaseline, usage ForkTurnUsage) ForkParityVerdict {
	b := baseline.withDefaults()
	frac := usage.PrefixCacheReadFraction()
	v := ForkParityVerdict{
		Fraction:        frac,
		Floor:           b.MinCacheReadFraction,
		HadParentPrefix: usage.HadParentPrefix,
	}
	switch {
	case !usage.HadParentPrefix:
		v.Reason = fmt.Sprintf("fork_cache_parity: no parent prefix to reuse (first-turn cache write is expected, not a regression); read fraction %.4f", frac)
	case frac+parityEps < b.MinCacheReadFraction:
		v.ColdWrite = true
		v.Reason = fmt.Sprintf("fork_cache_parity: fork prefix cache-read fraction %.4f fell below the floor %.4f — the fork COLD-WROTE a parent prefix it should have reused (a silent cache miss); re-measure the fork's tools[]/system prefix for byte-identical parity", frac, b.MinCacheReadFraction)
	default:
		v.Reason = fmt.Sprintf("fork_cache_parity: fork reused the parent prefix — cache-read fraction %.4f >= floor %.4f", frac, b.MinCacheReadFraction)
	}
	return v
}

// ForkParityBlocks is the fence's boolean decision, mirroring cachevalue.go's
// CacheValueGateBlocks: it returns (true, reason) iff the fork cold-wrote a parent prefix it
// should have reused, and (false, passNote) otherwise. This is the verb a live per-fork
// witness or a regression gate calls to flag a silent cache miss.
func ForkParityBlocks(baseline ForkParityBaseline, usage ForkTurnUsage) (bool, string) {
	v := CheckFork(baseline, usage)
	return v.ColdWrite, v.Reason
}

// ForkParityRegressions folds a batch of forked-turn usages against the baseline and returns
// the indices of the forks that cold-wrote their prefix — the regression fence over a fleet
// of forks (each index maps back to usages[i]). An empty slice means every fork with a parent
// prefix reused it; nil forks (none) is an empty result.
func ForkParityRegressions(baseline ForkParityBaseline, usages []ForkTurnUsage) []int {
	var out []int
	for i, u := range usages {
		if CheckFork(baseline, u).ColdWrite {
			out = append(out, i)
		}
	}
	return out
}

// ForkParityRow is one durable per-fork cache-parity witness record — the row a caller
// appends onto the gateway usage seam when this fence is wired live, tagged ForkParitySchema
// so it never shares a row shape with the review-fire ledger. It carries the realized split,
// the fraction, the floor, the lineage bit and the cold-write verdict, so the parity claim
// becomes a re-measurable ledger instead of a one-time citation.
type ForkParityRow struct {
	Schema              string  `json:"schema"`
	Seq                 int     `json:"seq"`
	CacheReadTokens     uint64  `json:"cache_read_tokens"`
	CacheCreationTokens uint64  `json:"cache_creation_tokens"`
	Fraction            float64 `json:"prefix_cache_read_fraction"`
	Floor               float64 `json:"floor"`
	HadParentPrefix     bool    `json:"had_parent_prefix"`
	ColdWrite           bool    `json:"cold_write"`
}

// NewForkParityRow builds the durable witness row for one forked turn from its usage and the
// baseline. It is PURE (returns the row; the caller persists it), so wiring the live witness
// onto the gateway usage seam is a trivial append and this fence stays unit-testable.
func NewForkParityRow(seq int, baseline ForkParityBaseline, usage ForkTurnUsage) ForkParityRow {
	v := CheckFork(baseline, usage)
	return ForkParityRow{
		Schema:              ForkParitySchema,
		Seq:                 seq,
		CacheReadTokens:     usage.CacheReadTokens,
		CacheCreationTokens: usage.CacheCreationTokens,
		Fraction:            v.Fraction,
		Floor:               v.Floor,
		HadParentPrefix:     v.HadParentPrefix,
		ColdWrite:           v.ColdWrite,
	}
}
