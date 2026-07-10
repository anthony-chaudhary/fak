package rsiloop

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// savingEps absorbs float-formatting noise so a re-derived figure can never spuriously fail
// an equality — the same tolerance discipline cacheparity/reviewgate hold.
const savingEps = 1e-9

func approxEq(a, b float64) bool {
	d := a - b
	if d < 0 {
		d = -d
	}
	return d <= savingEps
}

// TestDefaultForkSavingBasis pins the shipped write tier: the cold-written prefix is priced
// at the 5-minute cache-write multiplier, matching the Track-2 report's unattributed-creation
// convention (#2179). If someone drifts it off the canonical anchor this reds.
func TestDefaultForkSavingBasis(t *testing.T) {
	if got := DefaultForkSavingBasis().WriteMultiplier; got != cacheprice.Write5mMultiplier {
		t.Fatalf("default fork-saving write multiplier = %v, want %v (cacheprice.Write5mMultiplier)", got, cacheprice.Write5mMultiplier)
	}
}

// TestMeasureForkSaving_FullReuse is the ideal Hermes fork: a byte-identical parent prefix
// read entirely from cache, cold-writing nothing. The saving is the full read rebate and the
// net equals the gross — there is no write premium to net out.
func TestMeasureForkSaving_FullReuse(t *testing.T) {
	s := MeasureForkSaving(DefaultForkSavingBasis(), ForkTurnUsage{CacheReadTokens: 1_000, HadParentPrefix: true})

	if !s.Attributable {
		t.Fatalf("a fork with a parent prefix must be attributable: %+v", s)
	}
	if !approxEq(s.GrossSavedTokenEquiv, 900) {
		t.Fatalf("gross = %v, want 900 (1000 × 0.9)", s.GrossSavedTokenEquiv)
	}
	if !approxEq(s.WritePremiumTokenEquiv, 0) {
		t.Fatalf("write premium = %v, want 0 (nothing cold-written)", s.WritePremiumTokenEquiv)
	}
	if !approxEq(s.NetSavedTokenEquiv, s.GrossSavedTokenEquiv) {
		t.Fatalf("net (%v) must equal gross (%v) when nothing is cold-written", s.NetSavedTokenEquiv, s.GrossSavedTokenEquiv)
	}
	if !approxEq(s.CounterfactualTokenEquiv, 1_000) || !approxEq(s.NetSavedPct, 90) {
		t.Fatalf("counterfactual/netPct = %v/%v, want 1000/90.0", s.CounterfactualTokenEquiv, s.NetSavedPct)
	}
}

// TestMeasureForkSaving_MixedNetsOutWritePremium is the honest heart of #2875: a fork that
// mostly reused but cold-wrote a sliver has its gross rebate DOCKED by the write premium it
// paid on that sliver — net < gross. This is the #2797 gross/net split applied per fork.
func TestMeasureForkSaving_MixedNetsOutWritePremium(t *testing.T) {
	s := MeasureForkSaving(DefaultForkSavingBasis(), ForkTurnUsage{CacheReadTokens: 1_000, CacheCreationTokens: 100, HadParentPrefix: true})

	if !approxEq(s.GrossSavedTokenEquiv, 900) {
		t.Fatalf("gross = %v, want 900", s.GrossSavedTokenEquiv)
	}
	if !approxEq(s.WritePremiumTokenEquiv, 25) { // 100 × (1.25 − 1)
		t.Fatalf("write premium = %v, want 25 (100 × 0.25)", s.WritePremiumTokenEquiv)
	}
	if !approxEq(s.NetSavedTokenEquiv, 875) { // 900 − 25
		t.Fatalf("net = %v, want 875 (gross 900 − premium 25)", s.NetSavedTokenEquiv)
	}
	if s.NetSavedTokenEquiv >= s.GrossSavedTokenEquiv {
		t.Fatalf("net (%v) must be strictly below gross (%v) once anything is cold-written", s.NetSavedTokenEquiv, s.GrossSavedTokenEquiv)
	}
	if !approxEq(s.CounterfactualTokenEquiv, 1_100) {
		t.Fatalf("counterfactual = %v, want 1100 ((1000+100) × 1.0)", s.CounterfactualTokenEquiv)
	}
}

// TestMeasureForkSaving_ColdWriteOnlyGoesNegative is the anti-Hermes case a one-time asserted
// "~26% saving" can never express: a fork that cold-wrote a parent prefix it should have
// reused paid the write premium for NOTHING — its net saving is NEGATIVE (it lost money vs a
// plain fresh turn). The net keeps its sign (#1303: never floor at zero).
func TestMeasureForkSaving_ColdWriteOnlyGoesNegative(t *testing.T) {
	s := MeasureForkSaving(DefaultForkSavingBasis(), ForkTurnUsage{CacheCreationTokens: 1_000, HadParentPrefix: true})

	if !approxEq(s.GrossSavedTokenEquiv, 0) {
		t.Fatalf("gross = %v, want 0 (nothing read from cache)", s.GrossSavedTokenEquiv)
	}
	if !approxEq(s.WritePremiumTokenEquiv, 250) { // 1000 × 0.25
		t.Fatalf("write premium = %v, want 250", s.WritePremiumTokenEquiv)
	}
	if !approxEq(s.NetSavedTokenEquiv, -250) {
		t.Fatalf("net = %v, want -250 (a cold-write-only fork LOSES money)", s.NetSavedTokenEquiv)
	}
	if s.NetSavedTokenEquiv >= 0 {
		t.Fatalf("net must be negative for a cold-write-only fork, got %v", s.NetSavedTokenEquiv)
	}
	if !approxEq(s.NetSavedPct, -25) {
		t.Fatalf("netPct = %v, want -25.0", s.NetSavedPct)
	}
}

// TestMeasureForkSaving_NoParentPrefixNotAttributable is the confusion-risk fence: a genuine
// first-turn fork (no parent prefix to reuse) cold-writes its whole prefix EXPECTEDLY, so it
// is NOT a saving of zero — it is a saving that does not apply. Attributable is false and the
// figures stay zero, so the fold never books a spurious loss on a first turn.
func TestMeasureForkSaving_NoParentPrefixNotAttributable(t *testing.T) {
	s := MeasureForkSaving(DefaultForkSavingBasis(), ForkTurnUsage{CacheCreationTokens: 1_000, HadParentPrefix: false})

	if s.Attributable {
		t.Fatalf("a first-turn fork (no parent prefix) must NOT be attributable: %+v", s)
	}
	if s.NetSavedTokenEquiv != 0 || s.GrossSavedTokenEquiv != 0 || s.WritePremiumTokenEquiv != 0 {
		t.Fatalf("a non-attributable fork must carry zero saving figures, got %+v", s)
	}
	if !strings.Contains(s.Trailer(), "no reuse saving to measure") {
		t.Fatalf("first-turn trailer must say the saving does not apply, got %q", s.Trailer())
	}
}

// TestTrailerNetFirst pins the #2797 honesty ORDER: the trailer leads with the NET (the
// corrected headline) and labels the gross as an upper bound, so a reader can never mistake
// the optimistic number for the headline. It also carries the raw split so the figure is
// auditable, not asserted like Hermes' one-time citation.
func TestTrailerNetFirst(t *testing.T) {
	tr := MeasureForkSavingTrailer(DefaultForkSavingBasis(), ForkTurnUsage{CacheReadTokens: 1_000, CacheCreationTokens: 100, HadParentPrefix: true})

	netAt := strings.Index(tr, "net ")
	grossAt := strings.Index(tr, "gross upper bound")
	if netAt < 0 || grossAt < 0 {
		t.Fatalf("trailer must name both net and the gross upper bound: %q", tr)
	}
	if netAt > grossAt {
		t.Fatalf("net must lead the gross upper bound (net-first honesty): %q", tr)
	}
	if !strings.Contains(tr, "measured") {
		t.Fatalf("trailer must mark the saving as measured, not asserted: %q", tr)
	}
	if !strings.Contains(tr, "net +875 tok-eq") {
		t.Fatalf("trailer must carry the measured net figure: %q", tr)
	}
}

// TestForkSaving1hTier proves the write tier is honored: a 1h-attributed cold-write pays the
// 2.0× premium, so its net saving is docked twice as hard as the 5m default — the basis is a
// real input, not a hard-coded literal.
func TestForkSaving1hTier(t *testing.T) {
	usage := ForkTurnUsage{CacheReadTokens: 1_000, CacheCreationTokens: 100, HadParentPrefix: true}
	s := MeasureForkSaving(ForkSavingBasis{WriteMultiplier: cacheprice.Write1hMultiplier}, usage)

	if !approxEq(s.WritePremiumTokenEquiv, 100) { // 100 × (2.0 − 1)
		t.Fatalf("1h write premium = %v, want 100", s.WritePremiumTokenEquiv)
	}
	if !approxEq(s.NetSavedTokenEquiv, 800) { // 900 − 100
		t.Fatalf("1h net = %v, want 800", s.NetSavedTokenEquiv)
	}
}

// TestForkSavingBindsToParity is the cross-primitive consistency the issue's confusion-risk
// names: prefix-cache "parity" must be VERIFIED, not assumed. It binds MeasureForkSaving to
// its sibling CheckFork over the SAME ForkTurnUsage — a fork CheckFork clears at full parity
// nets its full rebate (net == gross), and a fork CheckFork flags as a cold-write has its net
// docked below gross by the write premium. The two primitives read one witnessed split.
func TestForkSavingBindsToParity(t *testing.T) {
	basis := DefaultForkSavingBasis()
	parity := DefaultForkParityBaseline()

	full := ForkTurnUsage{CacheReadTokens: 10_000, HadParentPrefix: true}
	if v := CheckFork(parity, full); v.ColdWrite {
		t.Fatalf("full-reuse fork wrongly flagged cold-write: %+v", v)
	}
	if fs := MeasureForkSaving(basis, full); !approxEq(fs.NetSavedTokenEquiv, fs.GrossSavedTokenEquiv) {
		t.Fatalf("a parity-clean fork must net its full rebate: net %v != gross %v", fs.NetSavedTokenEquiv, fs.GrossSavedTokenEquiv)
	}

	cold := ForkTurnUsage{CacheReadTokens: 1_000, CacheCreationTokens: 9_000, HadParentPrefix: true} // 0.10 << 0.9 floor
	if blocked, _ := ForkParityBlocks(parity, cold); !blocked {
		t.Fatalf("cold-write fork should be flagged by the parity fence")
	}
	fs := MeasureForkSaving(basis, cold)
	if fs.NetSavedTokenEquiv >= fs.GrossSavedTokenEquiv {
		t.Fatalf("a parity-flagged cold-write must dock net below gross: net %v, gross %v", fs.NetSavedTokenEquiv, fs.GrossSavedTokenEquiv)
	}
}

// TestNewForkSavingRow proves the durable witness row a live caller appends carries the schema
// tag, the measured split, and the gross/net saving — the one-time "~26%" citation turned into
// a re-measurable ledger row, distinct in shape from the fork-parity witness.
func TestNewForkSavingRow(t *testing.T) {
	row := NewForkSavingRow(3, DefaultForkSavingBasis(), ForkTurnUsage{CacheReadTokens: 1_000, CacheCreationTokens: 100, HadParentPrefix: true})

	if row.Schema != ForkSavingSchema {
		t.Fatalf("row schema = %q, want %q", row.Schema, ForkSavingSchema)
	}
	if row.Seq != 3 {
		t.Fatalf("row seq = %d, want 3", row.Seq)
	}
	if row.CacheReadTokens != 1_000 || row.CacheCreationTokens != 100 {
		t.Fatalf("row split not preserved: %+v", row)
	}
	if !approxEq(row.NetSavedTokenEquiv, 875) || !approxEq(row.GrossSavedTokenEquiv, 900) {
		t.Fatalf("row saving figures = net %v / gross %v, want 875 / 900", row.NetSavedTokenEquiv, row.GrossSavedTokenEquiv)
	}
	if !row.Attributable {
		t.Fatalf("row must record the fork as attributable: %+v", row)
	}
}

// TestMeasureForkSavingDeterministic guards purity: the same usage + basis prices identically
// every call, so the witness below is a fixed test (the #2819 discipline) and a live seam can
// replay a row and get the same saving.
func TestMeasureForkSavingDeterministic(t *testing.T) {
	usage := ForkTurnUsage{CacheReadTokens: 4_321, CacheCreationTokens: 765, HadParentPrefix: true}
	a := MeasureForkSaving(DefaultForkSavingBasis(), usage)
	b := MeasureForkSaving(DefaultForkSavingBasis(), usage)
	if a != b {
		t.Fatalf("MeasureForkSaving is not deterministic: %+v != %+v", a, b)
	}
}
