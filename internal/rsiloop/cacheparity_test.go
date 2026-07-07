package rsiloop

import "testing"

// TestPrefixCacheReadFraction pins the parity METRIC over its boundary cases: a full cache
// read is 1.0 (the byte-identical-prefix ideal), a full cold-write is 0.0 (the silent miss),
// a mix is the read share, and a fork with no cacheable prefix at all is 1.0 (no miss to
// observe — the fence keys the regression on lineage, not this degenerate ratio).
func TestPrefixCacheReadFraction(t *testing.T) {
	cases := []struct {
		name string
		u    ForkTurnUsage
		want float64
	}{
		{"full-parity", ForkTurnUsage{CacheReadTokens: 10_000}, 1},
		{"full-cold-write", ForkTurnUsage{CacheCreationTokens: 10_000}, 0},
		{"three-quarters", ForkTurnUsage{CacheReadTokens: 3_000, CacheCreationTokens: 1_000}, 0.75},
		{"no-cacheable-prefix", ForkTurnUsage{}, 1},
	}
	for _, c := range cases {
		if got := c.u.PrefixCacheReadFraction(); got != c.want {
			t.Fatalf("%s: PrefixCacheReadFraction = %v, want %v", c.name, got, c.want)
		}
	}
}

// TestDefaultForkParityBaseline guards the shipped floor itself: a byte-identical parent
// prefix should read near-1.0 from cache, so the floor sits high (0.9). If someone drops it
// toward 0 the fence would wave through a cold-write, so this asserts the shipped position —
// the same basis guard reviewgate_test holds on the cache-read marginal.
func TestDefaultForkParityBaseline(t *testing.T) {
	if got := DefaultForkParityBaseline().MinCacheReadFraction; got != 0.9 {
		t.Fatalf("default fork-parity floor = %v, want 0.9", got)
	}
}

// TestCheckForkFlagsColdWriteNotFirstTurn is the core done-condition witness (#2838): a fork
// that inherited a parent prefix but cold-wrote it (a silent cache miss) is FLAGGED, while a
// genuine first turn with no parent prefix — which cold-writes its whole prefix, expected —
// is NOT. Distinguishing the two is the confusion-risk fence the issue names.
func TestCheckForkFlagsColdWriteNotFirstTurn(t *testing.T) {
	base := DefaultForkParityBaseline()

	// A fork with a parent prefix that cold-wrote it: read fraction 0.0 << floor → cold-write.
	coldFork := ForkTurnUsage{CacheCreationTokens: 40_000, HadParentPrefix: true}
	if v := CheckFork(base, coldFork); !v.ColdWrite {
		t.Fatalf("cold-write fork not flagged: %+v", v)
	}

	// The SAME usage split on a genuine first turn (no parent prefix yet) is expected, not a
	// regression — the fence must not false-positive on it.
	firstTurn := ForkTurnUsage{CacheCreationTokens: 40_000, HadParentPrefix: false}
	if v := CheckFork(base, firstTurn); v.ColdWrite {
		t.Fatalf("first-turn cache write wrongly flagged as a regression: %+v", v)
	}

	// A fork that reused its parent prefix (near-full cache read) passes cleanly.
	reused := ForkTurnUsage{CacheReadTokens: 39_000, CacheCreationTokens: 1_000, HadParentPrefix: true}
	if v := CheckFork(base, reused); v.ColdWrite {
		t.Fatalf("healthy prefix-reusing fork wrongly flagged: %+v (fraction %.4f, floor %.4f)", v, v.Fraction, v.Floor)
	}
}

// TestForkParityBlocksAtFloor pins the fence boundary: a fork exactly at the floor passes
// (>= is honored, float noise absorbed by the epsilon), and one just below it reds.
func TestForkParityBlocksAtFloor(t *testing.T) {
	base := ForkParityBaseline{MinCacheReadFraction: 0.9}

	atFloor := ForkTurnUsage{CacheReadTokens: 9_000, CacheCreationTokens: 1_000, HadParentPrefix: true} // 0.90
	if blocked, reason := ForkParityBlocks(base, atFloor); blocked {
		t.Fatalf("fork exactly at floor should pass, blocked: %s", reason)
	}

	belowFloor := ForkTurnUsage{CacheReadTokens: 8_000, CacheCreationTokens: 2_000, HadParentPrefix: true} // 0.80
	blocked, reason := ForkParityBlocks(base, belowFloor)
	if !blocked {
		t.Fatalf("fork below floor should red")
	}
	if reason == "" {
		t.Fatalf("a blocked fork must carry a cold-write reason")
	}
}

// TestForkParityRegressionsBatch is the fleet-of-forks fence: over a mixed batch it returns
// exactly the indices of the forks that cold-wrote a parent prefix — the #2819 regression
// fence generalized to every forked turn, not just this one Hermes-derived case.
func TestForkParityRegressionsBatch(t *testing.T) {
	base := DefaultForkParityBaseline()
	usages := []ForkTurnUsage{
		{CacheReadTokens: 10_000, HadParentPrefix: true},                            // 0: full reuse — clean
		{CacheCreationTokens: 10_000, HadParentPrefix: true},                        // 1: cold-write — reds
		{CacheCreationTokens: 10_000, HadParentPrefix: false},                       // 2: first turn — expected
		{CacheReadTokens: 1_000, CacheCreationTokens: 9_000, HadParentPrefix: true}, // 3: 0.10 cold-write — reds
	}
	got := ForkParityRegressions(base, usages)
	want := []int{1, 3}
	if len(got) != len(want) {
		t.Fatalf("ForkParityRegressions = %v, want %v", got, want)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("ForkParityRegressions = %v, want %v", got, want)
		}
	}
}

// TestNewForkParityRow proves the durable witness row a live caller appends onto the gateway
// usage seam carries the schema tag, the realized split, the fraction/floor, the lineage bit
// and the cold-write verdict — the one-time "~26%" citation turned into a re-measurable row.
func TestNewForkParityRow(t *testing.T) {
	base := DefaultForkParityBaseline()
	usage := ForkTurnUsage{CacheReadTokens: 1_000, CacheCreationTokens: 9_000, HadParentPrefix: true}
	row := NewForkParityRow(7, base, usage)

	if row.Schema != ForkParitySchema {
		t.Fatalf("row schema = %q, want %q", row.Schema, ForkParitySchema)
	}
	if row.Seq != 7 {
		t.Fatalf("row seq = %d, want 7", row.Seq)
	}
	if row.CacheReadTokens != 1_000 || row.CacheCreationTokens != 9_000 {
		t.Fatalf("row usage split not preserved: %+v", row)
	}
	if row.Fraction != 0.1 {
		t.Fatalf("row fraction = %v, want 0.1", row.Fraction)
	}
	if !row.HadParentPrefix || !row.ColdWrite {
		t.Fatalf("row should record a parent-prefix cold-write: %+v", row)
	}
}
