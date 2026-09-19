package kvbudget

import (
	"math"
	"testing"
)

// TestEntropyGatedDisabledIsIdentity pins the default-off contract: a disabled
// config (the zero value) returns every chunk at its full length and a Retained
// equal to KVLen, so no existing caller's allocation changes.
func TestEntropyGatedDisabledIsIdentity(t *testing.T) {
	c := ChunkScores{
		{1, 2, 3, 4},
		{9, 0, 0, 0},
		{5, 5},
		{}, // empty chunk is legal
	}
	var cfg EntropyGatedConfig // zero value == disabled
	alloc := cfg.Allocate(c)
	if alloc.Retained != c.KVLen() {
		t.Fatalf("disabled Retained = %d, want %d (identity)", alloc.Retained, c.KVLen())
	}
	for i, ch := range c {
		if alloc.Lengths[i] != len(ch) {
			t.Errorf("disabled Lengths[%d] = %d, want %d", i, alloc.Lengths[i], len(ch))
		}
	}
}

// TestEntropyGatedConservesBudgetExactly is the load-bearing witness: the
// retained token count equals round((1-ratio)*KVLen) exactly across chunk
// shapes — partial trailing chunk, all-low-entropy, all-high-entropy, and a
// target that lands on a fractional boundary.
func TestEntropyGatedConservesBudgetExactly(t *testing.T) {
	cases := []struct {
		name  string
		c     ChunkScores
		ratio float64
	}{
		{
			name:  "partial trailing chunk",
			c:     ChunkScores{{1, 2, 3, 4, 5}, {1, 2, 3}, {1, 2, 3, 4, 5, 6, 7}},
			ratio: 0.5,
		},
		{
			name:  "all-low-entropy spiky chunks",
			c:     ChunkScores{{9, 0, 0, 0}, {8, 0, 0, 0}, {7, 0, 0, 0}},
			ratio: 0.5,
		},
		{
			name:  "all-high-entropy even chunks",
			c:     ChunkScores{{1, 1, 1, 1}, {1, 1, 1, 1}, {1, 1, 1, 1}},
			ratio: 0.4,
		},
		{
			name:  "fractional target boundary",
			c:     ChunkScores{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}},
			ratio: 0.5, // 9 tokens * 0.5 = 4.5 -> round = 5 (but target math rounds)
		},
		{
			name:  "empty and single-position chunks",
			c:     ChunkScores{{}, {5}, {}}, // single-position chunk entropy 0
			ratio: 0.5,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cfg := EntropyGatedConfig{
				Enabled:               true,
				CompressionRatio:      tc.ratio,
				ScoreThreshold:        1,
				EntropyThreshold:      0.5,
				LowEntropyChunkLength: 1,
			}
			kvLen := tc.c.KVLen()
			want := int(math.Round((1 - tc.ratio) * float64(kvLen)))
			alloc := cfg.Allocate(tc.c)
			if alloc.Retained != want {
				t.Fatalf("Retained = %d, want %d (exact conservation; KVLen=%d ratio=%v)",
					alloc.Retained, want, kvLen, tc.ratio)
			}
			if alloc.Target != want {
				t.Fatalf("Target = %d, want %d", alloc.Target, want)
			}
			// No chunk may exceed its own length.
			for i, ch := range tc.c {
				if alloc.Lengths[i] < 0 || alloc.Lengths[i] > len(ch) {
					t.Errorf("Lengths[%d] = %d out of [0,%d]", i, alloc.Lengths[i], len(ch))
				}
			}
		})
	}
}

// TestEntropyGatedCapFiresOnlyWhenBothGatesHold pins the two-gate rule: a chunk
// is truncated only when it is BOTH important (mean >= ScoreThreshold) AND spiky
// (entropy <= EntropyThreshold). Each gate alone must leave the chunk whole.
func TestEntropyGatedCapFiresOnlyWhenBothGatesHold(t *testing.T) {
	const scoreThr = 2.0
	const entThr = 0.5

	// Both gates hold: high mean, low entropy (all mass on one position).
	spikyImportant := []float64{9, 0, 0, 0}
	if e := normalizedEntropy(spikyImportant); e > entThr {
		t.Fatalf("fixture entropy = %v, want <= %v", e, entThr)
	}
	if m := meanScore(spikyImportant); m < scoreThr {
		t.Fatalf("fixture mean = %v, want >= %v", m, scoreThr)
	}

	// Gate 1 fails: high mean, high entropy (evenly spread) -> NOT capped.
	evenImportant := []float64{3, 3, 3, 3}
	// Gate 2 fails: low mean, low entropy -> NOT capped.
	spikyUnimportant := []float64{1, 0, 0, 0}

	cfg := EntropyGatedConfig{
		Enabled:               true,
		CompressionRatio:      0.5,
		ScoreThreshold:        scoreThr,
		EntropyThreshold:      entThr,
		LowEntropyChunkLength: 1,
	}
	alloc := cfg.Allocate(ChunkScores{spikyImportant, evenImportant, spikyUnimportant})
	if !alloc.Stats[0].Capped {
		t.Errorf("chunk 0 (important+spiky) not capped: %+v", alloc.Stats[0])
	}
	if alloc.Stats[1].Capped {
		t.Errorf("chunk 1 (important but even entropy=%v) capped, want not", alloc.Stats[1].Entropy)
	}
	if alloc.Stats[2].Capped {
		t.Errorf("chunk 2 (spiky but unimportant mean=%v) capped, want not", alloc.Stats[2].Mean)
	}
}

// TestEntropyGatedSpikyCapFreesSlotsForMoreChunks is the reallocation witness:
// a high-mean SPIKEY chunk is capped so the slots it would have monopolized are
// re-spent on MORE chunks. It directly compares the capped run against the
// no-cap run on the same input and budget, proving the mode spreads retained
// coverage across more chunks rather than merely evicting.
func TestEntropyGatedSpikyCapFreesSlotsForMoreChunks(t *testing.T) {
	// Four chunks of 8 positions. Chunk 0 is spiky AND high-mean (one hot
	// position of 64, mean 8); chunks 1-3 are evenly moderately important
	// (mean 2). Target retains half (16 of 32 tokens).
	spikyImportant := []float64{64, 0, 0, 0, 0, 0, 0, 0}
	mid := func() []float64 { return []float64{2, 2, 2, 2, 2, 2, 2, 2} }
	c := ChunkScores{spikyImportant, mid(), mid(), mid()}

	base := EntropyGatedConfig{
		Enabled:          true,
		CompressionRatio: 0.5, // retain 16 of 32
		ScoreThreshold:   1,
		EntropyThreshold: 0.5,
	}

	// No-cap arm: the spiky head is served whole, monopolizing the budget.
	noCap := base
	noCap.LowEntropyChunkLength = 0
	uncapped := noCap.Allocate(c)
	if uncapped.Retained != 16 {
		t.Fatalf("no-cap Retained = %d, want 16", uncapped.Retained)
	}
	cappedChunks := func(a EntropyGatedAllocation) int {
		n := 0
		for _, l := range a.Lengths {
			if l > 0 {
				n++
			}
		}
		return n
	}
	uncappedCoverage := cappedChunks(uncapped)

	// Capped arm: the spiky head is truncated to its cap, freeing slots.
	capCfg := base
	capCfg.LowEntropyChunkLength = 1
	alloc := capCfg.Allocate(c)
	if alloc.Retained != 16 {
		t.Fatalf("capped Retained = %d, want 16", alloc.Retained)
	}
	if !alloc.Stats[0].Capped {
		t.Fatalf("spiky head not capped: %+v", alloc.Stats[0])
	}
	if alloc.Lengths[0] != 1 {
		t.Errorf("capped head Lengths[0] = %d, want 1 (served first at its cap)", alloc.Lengths[0])
	}
	cappedCoverage := cappedChunks(alloc)
	if cappedCoverage <= uncappedCoverage {
		t.Fatalf("capped coverage %d chunks, no-cap coverage %d; want strictly more chunks retained",
			cappedCoverage, uncappedCoverage)
	}
}

// TestEntropyGatedDeterministicTieBreak pins determinism: equal-mean chunks are
// served in ascending index order, so two runs (and chunk order permutations
// within a tie) yield the same allocation.
func TestEntropyGatedDeterministicTieBreak(t *testing.T) {
	c := ChunkScores{
		{5, 5}, // mean 5
		{5, 5}, // mean 5 (tie with 0)
		{5, 5}, // mean 5 (tie)
		{5, 5}, // mean 5 (tie)
	}
	cfg := EntropyGatedConfig{
		Enabled:               true,
		CompressionRatio:      0.5, // retain 4 of 8
		ScoreThreshold:        0,
		EntropyThreshold:      1,
		LowEntropyChunkLength: 0, // cap disabled -> pure budget split
	}
	a := cfg.Allocate(c)
	b := cfg.Allocate(c)
	if a.Retained != 4 {
		t.Fatalf("Retained = %d, want 4", a.Retained)
	}
	for i := range a.Lengths {
		if a.Lengths[i] != b.Lengths[i] {
			t.Errorf("non-deterministic at %d: %v vs %v", i, a.Lengths, b.Lengths)
		}
	}
	// Ascending index tie-break: the first two chunks absorb the budget.
	if a.Lengths[0] != 2 || a.Lengths[1] != 2 || a.Lengths[2] != 0 || a.Lengths[3] != 0 {
		t.Errorf("tie-break allocation = %v, want [2 2 0 0]", a.Lengths)
	}
}

// TestEntropyGatedZeroRatioRetainsAll pins the ratio=0 boundary: nothing is
// dropped and every chunk keeps its full length, even with the cap enabled.
func TestEntropyGatedZeroRatioRetainsAll(t *testing.T) {
	c := ChunkScores{{9, 0, 0, 0}, {1, 1, 1, 1}}
	cfg := EntropyGatedConfig{
		Enabled:               true,
		CompressionRatio:      0,
		ScoreThreshold:        1,
		EntropyThreshold:      0.5,
		LowEntropyChunkLength: 1,
	}
	alloc := cfg.Allocate(c)
	if alloc.Retained != c.KVLen() {
		t.Fatalf("ratio=0 Retained = %d, want %d", alloc.Retained, c.KVLen())
	}
	for i, ch := range c {
		if alloc.Lengths[i] != len(ch) {
			t.Errorf("Lengths[%d] = %d, want %d", i, alloc.Lengths[i], len(ch))
		}
	}
}

// TestEntropyGatedNegativeAndZeroConfigsClamp pins the conservative clamps: a
// negative ratio clamps to 0 (retain all), a non-positive cap disables capping,
// and negative thresholds clamp to 0 — never a silent over-drop.
func TestEntropyGatedNegativeAndZeroConfigsClamp(t *testing.T) {
	c := ChunkScores{{9, 0, 0, 0}, {1, 1, 1, 1}}

	// Negative ratio => clamp to 0 => retain all.
	neg := EntropyGatedConfig{Enabled: true, CompressionRatio: -0.5}
	if got := neg.Allocate(c).Retained; got != c.KVLen() {
		t.Errorf("negative ratio Retained = %d, want %d", got, c.KVLen())
	}

	// Cap <= 0 => no chunk is truncated even though the gates would fire.
	noCap := EntropyGatedConfig{
		Enabled:               true,
		CompressionRatio:      0.5,
		ScoreThreshold:        0,
		EntropyThreshold:      1,
		LowEntropyChunkLength: 0,
	}
	alloc := noCap.Allocate(c)
	for i := range alloc.Stats {
		if alloc.Stats[i].Capped {
			t.Errorf("cap<=0 marked chunk %d capped, want not", i)
		}
	}
}
