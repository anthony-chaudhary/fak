package compute

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

func referenceDenseSlidingWindowAttention(
	q []float32,
	keys [][]float32,
	values [][]float32,
	headDim int,
	scale float32,
	windowSize int,
	queryPos int,
) []float32 {
	totalPositions := len(keys)
	winMin := queryPos - windowSize + 1
	if winMin < 0 {
		winMin = 0
	}
	winMax := queryPos

	scores := make([]float32, totalPositions)
	hasValid := false
	for j := 0; j < totalPositions; j++ {
		if j < winMin || j > winMax {
			scores[j] = float32(math.Inf(-1)) // masked
		} else {
			var d float32
			for e := 0; e < headDim; e++ {
				d += q[e] * keys[j][e]
			}
			scores[j] = d * scale
			hasValid = true
		}
	}

	if !hasValid {
		return make([]float32, headDim)
	}

	mx := float32(math.Inf(-1))
	for _, s := range scores {
		if s > mx {
			mx = s
		}
	}
	var sum float64
	probs := make([]float32, totalPositions)
	for j, s := range scores {
		if math.IsInf(float64(s), -1) {
			probs[j] = 0
		} else {
			e := math.Exp(float64(s - mx))
			probs[j] = float32(e)
			sum += e
		}
	}
	for j := range probs {
		if probs[j] > 0 {
			probs[j] = float32(float64(probs[j]) / sum)
		}
	}

	out := make([]float32, headDim)
	for j := 0; j < totalPositions; j++ {
		if probs[j] > 0 {
			w := probs[j]
			for e := 0; e < headDim; e++ {
				out[e] += w * values[j][e]
			}
		}
	}
	return out
}

func TestSlidingWindowTileSkipWitness(t *testing.T) {
	// First witness requirements (#9928):
	// 1. Exact dense-reference outputs on visible keys.
	// 2. Sentinel canaries in skipped tiles remain completely unaccessed.
	// 3. Matched long-context decode showing work proportional to window, not full context.

	rng := rand.New(rand.NewSource(4242))

	totalPositions := 8192
	windowSize := 512
	tileSize := 64
	headDim := 64
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	q := make([]float32, headDim)
	for e := range q {
		q[e] = rng.Float32()*2 - 1
	}

	keys := make([][]float32, totalPositions)
	values := make([][]float32, totalPositions)
	for j := range keys {
		keys[j] = make([]float32, headDim)
		values[j] = make([]float32, headDim)
		for e := range keys[j] {
			keys[j][e] = rng.Float32()*2 - 1
			values[j][e] = rng.Float32()*2 - 1
		}
	}

	queryPos := 7500 // late in the sequence: most tiles are far in the past and should be skipped!

	// Plant sentinel canaries on positions that should be skipped (e.g. j < queryPos - windowSize + 1)
	winMin := queryPos - windowSize + 1
	canaryPositions := make(map[int]bool)
	for j := 0; j < winMin-tileSize; j++ { // strictly in fully skipped tiles
		canaryPositions[j] = true
	}

	canaryAccessed := false
	accessHook := func(pos int) {
		if canaryPositions[pos] {
			canaryAccessed = true
		}
	}

	gotOut, receipt, err := ApplySlidingWindowTileMask(
		q, keys, values, headDim, scale, windowSize, tileSize, queryPos, accessHook,
	)
	if err != nil {
		t.Fatalf("ExecuteSlidingWindowTileAttention failed: %v", err)
	}

	// 2. Verify sentinel canaries: NO canary position in skipped tiles was ever accessed!
	if canaryAccessed {
		t.Fatal("sentinel canary was accessed; whole-tile skip violated")
	}

	// 1. Verify exact parity against dense reference oracle
	wantOut := referenceDenseSlidingWindowAttention(q, keys, values, headDim, scale, windowSize, queryPos)
	for e := 0; e < headDim; e++ {
		if math.Abs(float64(gotOut[e]-wantOut[e])) > 1e-5 {
			t.Fatalf("output mismatch at dim %d: got %v, want %v", e, gotOut[e], wantOut[e])
		}
	}

	// 3. Verify work is proportional to window size, not full context
	// For totalPositions=8192, tileSize=64 -> 128 total tiles
	// Window = 512 -> at most ~ (512/64 + 2) = 10 active tiles (processed)
	// Skipped tiles should be > 110 tiles (~90% skipped)
	if receipt.TotalTiles != 128 {
		t.Fatalf("expected 128 total tiles, got %d", receipt.TotalTiles)
	}
	if receipt.SkippedTiles < 110 {
		t.Fatalf("expected at least 110 skipped tiles, got %d (ratio %v)", receipt.SkippedTiles, receipt.SkipRatio)
	}
	if receipt.FullTiles+receipt.PartialTiles > 15 {
		t.Fatalf("work not bounded by window: processed %d tiles for window %d", receipt.FullTiles+receipt.PartialTiles, windowSize)
	}
	if receipt.SkipRatio < 0.85 {
		t.Fatalf("expected skip ratio >= 85%%, got %v", receipt.SkipRatio)
	}
}

func TestSlidingWindowTileBoundsSweep(t *testing.T) {
	// Boundary test across edge query positions
	testCases := []struct {
		totalPos int
		queryPos int
		window   int
		tileSize int
	}{
		{totalPos: 64, queryPos: 0, window: 16, tileSize: 16},
		{totalPos: 128, queryPos: 15, window: 32, tileSize: 16},
		{totalPos: 256, queryPos: 127, window: 64, tileSize: 16},
		{totalPos: 512, queryPos: 511, window: 128, tileSize: 32},
	}

	for i, tc := range testCases {
		t.Run(fmt.Sprintf("case_%d", i), func(t *testing.T) {
			tiles, receipt, err := DeriveSlidingWindowTiles(tc.totalPos, tc.queryPos, tc.window, tc.tileSize)
			if err != nil {
				t.Fatalf("DeriveSlidingWindowTiles failed: %v", err)
			}

			if receipt.TotalPositions != tc.totalPos || receipt.QueryPos != tc.queryPos {
				t.Fatalf("receipt mismatch: %+v", receipt)
			}
			if len(tiles) != receipt.TotalTiles {
				t.Fatalf("tile slice length %d != total_tiles %d", len(tiles), receipt.TotalTiles)
			}
			if receipt.SkippedTiles+receipt.FullTiles+receipt.PartialTiles != receipt.TotalTiles {
				t.Fatalf("tile counts don't sum to total: %+v", receipt)
			}
		})
	}
}
