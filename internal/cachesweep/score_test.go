package cachesweep

import (
	"math"
	"testing"
)

func TestTokenDensityHalfLifeScore_Basic(t *testing.T) {
	// Zero elapsed time: effectiveHits = 10. (10 + 1) * (1000/500) * 1.0 * 1.0 = 11 * 2 = 22.0
	score := TokenDensityHalfLifeScore(10, 0, 1000, 500, false, false)
	expected := 22.0
	if math.Abs(score-expected) > 1e-6 {
		t.Fatalf("expected score %v, got %v", expected, score)
	}
}

func TestTokenDensityHalfLifeScore_HalfLifeDecay(t *testing.T) {
	hits := 16
	// At exactly 6 hours (21600s), effectiveHits should decay from 16 to 8.
	// (8 + 1) * (100/100) = 9.0
	score := TokenDensityHalfLifeScore(hits, 21600.0, 100, 100, false, false)
	expected := 9.0
	if math.Abs(score-expected) > 1e-6 {
		t.Fatalf("expected score %v after 1 half-life, got %v", expected, score)
	}

	// At 12 hours (43200s), effectiveHits decays to 4. (4 + 1) * 1 = 5.0
	score12h := TokenDensityHalfLifeScore(hits, 43200.0, 100, 100, false, false)
	expected12h := 5.0
	if math.Abs(score12h-expected12h) > 1e-6 {
		t.Fatalf("expected score %v after 2 half-lives, got %v", expected12h, score12h)
	}
}

func TestTokenDensityHalfLifeScore_AnchorAndSuperseded(t *testing.T) {
	base := TokenDensityHalfLifeScore(5, 0, 500, 500, false, false)
	anchor := TokenDensityHalfLifeScore(5, 0, 500, 500, true, false)
	if math.Abs(anchor-(base*2.0)) > 1e-6 {
		t.Fatalf("expected anchor score %v to be 2x base %v", anchor, base)
	}

	superseded := TokenDensityHalfLifeScore(5, 0, 500, 500, false, true)
	if math.Abs(superseded-(base*0.2)) > 1e-6 {
		t.Fatalf("expected superseded score %v to be 0.2x base %v", superseded, base)
	}
}

func TestTokenDensityHalfLifeScore_AnchorSurvivesOverSuperseded(t *testing.T) {
	anchorScore := TokenDensityHalfLifeScore(10, 3600.0, 1000, 200, true, false)
	supersededScore := TokenDensityHalfLifeScore(10, 3600.0, 200, 1000, false, true)

	if anchorScore <= supersededScore {
		t.Fatalf("expected anchor score (%v) > superseded score (%v)", anchorScore, supersededScore)
	}

	ratio := anchorScore / supersededScore
	if ratio < 10.0 {
		t.Fatalf("expected anchor score to dominate superseded score by >10x, got ratio %v", ratio)
	}
}

func TestTokenDensityHalfLifeScore_EdgeCases(t *testing.T) {
	tests := []struct {
		name           string
		hits           int
		elapsedSeconds float64
		tokens         int
		bytes          int
		isAnchor       bool
		isSuperseded   bool
		expectedZero   bool
	}{
		{
			name:         "zero tokens",
			hits:         10,
			tokens:       0,
			bytes:        100,
			expectedZero: true,
		},
		{
			name:         "negative tokens",
			hits:         10,
			tokens:       -50,
			bytes:        100,
			expectedZero: true,
		},
		{
			name:         "zero bytes guarded to 1",
			hits:         0,
			tokens:       10,
			bytes:        0,
			expectedZero: false,
		},
		{
			name:           "negative elapsed seconds treated as zero elapsed",
			hits:           10,
			elapsedSeconds: -100.0,
			tokens:         100,
			bytes:          100,
			expectedZero:   false,
		},
		{
			name:         "negative hits clamped to 0",
			hits:         -5,
			tokens:       100,
			bytes:        100,
			expectedZero: false,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			s := TokenDensityHalfLifeScore(tc.hits, tc.elapsedSeconds, tc.tokens, tc.bytes, tc.isAnchor, tc.isSuperseded)
			if tc.expectedZero && s != 0.0 {
				t.Fatalf("expected 0.0, got %v", s)
			}
			if !tc.expectedZero && s <= 0.0 {
				t.Fatalf("expected positive score, got %v", s)
			}
		})
	}
}
