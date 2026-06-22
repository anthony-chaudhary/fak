package main

// Unit tests for the pure, deterministic helpers in agentdojoredteam:
//   - ratio: the ASR fraction n/d with a divide-by-zero guard.
//   - verdictLabel: the attack-succeeded boolean rendered as a stream label.
// Both are self-contained (stdlib only, no model/network/fixture), so the
// expected values below are computed by hand and the tests fail on regression.

import (
	"math"
	"testing"
)

func TestRatio(t *testing.T) {
	const eps = 1e-9
	tests := []struct {
		name string
		n    int
		d    int
		want float64
	}{
		{"zero numerator", 0, 10, 0.0},
		{"three tenths", 3, 10, 0.3},
		{"full", 10, 10, 1.0},
		{"quarter", 1, 4, 0.25},
		{"one third", 1, 3, 1.0 / 3.0},
		{"greater than one", 7, 2, 3.5},
		{"zero denominator guard", 5, 0, 0.0},
		{"zero over zero guard", 0, 0, 0.0},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ratio(tc.n, tc.d)
			if math.Abs(got-tc.want) > eps {
				t.Errorf("ratio(%d, %d) = %v, want %v", tc.n, tc.d, got, tc.want)
			}
		})
	}
}

func TestVerdictLabel(t *testing.T) {
	tests := []struct {
		name            string
		attackSucceeded bool
		want            string
	}{
		{"attack landed", true, "MISSED"},
		{"attack blocked", false, "caught"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := verdictLabel(tc.attackSucceeded); got != tc.want {
				t.Errorf("verdictLabel(%v) = %q, want %q", tc.attackSucceeded, got, tc.want)
			}
		})
	}
}
