// prefixlint_test.go — unit tests for the pure §A3 report renderer.
//
// renderReport is a deterministic, resource-free string formatter: given a
// cachemeta.StabilityReport and the per-turn prompt segments, it produces the
// human-facing prefix-stability report. These tests pin its exact output,
// covering both the "prefix broke" and "never broke" header branches and the
// per-turn volatile-ahead recommendation line.
package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

func TestRenderReport(t *testing.T) {
	tests := []struct {
		name  string
		turns [][]cachemeta.PromptSegment
		rep   cachemeta.StabilityReport
		want  string
	}{
		{
			name:  "broke at a turn, no per-turn lines",
			turns: nil, // empty turns => the per-turn recommendation loop is skipped
			rep: cachemeta.StabilityReport{
				Turns:             3,
				CacheableTokens:   300,
				LostTokens:        120,
				BrokeAtTurn:       1,
				RecoverableTokens: 600,
			},
			want: "prefix-stability report (3 turns)\n" +
				"  cacheable tokens across session : 300\n" +
				"  re-billed (lost) tokens         : 120\n" +
				"  prefix first broke at turn      : 1\n" +
				"  recoverable by reorder (uplift) : 600 tokens\n",
		},
		{
			name:  "never broke uses the clean-session line",
			turns: nil,
			rep: cachemeta.StabilityReport{
				Turns:             2,
				CacheableTokens:   500,
				LostTokens:        0,
				BrokeAtTurn:       -1,
				RecoverableTokens: 0,
			},
			want: "prefix-stability report (2 turns)\n" +
				"  cacheable tokens across session : 500\n" +
				"  re-billed (lost) tokens         : 0\n" +
				"  prefix first broke at turn      : (never — clean across the session)\n" +
				"  recoverable by reorder (uplift) : 0 tokens\n",
		},
		{
			name: "per-turn line emitted for a volatile-ahead turn",
			// One turn: a volatile segment ahead of a stable segment. RecommendLayout
			// hoists the single volatile segment to the tail, turning a 0-token front
			// prefix into a 100-token one (uplift +100), so the loop emits one line.
			turns: [][]cachemeta.PromptSegment{
				{
					{Kind: cachemeta.SegVolatile, Tokens: 6},
					{Kind: cachemeta.SegStable, Tokens: 100},
				},
			},
			rep: cachemeta.StabilityReport{
				Turns:             1,
				CacheableTokens:   0,
				LostTokens:        0,
				BrokeAtTurn:       -1,
				RecoverableTokens: 100,
			},
			want: "prefix-stability report (1 turns)\n" +
				"  cacheable tokens across session : 0\n" +
				"  re-billed (lost) tokens         : 0\n" +
				"  prefix first broke at turn      : (never — clean across the session)\n" +
				"  recoverable by reorder (uplift) : 100 tokens\n" +
				"  turn 0: move 1 volatile segment(s) to the tail -> +100 cacheable tokens\n",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := renderReport(tc.turns, tc.rep)
			if got != tc.want {
				t.Errorf("renderReport mismatch\n got: %q\nwant: %q", got, tc.want)
			}
		})
	}
}
