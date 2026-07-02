package session

import (
	"strings"
	"testing"
)

// ctxspike_test.go — the #2197 witness for the context-spike advisory fold. The
// acceptance shape: a sudden-and-material context jump folds to Spiked with the right
// numbers and a rendered nudge; suddenness alone (small delta), materiality alone
// (slow ratio), plateaus, shrinks, and baseline-less rings all fold quiet; the nudge
// self-extinguishes after one non-spiking turn; and the Table read is nil-safe and
// fail-open so the historical loop is byte-identical when nothing spikes.

// ringOf builds a ring by pushing each turn's context tokens in order (output tokens
// held at a nominal constant — the fold reads only the context axis).
func ringOf(contexts ...int) CostRing {
	var r CostRing
	for _, c := range contexts {
		r = r.push(TurnCost{OutputTokens: 100, ContextTokens: c})
	}
	return r
}

func TestSpikeAdvisoryQuietWithoutBaseline(t *testing.T) {
	cases := []struct {
		name string
		ring CostRing
	}{
		{"empty ring", CostRing{}},
		{"single turn", ringOf(120000)}, // a big FIRST window cannot be sudden
		{"prev turn had no context accounting", func() CostRing {
			var r CostRing
			r = r.push(TurnCost{OutputTokens: 500}) // output-only debit
			r = r.push(TurnCost{OutputTokens: 500, ContextTokens: 90000})
			return r
		}()},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := tc.ring.SpikeAdvisory(SpikePolicy{})
			if a != (SpikeAdvisory{}) {
				t.Fatalf("want zero advisory, got %+v", a)
			}
			if a.Nudge() != "" {
				t.Fatalf("zero advisory must render an empty nudge, got %q", a.Nudge())
			}
		})
	}
}

func TestSpikeAdvisoryNeedsBothFloors(t *testing.T) {
	cases := []struct {
		name       string
		prev, next int
	}{
		{"smooth growth", 30000, 33000},             // neither floor
		{"material but not sudden", 200000, 230000}, // +30k but ratio 1.15
		{"sudden but immaterial", 2000, 8000},       // 4x but +6k
		{"plateau", 90000, 90000},                   // no growth
		{"shrink (compaction is good, never nudged)", 90000, 20000},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			a := ringOf(tc.prev, tc.next).SpikeAdvisory(SpikePolicy{})
			if a.Spiked {
				t.Fatalf("%d->%d must not spike under defaults, got %+v", tc.prev, tc.next, a)
			}
			if a.Nudge() != "" {
				t.Fatalf("non-spiked advisory must render an empty nudge")
			}
			// The observed numbers still ride out for renderers.
			if a.PrevContext != tc.prev || a.LatestContext != tc.next {
				t.Fatalf("advisory numbers wrong: %+v", a)
			}
		})
	}
}

func TestSpikeAdvisorySpikes(t *testing.T) {
	a := ringOf(30000, 90000).SpikeAdvisory(SpikePolicy{})
	if !a.Spiked {
		t.Fatalf("30k->90k must spike under defaults, got %+v", a)
	}
	if a.PrevContext != 30000 || a.LatestContext != 90000 || a.DeltaTokens != 60000 {
		t.Fatalf("advisory numbers wrong: %+v", a)
	}
	if a.Ratio < 2.99 || a.Ratio > 3.01 {
		t.Fatalf("ratio: want ~3.0, got %g", a.Ratio)
	}
	n := a.Nudge()
	for _, want := range []string{"3.0x", "30000 -> 90000", "+60000", "Advisory only"} {
		if !strings.Contains(n, want) {
			t.Fatalf("nudge missing %q:\n%s", want, n)
		}
	}
	if n2 := a.Nudge(); n2 != n {
		t.Fatalf("nudge must be deterministic")
	}
}

// The advisory is latest-vs-previous by construction, so it fires exactly at the
// boundary after the spiking turn and goes quiet on the next non-spiking one — the
// model is nudged once per spike, not every turn the window stays big.
func TestSpikeAdvisorySelfExtinguishes(t *testing.T) {
	r := ringOf(30000, 90000)
	if !r.SpikeAdvisory(SpikePolicy{}).Spiked {
		t.Fatalf("spike boundary must fire")
	}
	r = r.push(TurnCost{OutputTokens: 100, ContextTokens: 92000}) // plateau turn
	if a := r.SpikeAdvisory(SpikePolicy{}); a.Spiked {
		t.Fatalf("plateau after spike must be quiet, got %+v", a)
	}
}

func TestSpikePolicyDefaultsAndOverrides(t *testing.T) {
	// Zero policy fills the documented defaults.
	p := SpikePolicy{}.withDefaults()
	if p.MinRatio != DefaultSpikeMinRatio || p.MinDeltaTokens != DefaultSpikeMinDeltaTokens {
		t.Fatalf("zero policy must fold to defaults, got %+v", p)
	}
	// A degenerate ratio floor (would fire every turn) falls back.
	if got := (SpikePolicy{MinRatio: 0.5}).withDefaults().MinRatio; got != DefaultSpikeMinRatio {
		t.Fatalf("degenerate MinRatio must fall back, got %g", got)
	}
	// A stricter explicit policy is honored: the default-spiking jump stays quiet.
	strict := SpikePolicy{MinRatio: 5, MinDeltaTokens: 100000}
	if ringOf(30000, 90000).SpikeAdvisory(strict).Spiked {
		t.Fatalf("strict policy must not spike on 3x/+60k")
	}
	// A looser one fires where the default would not.
	loose := SpikePolicy{MinRatio: 1.1, MinDeltaTokens: 1000}
	if !ringOf(30000, 34000).SpikeAdvisory(loose).Spiked {
		t.Fatalf("loose policy must spike on +4k/1.13x")
	}
}

func TestTableContextNudge(t *testing.T) {
	var nilTable *Table
	if got := nilTable.ContextNudge("tr"); got != "" {
		t.Fatalf("nil table must nudge nothing, got %q", got)
	}
	tb := NewTable()
	if got := tb.ContextNudge(""); got != "" {
		t.Fatalf("empty trace must nudge nothing")
	}
	if got := tb.ContextNudge("unseen"); got != "" {
		t.Fatalf("unseen session must nudge nothing, got %q", got)
	}
	// Two debited turns with a sudden material jump: the boundary read nudges.
	tb.DebitUsage("tr", Usage{OutputTokens: 200, ContextTokens: 30000})
	if got := tb.ContextNudge("tr"); got != "" {
		t.Fatalf("single-turn session must nudge nothing, got %q", got)
	}
	tb.DebitUsage("tr", Usage{OutputTokens: 200, ContextTokens: 90000})
	got := tb.ContextNudge("tr")
	if got == "" || !strings.Contains(got, "+60000") {
		t.Fatalf("spiking session must nudge with the delta, got %q", got)
	}
	// Read-only: asking again returns the same nudge, and a quiet turn silences it.
	if again := tb.ContextNudge("tr"); again != got {
		t.Fatalf("ContextNudge must be a pure read")
	}
	tb.DebitUsage("tr", Usage{OutputTokens: 200, ContextTokens: 91000})
	if after := tb.ContextNudge("tr"); after != "" {
		t.Fatalf("plateau turn must silence the nudge, got %q", after)
	}
}
