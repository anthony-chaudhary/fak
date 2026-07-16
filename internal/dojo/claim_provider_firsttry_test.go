package dojo

import "testing"

// Tests for the provider-firsttry/first_try_green_rate cross-provider KPI cell
// (#4494). They pin the registered claim (the RSI rewrite anchor) and the pure
// fold's failure classes: the per-provider rate, the no-retry semantics, and the
// honest UNMEASURED on an empty attempt ledger.

// TestProviderFirstTryClaimIsRegisteredEstimate pins the cell's one anchored
// literal. It is an ESTIMATE the RSI loop may recalibrate, never a floor: marking
// it IntentionalFloor would tell the loop this is a guarantee to defend rather
// than a theory to re-point at the measured rate (#4494's confusion risk).
func TestProviderFirstTryClaimIsRegisteredEstimate(t *testing.T) {
	c, ok := Registry.Lookup("provider-firsttry", "first_try_green_rate")
	if !ok {
		t.Fatal("provider-firsttry/first_try_green_rate did not resolve through the composed Lookup")
	}
	if c.Claimed != 0.5 {
		t.Fatalf("pinned claim drifted: Claimed = %v, want 0.5", c.Claimed)
	}
	if c.IntentionalFloor {
		t.Fatal("the cell is an estimate the RSI loop recalibrates, but it is marked IntentionalFloor")
	}
	if c.LowerIsBetter {
		t.Fatal("a higher first-try green rate is the good outcome, but the cell is marked LowerIsBetter")
	}
	if c.Basis == "" {
		t.Fatal("the cell carries no prose basis")
	}
	// The cell registers additively, so the central literal must not have grown.
	if _, central := Registry[claimKey{"provider-firsttry", "first_try_green_rate"}]; central {
		t.Fatal("the cell landed in the central Registry literal; it must register through the additive seam")
	}
}

// TestProviderFirstTryGreenSemantics pins the numerator's membership test: only a
// worker that went green on its FIRST attempt counts. A retry that eventually went
// green does not (the KPI is single-shot quality, not eventual success), and a
// worker that never went green never counts.
func TestProviderFirstTryGreenSemantics(t *testing.T) {
	for _, tc := range []struct {
		name string
		a    FirstTryAttempt
		want bool
	}{
		{"green on the first attempt", FirstTryAttempt{Provider: "claude", Attempts: 1, Green: true}, true},
		{"green only after a retry", FirstTryAttempt{Provider: "claude", Attempts: 2, Green: true}, false},
		{"red on the first attempt", FirstTryAttempt{Provider: "claude", Attempts: 1, Green: false}, false},
		{"red after many retries", FirstTryAttempt{Provider: "claude", Attempts: 5, Green: false}, false},
	} {
		if got := tc.a.FirstTryGreen(); got != tc.want {
			t.Fatalf("%s: FirstTryGreen() = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestProviderFirstTryEpisodesPerProvider is the cell's core witness: one episode
// per provider, each carrying the same registered claim and that provider's OWN
// realized first-try rate, in deterministic provider order — and no cross-provider
// contamination.
func TestProviderFirstTryEpisodesPerProvider(t *testing.T) {
	got := ProviderFirstTryEpisodes([]FirstTryAttempt{
		// claude: 3 of 4 attempts green with no retry -> 0.75
		{Provider: "claude", Attempts: 1, Green: true},
		{Provider: "claude", Attempts: 1, Green: true},
		{Provider: "claude", Attempts: 1, Green: true},
		{Provider: "claude", Attempts: 3, Green: true}, // retried: green, but not first-try
		// glm: 1 of 2 -> 0.5
		{Provider: "glm", Attempts: 1, Green: true},
		{Provider: "glm", Attempts: 1, Green: false},
	})
	if len(got) != 2 {
		t.Fatalf("want one episode per provider (claude, glm), got %d: %+v", len(got), got)
	}
	// Deterministic, sorted provider order.
	want := []struct {
		realized float64
		sample   int
	}{{0.75, 4}, {0.5, 2}}
	for i, in := range got {
		if in.Prediction.Lever != "provider-firsttry" || in.Prediction.Metric != "first_try_green_rate" {
			t.Fatalf("episode %d addresses the wrong cell: %+v", i, in.Prediction)
		}
		if in.Prediction.Claimed != 0.5 {
			t.Fatalf("episode %d must carry the ONE registered claim, got %v", i, in.Prediction.Claimed)
		}
		if in.Prediction.Unit != "fraction" {
			t.Fatalf("episode %d unit = %q, want fraction", i, in.Prediction.Unit)
		}
		if !in.Outcome.Measured {
			t.Fatalf("episode %d has attempts behind it but scored UNMEASURED: %+v", i, in.Outcome)
		}
		if in.Outcome.Realized != want[i].realized {
			t.Fatalf("episode %d realized = %v, want %v", i, in.Outcome.Realized, want[i].realized)
		}
		if in.Outcome.Sample != want[i].sample {
			t.Fatalf("episode %d sample = %d, want %d", i, in.Outcome.Sample, want[i].sample)
		}
		if in.Outcome.Provenance != Observed {
			t.Fatalf("episode %d provenance = %v, want Observed (the provider's own dispatch record)", i, in.Outcome.Provenance)
		}
	}
}

// TestProviderFirstTryEmptyLedgerIsUnmeasured pins #4494's honesty rule: no
// attempts yields ONE honest UNMEASURED episode, never a fabricated 0.0 rate that
// would slander every provider as never landing clean.
func TestProviderFirstTryEmptyLedgerIsUnmeasured(t *testing.T) {
	for _, tc := range []struct {
		name string
		in   []FirstTryAttempt
	}{
		{"a ledger with no rows at all", nil},
		{"rows with no provider to key by", []FirstTryAttempt{{Attempts: 1, Green: true}}},
		{"rows that were never dispatched", []FirstTryAttempt{{Provider: "claude", Attempts: 0, Green: true}}},
	} {
		got := ProviderFirstTryEpisodes(tc.in)
		if len(got) != 1 {
			t.Fatalf("%s: want exactly one UNMEASURED episode, got %d: %+v", tc.name, len(got), got)
		}
		if got[0].Outcome.Measured {
			t.Fatalf("%s: an empty attempt ledger must score UNMEASURED, got %+v", tc.name, got[0].Outcome)
		}
		if got[0].Outcome.Realized != 0 || got[0].Outcome.Source == "" {
			t.Fatalf("%s: an UNMEASURED episode must carry no number and an honest source, got %+v", tc.name, got[0].Outcome)
		}
		if got[0].Prediction.Claimed != 0.5 {
			t.Fatalf("%s: the UNMEASURED episode must still carry the registered claim, got %+v", tc.name, got[0].Prediction)
		}
	}
}

// TestProviderFirstTryEpisodesScoreAsUnmeasured proves the empty-ledger episode
// survives the real scorer as an UNMEASURED verdict — the honesty rule holds
// end-to-end through Score, not just in the fold's own struct.
func TestProviderFirstTryEpisodesScoreAsUnmeasured(t *testing.T) {
	in := ProviderFirstTryEpisodes(nil)[0]
	e := Score("attempt-ledger", in.Prediction, in.Outcome, DefaultCalibBand())
	if e.Verdict != VerdictUnmeasured {
		t.Fatalf("an empty attempt ledger must score %s, got %s (%+v)", VerdictUnmeasured, e.Verdict, e)
	}
}
