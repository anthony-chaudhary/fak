package dojo

import "testing"

// Tests for the token-economy/tokens_saved_ratio KPI cell (#4487). They pin the
// registered claim (the RSI rewrite anchor) and the pure fold's classes: the paired
// saving ratio, the honest UNMEASURED without a paired baseline, and the un-clamped
// negative saving that surfaces a levers-cost-more-than-baseline window.

// TestTokenEconomyClaimIsRegisteredEstimate pins the cell's one anchored literal.
// It is an ESTIMATE the RSI loop may recalibrate, never a floor: marking it
// IntentionalFloor would tell the loop this is a guarantee to defend rather than a
// theory to re-point at the measured ratio (#4487's confusion risk — it is a
// CALIBRATION target, not a quality gate).
func TestTokenEconomyClaimIsRegisteredEstimate(t *testing.T) {
	c, ok := Registry.Lookup("token-economy", "tokens_saved_ratio")
	if !ok {
		t.Fatal("token-economy/tokens_saved_ratio did not resolve through the composed Lookup")
	}
	if c.Claimed != 0.30 {
		t.Fatalf("pinned claim drifted: Claimed = %v, want 0.30", c.Claimed)
	}
	if c.IntentionalFloor {
		t.Fatal("the cell is an estimate the RSI loop recalibrates, but it is marked IntentionalFloor")
	}
	if c.LowerIsBetter {
		t.Fatal("a higher tokens-saved ratio is the good outcome, but the cell is marked LowerIsBetter")
	}
	if c.Basis == "" {
		t.Fatal("the cell carries no prose basis")
	}
	// The cell registers additively, so the central literal must not have grown.
	if _, central := Registry[claimKey{"token-economy", "tokens_saved_ratio"}]; central {
		t.Fatal("the cell landed in the central Registry literal; it must register through the additive seam")
	}
}

// TestTokenEconomyEpisodeMeasuresPairedRatio is the cell's core witness: a paired
// corpus (a no-lever OFF baseline AND fak's ON billing) folds one episode carrying
// the registered claim and the realized (off-on)/off saving, WITNESSED (fak's own
// controlled on/off experiment).
func TestTokenEconomyEpisodeMeasuresPairedRatio(t *testing.T) {
	got := TokenEconomyEpisodes(PairedTokenCorpus{OffBaselineTokens: 1000, OnTokens: 700, BaselinePaired: true})
	if len(got) != 1 {
		t.Fatalf("want exactly one episode, got %d: %+v", len(got), got)
	}
	in := got[0]
	if in.Prediction.Lever != "token-economy" || in.Prediction.Metric != "tokens_saved_ratio" {
		t.Fatalf("episode addresses the wrong cell: %+v", in.Prediction)
	}
	if in.Prediction.Claimed != 0.30 {
		t.Fatalf("episode must carry the ONE registered claim, got %v", in.Prediction.Claimed)
	}
	if in.Prediction.Unit != "fraction" {
		t.Fatalf("episode unit = %q, want fraction", in.Prediction.Unit)
	}
	if !in.Outcome.Measured {
		t.Fatalf("a paired corpus must measure, got %+v", in.Outcome)
	}
	if in.Outcome.Realized != 0.3 {
		t.Fatalf("realized = %v, want 0.3 ((1000-700)/1000 saved)", in.Outcome.Realized)
	}
	if in.Outcome.Sample != 1000 {
		t.Fatalf("sample = %d, want 1000 (the OFF-baseline denominator)", in.Outcome.Sample)
	}
	if in.Outcome.Provenance != Witnessed {
		t.Fatalf("provenance = %v, want Witnessed (fak's own on/off experiment)", in.Outcome.Provenance)
	}
}

// TestTokenEconomyNegativeSavingNotClamped pins the honesty rule that a levers-cost-
// MORE-than-baseline window (ON > OFF) surfaces as a negative saving rather than a
// clamped-to-zero "saved nothing" — the gym must not hide fak's levers costing more
// than they save. Through the scorer it lands OVER_CLAIM (reality fell short of the
// claimed saving).
func TestTokenEconomyNegativeSavingNotClamped(t *testing.T) {
	got := TokenEconomyEpisodes(PairedTokenCorpus{OffBaselineTokens: 1000, OnTokens: 1200, BaselinePaired: true})
	if len(got) != 1 || !got[0].Outcome.Measured {
		t.Fatalf("want one measured episode, got %+v", got)
	}
	if got[0].Outcome.Realized != -0.2 {
		t.Fatalf("realized = %v, want -0.2 ((1000-1200)/1000 — a negative saving surfaced, not clamped)", got[0].Outcome.Realized)
	}
	e := Score("paired-corpus", got[0].Prediction, got[0].Outcome, DefaultCalibBand())
	if e.Verdict != VerdictOverClaim {
		t.Fatalf("a negative saving vs a positive claim must score %s, got %s (%+v)", VerdictOverClaim, e.Verdict, e)
	}
}

// TestTokenEconomyUnmeasured pins #4487's done condition: a corpus with no paired
// OFF baseline (and a zero-denominator baseline) scores UNMEASURED, naming the
// missing baseline and carrying the ON-side token count as the sample — never a
// fabricated saving that would let the headline drift from billed reality.
func TestTokenEconomyUnmeasured(t *testing.T) {
	for _, tc := range []struct {
		name       string
		corpus     PairedTokenCorpus
		wantSample int
	}{
		{"ON side seen but no paired OFF baseline", PairedTokenCorpus{OnTokens: 5000, BaselinePaired: false}, 5000},
		{"a paired flag but a zero OFF denominator", PairedTokenCorpus{OnTokens: 5000, OffBaselineTokens: 0, BaselinePaired: true}, 5000},
		{"nothing billed at all", PairedTokenCorpus{}, 0},
	} {
		got := TokenEconomyEpisodes(tc.corpus)
		if len(got) != 1 {
			t.Fatalf("%s: want exactly one UNMEASURED episode, got %d: %+v", tc.name, len(got), got)
		}
		if got[0].Outcome.Measured {
			t.Fatalf("%s: must score UNMEASURED, got %+v", tc.name, got[0].Outcome)
		}
		if got[0].Outcome.Realized != 0 || got[0].Outcome.Source == "" {
			t.Fatalf("%s: an UNMEASURED episode must carry no number and an honest source, got %+v", tc.name, got[0].Outcome)
		}
		if got[0].Outcome.Sample != tc.wantSample {
			t.Fatalf("%s: UNMEASURED episode should carry the ON-side sample %d, got %+v", tc.name, tc.wantSample, got[0].Outcome)
		}
		if got[0].Prediction.Claimed != 0.30 {
			t.Fatalf("%s: the UNMEASURED episode must still carry the registered claim, got %+v", tc.name, got[0].Prediction)
		}
	}
}

// TestTokenEconomyEpisodeScoresAsUnmeasured proves the no-baseline episode survives
// the real scorer as an UNMEASURED verdict — the honesty rule holds end-to-end
// through Score, not just in the fold's own struct. This is the shape `fak dojo run`
// scores today, since no ordinary session corpus carries the no-lever OFF baseline.
func TestTokenEconomyEpisodeScoresAsUnmeasured(t *testing.T) {
	in := TokenEconomyEpisodes(PairedTokenCorpus{OnTokens: 5000, BaselinePaired: false})[0]
	e := Score("paired-corpus", in.Prediction, in.Outcome, DefaultCalibBand())
	if e.Verdict != VerdictUnmeasured {
		t.Fatalf("a corpus with no paired OFF baseline must score %s, got %s (%+v)", VerdictUnmeasured, e.Verdict, e)
	}
}
