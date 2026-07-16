package main

import (
	"math"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// fireBacktestCorpus is a frozen corpus of scored A4 receipts spanning the four backtest
// quadrants, priced through the real rsiloop.ScoreCompactionFire fold (not hand-set nets) so the
// fixture cannot drift from the cacheprice basis it measures. Read-mult 0.1×, write-mult 1.25×
// (cacheprice defaults): shedSaving = Shed×0.1×Horizon, burstCost = Burst×1.15, net = the two.
//
//   - A: gate-approved (margin 5), long realized horizon → net +3850  (TRUE POSITIVE)
//   - B: gate-approved (margin 1), session ended early    → net  −650  (FALSE POSITIVE)
//   - C: gate-refused  (margin −2), short horizon         → net  −850  (CORRECT BAIL)
//   - D: gate-approved (margin 3), healthy horizon        → net +1025  (TRUE POSITIVE)
func fireBacktestCorpus() []rsiloop.CompactionFireReceipt {
	return []rsiloop.CompactionFireReceipt{
		rsiloop.ScoreCompactionFire(rsiloop.CompactionFireObs{Seq: 1, PredictedHorizonMargin: 5, ShedTokens: 1000, BurstTokens: 1000, ObservedHorizonTurns: 50}),
		rsiloop.ScoreCompactionFire(rsiloop.CompactionFireObs{Seq: 2, PredictedHorizonMargin: 1, ShedTokens: 1000, BurstTokens: 1000, ObservedHorizonTurns: 5}),
		rsiloop.ScoreCompactionFire(rsiloop.CompactionFireObs{Seq: 3, PredictedHorizonMargin: -2, ShedTokens: 1000, BurstTokens: 1000, ObservedHorizonTurns: 3}),
		rsiloop.ScoreCompactionFire(rsiloop.CompactionFireObs{Seq: 4, PredictedHorizonMargin: 3, ShedTokens: 800, BurstTokens: 500, ObservedHorizonTurns: 20}),
	}
}

// TestBacktestCacheBurstPaysBackQuantifiesFalsePositives is the #2812 acceptance witness: the
// backtest of the untuned CacheBurstPaysBack gate (BaselineFirePolicy) over the frozen A4 corpus
// yields a MEASURED pay-back rate and a QUANTIFIED false-positive fire count — the two the issue's
// Acceptance names. Three fires were approved (A, B, D); one (B) did not pay back, so the measured
// rate is 2/3 and there is exactly one false-positive fire.
func TestBacktestCacheBurstPaysBackQuantifiesFalsePositives(t *testing.T) {
	bt := backtestCacheBurstPaysBack(fireBacktestCorpus(), rsiloop.BaselineFirePolicy())

	if bt.Corpus != 4 {
		t.Fatalf("corpus = %d, want 4", bt.Corpus)
	}
	if bt.Fires != 3 {
		t.Errorf("approved fires = %d, want 3 (A, B, D have PredictedHorizonMargin >= 0)", bt.Fires)
	}
	if bt.PaidBack != 2 {
		t.Errorf("paid-back fires = %d, want 2 (A, D)", bt.PaidBack)
	}
	if bt.FalsePositives != 1 {
		t.Errorf("false-positive fires = %d, want 1 (B: approved but realized net < 0)", bt.FalsePositives)
	}
	if want := 2.0 / 3.0; math.Abs(bt.PayBackRate()-want) > 1e-9 {
		t.Errorf("pay-back rate = %.6f, want %.6f", bt.PayBackRate(), want)
	}
	// The refused fire (C) had a negative realized net, so the gate was right to bail it.
	if bt.Bails != 1 || bt.CorrectBails != 1 || bt.MissedFires != 0 {
		t.Errorf("bails = %d (correct %d, missed %d), want 1/1/0", bt.Bails, bt.CorrectBails, bt.MissedFires)
	}
	// Realized net over the approved fires: +3850 (A) − 650 (B) + 1025 (D) = +4225 tok-eq.
	if math.Abs(bt.RealizedNet-4225) > 1e-6 {
		t.Errorf("realized net = %.1f, want 4225", bt.RealizedNet)
	}

	got := bt.Summary()
	for _, want := range []string{"pay-back rate 66.7%", "1 false-positive fires", "4 A4 receipts", "realized net +4225 tok-eq"} {
		if !strings.Contains(got, want) {
			t.Errorf("Summary() missing %q\n got: %s", want, got)
		}
	}
}

// TestBacktestMarginBailsTheFalsePositive shows the backtest is sensitive to the fire/bail
// threshold #2817 tunes: replaying the SAME A4 corpus at MinHorizonMargin 2 (a hedged gate) bails
// the thin-headroom false-positive fire B, lifting the measured pay-back rate to 100% and RAISING
// realized net — the evidence that would justify feeding a tuned margin back into the live gate.
func TestBacktestMarginBailsTheFalsePositive(t *testing.T) {
	corpus := fireBacktestCorpus()
	base := backtestCacheBurstPaysBack(corpus, rsiloop.BaselineFirePolicy())
	hedged := backtestCacheBurstPaysBack(corpus, rsiloop.FirePolicy{MinHorizonMargin: 2})

	if hedged.FalsePositives != 0 {
		t.Errorf("hedged false-positive fires = %d, want 0 (margin 2 bails B)", hedged.FalsePositives)
	}
	if hedged.PayBackRate() != 1.0 {
		t.Errorf("hedged pay-back rate = %.3f, want 1.0", hedged.PayBackRate())
	}
	if !(hedged.RealizedNet > base.RealizedNet) {
		t.Errorf("hedged realized net %.0f should exceed baseline %.0f (bailing a negative-net fire lifts net)", hedged.RealizedNet, base.RealizedNet)
	}
}

// TestBacktestEmptyCorpusIsZero pins the degenerate case: no receipts ⇒ no claim to score ⇒
// pay-back rate 0 and no false positives, never a divide-by-zero.
func TestBacktestEmptyCorpusIsZero(t *testing.T) {
	bt := backtestCacheBurstPaysBack(nil, rsiloop.BaselineFirePolicy())
	if bt.Fires != 0 || bt.FalsePositives != 0 || bt.PayBackRate() != 0 {
		t.Errorf("empty corpus = %+v, want zero fires/false-positives/rate", bt)
	}
}
