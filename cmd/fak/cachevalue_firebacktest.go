package main

// cachevalue_firebacktest.go — BACKTEST the CacheBurstPaysBack fire gate against a corpus of
// observed A4 compaction-fire receipts (#2812, epic #2783 workstream D: Score + RSI). The live
// gate in internal/agent/anthropic_compact.go fires a head-anchored cache burst only when
// CacheBurstPaysBack approves it — an EX-ANTE bet that the cached middle a compaction sheds
// forever (a per-turn read saving) repays the one-time cold re-write of the invalidated suffix
// within the remaining session horizon. That bet had never been checked against what the fires
// actually realized. This backtest replays the gate's own decision over a corpus of scored
// per-fire receipts (rsiloop.CompactionFireReceipt — the A4 receipt shape #2817 ships) and
// reports how often the "the burst pays back" claim held: the measured PAY-BACK RATE and the
// count of FALSE-POSITIVE fires (fires the gate approved whose realized net went negative because
// the session ended before the burst repaid). It is a MEASUREMENT ONLY (issue Out-of-scope): it
// does NOT change the predicate's logic; it prices the gate's accuracy so the #2817 tuning pass
// has a validated target.
//
// The gate decision is REUSED, not re-derived: a receipt's PredictedHorizonMargin is
// remainingTurns − CacheBurstBreakEvenTurns, and the untuned CacheBurstPaysBack fires iff that is
// >= 0 — which is exactly rsiloop.FirePolicy.Fires at MinHorizonMargin 0 (BaselineFirePolicy). So
// the backtest gates on the SAME predicate the live gate does, by construction — there is no
// second copy of the fire rule to drift. The realized outcome is the receipt's signed NetScore
// (>= 0 ⇒ the burst paid back), priced on the one canonical cacheprice basis (#2798), so a fire's
// verdict here and the Track-2 per-fire value agree by construction.
//
// GENERATION FRAME (gen/next). Promotion evidence (→ now): run this over a corpus of REAL per-fire
// receipts off live guard sessions (the A4 receipt seam, once internal/cachevaluereport ships it)
// — a measured fleet pay-back rate < 100% with a stable false-positive cluster is what justifies
// wiring the #2817-tuned MinHorizonMargin into the live gate. Demotion/retirement evidence: if the
// measured pay-back rate is ~100% across real corpora (no false-positive fires to bail), the
// ex-ante estimate is already unbiased and both this validation and the #2817 tuner fold away.
// Invalidating assumption: that a per-fire receipt can attribute the realized horizon to the fire
// that shed it — if the shed saving cannot be isolated per fire on the live seam (interleaved
// fires, unattributable session end), the ex-post net is unmeasurable and this stays a harness
// driven by a frozen corpus, not a witnessed fleet number.

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/rsiloop"
)

// cacheBurstBacktest is the 2×2 outcome of replaying one FirePolicy over a receipt corpus: each
// fired receipt is scored PAID-BACK (realized net >= 0) or a FALSE POSITIVE (net < 0), each bailed
// receipt a CORRECT BAIL (net < 0 — a value-destroying fire the gate avoided) or a MISS (net >= 0
// — value left on the table). Pay-back rate and false-positive fires are the two numbers the
// issue's Acceptance names; the bail columns complete the confusion matrix so a reader sees what
// the gate refused as well as what it fired.
type cacheBurstBacktest struct {
	Corpus         int     // receipts backtested
	Margin         int     // the FirePolicy.MinHorizonMargin replayed (0 = untuned CacheBurstPaysBack)
	Fires          int     // receipts the gate APPROVED (PredictedHorizonMargin >= Margin)
	PaidBack       int     // approved fires whose realized net >= 0 (the "pays back" claim held)
	FalsePositives int     // approved fires whose realized net < 0 (fired but did NOT pay back)
	Bails          int     // receipts the gate REFUSED (PredictedHorizonMargin < Margin)
	CorrectBails   int     // refused receipts whose realized net < 0 (correctly avoided)
	MissedFires    int     // refused receipts whose realized net >= 0 (value left on the table)
	RealizedNet    float64 // sum of realized net over the APPROVED fires, tok-eq on the cacheprice basis
}

// PayBackRate is the fraction of the gate's APPROVED fires whose burst actually paid back
// (realized net >= 0) — the headline accuracy of the CacheBurstPaysBack claim. 1.0 means every
// fire the gate approved repaid; a rate below 1.0 means the shortfall are false-positive fires the
// ex-ante estimate could not foresee. An empty fire set is 0 (the gate made no claim to score).
func (b cacheBurstBacktest) PayBackRate() float64 {
	if b.Fires == 0 {
		return 0
	}
	return float64(b.PaidBack) / float64(b.Fires)
}

// backtestCacheBurstPaysBack replays a FirePolicy's fire/bail decision over a corpus of scored A4
// receipts and tallies the confusion matrix. policy is the gate under test —
// rsiloop.BaselineFirePolicy() is the untuned CacheBurstPaysBack (MinHorizonMargin 0); a positive
// margin backtests the #2817-tuned gate over the same corpus. It is PURE (no I/O): the ex-ante
// decision is policy.Fires(r) and the ex-post outcome is r.NetScore(), so the same corpus
// reproduces the same numbers on any box (the determinism the RSI regression gate requires).
func backtestCacheBurstPaysBack(corpus []rsiloop.CompactionFireReceipt, policy rsiloop.FirePolicy) cacheBurstBacktest {
	bt := cacheBurstBacktest{Corpus: len(corpus), Margin: policy.MinHorizonMargin}
	for _, r := range corpus {
		paidBack := r.NetScore() >= 0
		if policy.Fires(r) {
			bt.Fires++
			bt.RealizedNet += r.NetScore()
			if paidBack {
				bt.PaidBack++
			} else {
				bt.FalsePositives++
			}
			continue
		}
		bt.Bails++
		if paidBack {
			bt.MissedFires++
		} else {
			bt.CorrectBails++
		}
	}
	return bt
}

// Summary renders the backtest as a one-line operator readout — pay-back rate and the two failure
// counts first (the #2797 honesty order: name what the gate got wrong before its win rate), then
// the bail split and the realized net. It is the rendered backtest report the issue names as the
// witness, and mirrors rsiloop.TuneResult.Summary so the validation and the tuner read alike.
func (b cacheBurstBacktest) Summary() string {
	return fmt.Sprintf(
		"CacheBurstPaysBack backtest (%d A4 receipts, MinHorizonMargin %d): pay-back rate %.1f%% (%d/%d approved fires paid back, %d false-positive fires); bailed %d (%d correct, %d missed); realized net %+.0f tok-eq",
		b.Corpus, b.Margin,
		b.PayBackRate()*100, b.PaidBack, b.Fires, b.FalsePositives,
		b.Bails, b.CorrectBails, b.MissedFires,
		b.RealizedNet,
	)
}
