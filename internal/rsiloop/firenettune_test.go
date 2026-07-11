package rsiloop

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cacheprice"
)

// approxEq (shared with forksaving_test.go) compares token-equivalent floats within a tolerance
// far tighter than any figure these fixtures separate on.

// TestScoreCompactionFireNetSign witnesses the pure fold: the net score is B (shed read-rebate
// over the realized horizon) minus A2 (one-time burst premium), on the canonical cacheprice
// basis, and it KEEPS ITS SIGN — a value-destroying fire scores negative.
func TestScoreCompactionFireNetSign(t *testing.T) {
	// A short-horizon fire that cold-wrote as much as it shed: the burst premium eats the thin
	// rebate, so the realized net is negative.
	neg := ScoreCompactionFire(CompactionFireObs{
		Seq: 3, PredictedHorizonMargin: 1,
		ShedTokens: 1000, BurstTokens: 1000, ObservedHorizonTurns: 2,
	})
	wantShed := 1000 * cacheprice.ReadMultiplier * 2                       // 200
	wantBurst := 1000 * (cacheprice.Write5mMultiplier - cacheprice.ReadMultiplier) // 1150
	if !approxEq(neg.ShedSavingTokenEquiv, wantShed) {
		t.Fatalf("shed saving = %v, want %v", neg.ShedSavingTokenEquiv, wantShed)
	}
	if !approxEq(neg.BurstCostTokenEquiv, wantBurst) {
		t.Fatalf("burst cost = %v, want %v", neg.BurstCostTokenEquiv, wantBurst)
	}
	if !approxEq(neg.NetScore(), wantShed-wantBurst) || neg.NetScore() >= 0 {
		t.Fatalf("net = %v, want %v and negative", neg.NetScore(), wantShed-wantBurst)
	}
	// Default write tier is applied when the caller leaves WriteMultiplier zero.
	if neg.WriteMultiplier != cacheprice.Write5mMultiplier {
		t.Fatalf("write mult = %v, want default %v", neg.WriteMultiplier, cacheprice.Write5mMultiplier)
	}
	// A long-horizon fire earns its rebate many times over: strongly positive net.
	pos := ScoreCompactionFire(CompactionFireObs{
		Seq: 1, PredictedHorizonMargin: 20,
		ShedTokens: 1000, BurstTokens: 1000, ObservedHorizonTurns: 30,
	})
	if pos.NetScore() <= 0 {
		t.Fatalf("long-horizon net = %v, want positive", pos.NetScore())
	}
}

// acceptanceCorpus is the fixed #2817 corpus: two long-horizon fires the gate rightly took
// (positive net) and one thin-headroom fire it approved on a predicted horizon the session did
// not deliver (negative net). The predicted margin — the ex-ante feature — correlates with the
// realized net, which is the premise the whole issue rests on: per-fire net is a learnable signal.
func acceptanceCorpus() []CompactionFireReceipt {
	return []CompactionFireReceipt{
		ScoreCompactionFire(CompactionFireObs{Seq: 1, PredictedHorizonMargin: 20, ShedTokens: 1000, BurstTokens: 1000, ObservedHorizonTurns: 30}), // net +1850
		ScoreCompactionFire(CompactionFireObs{Seq: 2, PredictedHorizonMargin: 15, ShedTokens: 800, BurstTokens: 500, ObservedHorizonTurns: 25}),   // net +1425
		ScoreCompactionFire(CompactionFireObs{Seq: 3, PredictedHorizonMargin: 1, ShedTokens: 1000, BurstTokens: 1000, ObservedHorizonTurns: 2}),   // net -950
	}
}

// TestTuneRaisesMeanPerFireNet is the acceptance witness (#2817): over the corpus, mean per-fire
// net RISES after tuning, and the tuned threshold BAILS the known negative-net fire while keeping
// the positive-net ones.
func TestTuneRaisesMeanPerFireNet(t *testing.T) {
	corpus := acceptanceCorpus()
	res := TuneFirePolicy(corpus)

	if res.TunedMeanNet <= res.BaselineMeanNet {
		t.Fatalf("tuned mean net %v did not rise above baseline %v", res.TunedMeanNet, res.BaselineMeanNet)
	}
	if res.Lift() <= 0 {
		t.Fatalf("lift = %v, want positive", res.Lift())
	}
	// Baseline fires everything (all receipts are gate-approved: margin >= 0).
	if !approxEq(res.BaselineMeanNet, (1850+1425-950)/3.0) {
		t.Fatalf("baseline mean = %v, want %v", res.BaselineMeanNet, (1850+1425-950)/3.0)
	}
	// Tuned must bail the negative-net fire (Seq 3, margin 1) and keep the two positive ones.
	if res.Tuned.MinHorizonMargin <= 1 {
		t.Fatalf("tuned margin = %d, want > 1 so the thin-headroom fire bails", res.Tuned.MinHorizonMargin)
	}
	for _, r := range corpus {
		fires := res.Tuned.Fires(r)
		if r.Seq == 3 && fires {
			t.Fatalf("tuned policy still fires the negative-net receipt (Seq 3)")
		}
		if (r.Seq == 1 || r.Seq == 2) && !fires {
			t.Fatalf("tuned policy bails a positive-net receipt (Seq %d)", r.Seq)
		}
	}
	// The realized mean over only the surviving (positive) fires.
	if !approxEq(res.TunedMeanNet, (1850+1425)/3.0) {
		t.Fatalf("tuned mean = %v, want %v", res.TunedMeanNet, (1850+1425)/3.0)
	}
	if res.Summary() == "" {
		t.Fatal("empty tuning summary")
	}
}

// TestTuneNeverDegrades witnesses the demotion-safe property: on a corpus with no negative-net
// fire (the ex-ante estimate is unbiased — no cluster to bail), tuning keeps the baseline rather
// than over-suppressing profitable fires, so the gate is never made worse.
func TestTuneNeverDegrades(t *testing.T) {
	corpus := []CompactionFireReceipt{
		ScoreCompactionFire(CompactionFireObs{Seq: 1, PredictedHorizonMargin: 10, ShedTokens: 1000, BurstTokens: 500, ObservedHorizonTurns: 30}), // net +2425
		ScoreCompactionFire(CompactionFireObs{Seq: 2, PredictedHorizonMargin: 5, ShedTokens: 1000, BurstTokens: 500, ObservedHorizonTurns: 20}),  // net +1425
	}
	res := TuneFirePolicy(corpus)
	if res.Tuned.MinHorizonMargin != 0 {
		t.Fatalf("tuned margin = %d on an all-positive corpus, want 0 (no suppression)", res.Tuned.MinHorizonMargin)
	}
	if !approxEq(res.TunedMeanNet, res.BaselineMeanNet) || res.Lift() != 0 {
		t.Fatalf("tuning degraded/changed an unbiased corpus: baseline %v tuned %v", res.BaselineMeanNet, res.TunedMeanNet)
	}
}

// TestNewCompactionFireScoreRow witnesses the durable witness row: it carries the schema tag and
// the same scored net as the pure fold, ready to append onto the live A4 receipt seam.
func TestNewCompactionFireScoreRow(t *testing.T) {
	obs := CompactionFireObs{Seq: 7, PredictedHorizonMargin: 3, ShedTokens: 500, BurstTokens: 200, ObservedHorizonTurns: 10}
	row := NewCompactionFireScoreRow(obs)
	if row.Schema != CompactionFireScoreSchema {
		t.Fatalf("schema = %q, want %q", row.Schema, CompactionFireScoreSchema)
	}
	if !approxEq(row.NetScore(), ScoreCompactionFire(obs).NetScore()) {
		t.Fatalf("row net %v != fold net %v", row.NetScore(), ScoreCompactionFire(obs).NetScore())
	}
}
