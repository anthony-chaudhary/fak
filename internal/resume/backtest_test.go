package resume

import (
	"math"
	"testing"
)

// turn is a tiny constructor so the tests read as token-count facts, not struct literals.
func turn(unix int64, in, cc, cr int) ObservedTurn {
	return ObservedTurn{UnixSeconds: unix, InputTokens: in, CacheCreationTokens: cc, CacheReadTokens: cr}
}

// TestBacktestWarmPairAgrees: two turns 30s apart (within the 5m TTL) where the later turn
// re-served essentially the whole prior prefix is an observed WARM the projection also calls
// warm — an agreement, and not a confirmed-cold boundary.
func TestBacktestWarmPairAgrees(t *testing.T) {
	// prev prompt = 2+1000+19000 = 20002; cur re-reads 20000 of it (recovery ~1.0 => warm).
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 19000),
		turn(30, 2, 500, 20000),
	}}
	r := Backtest(sess, TTL5m, DefaultRecoveryBand())
	if r.Scored != 1 || r.Agree != 1 || r.Disagree != 0 {
		t.Fatalf("scored=%d agree=%d disagree=%d, want 1/1/0", r.Scored, r.Agree, r.Disagree)
	}
	if r.ConfirmedCold != 0 {
		t.Errorf("confirmed cold = %d, want 0 on a warm pair", r.ConfirmedCold)
	}
	if !approx(r.Accuracy, 1.0) {
		t.Errorf("accuracy = %.3f, want 1.0", r.Accuracy)
	}
}

// TestBacktestColdPairAgreesAndValidatesCost: two turns 2h apart where the later re-served
// nothing (recovery 0 => observed cold) and re-wrote essentially its whole prompt. The
// projection also calls it cold (gap >> TTL), so it agrees AND it is a confirmed-cold boundary
// whose write ratio validates the cold-cost premise (~1.0).
func TestBacktestColdPairAgreesAndValidatesCost(t *testing.T) {
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 49000),   // prior prompt = 50002
		turn(7200, 100, 49900, 0), // 2h later: cache_read 0 (cold), wrote ~the whole prompt
	}}
	r := Backtest(sess, TTL5m, DefaultRecoveryBand())
	if r.Agree != 1 || r.Disagree != 0 {
		t.Fatalf("agree=%d disagree=%d, want 1/0", r.Agree, r.Disagree)
	}
	if r.ConfirmedCold != 1 {
		t.Fatalf("confirmed cold = %d, want 1", r.ConfirmedCold)
	}
	// cache_creation 49900 / prompt 50000 = 0.998 — a near-total re-write, the cold premise.
	if r.ColdWriteRatioMean < 0.95 {
		t.Errorf("cold write ratio = %.3f, want >= 0.95 (whole resident re-written)", r.ColdWriteRatioMean)
	}
}

// TestBacktestProjColdObsWarm: a gap PAST the 5m TTL where the prefix was nonetheless fully
// re-served — the empirical case the corpus shows (the documented 5m TTL is a floor, the real
// reuse window is longer). The projection calls cold, reality is warm: a counted disagreement
// in the proj-cold/obs-warm direction (the projection would burst a still-warm cache).
func TestBacktestProjColdObsWarm(t *testing.T) {
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 19000),  // prior prompt 20002
		turn(600, 2, 500, 20000), // 10 min later, still fully served from cache
	}}
	r := Backtest(sess, TTL5m, DefaultRecoveryBand())
	if r.Disagree != 1 || r.ProjColdObsWarm != 1 {
		t.Fatalf("disagree=%d projColdObsWarm=%d, want 1/1", r.Disagree, r.ProjColdObsWarm)
	}
	if r.ProjWarmObsCold != 0 {
		t.Errorf("projWarmObsCold = %d, want 0", r.ProjWarmObsCold)
	}
}

// TestBacktestProjWarmObsCold: a gap WITHIN the TTL whose later turn nonetheless re-served
// nothing (a breakpoint move or a fresh uncached injection). The projection calls warm,
// reality is cold: the opposite-direction disagreement.
func TestBacktestProjWarmObsCold(t *testing.T) {
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 19000), // prior prompt 20002
		turn(60, 20000, 0, 0),   // 1 min later, nothing served from cache (recovery 0)
	}}
	r := Backtest(sess, TTL5m, DefaultRecoveryBand())
	if r.Disagree != 1 || r.ProjWarmObsCold != 1 {
		t.Fatalf("disagree=%d projWarmObsCold=%d, want 1/1", r.Disagree, r.ProjWarmObsCold)
	}
	// gap < TTL so the projection said warm, not cold: not a confirmed-cold boundary.
	if r.ConfirmedCold != 0 {
		t.Errorf("confirmed cold = %d, want 0 (projection said warm)", r.ConfirmedCold)
	}
}

// TestBacktestAmbiguousExcluded: a partial re-serve (recovery in the dead-zone) is counted as
// a pair and bucketed, but excluded from the accuracy denominator — never scored on a guess.
func TestBacktestAmbiguousExcluded(t *testing.T) {
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 19000), // prior prompt 20002
		turn(30, 2, 6000, 6000), // recovery 6000/20002 = 0.30, in (0.05, 0.50): ambiguous
	}}
	r := Backtest(sess, TTL5m, DefaultRecoveryBand())
	if r.Pairs != 1 {
		t.Fatalf("pairs = %d, want 1", r.Pairs)
	}
	if r.Scored != 0 || r.Ambiguous != 1 {
		t.Errorf("scored=%d ambiguous=%d, want 0/1", r.Scored, r.Ambiguous)
	}
	if r.Accuracy != 0 {
		t.Errorf("accuracy = %.3f, want 0 (nothing scored)", r.Accuracy)
	}
}

// TestBacktestBucketsTallyByGap: boundaries land in the right wall-clock gap bucket, so the
// per-gap survival curve is correct.
func TestBacktestBucketsTallyByGap(t *testing.T) {
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 19000),
		turn(30, 2, 500, 20000),       // gap 30 -> bucket [0,60)
		turn(630, 2, 500, 20000),      // gap 600 -> bucket [300,900)
		turn(630+4000, 2, 500, 20000), // gap 4000 -> bucket [3600,18000)
	}}
	r := Backtest(sess, TTL5m, DefaultRecoveryBand())
	got := map[int64]int{}
	for _, b := range r.Buckets {
		if b.N > 0 {
			got[b.LoSeconds] = b.N
		}
	}
	for lo, want := range map[int64]int{0: 1, 300: 1, 3600: 1} {
		if got[lo] != want {
			t.Errorf("bucket lo=%d n=%d, want %d", lo, got[lo], want)
		}
	}
}

// TestBacktestSortsUnordered: turns passed out of chronological order are sorted on a copy, so
// the gap is computed correctly and the caller's slice is left untouched.
func TestBacktestSortsUnordered(t *testing.T) {
	in := []ObservedTurn{
		turn(7200, 100, 49900, 0),
		turn(0, 2, 1000, 49000),
	}
	r := Backtest([][]ObservedTurn{in}, TTL5m, DefaultRecoveryBand())
	if r.ConfirmedCold != 1 {
		t.Errorf("confirmed cold = %d, want 1 (turns should sort to a 2h cold gap)", r.ConfirmedCold)
	}
	if in[0].UnixSeconds != 7200 {
		t.Errorf("caller slice was reordered: in[0].UnixSeconds = %d, want 7200", in[0].UnixSeconds)
	}
}

// TestBacktestTotalOnEmpty: no sessions, or single-turn sessions, yield a defined zeroed
// report (no panic, no division by zero).
func TestBacktestTotalOnEmpty(t *testing.T) {
	r := Backtest(nil, TTL5m, DefaultRecoveryBand())
	if r.Pairs != 0 || r.Accuracy != 0 || len(r.Buckets) == 0 {
		t.Errorf("empty backtest = %+v, want zeroed counts with a defined bucket ladder", r)
	}
	one := Backtest([][]ObservedTurn{{turn(0, 1, 2, 3)}}, TTL5m, DefaultRecoveryBand())
	if one.Pairs != 0 {
		t.Errorf("single-turn session pairs = %d, want 0", one.Pairs)
	}
}

// TestBacktestDeterministic: same observed sessions in, same residual out.
func TestBacktestDeterministic(t *testing.T) {
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 19000),
		turn(600, 2, 500, 20000),
		turn(7800, 100, 25000, 0),
	}}
	a := Backtest(sess, TTL5m, DefaultRecoveryBand())
	b := Backtest(sess, TTL5m, DefaultRecoveryBand())
	if a.Accuracy != b.Accuracy || a.Agree != b.Agree || a.ProjColdObsWarm != b.ProjColdObsWarm ||
		!approx(a.ColdWriteRatioMean, b.ColdWriteRatioMean) {
		t.Errorf("Backtest is not deterministic:\n a=%+v\n b=%+v", a, b)
	}
}

// guard against accidental NaN in the cold ratio when there are no cold boundaries.
func TestBacktestNoColdRatioIsZeroNotNaN(t *testing.T) {
	sess := [][]ObservedTurn{{
		turn(0, 2, 1000, 19000),
		turn(30, 2, 500, 20000),
	}}
	r := Backtest(sess, TTL5m, DefaultRecoveryBand())
	if math.IsNaN(r.ColdWriteRatioMean) || r.ColdWriteRatioMean != 0 {
		t.Errorf("cold write ratio = %v, want 0 (no cold boundaries)", r.ColdWriteRatioMean)
	}
}
