package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/model"
)

// TestDedupBenchArmsCoalesceAndMatchFullPrefill pins the properties every reported
// number depends on: the fusion arm REALLY coalesces (exactly one leader, N-1
// followers), every arm's logits are bit-identical to a fresh full prefill (each arm
// gates this internally and returns ok=false otherwise), per-request TTFT is recorded
// for all N requests, and the token accounting is honest — fusion, nba and cascade all
// prefill exactly P + N*S tokens.
//
// fak-test:runtime fast est=3000ms
func TestDedupBenchArmsCoalesceAndMatchFullPrefill(t *testing.T) {
	const n, prefix, suffix = 6, 96, 24
	m := model.NewSynthetic(benchConfig(128, 4))
	shared, suffixes, prompts := buildFamily(n, prefix, suffix, m.Cfg.VocabSize, 7)
	oracle := make([][]float32, n)
	for i, p := range prompts {
		oracle[i] = m.NewSession().Prefill(p)
	}
	wantShared := prefix + n*suffix

	fusion, ok := armFusion(m, prompts, oracle, prefix)
	if !ok {
		t.Fatal("fusion arm failed its correctness/integrity gate")
	}
	if fusion.Leaders != 1 || fusion.Followers != int64(n-1) {
		t.Fatalf("fusion coalescing = leaders %d followers %d, want 1/%d", fusion.Leaders, fusion.Followers, n-1)
	}
	nba, ok := armWarmCacheNBA(m, prompts, oracle, prefix, 3)
	if !ok {
		t.Fatal("nba arm failed its correctness gate")
	}
	if nba.ModeledMS != 3 {
		t.Fatalf("nba modeled sched step = %v, want 3", nba.ModeledMS)
	}
	for i := 1; i < n; i++ {
		if nba.TTFT[i] < nba.TTFT[0]+3 {
			t.Fatalf("nba follower %d TTFT %.3f does not include the modeled step over leader %.3f", i, nba.TTFT[i], nba.TTFT[0])
		}
	}
	cascade, ok := armCascade(m, shared, suffixes, oracle)
	if !ok {
		t.Fatal("cascade arm failed its correctness gate")
	}
	naive, ok := armNaive(m, prompts, oracle)
	if !ok {
		t.Fatal("naive arm failed its correctness gate")
	}

	for name, r := range map[string]repResult{"fusion": fusion, "nba": nba, "cascade": cascade, "naive": naive} {
		if len(r.TTFT) != n {
			t.Fatalf("%s recorded %d TTFTs, want %d", name, len(r.TTFT), n)
		}
		for i, v := range r.TTFT {
			if v <= 0 {
				t.Fatalf("%s TTFT[%d] = %v, want > 0", name, i, v)
			}
		}
	}
	for name, r := range map[string]repResult{"fusion": fusion, "nba": nba, "cascade": cascade} {
		if r.PrefillTok != wantShared {
			t.Fatalf("%s prefill tokens = %d, want P+N*S = %d", name, r.PrefillTok, wantShared)
		}
	}
	if naive.PrefillTok != n*(prefix+suffix) {
		t.Fatalf("naive prefill tokens = %d, want %d", naive.PrefillTok, n*(prefix+suffix))
	}
	// Cascade lands every request at once: leader TTFT == follower TTFT.
	for i := range cascade.TTFT {
		if cascade.TTFT[i] != cascade.TTFT[0] {
			t.Fatalf("cascade TTFT[%d] = %v != leader %v", i, cascade.TTFT[i], cascade.TTFT[0])
		}
	}
}

func TestDedupBenchSummaryStats(t *testing.T) {
	reps := []repResult{
		{TTFT: []float64{1, 4, 4}},
		{TTFT: []float64{3, 6, 6}},
		{TTFT: []float64{2, 5, 5}},
	}
	s := summarize(reps)
	if s.LeaderTTFT != (spread{Median: 2, Min: 1, Max: 3}) {
		t.Fatalf("leader spread = %+v", s.LeaderTTFT)
	}
	if s.FollowerMeanTTFT.Median != 5 || s.Makespan.Median != 5 || s.Makespan.Max != 6 {
		t.Fatalf("follower/makespan = %+v / %+v", s.FollowerMeanTTFT, s.Makespan)
	}
	if s.FollowerP50TTFT != 5 {
		t.Fatalf("follower p50 = %v, want 5", s.FollowerP50TTFT)
	}
	r := ratioOf(armSummary{Makespan: spread{Median: 10}}, armSummary{Makespan: spread{Median: 5}})
	if r.Makespan != 2 || r.LeaderTTFT != 0 {
		t.Fatalf("ratio = %+v", r)
	}
}

// TestDedupBenchBitEqualRejectsDivergence makes the correctness gate non-vacuous.
func TestDedupBenchBitEqualRejectsDivergence(t *testing.T) {
	a := []float32{1, 2, 3}
	if !bitEqual(a, []float32{1, 2, 3}) {
		t.Fatal("bitEqual rejected identical slices")
	}
	if bitEqual(a, []float32{1, 2, 4}) {
		t.Fatal("bitEqual accepted a divergent slice")
	}
	if bitEqual(a, []float32{1, 2}) {
		t.Fatal("bitEqual accepted a length mismatch")
	}
}
