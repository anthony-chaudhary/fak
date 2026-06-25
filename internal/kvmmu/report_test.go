package kvmmu_test

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/kvmmu"
)

// report_test.go — acceptance for issue #857 (post-hoc session attention report).
//
// The report is a pure fold over the accumulator (#855) plus a recorded per-turn S/N curve, so the
// tests drive the accumulator directly with a known mass stream and hand a known curve in — the fold
// math (hottest, dead weight, integrated S/N, the bloat marker) is what is under test.

// buildSession folds a fixed multi-turn mass stream into a post-hoc (λ=1) accumulator: a "hot" span
// attended every turn, a big "blob" that is resident the whole session but never attended (the dead
// weight), and a "warm" span attended early then idle.
func buildSession() *kvmmu.AttentionAccumulator {
	a := kvmmu.NewAttentionAccumulator(1.0, 0) // λ=1: post-hoc cumulative
	a.Observe(map[string]float64{"hot": 0.8, "warm": 0.6, "blob": 0.0})
	a.Observe(map[string]float64{"hot": 0.9, "warm": 0.3, "blob": 0.0})
	a.Observe(map[string]float64{"hot": 0.7, "warm": 0.0, "blob": 0.0})
	a.Observe(map[string]float64{"hot": 0.85, "warm": 0.0, "blob": 0.0})
	return a
}

// TestReportNamesDeadWeight: the report's dead-weight list names the big-but-never-attended blob, not
// the hot span — the "was this 9000-token blob ever worth its residency?" answer.
func TestReportNamesDeadWeight(t *testing.T) {
	a := buildSession()
	cost := map[string]int{"hot": 200, "warm": 150, "blob": 9000}
	curve := []kvmmu.TurnSN{
		{Turn: 1, Ratio: 0.8, Cost: 9350, CacheHit: 0.5},
		{Turn: 2, Ratio: 0.6, Cost: 9350, CacheHit: 0.7},
	}
	r := kvmmu.BuildSessionAttentionReport(a, curve, cost, 2, 1e-9)

	if len(r.DeadWeight) == 0 || r.DeadWeight[0].ID != "blob" {
		t.Fatalf("dead weight head = %+v, want blob (cold + biggest)", r.DeadWeight)
	}
	// The hot span must NOT be flagged dead weight.
	for _, d := range r.DeadWeight {
		if d.ID == "hot" {
			t.Errorf("hot span wrongly flagged as dead weight: %+v", d)
		}
	}
	// Hottest head is the most-attended span.
	if len(r.Hottest) == 0 || r.Hottest[0].ID != "hot" {
		t.Fatalf("hottest head = %+v, want hot", r.Hottest)
	}
	// blob carries its cost into the report (the 9000-token residency).
	if r.DeadWeight[0].Cost != 9000 {
		t.Errorf("blob cost = %d, want 9000", r.DeadWeight[0].Cost)
	}
	// blob was never hot → FirstHot/LastHot zero (dead the whole session).
	if r.DeadWeight[0].FirstHot != 0 || r.DeadWeight[0].LastHot != 0 {
		t.Errorf("blob hot-turns = (%d,%d), want (0,0) never attended", r.DeadWeight[0].FirstHot, r.DeadWeight[0].LastHot)
	}
}

// TestDeadWeightTail: a span attended early then idle has a dead-weight tail (LastHot..end), which the
// report surfaces via FirstHot/LastHot — "for how many turns was it dead weight?"
func TestDeadWeightTail(t *testing.T) {
	a := buildSession() // warm: hot turns 1-2, idle turns 3-4
	r := kvmmu.BuildSessionAttentionReport(a, nil, nil, 10, 0.0)
	var warm *kvmmu.SpanReport
	for i := range r.Hottest {
		if r.Hottest[i].ID == "warm" {
			warm = &r.Hottest[i]
		}
	}
	if warm == nil {
		t.Fatal("warm span missing from report")
	}
	if warm.FirstHot != 1 || warm.LastHot != 2 {
		t.Errorf("warm hot-turns = (%d,%d), want (1,2) — dead from turn 3 on", warm.FirstHot, warm.LastHot)
	}
	if r.Turns != 4 {
		t.Errorf("report Turns = %d, want 4", r.Turns)
	}
}

// TestIntegratedSNCostWeighted: the integrated S/N is the cost-weighted mean of the per-turn ratios —
// a bloated (high-cost) turn and a lean (low-cost) turn are not equal votes.
func TestIntegratedSNCostWeighted(t *testing.T) {
	curve := []kvmmu.TurnSN{
		{Turn: 1, Ratio: 0.9, Cost: 100}, // lean, high S/N
		{Turn: 2, Ratio: 0.3, Cost: 900}, // bloated, low S/N — dominates the weight
	}
	a := kvmmu.NewAttentionAccumulator(1.0, 0)
	a.Observe(map[string]float64{"x": 0.5})
	r := kvmmu.BuildSessionAttentionReport(a, curve, nil, 5, 0.0)

	// Σ Ratio·Cost / Σ Cost = (0.9·100 + 0.3·900) / 1000 = (90 + 270)/1000 = 0.36.
	want := (0.9*100 + 0.3*900) / 1000.0
	if d := math.Abs(r.IntegratedSN - want); d > 1e-9 {
		t.Errorf("IntegratedSN = %v, want %v (Δ=%v)", r.IntegratedSN, want, d)
	}
	// Sanity: the cost-weighted mean is pulled well below the simple mean (0.6) by the bloated turn.
	if r.IntegratedSN >= 0.6 {
		t.Errorf("IntegratedSN %v should be < simple mean 0.6 (bloated turn dominates)", r.IntegratedSN)
	}
}

// TestBloatedSinceDetectsDivergence: S/N(t) falling while cache-hit climbs is the bloat pathology —
// the report flags the turn it begins.
func TestBloatedSinceDetectsDivergence(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(1.0, 0)
	a.Observe(map[string]float64{"x": 0.5})
	// S/N declines 0.9→0.4 while cache-hit climbs 0.5→0.9 across the whole run.
	curve := []kvmmu.TurnSN{
		{Turn: 1, Ratio: 0.9, Cost: 100, CacheHit: 0.5},
		{Turn: 2, Ratio: 0.8, Cost: 200, CacheHit: 0.7},
		{Turn: 3, Ratio: 0.6, Cost: 300, CacheHit: 0.8},
		{Turn: 4, Ratio: 0.4, Cost: 400, CacheHit: 0.9},
	}
	r := kvmmu.BuildSessionAttentionReport(a, curve, nil, 5, 0.0)
	if r.BloatedSince != 1 {
		t.Errorf("BloatedSince = %d, want 1 (whole run bloats as cache-hit climbs)", r.BloatedSince)
	}
}

// TestBloatedSinceHealthyRun: a run whose S/N holds or improves is not flagged.
func TestBloatedSinceHealthyRun(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(1.0, 0)
	a.Observe(map[string]float64{"x": 0.5})
	curve := []kvmmu.TurnSN{
		{Turn: 1, Ratio: 0.7, Cost: 100, CacheHit: 0.5},
		{Turn: 2, Ratio: 0.8, Cost: 100, CacheHit: 0.6}, // S/N improving
		{Turn: 3, Ratio: 0.85, Cost: 100, CacheHit: 0.7},
	}
	if got := kvmmu.BuildSessionAttentionReport(a, curve, nil, 5, 0.0).BloatedSince; got != -1 {
		t.Errorf("BloatedSince = %d, want -1 (healthy run, S/N rising)", got)
	}
}

// TestBloatedSincePartialWindow: only the trailing window where the pathology holds is flagged, not an
// earlier healthy stretch.
func TestBloatedSincePartialWindow(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(1.0, 0)
	a.Observe(map[string]float64{"x": 0.5})
	curve := []kvmmu.TurnSN{
		{Turn: 1, Ratio: 0.5, Cost: 100, CacheHit: 0.9}, // healthy-ish, cache-hit high then drops
		{Turn: 2, Ratio: 0.8, Cost: 100, CacheHit: 0.4}, // S/N up — breaks any window before here
		{Turn: 3, Ratio: 0.7, Cost: 100, CacheHit: 0.5}, // decline begins here, cache-hit climbing
		{Turn: 4, Ratio: 0.5, Cost: 100, CacheHit: 0.8},
	}
	if got := kvmmu.BuildSessionAttentionReport(a, curve, nil, 5, 0.0).BloatedSince; got != 2 {
		t.Errorf("BloatedSince = %d, want 2 (decline starts at turn 2's peak)", got)
	}
}

// TestReportEmptyCurve: no curve → honest zeros, no divide-by-zero, no false bloat flag.
func TestReportEmptyCurve(t *testing.T) {
	a := kvmmu.NewAttentionAccumulator(1.0, 0)
	a.Observe(map[string]float64{"x": 0.5})
	r := kvmmu.BuildSessionAttentionReport(a, nil, nil, 5, 0.0)
	if r.IntegratedSN != 0 {
		t.Errorf("IntegratedSN = %v, want 0 (empty curve)", r.IntegratedSN)
	}
	if r.BloatedSince != -1 {
		t.Errorf("BloatedSince = %d, want -1 (empty curve)", r.BloatedSince)
	}
}

// TestReportDeterministic: same inputs yield the same report (the post-hoc analyst is replayable).
func TestReportDeterministic(t *testing.T) {
	cost := map[string]int{"hot": 200, "warm": 150, "blob": 9000}
	curve := []kvmmu.TurnSN{{Turn: 1, Ratio: 0.8, Cost: 9350, CacheHit: 0.5}}
	r1 := kvmmu.BuildSessionAttentionReport(buildSession(), curve, cost, 3, 1e-9)
	r2 := kvmmu.BuildSessionAttentionReport(buildSession(), curve, cost, 3, 1e-9)
	if r1.IntegratedSN != r2.IntegratedSN || r1.BloatedSince != r2.BloatedSince || len(r1.Hottest) != len(r2.Hottest) {
		t.Fatalf("reports differ: %+v vs %+v", r1, r2)
	}
	for i := range r1.Hottest {
		if r1.Hottest[i] != r2.Hottest[i] {
			t.Errorf("hottest[%d] differs: %+v vs %+v", i, r1.Hottest[i], r2.Hottest[i])
		}
	}
}
