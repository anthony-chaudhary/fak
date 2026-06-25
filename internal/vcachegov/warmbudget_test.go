package vcachegov

import (
	"math"
	"testing"
)

// warmbudget_test.go pins the §5.5 rate-limit warm-budget math to the worked
// example in the design note (a half-used 4000 RPM / 400k TPM tier), so the
// crossover, sustainable-set, and TPM-bound numbers reproduce to the integer.

func halfUsed4000Tier() RateLimit {
	// §5.5 worked example: R=4000, X=400000, half-utilized → R_real=2000, X_real=200000.
	return RateLimit{TierRPM: 4000, TierTPM: 400000, RealRPM: 2000, RealTPM: 200000}
}

func TestPlanWarmBudgetCrossoverIs100Tokens(t *testing.T) {
	// §5.5: P* = (X − X_real)/(R − R_real) = 200000/2000 = 100 tokens.
	b := PlanWarmBudget(halfUsed4000Tier(), 4096, TTL5MinutesMillis)
	if math.Abs(b.CrossoverTokens-100.0) > 1e-9 {
		t.Fatalf("CrossoverTokens = %v, want 100 (§5.5 P*)", b.CrossoverTokens)
	}
}

func TestPlanWarmBudgetAnchorRowsMatchDesignTable(t *testing.T) {
	// §5.5 table: at the half-used 4000/400k tier, every realistic anchor is
	// TPM-bound, and the sustainable set scales as warms/min × T.
	cases := []struct {
		anchor          float64
		wantWarmsPerMin float64 // ~value in the doc table
		wantSet         int     // ~value in the doc table
	}{
		{4096, 48.83, 244}, // "~48 warms/min, ~244 anchors (×5 min)"
		{16384, 12.21, 61}, // "~12 warms/min, ~61 anchors"
		{65536, 3.05, 15},  // "~3 warms/min, ~15 anchors"
	}
	for _, c := range cases {
		b := PlanWarmBudget(halfUsed4000Tier(), c.anchor, TTL5MinutesMillis)
		if !b.TPMBound {
			t.Errorf("anchor=%v: want TPM-bound (P>P*=100)", c.anchor)
		}
		if math.Abs(b.WarmsPerMin-c.wantWarmsPerMin) > 0.5 {
			t.Errorf("anchor=%v: WarmsPerMin=%v, want ~%v", c.anchor, b.WarmsPerMin, c.wantWarmsPerMin)
		}
		if b.SustainableSet != c.wantSet {
			t.Errorf("anchor=%v: SustainableSet=%d, want %d", c.anchor, b.SustainableSet, c.wantSet)
		}
	}
	// 1h TTL buys 12× the set for the same headroom (§5.5/§10).
	b5m := PlanWarmBudget(halfUsed4000Tier(), 4096, TTL5MinutesMillis)
	b1h := PlanWarmBudget(halfUsed4000Tier(), 4096, TTL1HourMillis)
	if b1h.SustainableSet < b5m.SustainableSet*11 {
		t.Errorf("1h sustainable set = %d, want ~12× the 5m set (%d)", b1h.SustainableSet, b5m.SustainableSet)
	}
}

func TestPlanWarmBudgetRPMBoundBelowCrossover(t *testing.T) {
	// §5.5: below P* (~100 tok) the tier is RPM-starved. A 50-token "anchor" is
	// RPM-bound (TPMBound false), and warms clamp to availRPM, not availTPM/P.
	b := PlanWarmBudget(halfUsed4000Tier(), 50, TTL5MinutesMillis)
	if b.TPMBound {
		t.Fatal("anchor=50 (<P*=100) must be RPM-bound, not TPM-bound")
	}
	if math.Abs(b.WarmsPerMin-2000) > 1e-9 { // min(2000, 200000/50=4000) = 2000
		t.Fatalf("RPM-bound warms = %v, want 2000 (availRPM)", b.WarmsPerMin)
	}
}

func TestPlanWarmBudgetZeroOnNoHeadroom(t *testing.T) {
	// Saturated tier: real traffic uses the whole quota → no warming budget. This
	// is the "degrade by warming fewer, never 429" floor: zero warms, not negative.
	saturated := RateLimit{TierRPM: 4000, TierTPM: 400000, RealRPM: 4000, RealTPM: 400000}
	b := PlanWarmBudget(saturated, 4096, TTL5MinutesMillis)
	if b.WarmsPerMin != 0 || b.SustainableSet != 0 {
		t.Fatalf("saturated tier → warms=%v set=%d, want 0/0", b.WarmsPerMin, b.SustainableSet)
	}
}

func TestRankDropsSecretsAndOrdersByWorkingSetWeight(t *testing.T) {
	// §5.2: rank by frequency × size × reuse-density; secrets never enter the warm set.
	cands := []WarmCandidate{
		{Key: "a", Frequency: 10, Size: 4096, ReuseDensity: 2, Secret: Cacheable},      // score 81920
		{Key: "b", Frequency: 100, Size: 4096, ReuseDensity: 5, Secret: Secret},        // DROPPED
		{Key: "c", Frequency: 50, Size: 8192, ReuseDensity: 1, Secret: Cacheable},      // score 409600 → head
		{Key: "d", Frequency: 1, Size: 1024, ReuseDensity: 1, Secret: Cacheable},       // score 1024 → tail
		{Key: "e", Frequency: 1, Size: 1024, ReuseDensity: 1, Secret: SecretRegulated}, // DROPPED
	}
	ranked := Rank(cands)
	if len(ranked) != 3 {
		t.Fatalf("Rank kept %d, want 3 (2 secrets dropped)", len(ranked))
	}
	if ranked[0].Key != "c" {
		t.Errorf("head = %q, want c (highest working-set weight)", ranked[0].Key)
	}
	if ranked[len(ranked)-1].Key != "d" {
		t.Errorf("tail = %q, want d", ranked[len(ranked)-1].Key)
	}
}

func TestScheduleDegradesByWarmingFewerNever429(t *testing.T) {
	// The load-bearing acceptance clause: when headroom is short, Schedule truncates
	// to the sustainable set rather than issuing a warm that would 429 real traffic.
	cands := []WarmCandidate{
		{Key: "h1", Frequency: 10, Size: 4096, ReuseDensity: 5, Secret: Cacheable},
		{Key: "h2", Frequency: 9, Size: 4096, ReuseDensity: 5, Secret: Cacheable},
		{Key: "h3", Frequency: 8, Size: 4096, ReuseDensity: 5, Secret: Cacheable},
		{Key: "h4", Frequency: 7, Size: 4096, ReuseDensity: 5, Secret: Cacheable},
	}
	// Tight budget: only 2 anchors sustainable this window.
	budget := WarmBudget{SustainableSet: 2}
	got := Schedule(cands, budget)
	if len(got) != 2 {
		t.Fatalf("Schedule returned %d, want 2 (degrade to budget)", len(got))
	}
	if got[0].Key != "h1" || got[1].Key != "h2" {
		t.Errorf("Schedule kept %q,%q; want the two highest-ranked h1,h2", got[0].Key, got[1].Key)
	}
	// Budget larger than the candidate pool → warm all of them, never more.
	budget.SustainableSet = 99
	got = Schedule(cands, budget)
	if len(got) != 4 {
		t.Fatalf("Schedule with room returned %d, want all 4", len(got))
	}
	// Zero budget → warm nothing (no 429).
	budget.SustainableSet = 0
	got = Schedule(cands, budget)
	if len(got) != 0 {
		t.Fatalf("zero-budget Schedule returned %d, want 0", len(got))
	}
}

func TestScheduleNeverAdmitsSecretsEvenWithRoom(t *testing.T) {
	// Even with unlimited budget, a secret candidate is never warmed (Law D4).
	cands := []WarmCandidate{
		{Key: "ok", Frequency: 1, Size: 4096, ReuseDensity: 1, Secret: Cacheable},
		{Key: "key", Frequency: 1000, Size: 4096, ReuseDensity: 100, Secret: Secret},
	}
	got := Schedule(cands, WarmBudget{SustainableSet: 99})
	if len(got) != 1 || got[0].Key != "ok" {
		t.Fatalf("Schedule admitted a secret: got %+v", got)
	}
}
