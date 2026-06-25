package vcachecal

import (
	"math"
	"testing"
)

// vcachecal_test.go pins each M1 acceptance criterion (issue #716) to the decimal. The
// four scopes: the warmth-belief estimator (§7), the probe harness (Law D2), the LRU
// probe budget (observer-perturbs-state), the concentration gate (§5.2), and the
// prediction-error report (false-warm / false-cold rates, not assumed zero).
//
// The estimator drives cachemeta.Lifecycle at TierProvider; its State/Tier are named
// string types, so the tests cast to string and compare to the literal values. This
// keeps the test free of a cachemeta import while still pinning the exact transitions.

const (
	resident = "resident"
	expiring = "expiring"
	expired  = "expired"
	provider = "provider"
)

// --- Scope 1: the warmth-belief estimator (§7) ---

func TestNewBeliefIsWarm(t *testing.T) {
	b := NewBelief(0)
	if !b.IsWarm() {
		t.Fatal("a freshly written prefix is believed warm")
	}
	if string(b.Lifecycle.Tier) != provider {
		t.Errorf("belief tier = %q, want %q", b.Lifecycle.Tier, provider)
	}
}

func TestAdvanceDecaysWarmToCoolingToCold(t *testing.T) {
	policy := BeliefPolicy{ProviderTTLMillis: 300_000, GraceMillis: 30_000}
	b := NewBelief(0) // Resident, EnteredTierMillis = 0

	// One ms before the TTL: still Resident (warm).
	if b2 := b.Advance(policy, 299_999); !b2.IsWarm() || string(b2.Lifecycle.State) != resident {
		t.Errorf("at TTL-1ms: state=%q warm=%v, want resident/warm", b2.Lifecycle.State, b2.IsWarm())
	}
	// At the TTL: Resident -> Expiring (warm -> cooling).
	b2 := b.Advance(policy, 300_000)
	if string(b2.Lifecycle.State) != expiring || !b2.IsWarm() {
		t.Errorf("at TTL: state=%q warm=%v, want expiring/warm (cooling)", b2.Lifecycle.State, b2.IsWarm())
	}
	// Inside the grace window: still Expiring.
	if b3 := b2.Advance(policy, 329_999); string(b3.Lifecycle.State) != expiring {
		t.Errorf("inside grace: state=%q, want expiring", b3.Lifecycle.State)
	}
	// Past the grace window: Expiring -> Expired (cold). §7: "we can't see eviction".
	b3 := b2.Advance(policy, 330_000)
	if string(b3.Lifecycle.State) != expired || b3.IsWarm() {
		t.Errorf("past grace: state=%q warm=%v, want expired/cold", b3.Lifecycle.State, b3.IsWarm())
	}
}

func TestObserveConfirmedReadRevivesAndResetsTTLClock(t *testing.T) {
	policy := BeliefPolicy{ProviderTTLMillis: 300_000, GraceMillis: 30_000}
	b := NewBelief(0)

	// A confirmed read at t=100_000 revives and RESETS the TTL clock (§7).
	b, _ = b.Observe(policy, 100_000, 500)
	if string(b.Lifecycle.State) != resident || b.Confirmed != 1 {
		t.Errorf("after confirmed read: state=%q confirmed=%d, want resident/1", b.Lifecycle.State, b.Confirmed)
	}
	// Because the TTL clock restarted at t=100_000, t=399_999 (TTL-1ms later) is still
	// Resident. WITHOUT the reset it would already be Expiring (399_999 >= 300_000 from 0).
	if b2 := b.Advance(policy, 399_999); string(b2.Lifecycle.State) != resident {
		t.Errorf("TTL-clock reset failed: state at 399999=%q, want resident (clock restarted at 100000)", b2.Lifecycle.State)
	}
	// ...and t=400_000 (exactly TTL after the read) is Expiring.
	if b2 := b.Advance(policy, 400_000); string(b2.Lifecycle.State) != expiring {
		t.Errorf("state at 400000=%q, want expiring (TTL elapsed since the confirmed read)", b2.Lifecycle.State)
	}
}

func TestObserveRevivesFromExpiring(t *testing.T) {
	policy := BeliefPolicy{ProviderTTLMillis: 300_000, GraceMillis: 30_000}
	b := NewBelief(0).Advance(policy, 300_000) // Expiring (cooling)
	b, out := b.Observe(policy, 300_000, 43995)
	if string(b.Lifecycle.State) != resident {
		t.Errorf("read during grace: state=%q, want resident (Touch revives)", b.Lifecycle.State)
	}
	if out.Class() != TrueWarm {
		t.Errorf("predicted warm + read>0 = %s, want true_warm", out.Class())
	}
}

func TestObserveZeroOnBelievedWarmDemotesAndRecordsFalseWarm(t *testing.T) {
	policy := BeliefPolicy{ProviderTTLMillis: 300_000, GraceMillis: 30_000}
	b := NewBelief(0) // believed warm
	b, out := b.Observe(policy, 1000, 0)
	// Rule A1: believed warm + cache_read=0 -> demote to cold AT ONCE + record divergence.
	if b.IsWarm() || string(b.Lifecycle.State) != expired {
		t.Errorf("zero-on-warm: state=%q warm=%v, want demoted to expired/cold", b.Lifecycle.State, b.IsWarm())
	}
	if b.FalseWarm != 1 {
		t.Errorf("FalseWarm=%d, want 1 (the lethal HIT-vs-MISS divergence)", b.FalseWarm)
	}
	if out.Class() != FalseWarm {
		t.Errorf("class=%s, want false_warm", out.Class())
	}
}

func TestObserveWarmOnBelievedColdRecordsFalseCold(t *testing.T) {
	policy := BeliefPolicy{ProviderTTLMillis: 300_000, GraceMillis: 30_000}
	// Two advances: Resident -> Expiring at the TTL, then Expiring -> Expired past the
	// grace window (cachemeta.Advance does one transition per call). Now the belief is
	// genuinely cold (Expired).
	b := NewBelief(0).Advance(policy, 300_000).Advance(policy, 330_000)
	if b.IsWarm() {
		t.Fatalf("precondition: belief should be cold, state=%q", b.Lifecycle.State)
	}
	b, out := b.Observe(policy, 330_000, 500)
	if !b.IsWarm() {
		t.Errorf("read on cold: should revive to warm, state=%q", b.Lifecycle.State)
	}
	if b.FalseCold != 1 {
		t.Errorf("FalseCold=%d, want 1 (a warming chance the belief missed)", b.FalseCold)
	}
	if out.Class() != FalseCold {
		t.Errorf("class=%s, want false_cold (invisible regret, made visible)", out.Class())
	}
}

// --- Scope 4: the prediction-error report (rates reported, not assumed zero) ---

func TestPredictionErrorRates(t *testing.T) {
	policy := BeliefPolicy{ProviderTTLMillis: 300_000, GraceMillis: 30_000}
	var err PredictionError

	// TrueWarm: predict warm, read>0.
	b := NewBelief(0)
	b, out := b.Observe(policy, 1000, 500)
	err.Add(out)
	// FalseWarm: predict warm, read=0.
	b = NewBelief(0)
	b, out = b.Observe(policy, 1000, 0)
	err.Add(out)
	// TrueCold: predict cold, read=0.
	b = NewBelief(0).Advance(policy, 300_000).Advance(policy, 330_000) // cold
	b, out = b.Observe(policy, 331_000, 0)
	err.Add(out)
	// FalseCold: predict cold, read>0.
	b = NewBelief(0).Advance(policy, 300_000).Advance(policy, 330_000) // cold
	b, out = b.Observe(policy, 331_000, 500)
	err.Add(out)

	if err.Total != 4 || err.TrueWarm != 1 || err.FalseWarm != 1 || err.TrueCold != 1 || err.FalseCold != 1 {
		t.Fatalf("counts = %+v, want 1 each", err)
	}
	// Of 2 predicted warm, 1 missed -> 0.5 false-warm rate.
	if got := err.FalseWarmRate(); got != 0.5 {
		t.Errorf("FalseWarmRate = %g, want 0.5", got)
	}
	// Of 2 predicted cold, 1 was actually warm -> 0.5 false-cold rate.
	if got := err.FalseColdRate(); got != 0.5 {
		t.Errorf("FalseColdRate = %g, want 0.5", got)
	}
}

func TestPredictionErrorRatesAssumedZeroOnlyWhenEmpty(t *testing.T) {
	var err PredictionError
	// An empty report reports zero rates (not a panic, not a fabricated non-zero).
	if err.FalseWarmRate() != 0 || err.FalseColdRate() != 0 {
		t.Errorf("empty report rates = %g/%g, want 0/0", err.FalseWarmRate(), err.FalseColdRate())
	}
}

// --- Scope 2: the probe harness — FitCalibration fits T, M_min, r from replay ---

func TestFitCalibrationFromReplay(t *testing.T) {
	hypo := DefaultHypothesis() // TTL 300000, M_min 1024, r 0.1
	// Replay ladder: hits at 30s/2m/5m, a miss at 10m (bounds TTL above); a prefix sweep
	// hits at 2048/4096 and misses at 512 (bounds M_min below); one sample itemizes the
	// cached-read bill (1000 equiv for 10000 cached tokens -> r = 0.1).
	samples := []ProbeSample{
		{Provider: "anthropic", ModelID: "opus-4.8", Endpoint: "x", DelayMillis: 30_000, PrefixTokens: 4096, CachedTokens: 10000, ReadCostEquiv: 1000},
		{Provider: "anthropic", ModelID: "opus-4.8", Endpoint: "x", DelayMillis: 120_000, PrefixTokens: 4096, CachedTokens: 10000},
		{Provider: "anthropic", ModelID: "opus-4.8", Endpoint: "x", DelayMillis: 300_000, PrefixTokens: 2048, CachedTokens: 8000},
		{Provider: "anthropic", ModelID: "opus-4.8", Endpoint: "x", DelayMillis: 600_000, PrefixTokens: 4096, CachedTokens: 0}, // miss at 10m
		{Provider: "anthropic", ModelID: "opus-4.8", Endpoint: "x", DelayMillis: 30_000, PrefixTokens: 512, CachedTokens: 0},   // below min
	}
	c := FitCalibration(samples, hypo)

	// T = the longest confirmed-warm delay (5m); the 10m miss bounds it above.
	if c.TTLMillis != 300_000 || !c.TTLMeasured {
		t.Errorf("TTL = %d (measured=%v), want 300000/true", c.TTLMillis, c.TTLMeasured)
	}
	// M_min = the smallest confirmed-cacheable prefix (2048); the 512 miss bounds it below.
	if c.MinPrefixTokens != 2048 || !c.MinPrefixMeasured {
		t.Errorf("M_min = %d (measured=%v), want 2048/true", c.MinPrefixTokens, c.MinPrefixMeasured)
	}
	// r = ReadCostEquiv / CachedTokens = 1000/10000 = 0.1, measured.
	if c.ReadMult != 0.1 || !c.ReadMultMeasured {
		t.Errorf("r = %g (measured=%v), want 0.1/true", c.ReadMult, c.ReadMultMeasured)
	}
	if c.Provider != "anthropic" || c.ModelID != "opus-4.8" {
		t.Errorf("provider/model = %q/%q, want anthropic/opus-4.8", c.Provider, c.ModelID)
	}
}

func TestFitCalibrationFallsBackToHypothesisWhenUnmeasured(t *testing.T) {
	hypo := Hypothesis{TTLMillis: 123, MinPrefixTokens: 456, ReadMult: 0.25}
	// Only misses — no confirmed-warm sample: every constant stays on the hypothesis and
	// is flagged unmeasured (calibrate-don't-assume: the caller sees it did not measure).
	c := FitCalibration([]ProbeSample{
		{Provider: "p", DelayMillis: 1, PrefixTokens: 10, CachedTokens: 0},
	}, hypo)
	if c.TTLMeasured || c.MinPrefixMeasured || c.ReadMultMeasured {
		t.Errorf("measured flags = %v/%v/%v, want all false (only misses observed)",
			c.TTLMeasured, c.MinPrefixMeasured, c.ReadMultMeasured)
	}
	if c.TTLMillis != 123 || c.MinPrefixTokens != 456 || c.ReadMult != 0.25 {
		t.Errorf("constants = %d/%d/%g, want hypothesis 123/456/0.25", c.TTLMillis, c.MinPrefixTokens, c.ReadMult)
	}
}

func TestFromCalibrationBuildsBeliefPolicyFromMeasuredTTL(t *testing.T) {
	c := Calibration{TTLMillis: 600_000, TTLMeasured: true}
	bp := FromCalibration(c)
	if bp.ProviderTTLMillis != 600_000 {
		t.Errorf("policy TTL = %d, want 600000 (the measured T, not the hypothesis)", bp.ProviderTTLMillis)
	}
}

// --- Scope 2: the LRU probe budget (observer-perturbs-state) ---

func TestProbeBudgetEvictsColdestLRUVictim(t *testing.T) {
	b := NewProbeBudget(2)
	if ok, ev := b.AdmitProbe("a"); !ok || ev != "" {
		t.Errorf("admit a = %v/%q, want true/empty", ok, ev)
	}
	if ok, ev := b.AdmitProbe("b"); !ok || ev != "" {
		t.Errorf("admit b = %v/%q, want true/empty", ok, ev)
	}
	// Full: the LRU victim "a" (cold) is evicted to make room for "c".
	if ok, ev := b.AdmitProbe("c"); !ok || ev != "a" {
		t.Errorf("admit c = %v/%q, want true/\"a\" (cold LRU victim evicted)", ok, ev)
	}
}

func TestProbeBudgetRefusesToEvictBelievedWarm(t *testing.T) {
	b := NewProbeBudget(2)
	b.AdmitProbe("a")
	b.AdmitProbe("b")
	b.MarkWarm("a") // the estimator believes "a" warm -> protected
	// "a" is the LRU victim but it is warm: the probe is REFUSED rather than evicting it.
	if ok, ev := b.AdmitProbe("c"); ok || ev != "" {
		t.Errorf("admit c = %v/%q, want false/empty (refused: would evict believed-warm \"a\")", ok, ev)
	}
}

func TestProbeBudgetRetouchRefreshesLRUPosition(t *testing.T) {
	b := NewProbeBudget(2)
	b.AdmitProbe("a")
	b.AdmitProbe("b")
	// Re-probing "a" refreshes its LRU position, so "b" becomes the next victim.
	if ok, _ := b.AdmitProbe("a"); !ok {
		t.Error("re-probe of a present slot must always be admitted")
	}
	if ok, ev := b.AdmitProbe("c"); !ok || ev != "b" {
		t.Errorf("admit c after re-touch = %v/%q, want true/\"b\"", ok, ev)
	}
}

func TestProbeBudgetZeroCapacityRefuses(t *testing.T) {
	b := NewProbeBudget(0)
	if ok, _ := b.AdmitProbe("a"); ok {
		t.Error("zero-capacity budget must refuse every probe")
	}
}

// --- Scope 3: the concentration gate (§5.2 — measure s before trusting vCache) ---

// zipfWeights returns perfect-Zipf weights w_i = i^-s for i = 1..n (already ranked
// descending, as FitConcentration requires).
func zipfWeights(s float64, n int) []RankedVBlock {
	out := make([]RankedVBlock, n)
	for i := 0; i < n; i++ {
		w := math.Pow(float64(i+1), -s)
		out[i] = RankedVBlock{Key: "a", Frequency: w, Size: 1, ReuseDensity: 1}
	}
	return out
}

func TestFitConcentrationRecoversZipfS(t *testing.T) {
	// A perfect Zipf workload: log(weight) = C - s·log(rank), so the log-log regression
	// recovers s to the float.
	c := FitConcentration(zipfWeights(1.74, 1000))
	if !c.Measured {
		t.Fatal("Measured=false, want true for a non-empty workload")
	}
	if math.Abs(c.ZipfS-1.74) > 1e-9 {
		t.Errorf("ZipfS = %g, want 1.74 (exact recovery for perfect Zipf)", c.ZipfS)
	}
	// s > 1 -> NOT defeated; a small anchor set captures most of the volume.
	if c.Defeated {
		t.Error("s=1.74 must NOT be defeated (skewed workload)")
	}
	if cov := c.TopNCoverage[7]; cov < 0.5 {
		t.Errorf("top-7 coverage at s=1.74 = %.3f, want > 0.5 (concentrated)", cov)
	}
}

func TestFitConcentrationFlatWorkloadDefeated(t *testing.T) {
	// A flat workload (s <= 1) is structurally defeated (§5.2 corollary).
	for _, s := range []float64{0.8, 1.0} {
		c := FitConcentration(zipfWeights(s, 1000))
		if math.Abs(c.ZipfS-s) > 1e-9 {
			t.Errorf("s=%.2f: ZipfS = %g, want %g (exact recovery)", s, c.ZipfS, s)
		}
		if !c.Defeated {
			t.Errorf("s=%.2f must be defeated (s<=1 -> vCache will not help)", s)
		}
	}
	// The s=1 headline number from §5.2: top-7 of 1000 covers ~34.6%.
	c := FitConcentration(zipfWeights(1.0, 1000))
	if cov := c.TopNCoverage[7]; math.Abs(cov-0.346) > 0.01 {
		t.Errorf("top-7 coverage at s=1.0 = %.3f, want ~0.346 (the §5.2 flat headline)", cov)
	}
}

func TestFitConcentrationEmptyIsDefeated(t *testing.T) {
	c := FitConcentration(nil)
	if !c.Defeated || c.Measured {
		t.Errorf("empty workload: defeated=%v measured=%v, want true/false", c.Defeated, c.Measured)
	}
}
