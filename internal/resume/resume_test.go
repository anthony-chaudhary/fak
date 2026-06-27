package resume

import (
	"math"
	"testing"
)

// opusPricing is the base per-MTok price used across the tests (Opus 4.8 = {5, 25}).
var opusPricing = Pricing{InputPerMTokUSD: 5, OutputPerMTokUSD: 25}

func approx(a, b float64) bool { return math.Abs(a-b) < 1e-9 }

func strat(r Report, s Strategy) StrategyCost {
	for _, c := range r.Strategies {
		if c.Strategy == s {
			return c
		}
	}
	return StrategyCost{}
}

// TestHeadline250kColdResume is the goal in one test: resume a 250k-token session that has
// been idle 2 hours. The 5-minute prompt cache has aged out, so the posture is COLD, the
// full re-prefill is a cache WRITE of all 250k tokens, and the kernel recommends a CUT.
func TestHeadline250kColdResume(t *testing.T) {
	r := Plan(Input{
		ResidentTokens: 250000,
		IdleSeconds:    7200, // 2h >> 300s TTL
		TTL:            TTL5m,
		Pricing:        opusPricing,
		HorizonTurns:   20,
	})
	if r.Posture != PostureCold {
		t.Fatalf("posture = %q, want cold (idle 7200s exceeds 300s TTL)", r.Posture)
	}
	if r.PostureReason != "idle_exceeds_ttl" {
		t.Errorf("posture reason = %q, want idle_exceeds_ttl", r.PostureReason)
	}
	if r.Recommended != StrategyCut || r.Reason != ReasonColdPrefillShed {
		t.Errorf("recommend = (%q,%q), want (cut, cold_prefill_shed)", r.Recommended, r.Reason)
	}

	// The full cold re-prefill prices the resident at the CALIBRATED cold-write multiplier —
	// ColdWriteShare of the resident at the 5m write premium, the remainder as plain base input —
	// not the naive whole-resident-at-the-write-premium it over-stated before (#955).
	full := strat(r, StrategyResumeFull)
	if want := 250000 * (5.0 / 1e6) * TTL5m.coldWriteMultiplier(); !approx(full.ColdReprefillUSD, want) {
		t.Errorf("full cold re-prefill = %.6f, want %.6f", full.ColdReprefillUSD, want)
	}
	// The calibration is a strict reduction from the old whole-resident-at-the-write pricing.
	if naive := 250000 * (5.0 / 1e6) * CacheWrite5mMultiplier; !(full.ColdReprefillUSD < naive) {
		t.Errorf("calibrated cold re-prefill %.6f should be below the naive whole-write %.6f", full.ColdReprefillUSD, naive)
	}
	if full.ContextKeptFraction != 1.0 {
		t.Errorf("full keeps fraction %.3f, want 1.0", full.ContextKeptFraction)
	}

	// The cut sheds to the default 48k budget — a far cheaper cold turn.
	cut := strat(r, StrategyCut)
	if cut.PrefillTokens != DefaultShedBudgetTokens {
		t.Errorf("cut prefill = %d, want %d", cut.PrefillTokens, DefaultShedBudgetTokens)
	}
	if !(cut.ColdReprefillUSD < full.ColdReprefillUSD) {
		t.Errorf("cut cold re-prefill %.6f not < full %.6f", cut.ColdReprefillUSD, full.ColdReprefillUSD)
	}
	// Reset is cheaper still, and reported even though it is not recommended.
	reset := strat(r, StrategyReset)
	if !(reset.ColdReprefillUSD < cut.ColdReprefillUSD) {
		t.Errorf("reset %.6f not < cut %.6f", reset.ColdReprefillUSD, cut.ColdReprefillUSD)
	}
	// Following the recommendation saves money over resuming the whole transcript.
	if !(r.RecommendedSavingsUSD > 0) {
		t.Errorf("recommended savings = %.6f, want > 0", r.RecommendedSavingsUSD)
	}
	if want := full.HorizonUSD - cut.HorizonUSD; !approx(r.RecommendedSavingsUSD, want) {
		t.Errorf("recommended savings = %.6f, want full-cut horizon delta %.6f", r.RecommendedSavingsUSD, want)
	}
}

// TestWarmResumeKeepsFullWhenHorizonShort: idle within the TTL and few turns left -> the
// prefix may still be cached, so keep the whole transcript; bursting it to shed would cost
// more than it saves.
func TestWarmResumeKeepsFullWhenHorizonShort(t *testing.T) {
	r := Plan(Input{
		ResidentTokens: 250000,
		IdleSeconds:    60, // within the 300s TTL
		TTL:            TTL5m,
		Pricing:        opusPricing,
		HorizonTurns:   3, // short horizon: below the break-even
	})
	if r.Posture != PostureWarm {
		t.Fatalf("posture = %q, want warm", r.Posture)
	}
	if r.Recommended != StrategyResumeFull || r.Reason != ReasonWarmPrefixIntact {
		t.Errorf("recommend = (%q,%q), want (resume_full, warm_prefix_intact)", r.Recommended, r.Reason)
	}
	if r.RecommendedSavingsUSD != 0 {
		t.Errorf("savings = %.6f, want 0 when recommending full", r.RecommendedSavingsUSD)
	}
	// On a warm prefix the full first turn bills as a READ, not a write.
	full := strat(r, StrategyResumeFull)
	wantRead := 250000*(5.0/1e6)*CacheReadMultiplier + float64(r.OutputTokensPerTurn)*(25.0/1e6)
	if !approx(full.FirstTurnUSD, wantRead) {
		t.Errorf("warm full first turn = %.6f, want read-priced %.6f", full.FirstTurnUSD, wantRead)
	}
}

// TestWarmResumeCutsWhenHorizonPaysBurst: warm prefix but a long horizon -> the per-turn
// read savings on the shed tokens repay the one-time burst, so CUT is recommended.
func TestWarmResumeCutsWhenHorizonPaysBurst(t *testing.T) {
	r := Plan(Input{
		ResidentTokens: 250000,
		IdleSeconds:    60,
		TTL:            TTL5m,
		Pricing:        opusPricing,
		HorizonTurns:   500, // well above the break-even
	})
	if r.Posture != PostureWarm {
		t.Fatalf("posture = %q, want warm", r.Posture)
	}
	if r.HorizonTurns <= r.BreakEvenTurns {
		t.Fatalf("test needs horizon (%d) > break-even (%d)", r.HorizonTurns, r.BreakEvenTurns)
	}
	if r.Recommended != StrategyCut || r.Reason != ReasonWarmHorizonPaysBurst {
		t.Errorf("recommend = (%q,%q), want (cut, warm_horizon_pays_burst)", r.Recommended, r.Reason)
	}
}

// TestUnknownIdleResumesFull: with idle unknown the package cannot tell warm from cold, so
// it never sheds on a guess — it recommends RESUME_FULL.
func TestUnknownIdleResumesFull(t *testing.T) {
	r := Plan(Input{
		ResidentTokens: 250000,
		IdleSeconds:    -1, // unknown
		TTL:            TTL5m,
		Pricing:        opusPricing,
	})
	if r.Posture != PostureUnknown {
		t.Fatalf("posture = %q, want unknown", r.Posture)
	}
	if r.Recommended != StrategyResumeFull || r.Reason != ReasonUnknownIdle {
		t.Errorf("recommend = (%q,%q), want (resume_full, unknown_idle)", r.Recommended, r.Reason)
	}
	// Unknown is priced as the cold worst case so the operator sees the exposure.
	full := strat(r, StrategyResumeFull)
	if !approx(full.FirstTurnUSD-float64(r.OutputTokensPerTurn)*(25.0/1e6), full.ColdReprefillUSD) {
		t.Errorf("unknown idle should price the first turn as a cold write")
	}
}

// TestSmallSessionResumesFull: a transcript that already fits the shed budget has nothing
// to shed, so even when cold it resumes full.
func TestSmallSessionResumesFull(t *testing.T) {
	r := Plan(Input{
		ResidentTokens: 10000, // < 48k default shed budget
		IdleSeconds:    7200,  // cold
		TTL:            TTL5m,
		Pricing:        opusPricing,
	})
	if r.Recommended != StrategyResumeFull || r.Reason != ReasonSmallSession {
		t.Errorf("recommend = (%q,%q), want (resume_full, small_session)", r.Recommended, r.Reason)
	}
	if r.BreakEvenTurns != 0 {
		t.Errorf("break-even = %d, want 0 (nothing to shed)", r.BreakEvenTurns)
	}
	// Cut clamps to the resident size, so it equals full here.
	if got := strat(r, StrategyCut).PrefillTokens; got != 10000 {
		t.Errorf("cut prefill = %d, want clamped to 10000", got)
	}
}

// TestBreakEvenMatchesExplainer reproduces the worked example in
// docs/explainers/long-sessions-keep-the-cache-hit.md: dropping 20k cached tokens while
// invalidating a 40k warm suffix takes 23 future turns to pay back, at 5m economics. In the
// resume framing the kept tail (invalidated suffix) is the shed budget = 40k and the shed
// portion (dropped cached) is 20k, i.e. resident = 60k.
func TestBreakEvenMatchesExplainer(t *testing.T) {
	got := breakEvenTurns(60000, 40000, TTL5m)
	if got != 23 {
		t.Errorf("break-even = %d, want 23 (the explainer's worked example)", got)
	}
}

// TestOneHourTTLRaisesCostAndCutoff: the 1h tier doubles the write premium and widens the
// cold cutoff, so a 30-minute idle that is COLD under 5m is WARM under 1h.
func TestOneHourTTLChangesPostureAndCost(t *testing.T) {
	cold5m := Plan(Input{ResidentTokens: 250000, IdleSeconds: 1800, TTL: TTL5m, Pricing: opusPricing})
	if cold5m.Posture != PostureCold {
		t.Errorf("30min idle under 5m TTL = %q, want cold", cold5m.Posture)
	}
	warm1h := Plan(Input{ResidentTokens: 250000, IdleSeconds: 1800, TTL: TTL1h, Pricing: opusPricing})
	if warm1h.Posture != PostureWarm {
		t.Errorf("30min idle under 1h TTL = %q, want warm", warm1h.Posture)
	}
	// 1h write premium is 2.0x vs 1.25x for 5m.
	full1h := Plan(Input{ResidentTokens: 250000, IdleSeconds: 7200, TTL: TTL1h, Pricing: opusPricing})
	full5h := Plan(Input{ResidentTokens: 250000, IdleSeconds: 7200, TTL: TTL5m, Pricing: opusPricing})
	if !(strat(full1h, StrategyResumeFull).ColdReprefillUSD > strat(full5h, StrategyResumeFull).ColdReprefillUSD) {
		t.Errorf("1h cold re-prefill should exceed 5m (2.0x vs 1.25x write)")
	}
}

// TestColdWriteShareCalibratesColdCost pins the calibrated cold-write-share factor (#955). The
// naive projection priced the whole resident at the write premium (implicit share 1.0); the
// factor is fit from the back-test (BacktestReport.FirstTurnColdWriteShareMean) so the cold turn
// is priced at ColdWriteShare of the resident at the write premium and the remainder as plain
// base input. It checks three things: the factor is a measured fraction (not a magic 1.0); the
// effective cold multiplier is the share-weighted blend whose saving vs the naive write is
// exactly the premium on the un-cached remainder; and the constant is RE-DERIVABLE — a corpus
// whose cold first-turn resumes re-cache exactly ColdWriteShare of the prompt reproduces it.
func TestColdWriteShareCalibratesColdCost(t *testing.T) {
	if !(ColdWriteShare > 0 && ColdWriteShare < 1.0) {
		t.Fatalf("ColdWriteShare = %v, want a calibrated fraction in (0,1) fit from the back-test, not a magic 1.0", ColdWriteShare)
	}
	for _, ttl := range []CacheTTL{TTL5m, TTL1h} {
		eff := ttl.coldWriteMultiplier()
		wm := ttl.WriteMultiplier()
		// The blend sits strictly between plain base input (1.0) and the raw write premium.
		if !(eff > 1.0 && eff < wm) {
			t.Errorf("%s cold multiplier %.4f should sit in (1.0, %.4f)", ttl, eff, wm)
		}
		// The calibration's saving over the naive whole-write is exactly the write premium that
		// the un-cached remainder (1-ColdWriteShare) no longer pays — a real identity, not a
		// restatement of the formula under test.
		if want := (1 - ColdWriteShare) * (wm - 1.0); !approx(wm-eff, want) {
			t.Errorf("%s cold saving %.6f, want premium on the un-cached remainder %.6f", ttl, wm-eff, want)
		}
	}
	// Re-derivability: a back-test over a cold first-turn resume that re-cached exactly
	// ColdWriteShare of its prompt reproduces the constant from billed reality.
	prompt := 100000
	creation := int(float64(prompt) * ColdWriteShare) // cache_creation = share * prompt
	input := prompt - creation                        // the remainder, sent as plain input (cache_read 0 => cold)
	rep := Backtest([][]ObservedTurn{{
		{UnixSeconds: 0, InputTokens: input, CacheCreationTokens: creation, CacheReadTokens: 0},
	}}, TTL5m, DefaultRecoveryBand())
	if rep.FirstTurnCold != 1 {
		t.Fatalf("back-test should see 1 cold first-turn resume, got %d", rep.FirstTurnCold)
	}
	if !approx(rep.FirstTurnColdWriteShareMean, ColdWriteShare) {
		t.Errorf("back-test cold-write share %.6f should re-derive ColdWriteShare %.6f", rep.FirstTurnColdWriteShareMean, ColdWriteShare)
	}
}

// TestDeterministic: same input, same report (the "deterministic process" contract).
func TestDeterministic(t *testing.T) {
	in := Input{ResidentTokens: 123456, IdleSeconds: 999, TTL: TTL5m, Pricing: opusPricing, HorizonTurns: 17}
	a, b := Plan(in), Plan(in)
	if a.Recommended != b.Recommended || a.Reason != b.Reason || !approx(a.RecommendedSavingsUSD, b.RecommendedSavingsUSD) {
		t.Errorf("Plan is not deterministic: %+v vs %+v", a, b)
	}
	for i := range a.Strategies {
		if a.Strategies[i] != b.Strategies[i] {
			t.Errorf("strategy %d differs across identical inputs", i)
		}
	}
}

// TestTotalOnDegenerateInput: zero/negative axes never panic and yield a defined report.
func TestTotalOnDegenerateInput(t *testing.T) {
	r := Plan(Input{}) // all zero
	if len(r.Strategies) != 3 {
		t.Fatalf("want 3 strategies even on empty input, got %d", len(r.Strategies))
	}
	if r.TTL != TTL5m {
		t.Errorf("empty TTL should default to 5m, got %q", r.TTL)
	}
	if r.HorizonTurns != DefaultHorizonTurns {
		t.Errorf("empty horizon should default to %d, got %d", DefaultHorizonTurns, r.HorizonTurns)
	}
	// Negative resident clamps to 0; costs are all zero but defined.
	neg := Plan(Input{ResidentTokens: -5, Pricing: opusPricing})
	if strat(neg, StrategyResumeFull).ColdReprefillUSD != 0 {
		t.Errorf("negative resident should clamp to a zero-cost plan")
	}
}
