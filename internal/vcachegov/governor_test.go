package vcachegov

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
)

// governor_test.go pins the §5.4 / §10 pin-lazy-evict classifier to the exact
// numbers the design note's reconciliation panel derived, so a regression in the
// cutoff math is caught as a test failure rather than a silent mis-pin.

func TestPinThresholdMatchesDesignNote(t *testing.T) {
	// §10: with the Anthropic-5m write cost w=1.25, L=1, μ=0 the cutoff is
	// ln(2.25) = 0.8109 — the doc's "≈0.81 → ~1 req/TTL".
	got := PinThreshold(WriteMult5Minutes, 1.0, 0.0)
	if math.Abs(got-math.Log(2.25)) > 1e-12 || math.Abs(got-0.81093) > 1e-4 {
		t.Fatalf("PinThreshold(1.25,1,0) = %v, want ln(2.25)≈0.8109", got)
	}
	// §10: under rate pressure (μ=2, L=1) the cutoff rises to ln(4.25)=1.447 — the
	// doc's "~1.2–1.5".
	got = PinThreshold(WriteMult5Minutes, 1.0, 2.0)
	if math.Abs(got-math.Log(4.25)) > 1e-12 || got < 1.2 || got > 1.5 {
		t.Fatalf("PinThreshold(1.25,1,2) = %v, want ln(4.25)≈1.447 (in [1.2,1.5])", got)
	}
	// §10: pure cache dollars (L=0) → lazy weakly dominates for ALL λ, so pinning
	// never fires (threshold +Inf).
	if PinThreshold(WriteMult5Minutes, 0.0, 0.0) != math.Inf(1) {
		t.Fatal("PinThreshold with L=0 must be +Inf (pure dollars: never pin)")
	}
}

func TestClassifyRideNaturalWhenLambdaTGEQ1(t *testing.T) {
	// λT ≥ 1 → natural traffic refreshes the prefix within every TTL window.
	// λ=1/300s (one per 5m), T=5m → λT = 1.0 exactly.
	s := PrefixStats{
		ArrivalRatePerSec: 1.0 / 300.0,
		TTLMillis:         TTL5MinutesMillis,
		WriteMult:         WriteMult5Minutes,
		Secret:            Cacheable,
		LastAccessMillis:  0,
		NowMillis:         1_000, // recently touched, not cold
	}
	if d := Classify(s); d != DecisionRideNatural {
		t.Fatalf("λT=1.0 → %s, want ride_natural", d)
	}
	// λT > 1 also rides natural even if pinning would otherwise be justified.
	s.ArrivalRatePerSec = 5.0 / 300.0 // λT = 5
	if d := Classify(s); d != DecisionRideNatural {
		t.Fatalf("λT=5.0 → %s, want ride_natural", d)
	}
}

func TestClassifyHeartbeatPinOnlyWhenValueTipsPastLazy(t *testing.T) {
	// λT < 1. On pure cache dollars (L=0) lazy rebuild weakly dominates for all λ.
	s := PrefixStats{
		ArrivalRatePerSec: 0.5 / 300.0, // λT = 0.5
		TTLMillis:         TTL5MinutesMillis,
		WriteMult:         WriteMult5Minutes,
		LatencyValue:      0, // pure dollars
		NowMillis:         1_000,
	}
	if d := Classify(s); d != DecisionLazyRebuild {
		t.Fatalf("λT=0.5, L=0 (pure dollars) → %s, want lazy_rebuild", d)
	}
	// Same λT, but now latency value L=1 clears the §10 cutoff (0.811): 0.5 < 0.811
	// → still lazy (below cutoff).
	s.LatencyValue = 1.0
	if d := Classify(s); d != DecisionLazyRebuild {
		t.Fatalf("λT=0.5 < cutoff 0.811 → %s, want lazy_rebuild", d)
	}
	// Raise λT to 0.9 (still <1 so not ride-natural, but >0.811 cutoff) → pin.
	s.ArrivalRatePerSec = 0.9 / 300.0 // λT = 0.9
	if d := Classify(s); d != DecisionHeartbeatPin {
		t.Fatalf("λT=0.9 > cutoff 0.811 (and <1) → %s, want heartbeat_pin", d)
	}
}

func TestClassifyPinRisesUnderRatePressure(t *testing.T) {
	// Under rate pressure μ=2 the cutoff rises to 1.447, so λT=0.9 no longer pins
	// even with L=1: the prefix must be hotter to justify the heartbeat when
	// rate-limit headroom is scarce.
	s := PrefixStats{
		ArrivalRatePerSec: 0.9 / 300.0, // λT = 0.9
		TTLMillis:         TTL5MinutesMillis,
		WriteMult:         WriteMult5Minutes,
		LatencyValue:      1.0,
		RateShadowPrice:   2.0, // cutoff → 1.447
		NowMillis:         1_000,
	}
	if d := Classify(s); d != DecisionLazyRebuild {
		t.Fatalf("λT=0.9 < pressured cutoff 1.447 → %s, want lazy_rebuild", d)
	}
	// λT=0.97 (still <1) clears 1.447? No: 0.97<1.447 → lazy. Need λT in (1.447,1):
	// impossible (cutoff>1), so under this μ nothing below ride-natural pins —
	// exactly the doc's "rising under rate pressure".
	s.ArrivalRatePerSec = 1.2 / 300.0 // λT = 1.2 ≥ 1 → ride natural instead
	if d := Classify(s); d != DecisionRideNatural {
		t.Fatalf("λT=1.2 ≥ 1 even under μ=2 → %s, want ride_natural", d)
	}
}

func TestClassifyEvictsColdIdlePastTTL(t *testing.T) {
	// Cold: no arrivals, idle past the TTL → evict from the manifest.
	s := PrefixStats{
		ArrivalRatePerSec: 0,
		TTLMillis:         TTL5MinutesMillis,
		Secret:            Cacheable,
		LastAccessMillis:  0,
		NowMillis:         TTL5MinutesMillis + 1, // 1ms past TTL, idle
	}
	if d := Classify(s); d != DecisionEvict {
		t.Fatalf("cold idle-past-TTL → %s, want evict", d)
	}
	// Idle but still within TTL → not evicted (lazy: might still be reused).
	s.NowMillis = TTL5MinutesMillis - 1
	if d := Classify(s); d != DecisionLazyRebuild {
		t.Fatalf("λ=0 but within TTL → %s, want lazy_rebuild (not evict)", d)
	}
}

func TestClassifyRefusesSecretsBeforeEconomics(t *testing.T) {
	// A secret prefix short-circuits to NoCache BEFORE any λT math runs, even when
	// it is the hottest prefix in the set (Law D4: never warm secrets).
	hot := PrefixStats{
		ArrivalRatePerSec: 10.0 / 300.0, // λT = 10 — would ride-natural if cacheable
		TTLMillis:         TTL5MinutesMillis,
		Secret:            Secret,
		NowMillis:         1_000,
	}
	if d := Classify(hot); d != DecisionNoCache {
		t.Fatalf("hot secret → %s, want no_cache (Law D4)", d)
	}
	// Regulated content → explicit-cache-with-deletion path, not the implicit warm.
	hot.Secret = SecretRegulated
	if d := Classify(hot); d != DecisionExplicitCache {
		t.Fatalf("hot regulated → %s, want explicit_cache", d)
	}
}

func TestPreferLongTTLForBurstyLongIdleGap(t *testing.T) {
	// §5.4/§10: 1h is 7.5× cheaper to HOLD for a pinned prefix over a 1h horizon
	// with idle gaps exceeding the 5m TTL (2·P vs ~15·P, 12× fewer heartbeats).
	if !PreferLongTTL(true, TTL1HourMillis, 10*60*1000) {
		t.Fatal("pinned, 1h horizon, 10m gaps → want 1h TTL preferred")
	}
	// A 5m horizon doesn't amortize the pricier 1h write (1×1.25=1.25 < 2.0).
	if PreferLongTTL(true, TTL5MinutesMillis, 10*60*1000) {
		t.Fatal("pinned but 5m horizon → want 5m kept (cheaper write)")
	}
	// Idle gaps inside the 5m TTL need no 1h escape — natural traffic holds 5m.
	if PreferLongTTL(true, TTL1HourMillis, 3*60*1000) {
		t.Fatal("gaps < 5m → want 5m (1h unneeded)")
	}
	// A non-pinned prefix has no heartbeat stream to amortize.
	if PreferLongTTL(false, TTL1HourMillis, 10*60*1000) {
		t.Fatal("not pinned → 1h never preferred")
	}
}

func TestPrefixStatsFromLifecycleReusesArrivalRate(t *testing.T) {
	// The cachemeta bridge: ArrivalRatePerSec comes from Lifecycle.AccessRatePerSec.
	// An entry admitted 1000s ago with 5 accesses → rate 5/1000 = 0.005/s.
	lc := cachemeta.NewLifecycle(cachemeta.TierProvider, 0)
	lc.Accesses = 5
	s := PrefixStatsFromLifecycle(lc, TTL5MinutesMillis, WriteMult5Minutes, 1.0, 0.0, Cacheable, 1_000_000)
	// λT = 0.005 * 300 = 1.5 ≥ 1 → ride natural (proving the lifecycle-derived rate
	// drove the classification).
	if d := Classify(s); d != DecisionRideNatural {
		t.Fatalf("lifecycle-derived λT=1.5 → %s, want ride_natural", d)
	}
}

func TestIsPinnedOnlyForHeartbeat(t *testing.T) {
	if !DecisionHeartbeatPin.IsPinned() {
		t.Fatal("heartbeat_pin must be IsPinned")
	}
	for _, d := range []GovernorDecision{DecisionRideNatural, DecisionLazyRebuild, DecisionEvict, DecisionNoCache, DecisionExplicitCache} {
		if d.IsPinned() {
			t.Fatalf("%s must NOT be IsPinned", d)
		}
	}
}
