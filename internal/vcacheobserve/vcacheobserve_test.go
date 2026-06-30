package vcacheobserve

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachechain"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

// twoFamilies is a tiny but realistic two-session run: each family warms a ~40k
// system prefix on turn 1 (cache_creation) and reads it back on later turns
// (cache_read). Family "alpha" is busier than "beta".
func twoFamilies() []Turn {
	const sec = 1000
	return []Turn{
		// alpha: warm then three reads
		{Family: "alpha", UnixMillis: 0, InputTokens: 100, CacheCreation: 40000, Ephemeral1h: 40000},
		{Family: "alpha", UnixMillis: 10 * sec, InputTokens: 50, CacheRead: 40000, CacheCreation: 500, Ephemeral1h: 500},
		{Family: "alpha", UnixMillis: 20 * sec, InputTokens: 50, CacheRead: 40000, CacheCreation: 500, Ephemeral1h: 500},
		{Family: "alpha", UnixMillis: 30 * sec, InputTokens: 50, CacheRead: 40000, CacheCreation: 500, Ephemeral1h: 500},
		// beta: warm then one read
		{Family: "beta", UnixMillis: 5 * sec, InputTokens: 100, CacheCreation: 30000, Ephemeral1h: 30000},
		{Family: "beta", UnixMillis: 15 * sec, InputTokens: 50, CacheRead: 30000, CacheCreation: 400, Ephemeral1h: 400},
	}
}

func TestObserveGroupsFamiliesAndProvesSavings(t *testing.T) {
	r := Observe(twoFamilies(), DefaultMultipliers())
	if r.Turns != 6 {
		t.Fatalf("turns: got %d want 6", r.Turns)
	}
	if r.FamilyCount != 2 {
		t.Fatalf("family count: got %d want 2", r.FamilyCount)
	}
	if r.Aggregate.Status != vcachegov.ProofProven {
		t.Fatalf("aggregate should PROVE realized savings, got %s (%s)", r.Aggregate.Status, r.Aggregate.Reason)
	}
	if r.Aggregate.SavedTokenEquiv <= 0 {
		t.Fatalf("expected positive realized savings, got %.1f", r.Aggregate.SavedTokenEquiv)
	}
	if r.HitRate <= 0 || r.HitRate > 1 {
		t.Fatalf("hit rate out of range: %v", r.HitRate)
	}
	// Per-family economics must reconcile: the family savings sum to the aggregate.
	var sum float64
	for _, f := range r.Families {
		sum += f.Economics.SavedTokenEquiv
	}
	if diff := sum - r.Aggregate.SavedTokenEquiv; diff > 1e-6 || diff < -1e-6 {
		t.Fatalf("per-family savings %.4f != aggregate %.4f", sum, r.Aggregate.SavedTokenEquiv)
	}
}

func TestObserveOwnerSlicesSeparateProviderFromFak(t *testing.T) {
	r := Observe(twoFamilies(), DefaultMultipliers())
	if len(r.OwnerSlices) != 2 {
		t.Fatalf("owner slice count: got %d want 2", len(r.OwnerSlices))
	}
	provider := r.OwnerSlices[0]
	if provider.Owner != "provider" || provider.Mechanism != "prompt_cache" || provider.Provenance != Observed {
		t.Fatalf("provider owner slice not explicit: %+v", provider)
	}
	if provider.SavedTokenEquiv != r.Aggregate.SavedTokenEquiv {
		t.Fatalf("provider saved token-equiv: got %.1f want %.1f", provider.SavedTokenEquiv, r.Aggregate.SavedTokenEquiv)
	}
	if provider.CacheReadTokens != r.Aggregate.CacheReadTokens || provider.CacheCreationTokens != r.Aggregate.CacheCreationTokens {
		t.Fatalf("provider raw counters do not reconcile: %+v vs aggregate %+v", provider, r.Aggregate)
	}

	fak := r.OwnerSlices[1]
	if fak.Owner != "fak" || fak.Mechanism != "authored_cache" || fak.Provenance != NotObserved {
		t.Fatalf("fak owner slice should be explicitly not observed: %+v", fak)
	}
	if fak.SavedTokenEquiv != 0 {
		t.Fatalf("fak telemetry slice must not invent savings, got %.1f", fak.SavedTokenEquiv)
	}
}

func TestObserveWarmthBeliefNeverFalseWarms(t *testing.T) {
	r := Observe(twoFamilies(), DefaultMultipliers())
	// Every read in this fixture is genuine, so the estimator must never predict a
	// warm that the provider then misses (Law A1: the lethal direction stays 0).
	if r.Prediction.FalseWarm != 0 {
		t.Fatalf("false-warm must be 0 on all-genuine reads, got %d", r.Prediction.FalseWarm)
	}
	if r.Prediction.Total != 6 {
		t.Fatalf("prediction total: got %d want 6", r.Prediction.Total)
	}
	if got := r.Prediction.FalseWarmRate(); got != 0 {
		t.Fatalf("false-warm rate must be 0, got %v", got)
	}
}

func TestObserveWithCalibrationFeedsTTLAndReadMultiplier(t *testing.T) {
	const min = 60 * 1000
	turns := []Turn{
		{Family: "cal", UnixMillis: 0, InputTokens: 100, CacheRead: 40000},
		{Family: "cal", UnixMillis: 10 * min, InputTokens: 100, CacheCreation: 1},
	}
	defaulted := Observe(turns, DefaultMultipliers())
	if defaulted.Prediction.FalseWarm != 0 {
		t.Fatalf("default 5m TTL should expire before the 10m miss, got false_warm=%d", defaulted.Prediction.FalseWarm)
	}

	calibrated := ObserveWithOptions(turns, Options{
		Calibration: vcachecal.Calibration{
			TTLMillis:        20 * min,
			TTLMeasured:      true,
			MinPrefixTokens:  1024,
			ReadMult:         0.25,
			ReadMultMeasured: true,
		},
	})
	if calibrated.Prediction.FalseWarm != 1 {
		t.Fatalf("measured 20m TTL should keep belief warm and flag the 10m miss, got %+v", calibrated.Prediction)
	}
	if calibrated.Aggregate.ActualTokenEquiv <= defaulted.Aggregate.ActualTokenEquiv {
		t.Fatalf("measured read_mult=0.25 should price cached reads above default: got %.1f <= %.1f",
			calibrated.Aggregate.ActualTokenEquiv, defaulted.Aggregate.ActualTokenEquiv)
	}
}

func TestObserveRecallRefusesSingleUnit(t *testing.T) {
	r := Observe(twoFamilies(), DefaultMultipliers())
	// The account's mean prefix is tens of thousands of tokens; recalling one 10-token
	// unit by replay is a large loss, so the cost gate must REFUSE it (§11.0).
	if r.Recall.Status != vcachechain.ProofRefuted {
		t.Fatalf("single-unit recall should be refused, got %s", r.Recall.Status)
	}
	if r.Recall.LossRatio <= 1 {
		t.Fatalf("expected a >1x loss ratio, got %v", r.Recall.LossRatio)
	}
}

func TestObserveGovernorRidesNaturalForBusyFamily(t *testing.T) {
	r := Observe(twoFamilies(), DefaultMultipliers())
	var alpha *Family
	for i := range r.Families {
		if r.Families[i].Key == "alpha" {
			alpha = &r.Families[i]
		}
	}
	if alpha == nil {
		t.Fatal("alpha family missing")
	}
	// alpha sees 4 turns in 30s — far more than one per 5m TTL (λT≫1) — so the
	// governor must ride natural traffic, not pin.
	if alpha.GovernorDecision != vcachegov.DecisionRideNatural {
		t.Fatalf("busy family should ride natural, got %s", alpha.GovernorDecision)
	}
}

func TestObservePanelsCoverEverySubConcept(t *testing.T) {
	r := Observe(twoFamilies(), DefaultMultipliers())
	want := []string{
		"base provider cache", "M2 star anchors", "M1 concentration", "M1 warmth belief",
		"M3 dedicated warming", "M4 chains & recall", "M5 governor", "score composite",
		"cachemeta canonicalization",
	}
	if len(r.Panels) != len(want) {
		t.Fatalf("panel count: got %d want %d", len(r.Panels), len(want))
	}
	for i, name := range want {
		if r.Panels[i].Name != name {
			t.Fatalf("panel %d: got %q want %q", i, r.Panels[i].Name, name)
		}
		if r.Panels[i].Provenance != Observed && r.Panels[i].Provenance != Decision {
			t.Fatalf("panel %q has no provenance label", name)
		}
		if r.Panels[i].Verdict == "" {
			t.Fatalf("panel %q has no verdict", name)
		}
	}
}

func TestObserveEmptyIsSafe(t *testing.T) {
	r := Observe(nil, DefaultMultipliers())
	if r.Turns != 0 || r.FamilyCount != 0 {
		t.Fatalf("empty run should be empty, got %d turns / %d families", r.Turns, r.FamilyCount)
	}
}
