package vcachescore

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/vcachecal"
	"github.com/anthony-chaudhary/fak/internal/vcachegov"
)

func TestDefaultScorePassesTwoXAndBuildsIndex(t *testing.T) {
	rep := Score(DefaultInput())
	if !rep.TwoXBetter || rep.Status != "2x_ready" {
		t.Fatalf("default status=%s twoX=%v multiplier=%g", rep.Status, rep.TwoXBetter, rep.ActiveMultiplier)
	}
	if rep.ActiveMultiplier < 2 {
		t.Fatalf("active multiplier=%g, want >=2", rep.ActiveMultiplier)
	}
	if rep.Index.AnchorCount <= 0 || rep.Index.Coverage < rep.Index.TargetCoverage {
		t.Fatalf("index=%+v, want a target-covering hot-anchor index", rep.Index)
	}
	if rep.Prediction.FalseWarmRate != 0 {
		t.Fatalf("false-warm rate=%g, want 0 when no prediction errors observed", rep.Prediction.FalseWarmRate)
	}
}

func TestFlatWorkloadFailsTwoXGateEvenWhenStarSaves(t *testing.T) {
	in := DefaultInput()
	in.Ranked = SyntheticZipfWorkload(1.0, 1000)
	rep := Score(in)
	if rep.TwoXBetter {
		t.Fatalf("flat workload should not pass 2x gate: %+v", rep)
	}
	if !rep.Concentration.Defeated {
		t.Fatalf("s=1 workload should be defeated: %+v", rep.Concentration)
	}
	if !contains(rep.Actions, "manufacture skew") {
		t.Fatalf("actions=%v, want manufacture-skew guidance", rep.Actions)
	}
}

func TestTelemetryOverridesActiveMultiplier(t *testing.T) {
	in := DefaultInput()
	in.TelemetryRows = []vcachegov.TelemetryRow{
		{InputTokens: 86, CacheReadInputTokens: 1920},
	}
	rep := Score(in)
	if rep.Observed == nil || rep.ActiveSource != "telemetry" {
		t.Fatalf("observed/source = %+v/%q, want telemetry active", rep.Observed, rep.ActiveSource)
	}
	if rep.Observed.Status != vcachegov.ProofProven {
		t.Fatalf("observed status=%s reason=%s, want proven", rep.Observed.Status, rep.Observed.Reason)
	}
	if rep.ActiveMultiplier < 7 {
		t.Fatalf("telemetry multiplier=%g, want observed cached-token gain", rep.ActiveMultiplier)
	}
}

func TestEconomicsBlockReportsHitReadRebateCostFromTelemetry(t *testing.T) {
	// No telemetry: the observed economics block is absent (a hit/read/rebate
	// value must always have a provider witness behind it).
	if rep := Score(DefaultInput()); rep.Economics != nil {
		t.Fatalf("economics=%+v, want nil without telemetry", rep.Economics)
	}

	in := DefaultInput()
	in.TelemetryReadMult = 0.1
	in.TelemetryRows = []vcachegov.TelemetryRow{
		{InputTokens: 86, CacheReadInputTokens: 1920},
	}
	rep := Score(in)
	e := rep.Economics
	if e == nil {
		t.Fatalf("economics is nil, want observed block from telemetry")
	}
	if e.Source != "telemetry" || e.Witness != "observed" {
		t.Fatalf("economics provenance = %q/%q, want telemetry/observed", e.Source, e.Witness)
	}
	// hit = read / baseline; baseline is the whole prompt cold = 86 + 1920.
	const baseline = 2006.0
	if e.BaselineTokenEquiv != baseline || e.CacheReadTokens != 1920 || e.CacheCreationTokens != 0 {
		t.Fatalf("read/baseline = %g/%g write=%g, want 1920/2006/0", e.CacheReadTokens, e.BaselineTokenEquiv, e.CacheCreationTokens)
	}
	if got, want := e.HitRate, 1920.0/baseline; got != want {
		t.Fatalf("hit rate = %g, want %g", got, want)
	}
	// cost = baseline - (1-read_mult)*read; rebate = baseline - cost.
	wantCost := baseline - 0.9*1920
	if e.CostTokenEquiv != wantCost {
		t.Fatalf("cost = %g, want %g (baseline - 0.9*read)", e.CostTokenEquiv, wantCost)
	}
	if e.RebateTokenEquiv != baseline-wantCost {
		t.Fatalf("rebate = %g, want %g (baseline - cost)", e.RebateTokenEquiv, baseline-wantCost)
	}
	// The reported economics multiplier is the realized 2x gate number.
	if e.Multiplier != rep.ActiveMultiplier {
		t.Fatalf("economics multiplier=%g, active=%g, want equal", e.Multiplier, rep.ActiveMultiplier)
	}
	if got, want := e.Multiplier, baseline/wantCost; got != want {
		t.Fatalf("multiplier=%g, want %g (baseline/cost)", got, want)
	}
}

func TestProviderTelemetryDoesNotCountAsAgenticActivation(t *testing.T) {
	in := DefaultInput()
	in.TelemetryRows = []vcachegov.TelemetryRow{
		{InputTokens: 86, CacheReadInputTokens: 1920},
	}
	rep := Score(in)
	if rep.AgenticActivation.Active || rep.AgenticActivation.Total != 0 {
		t.Fatalf("provider telemetry must not count as fak-authored activation: %+v", rep.AgenticActivation)
	}
	if !rep.Planes.ProviderObserved.Available || rep.Planes.ProviderObserved.Provenance != "OBSERVED" {
		t.Fatalf("provider plane=%+v, want observed provider evidence", rep.Planes.ProviderObserved)
	}
	if rep.Planes.KernelWitnessed.Available || rep.Planes.ContextWitnessed.Available {
		t.Fatalf("kernel/context planes must stay missing without witnesses: %+v", rep.Planes)
	}
	if rep.DefaultUsefulness.Schema != "fak.cache.default_usefulness.v1" {
		t.Fatalf("default-usefulness schema=%q", rep.DefaultUsefulness.Schema)
	}
	if rep.DefaultUsefulness.Facets.AgenticActivation != 0 {
		t.Fatalf("agentic activation facet=%d, want 0 for provider-only telemetry", rep.DefaultUsefulness.Facets.AgenticActivation)
	}
	if !strings.Contains(rep.DefaultUsefulness.Reason, "provider rebate observed") {
		t.Fatalf("reason=%q, want provider-only caveat", rep.DefaultUsefulness.Reason)
	}
	if !rep.ColdPathCorrect || !rep.DefaultUsefulness.ColdPathCorrect {
		t.Fatalf("cold path must remain correct even when provider cache misses: report=%v default=%v", rep.ColdPathCorrect, rep.DefaultUsefulness.ColdPathCorrect)
	}
}

func TestSuppliedAgenticActivationIsScoredSeparately(t *testing.T) {
	in := DefaultInput()
	in.TelemetryRows = []vcachegov.TelemetryRow{
		{InputTokens: 86, CacheReadInputTokens: 1920},
	}
	in.AgenticActivation = AgenticActivationInput{
		KernelKVEvents:          2,
		ContextEvents:           1,
		ProviderVCacheDecisions: 3,
		ExternalEngineEvents:    4,
	}
	rep := Score(in)
	if !rep.AgenticActivation.Active || rep.AgenticActivation.Total != 10 {
		t.Fatalf("activation=%+v, want supplied fak-authored events counted", rep.AgenticActivation)
	}
	if rep.DefaultUsefulness.Facets.AgenticActivation != 20 {
		t.Fatalf("activation facet=%d, want full 20 when any fak-authored cache events fired", rep.DefaultUsefulness.Facets.AgenticActivation)
	}
	if !strings.Contains(rep.DefaultUsefulness.Reason, "fak-authored cache activation") {
		t.Fatalf("reason=%q, want activation-present reason", rep.DefaultUsefulness.Reason)
	}
}

func TestKernelKVWitnessPopulatesSeparatePlane(t *testing.T) {
	in := DefaultInput()
	in.KernelKV = PlaneEvidenceInput{
		Available:          true,
		BaselineTokenEquiv: 1000,
		SavedTokenEquiv:    900,
		CostTokenEquiv:     100,
		Reason:             "test kernel KV reuse",
	}
	in.AgenticActivation = AgenticActivationInput{KernelKVEvents: 2}

	rep := Score(in)
	if rep.Observed != nil || rep.Planes.ProviderObserved.Available {
		t.Fatalf("kernel-only witness must not invent provider telemetry: observed=%+v provider=%+v", rep.Observed, rep.Planes.ProviderObserved)
	}
	kernel := rep.Planes.KernelWitnessed
	if !kernel.Available || kernel.Provenance != "WITNESSED" {
		t.Fatalf("kernel plane=%+v, want WITNESSED available", kernel)
	}
	if kernel.SavedTokenEquiv != 900 || kernel.BaselineTokenEquiv != 1000 || kernel.CostTokenEquiv != 100 {
		t.Fatalf("kernel economics=%+v, want 900 saved / 1000 baseline / 100 cost", kernel)
	}
	if kernel.Multiplier != 10 {
		t.Fatalf("kernel multiplier=%g, want 10", kernel.Multiplier)
	}
	if rep.DefaultUsefulness.Facets.NetRealizedValue == 0 {
		t.Fatalf("default-usefulness should credit realized kernel KV value: %+v", rep.DefaultUsefulness)
	}
	if rep.DefaultUsefulness.Facets.DefaultCoverage <= 1 {
		t.Fatalf("default coverage=%d, want kernel plane to add coverage beyond forecast", rep.DefaultUsefulness.Facets.DefaultCoverage)
	}
}

func TestContextWitnessPopulatesSeparatePlane(t *testing.T) {
	in := DefaultInput()
	in.Context = PlaneEvidenceInput{
		Available:          true,
		BaselineTokenEquiv: 1800,
		SavedTokenEquiv:    800,
		CostTokenEquiv:     1000,
		Reason:             "test O(1) context shed",
	}
	in.AgenticActivation = AgenticActivationInput{ContextEvents: 1}

	rep := Score(in)
	if rep.Observed != nil || rep.Planes.ProviderObserved.Available || rep.Planes.KernelWitnessed.Available {
		t.Fatalf("context-only witness must not invent provider/kernel planes: observed=%+v planes=%+v", rep.Observed, rep.Planes)
	}
	context := rep.Planes.ContextWitnessed
	if !context.Available || context.Provenance != "WITNESSED" {
		t.Fatalf("context plane=%+v, want WITNESSED available", context)
	}
	if context.SavedTokenEquiv != 800 || context.BaselineTokenEquiv != 1800 || context.CostTokenEquiv != 1000 {
		t.Fatalf("context economics=%+v, want 800 saved / 1800 baseline / 1000 cost", context)
	}
	if rep.AgenticActivation.ContextEvents != 1 || !rep.AgenticActivation.Active {
		t.Fatalf("context activation=%+v, want one active context event", rep.AgenticActivation)
	}
	if rep.DefaultUsefulness.Facets.NetRealizedValue == 0 {
		t.Fatalf("default-usefulness should credit realized context value: %+v", rep.DefaultUsefulness)
	}
}

func TestSavedOnlyWitnessDoesNotEarnNetValueCredit(t *testing.T) {
	in := DefaultInput()
	in.Context = PlaneEvidenceInput{
		Available:       true,
		SavedTokenEquiv: 800,
		Reason:          "shed tokens without resident denominator",
	}
	in.AgenticActivation = AgenticActivationInput{ContextEvents: 1}

	rep := Score(in)
	if !rep.Planes.ContextWitnessed.Available || rep.Planes.ContextWitnessed.BaselineTokenEquiv != 0 {
		t.Fatalf("saved-only context plane=%+v, want available without inferred baseline", rep.Planes.ContextWitnessed)
	}
	if rep.DefaultUsefulness.Facets.NetRealizedValue != 0 {
		t.Fatalf("saved-only evidence must not earn net-value credit: %+v", rep.DefaultUsefulness)
	}
}

func TestExternalEngineHitRateIsPlaneEvidenceNotValue(t *testing.T) {
	in := DefaultInput()
	in.ExternalEngine = PlaneEvidenceInput{
		Available:  true,
		Provenance: "OBSERVED",
		HitRate:    0.72,
		Reason:     "vLLM prefix cache hit rate observed",
	}

	rep := Score(in)
	external := rep.Planes.ExternalEngineObserved
	if !external.Available || external.Provenance != "OBSERVED" {
		t.Fatalf("external plane=%+v, want OBSERVED available", external)
	}
	if external.HitRate != 0.72 {
		t.Fatalf("external hit rate=%g, want 0.72", external.HitRate)
	}
	if rep.DefaultUsefulness.Facets.NetRealizedValue != 0 {
		t.Fatalf("hit-rate-only evidence must not earn token-value credit: %+v", rep.DefaultUsefulness)
	}
	if rep.DefaultUsefulness.Facets.DefaultCoverage <= 1 {
		t.Fatalf("default coverage=%d, want external plane to add coverage beyond forecast", rep.DefaultUsefulness.Facets.DefaultCoverage)
	}
	if !strings.Contains(rep.DefaultUsefulness.Reason, "no realized value witness") {
		t.Fatalf("reason=%q, want hit-rate-only caveat", rep.DefaultUsefulness.Reason)
	}
}

func TestFalseWarmRateFailsTwoXGate(t *testing.T) {
	in := DefaultInput()
	in.Prediction = vcachecal.PredictionError{Total: 100, TrueWarm: 90, FalseWarm: 10}
	rep := Score(in)
	if rep.TwoXBetter {
		t.Fatalf("false-warm workload should not pass 2x gate: %+v", rep)
	}
	if rep.Prediction.FalseWarmRate != 0.1 {
		t.Fatalf("false-warm rate=%g, want 0.1", rep.Prediction.FalseWarmRate)
	}
	if !contains(rep.Risks, "false-warm rate") {
		t.Fatalf("risks=%v, want false-warm risk", rep.Risks)
	}
}

func TestPlanIndexChoosesSmallestTargetCoveringAnchorSet(t *testing.T) {
	c := vcachecal.FitConcentration([]vcachecal.RankedVBlock{
		{Key: "a", Frequency: 60, Size: 1, ReuseDensity: 1},
		{Key: "b", Frequency: 30, Size: 1, ReuseDensity: 1},
		{Key: "c", Frequency: 10, Size: 1, ReuseDensity: 1},
	})
	p := PlanIndex(c, 0.85)
	if p.AnchorCount != 2 || p.Coverage != 0.9 {
		t.Fatalf("index plan=%+v, want top-2 at 90%% coverage", p)
	}
}

func TestNormalizeRankedSortsAndDefaultsAnchorRows(t *testing.T) {
	ranked := NormalizeRanked([]vcachecal.RankedVBlock{
		{Key: "tail", Frequency: 10},
		{Key: "head", Frequency: 60},
		{Key: "dead", Frequency: 0},
		{Key: "mid", Frequency: 15, Size: 2},
	})
	if len(ranked) != 3 {
		t.Fatalf("ranked=%+v, want non-positive rows removed", ranked)
	}
	if ranked[0].Key != "head" || ranked[1].Key != "mid" || ranked[2].Key != "tail" {
		t.Fatalf("ranked order=%+v, want descending weight", ranked)
	}
	if ranked[0].Size != 1 || ranked[0].ReuseDensity != 1 {
		t.Fatalf("defaults not filled: %+v", ranked[0])
	}
}

func TestBuildIndexArtifactSelectsPayloadFreeHotAnchors(t *testing.T) {
	artifact := BuildIndexArtifact([]vcachecal.RankedVBlock{
		{Key: "tail", Frequency: 10},
		{Key: "head", Frequency: 60},
		{Key: "mid", Frequency: 30},
	}, 0.85)
	if artifact.Schema != "fak.vcache.anchor_index.v1" {
		t.Fatalf("schema=%q", artifact.Schema)
	}
	if artifact.AnchorCount != 2 || artifact.Coverage != 0.9 || len(artifact.Entries) != 2 {
		t.Fatalf("artifact=%+v, want top-2 at 90%% coverage", artifact)
	}
	if artifact.Entries[0].Key != "head" || artifact.Entries[1].Key != "mid" {
		t.Fatalf("entries=%+v, want ranked hot anchors without payload", artifact.Entries)
	}
}

func contains(items []string, needle string) bool {
	for _, item := range items {
		if strings.Contains(item, needle) {
			return true
		}
	}
	return false
}
