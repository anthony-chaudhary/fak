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
