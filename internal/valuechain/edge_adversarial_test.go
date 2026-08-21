package valuechain

import (
	"math"
	"strings"
	"testing"
)

func TestVerticalValueChainEdgeAndAdversarialInputs(t *testing.T) {
	baseManifest := Manifest{Schema: "fak-value-chain/1", ChainID: "chain", Outcome: Outcome{Unit: "tasks"}, Stages: []Stage{{ID: "work", Name: "Work"}}, Arms: []Arm{{ID: "baseline", Label: "Baseline"}}}
	validInput := Input{Observations: []Observation{{StageID: "work", ArmID: "baseline", Runs: 1, CostUSD: float64Ptr(1), InputTokens: 1, OutputTokens: 1, LatencyMS: 1, OutcomeUnits: 1}}}
	cases := []struct {
		name   string
		mutate func(*Manifest, *Input)
		want   string
	}{
		{name: "empty", mutate: func(m *Manifest, in *Input) { *m = Manifest{}; *in = Input{} }, want: "schema"},
		{name: "oversized stage id", mutate: func(m *Manifest, _ *Input) { m.Stages[0].ID = strings.Repeat("x", 1<<20) }, want: "stage"},
		{name: "hostile relation target", mutate: func(m *Manifest, _ *Input) { m.Stages[0].Consumes = []string{"../../escape"} }, want: "unknown"},
		{name: "nan cost", mutate: func(_ *Manifest, in *Input) { in.Observations[0].CostUSD = float64Ptr(math.NaN()) }, want: "finite"},
		{name: "infinite outcome", mutate: func(_ *Manifest, in *Input) { in.Observations[0].OutcomeUnits = math.Inf(1) }, want: "finite"},
		{name: "duplicate observation", mutate: func(_ *Manifest, in *Input) { in.Observations = append(in.Observations, in.Observations[0]) }, want: "duplicate"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := baseManifest
			m.Stages = append([]Stage(nil), baseManifest.Stages...)
			in := Input{Observations: append([]Observation(nil), validInput.Observations...)}
			tc.mutate(&m, &in)
			_, err := Audit(m, in)
			if err == nil || !strings.Contains(strings.ToLower(err.Error()), tc.want) {
				t.Fatalf("Audit error=%v want %q", err, tc.want)
			}
		})
	}
}
