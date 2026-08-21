package valuechain

import (
	"math"
	"strings"
	"testing"
)

func TestVerticalValueChainEdgeAndAdversarialInputs(t *testing.T) {
	baseManifest := Manifest{
		Schema:   Schema,
		Name:     "chain",
		Stages:   []Stage{{ID: "work", Kind: "outcome"}},
		Arms:     []Arm{{ID: "baseline", Default: true}},
		Outcomes: []Outcome{{ID: "tasks", Unit: "task"}},
	}
	validInput := Input{Schema: Schema, Observations: []Observation{{
		ID: "observation", TraceID: "trace", StageID: "work", Arm: "baseline", Turns: 1,
		CostUSD: f(1), Outcomes: map[string]float64{"tasks": 1}, Provenance: "test-meter",
	}}}
	cases := []struct {
		name   string
		mutate func(*Manifest, *Input)
		want   string
	}{
		{name: "empty", mutate: func(m *Manifest, in *Input) { *m = Manifest{}; *in = Input{} }, want: "schema"},
		{name: "oversized stage id", mutate: func(m *Manifest, _ *Input) { m.Stages[0].ID = strings.Repeat("x", 1<<20) }, want: "stage"},
		{name: "hostile relation target", mutate: func(m *Manifest, _ *Input) { m.Stages[0].DependsOn = []string{"../../escape"} }, want: "unknown"},
		{name: "nan cost", mutate: func(_ *Manifest, in *Input) { in.Observations[0].CostUSD = f(math.NaN()) }, want: "finite"},
		{name: "infinite outcome", mutate: func(_ *Manifest, in *Input) { in.Observations[0].Outcomes = map[string]float64{"tasks": math.Inf(1)} }, want: "finite"},
		{name: "duplicate observation", mutate: func(_ *Manifest, in *Input) { in.Observations = append(in.Observations, in.Observations[0]) }, want: "duplicate"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := baseManifest
			m.Stages = append([]Stage(nil), baseManifest.Stages...)
			in := Input{Schema: validInput.Schema, Observations: append([]Observation(nil), validInput.Observations...)}
			tc.mutate(&m, &in)
			_, err := Audit(m, in)
			if err == nil || !strings.Contains(strings.ToLower(err.Error()), tc.want) {
				t.Fatalf("Audit error=%v want %q", err, tc.want)
			}
		})
	}
}
