package perfrsiscore

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"strings"
	"testing"
)

func fixture(t *testing.T) Evidence {
	t.Helper()
	e, err := Load("testdata/complete.json")
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestAll16ExactlyOnceAndDeterministic(t *testing.T) {
	e := fixture(t)
	if len(e.Dimensions) != 16 {
		t.Fatalf("got %d", len(e.Dimensions))
	}
	a := Score(e)
	b := Score(e)
	aj, _ := json.Marshal(a)
	bj, _ := json.Marshal(b)
	if !bytes.Equal(aj, bj) {
		t.Fatal("nondeterministic output")
	}
}
func TestUnknownDebtAndDominantBottleneck(t *testing.T) {
	e := fixture(t)
	e.Dimensions[0].Current = nil
	e.Dimensions[1].Current = nil
	r := Score(e)
	if r.UnknownDebt != 2 || r.DominantBottleneck != "evaluation_latency" {
		t.Fatalf("debt=%d bottleneck=%s", r.UnknownDebt, r.DominantBottleneck)
	}
}
func TestBottleneckSelectionUsesLowestRatio(t *testing.T) {
	r := Score(fixture(t))
	if r.DominantBottleneck != "evaluation_latency" {
		t.Fatal(r.DominantBottleneck)
	}
}
func TestUnsaturatedRatioAnd100x(t *testing.T) {
	e := fixture(t)
	v := 250.0
	e.Dimensions[1].Current = &v
	r := Score(e)
	if *r.Dimensions[1].NormalizedRatio != 2.5 || r.TargetMultiplier != 100 {
		t.Fatalf("%+v", r)
	}
}
func TestInvalidValues(t *testing.T) {
	for _, v := range []float64{-1, math.NaN(), math.Inf(1)} {
		e := fixture(t)
		e.Dimensions[0].Current = &v
		b, _ := json.Marshal(e)
		if _, err := Decode(bytes.NewReader(b)); err == nil {
			t.Fatalf("accepted %v", v)
		}
	}
}
func TestRenderersAndComparison(t *testing.T) {
	e := fixture(t)
	r := Score(e)
	prior := r
	prior.Snapshot = "prior"
	old := .1
	prior.Dimensions[0].NormalizedRatio = &old
	if err := Compare(&r, prior); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(RenderHuman(r), "dominant bottleneck") || !strings.Contains(RenderMarkdown(r), "Normalized ratio") {
		t.Fatal("renderer missing fields")
	}
	b, err := MarshalJSON(r)
	if err != nil || !bytes.Contains(b, []byte(`"comparison"`)) {
		t.Fatalf("json: %v", err)
	}
}
func TestNativeProvenanceAndNoLlamaFallback(t *testing.T) {
	e := fixture(t)
	e.Dimensions[2].EvidenceKind = "native_benchmark"
	e.Dimensions[2].Engine = ""
	b, _ := json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted unnamed native engine")
	}
	e.Dimensions[2].Engine = "llama.cpp"
	b, _ = json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted llama fallback")
	}
	e.Dimensions[2].Engine = "fak-native/qwen3.8"
	b, _ = json.Marshal(e)
	if _, err := Decode(bytes.NewReader(b)); err != nil {
		t.Fatal(err)
	}
}
func TestFixtureIsVersioned(t *testing.T) {
	b, err := os.ReadFile("testdata/complete.json")
	if err != nil || !bytes.Contains(b, []byte(EvidenceSchema)) {
		t.Fatal(err)
	}
}

func cycleEvidence(t *testing.T) Evidence {
	t.Helper()
	e := fixture(t)
	e.Cycle = &Cycle{
		Schema:                CycleSchema,
		Engine:                "fak-native/qwen3.8",
		IdeaAt:                "2026-08-28T10:00:00Z",
		QueueAt:               "2026-08-28T10:10:00Z",
		ExecutionAt:           "2026-08-28T10:30:00Z",
		EvaluationAt:          "2026-08-28T11:00:00Z",
		LandingAt:             "2026-08-28T11:30:00Z",
		LearningAt:            "2026-08-28T12:00:00Z",
		OperatorActiveSeconds: 1800,
	}
	for i := range e.Dimensions {
		switch e.Dimensions[i].ID {
		case "cycle_time", "evaluation_latency":
			e.Dimensions[i].Unit = "hours"
			e.Dimensions[i].Current = nil
		case "experiment_throughput":
			e.Dimensions[i].Unit = "experiments/day"
			e.Dimensions[i].Current = nil
		case "automation_coverage":
			e.Dimensions[i].Unit = "percent"
			e.Dimensions[i].Current = nil
		}
	}
	return e
}

func decodeCycleEvidence(t *testing.T, e Evidence) (Evidence, error) {
	t.Helper()
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	return Decode(bytes.NewReader(b))
}

func TestCycleDerivesFourDimensions(t *testing.T) {
	e, err := decodeCycleEvidence(t, cycleEvidence(t))
	if err != nil {
		t.Fatal(err)
	}
	want := map[string]float64{
		"cycle_time":            2,
		"evaluation_latency":    .5,
		"experiment_throughput": 12,
		"automation_coverage":   75,
	}
	r := Score(e)
	for _, d := range r.Dimensions {
		expected, ok := want[d.ID]
		if !ok {
			continue
		}
		if d.Status == "UNKNOWN" || d.Current == nil || *d.Current != expected {
			t.Errorf("%s: status=%s current=%v, want %v", d.ID, d.Status, d.Current, expected)
		}
		if d.Source != "cycle:"+CycleSchema {
			t.Errorf("%s source=%q", d.ID, d.Source)
		}
	}
}

func TestCycleRejectsInvalidRows(t *testing.T) {
	tests := []struct {
		name   string
		mutate func(*Cycle)
	}{
		{"unsupported schema", func(c *Cycle) { c.Schema = "fak-performance-rsi-cycle/2" }},
		{"missing stage", func(c *Cycle) { c.QueueAt = "" }},
		{"malformed timestamp", func(c *Cycle) { c.ExecutionAt = "yesterday" }},
		{"reordered stages", func(c *Cycle) { c.LandingAt = "2026-08-28T10:45:00Z" }},
		{"negative operator duration", func(c *Cycle) { c.OperatorActiveSeconds = -1 }},
		{"non-native engine", func(c *Cycle) { c.Engine = "llama.cpp" }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := cycleEvidence(t)
			tc.mutate(e.Cycle)
			if _, err := decodeCycleEvidence(t, e); err == nil {
				t.Fatal("accepted invalid cycle")
			}
		})
	}
}

func TestCycleRejectsMissingOperatorDuration(t *testing.T) {
	e := cycleEvidence(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	b = bytes.Replace(b, []byte(`,"operator_active_seconds":1800`), nil, 1)
	if _, err := Decode(bytes.NewReader(b)); err == nil {
		t.Fatal("accepted missing operator_active_seconds")
	}
}

func improvementEvidence(t *testing.T) Evidence {
	t.Helper()
	e, err := Load("../../docs/_witnesses/issue-9781-performance-rsi-improvement.json")
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func TestImprovementDerivesReceiptDimensionsOnly(t *testing.T) {
	e := improvementEvidence(t)
	baseline, err := Load("testdata/complete.json")
	if err != nil {
		t.Fatal(err)
	}
	prior := make(map[string]Dimension, len(baseline.Dimensions))
	for _, d := range baseline.Dimensions {
		prior[d.ID] = d
	}
	want := map[string]float64{
		"improvement_yield": 20, "receipt_coverage": 100,
		"quality_gate_coverage": 100, "attribution_quality": 100,
	}
	for _, d := range e.Dimensions {
		value, derived := want[d.ID]
		if derived {
			if d.Current == nil || *d.Current != value || d.EvidenceKind != "improvement_receipt" || d.Engine != "fak-native/qwen3.8" {
				t.Errorf("%s not deterministically derived: %+v", d.ID, d)
			}
			continue
		}
		p := prior[d.ID]
		if (d.Current == nil) != (p.Current == nil) || (d.Current != nil && *d.Current != *p.Current) ||
			d.Source != p.Source || d.EvidenceKind != p.EvidenceKind || d.Engine != p.Engine {
			t.Errorf("unrelated dimension %s changed: got %+v want %+v", d.ID, d, p)
		}
	}
}

func TestImprovementRejectsInvalidReceipts(t *testing.T) {
	boolp := func(v bool) *bool { return &v }
	tests := []struct {
		name string
		edit func(*Improvement)
		want string
	}{
		{"schema", func(r *Improvement) { r.Schema = "fak-performance-rsi-improvement/2" }, "schema"},
		{"units", func(r *Improvement) { r.Baseline.Unit = "seconds" }, "milliseconds"},
		{"quality parity", func(r *Improvement) { r.Quality.Parity = boolp(false) }, "quality"},
		{"missing quality parity", func(r *Improvement) { r.Quality.Parity = nil }, "quality"},
		{"envelope", func(r *Improvement) { r.CandidateEnvelope.BatchSize = 2 }, "matched"},
		{"overhead excluded", func(r *Improvement) { r.NetTrueGain.IncludesOverhead = boolp(false) }, "include"},
		{"causal binding", func(r *Improvement) { r.Causal.IsolatesChange = nil }, "causal"},
		{"engine", func(r *Improvement) { r.Engine = "llama.cpp" }, "fak-native"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			e := improvementEvidence(t)
			tc.edit(e.Improvement)
			b, err := json.Marshal(e)
			if err != nil {
				t.Fatal(err)
			}
			_, err = Decode(bytes.NewReader(b))
			if err == nil || !strings.Contains(err.Error(), tc.want) {
				t.Fatalf("err=%v, want substring %q", err, tc.want)
			}
		})
	}
}

func TestImprovementReceiptStrictJSON(t *testing.T) {
	e := improvementEvidence(t)
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatal(err)
	}
	var raw map[string]any
	if err := json.Unmarshal(b, &raw); err != nil {
		t.Fatal(err)
	}
	raw["improvement"].(map[string]any)["unexpected"] = true
	b, _ = json.Marshal(raw)
	if _, err := Decode(bytes.NewReader(b)); err == nil || !strings.Contains(err.Error(), "unknown field") {
		t.Fatalf("expected strict JSON refusal, got %v", err)
	}
}
