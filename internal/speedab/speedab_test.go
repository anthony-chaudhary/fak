package speedab

import (
	"math"
	"testing"
)

func TestGradeNetTruePinnedPairs(t *testing.T) {
	m := Manifest{Schema: Schema, Runs: []Run{
		{ID: "if", WorkClass: "interactive", Speed: "fast", Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 60, ToolCalls: 6, TurnaroundMS: []float64{5, 7, 9, 11, 13}, Quality: "pass", Witness: "if.json"},
		{ID: "is", WorkClass: "interactive", Speed: "standard", Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 60, ToolCalls: 5, TurnaroundMS: []float64{10, 14, 18, 22, 26}, Quality: "pass", Witness: "is.json"},
		{ID: "gf", WorkClass: "grind", Speed: "fast", Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 120, ToolCalls: 20, TurnaroundMS: []float64{5, 6, 7, 8, 9}, Quality: "pass", Witness: "gf.json"},
		{ID: "gs", WorkClass: "grind", Speed: "standard", Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 120, ToolCalls: 10, TurnaroundMS: []float64{10, 12, 14, 16, 18}, Quality: "pass", Witness: "gs.json"},
	}}
	r := Grade(m)
	if r.Verdict != "NET_TRUE" || len(r.Comparisons) != 2 {
		t.Fatalf("report=%+v", r)
	}
	for _, c := range r.Comparisons {
		if c.P50Delta >= 0 || c.P90Delta >= 0 || c.Verdict != "NET_TRUE" {
			t.Fatalf("comparison=%+v", c)
		}
	}
}

// Runtime estimate is unmeasured; only four-run in-memory manifests are graded.
// fak-test:runtime fast est=10ms lane=default
func TestGradeRejectsInvalidMeasurements(t *testing.T) {
	manifest := func() Manifest {
		m := Manifest{Schema: Schema}
		for _, class := range []string{"interactive", "grind"} {
			m.Runs = append(m.Runs,
				Run{ID: class + "-fast", WorkClass: class, Speed: "fast", Model: "m", Account: "a", Revision: "r", DurationSeconds: 60, ToolCalls: 2, TurnaroundMS: []float64{1, 2}, Quality: "pass", Witness: "w"},
				Run{ID: class + "-standard", WorkClass: class, Speed: "standard", Model: "m", Account: "a", Revision: "r", DurationSeconds: 60, ToolCalls: 1, TurnaroundMS: []float64{10, 20}, Quality: "pass", Witness: "w"},
			)
		}
		return m
	}
	for _, tc := range []struct {
		name   string
		mutate func([]Run)
	}{
		{"negative latency", func(r []Run) { r[0].TurnaroundMS = []float64{-2, -1} }},
		{"NaN latency", func(r []Run) { r[0].TurnaroundMS = []float64{math.NaN(), math.NaN()} }},
		{"positive infinite latency", func(r []Run) { r[0].TurnaroundMS[0] = math.Inf(1) }},
		{"negative infinite latency", func(r []Run) { r[0].TurnaroundMS = []float64{math.Inf(-1), math.Inf(-1)} }},
		{"NaN duration", func(r []Run) { r[0].DurationSeconds = math.NaN() }},
		{"infinite duration", func(r []Run) { r[0].DurationSeconds = math.Inf(1) }},
		{"negative count", func(r []Run) { r[0].ToolCalls = -1 }},
		{"throughput overflow", func(r []Run) { r[0].DurationSeconds = math.SmallestNonzeroFloat64 }},
		{"throughput overflow with zero baseline", func(r []Run) {
			r[0].DurationSeconds = math.SmallestNonzeroFloat64
			r[1].ToolCalls = 0
		}},
		{"latency delta overflow", func(r []Run) {
			r[0].TurnaroundMS = []float64{math.MaxFloat64 / 2, math.MaxFloat64 / 2}
			r[1].TurnaroundMS = []float64{math.MaxFloat64, math.MaxFloat64}
		}},
		{"throughput delta overflow", func(r []Run) { r[1].DurationSeconds = math.MaxFloat64 }},
	} {
		t.Run(tc.name, func(t *testing.T) {
			m := manifest()
			tc.mutate(m.Runs)
			if got := Grade(m); got.Verdict != "NOT_YET" || len(got.Comparisons) != 0 {
				t.Fatalf("invalid measurements admitted: %+v", got)
			}
			if err := Validate(m); err == nil {
				t.Fatal("Validate accepted invalid measurements")
			}
		})
	}
	for _, zero := range []bool{false, true} {
		m := manifest()
		if zero {
			m.Runs[0].TurnaroundMS = []float64{0, 0}
			m.Runs[0].ToolCalls = 0
		}
		if got := Grade(m); got.Verdict != "NET_TRUE" {
			t.Fatalf("valid non-negative measurements refused (zero=%v): %+v", zero, got)
		}
		if err := Validate(m); err != nil {
			t.Fatalf("valid measurements failed validation: %v", err)
		}
	}
}

func TestGradeRejectsUnpinnedAndUndersampled(t *testing.T) {
	base := Run{WorkClass: "interactive", Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 60, ToolCalls: 1, TurnaroundMS: []float64{1, 2}, Quality: "pass", Witness: "x"}
	fast := base
	fast.Speed = "fast"
	standard := base
	standard.Speed = "standard"
	standard.Model = "sonnet"
	if got := Grade(Manifest{Schema: Schema, Runs: []Run{fast, standard}}); got.Verdict != "NOT_YET" {
		t.Fatalf("unpinned=%+v", got)
	}
	standard.Model = "opus"
	fast.TurnaroundMS = []float64{1}
	if got := Grade(Manifest{Schema: Schema, Runs: []Run{fast, standard}}); got.Verdict != "NOT_YET" {
		t.Fatalf("undersampled=%+v", got)
	}
}

func TestGradeRejectsQualityRegression(t *testing.T) {
	base := Run{Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 60, ToolCalls: 1, TurnaroundMS: []float64{10, 20}, Quality: "pass", Witness: "x"}
	runs := []Run{}
	for _, class := range []string{"interactive", "grind"} {
		fast := base
		fast.ID, fast.WorkClass, fast.Speed, fast.TurnaroundMS = class+"-fast", class, "fast", []float64{1, 2}
		standard := base
		standard.ID, standard.WorkClass, standard.Speed = class+"-standard", class, "standard"
		runs = append(runs, fast, standard)
	}
	runs[0].Quality = "fail"
	got := Grade(Manifest{Schema: Schema, Runs: runs})
	if got.Verdict != "NOT_YET" || got.Comparisons[1].Reason != "quality witness failed" {
		t.Fatalf("report=%+v", got)
	}
}
func TestGradeRequiresBothClassesAndUniqueIDs(t *testing.T) {
	base := Run{ID: "one", WorkClass: "interactive", Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 60, ToolCalls: 1, TurnaroundMS: []float64{1, 2}, Quality: "pass", Witness: "x"}
	fast := base
	fast.Speed = "fast"
	standard := base
	standard.ID = "two"
	standard.Speed = "standard"
	if got := Grade(Manifest{Schema: Schema, Runs: []Run{fast, standard}}); got.Verdict != "NOT_YET" || got.Reason != "both interactive and grind work classes are required" {
		t.Fatalf("missing class=%+v", got)
	}
	standard.ID = fast.ID
	if got := Grade(Manifest{Schema: Schema, Runs: []Run{fast, standard}}); got.Verdict != "NOT_YET" || got.Reason != "every run needs a unique id" {
		t.Fatalf("duplicate ids=%+v", got)
	}
}
