package speedab

import "testing"

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
	base := Run{WorkClass: "interactive", Model: "opus", Account: "seat", Revision: "abc", DurationSeconds: 60, ToolCalls: 1, TurnaroundMS: []float64{10, 20}, Quality: "pass", Witness: "x"}
	fast := base
	fast.Speed = "fast"
	fast.TurnaroundMS = []float64{1, 2}
	fast.Quality = "fail"
	standard := base
	standard.Speed = "standard"
	if got := Grade(Manifest{Schema: Schema, Runs: []Run{fast, standard}}); got.Verdict != "NOT_YET" || got.Comparisons[0].Reason != "quality witness failed" {
		t.Fatalf("report=%+v", got)
	}
}
