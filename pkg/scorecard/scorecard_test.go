package scorecard

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestGradeStdBoundaries(t *testing.T) {
	cases := []struct {
		score float64
		want  string
	}{
		{90, "A"}, {89.9, "B"}, {80, "B"}, {79.9, "C"}, {70, "C"},
		{69.9, "D"}, {60, "D"}, {59.9, "F"}, {0, "F"}, {100, "A"},
	}
	for _, c := range cases {
		if got := GradeStd(c.score); got != c.want {
			t.Errorf("GradeStd(%g)=%q want %q", c.score, got, c.want)
		}
	}
}

func TestGradeStrictBoundaries(t *testing.T) {
	cases := []struct {
		score float64
		want  string
	}{
		{95, "A"}, {94.9, "B"}, {85, "B"}, {84.9, "C"}, {75, "C"},
		{74.9, "D"}, {60, "D"}, {59.9, "F"},
	}
	for _, c := range cases {
		if got := GradeStrict(c.score); got != c.want {
			t.Errorf("GradeStrict(%g)=%q want %q", c.score, got, c.want)
		}
	}
	// The strict curve must be stricter than std at the A and B edges: 90 is an A under
	// std but only a B under strict; 80 is a B under std but only a C under strict.
	if GradeStrict(90) != "B" {
		t.Errorf("strict 90 should be B (std is A), got %q", GradeStrict(90))
	}
	if GradeStrict(80) != "C" {
		t.Errorf("strict 80 should be C (std is B), got %q", GradeStrict(80))
	}
}

func TestRound1(t *testing.T) {
	cases := []struct {
		in   float64
		want float64
	}{
		{66.666666, 66.7}, {100, 100}, {0, 0}, {33.34, 33.3}, {99.95, 100},
	}
	for _, c := range cases {
		if got := Round1(c.in); got != c.want {
			t.Errorf("Round1(%g)=%g want %g", c.in, got, c.want)
		}
	}
}

func TestFoldDebtIsSumOfDefects(t *testing.T) {
	kpis := []KPI{
		{Key: "a", Group: "g", Score: 100},
		{Key: "b", Group: "g", Score: 0, Defects: []string{"d1", "d2"}},
		{Key: "c", Group: "g", Score: 70, Soft: []string{"s1"}},
	}
	p := Fold("fak-x-scorecard/1", kpis, "x_debt", nil, Messages{
		Finding: "has debt", FindingClean: "clean",
		NextAction: "fix it", NextActionClean: "hold",
		Grade: GradeStrict,
	})
	if got := p.Corpus["x_debt"]; got != 2 {
		t.Errorf("x_debt=%v want 2 (sum of defects, soft excluded)", got)
	}
	if p.OK {
		t.Error("ok should be false with debt>0")
	}
	if p.Verdict != "ACTION" {
		t.Errorf("verdict=%q want ACTION", p.Verdict)
	}
	if p.Finding != "has debt" || p.NextAction != "fix it" {
		t.Errorf("debt prose not selected: finding=%q next=%q", p.Finding, p.NextAction)
	}
	// composite = mean(100,0,70) = 56.667 -> F under strict
	if p.Corpus["grade"] != "F" {
		t.Errorf("grade=%v want F", p.Corpus["grade"])
	}
	if p.Corpus["score"] != Round1((100+0+70)/3.0) {
		t.Errorf("score=%v want %v", p.Corpus["score"], Round1((100+0+70)/3.0))
	}
}

func TestFoldCleanGatesOK(t *testing.T) {
	kpis := []KPI{{Key: "a", Group: "g", Score: 100}, {Key: "b", Group: "g", Score: 100}}
	p := Fold("fak-x-scorecard/1", kpis, "x_debt", nil, Messages{
		Finding: "debt", FindingClean: "all honest",
		NextAction: "fix", NextActionClean: "hold the line",
	})
	if !p.OK || p.Verdict != "OK" {
		t.Errorf("clean card should be ok/OK, got ok=%v verdict=%q", p.OK, p.Verdict)
	}
	if p.Finding != "all honest" || p.NextAction != "hold the line" {
		t.Errorf("clean prose not selected: finding=%q next=%q", p.Finding, p.NextAction)
	}
	if p.Reason != "clean" {
		t.Errorf("reason=%q want clean", p.Reason)
	}
	if p.Corpus["x_debt"] != 0 {
		t.Errorf("x_debt=%v want 0", p.Corpus["x_debt"])
	}
}

func TestFoldExtraCorpusMerged(t *testing.T) {
	p := Fold("fak-x-scorecard/1", []KPI{{Key: "a", Score: 100}}, "x_debt", nil, Messages{
		ExtraCorpus: map[string]any{"surfaces": 2, "seen": 5},
	})
	if p.Corpus["surfaces"] != 2 || p.Corpus["seen"] != 5 {
		t.Errorf("extra corpus not merged: %v", p.Corpus)
	}
	// the kernel-written keys must still be present and not clobbered
	for _, k := range []string{"score", "grade", "x_debt"} {
		if _, ok := p.Corpus[k]; !ok {
			t.Errorf("kernel key %q missing from corpus", k)
		}
	}
}

func TestFoldWeightedMean(t *testing.T) {
	// group "heavy" weighs 3, "light" weighs 1: composite = (3*60 + 1*100)/4 = 70
	kpis := []KPI{{Key: "a", Group: "heavy", Score: 60}, {Key: "b", Group: "light", Score: 100}}
	p := Fold("fak-x/1", kpis, "x_debt", map[string]float64{"heavy": 3, "light": 1}, Messages{})
	if p.Corpus["score"] != 70.0 {
		t.Errorf("weighted score=%v want 70", p.Corpus["score"])
	}
}

func TestFoldKPIsMarshalEmptyNotNull(t *testing.T) {
	p := Fold("fak-x/1", []KPI{{Key: "a", Score: 100}}, "x_debt", nil, Messages{})
	b, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	s := string(b)
	if !strings.Contains(s, `"defects":[]`) || !strings.Contains(s, `"soft":[]`) {
		t.Errorf("KPI defects/soft must marshal as [] not null: %s", s)
	}
	// the envelope keys the control-pane fold reads
	for _, key := range []string{`"ok":`, `"verdict":`, `"corpus":`, `"x_debt":`, `"grade":`} {
		if !strings.Contains(s, key) {
			t.Errorf("payload missing control-pane key %s", key)
		}
	}
}

func TestHasAny(t *testing.T) {
	if !HasAny("The CACHE_READ counter", []string{"cache_read"}) {
		t.Error("HasAny must be case-insensitive substring")
	}
	if HasAny("nothing here", []string{"absent"}) {
		t.Error("HasAny false positive")
	}
}

func TestClip(t *testing.T) {
	if got := Clip("  a   b\tc  ", 90); got != "a b c" {
		t.Errorf("Clip whitespace collapse=%q want %q", got, "a b c")
	}
	long := "abcdefghij"
	if got := Clip(long, 5); got != "abcd..." {
		t.Errorf("Clip(%q,5)=%q want %q", long, got, "abcd...")
	}
}

func TestCompareLine(t *testing.T) {
	p := Fold("fak-x/1", []KPI{{Key: "a", Score: 0, Defects: []string{"d"}}}, "x_debt", nil, Messages{})
	out := Compare(p, map[string]any{"corpus": map[string]any{"x_debt": float64(3)}}, "x_debt")
	if !strings.Contains(out, "x_debt 3 -> 1 (improved by 2)") {
		t.Errorf("compare line wrong: %q", out)
	}
}
