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

func TestValueFromScoreAndRound3(t *testing.T) {
	if got := ValueFromScore(66.7); got != 0.667 {
		t.Fatalf("ValueFromScore(66.7)=%g want 0.667", got)
	}
	if got := Round3(0.666666); got != 0.667 {
		t.Fatalf("Round3(0.666666)=%g want 0.667", got)
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
	if p.Corpus["value"] != Round3(ValueFromScore((100+0+70)/3.0)) {
		t.Errorf("value=%v want %v", p.Corpus["value"], Round3(ValueFromScore((100+0+70)/3.0)))
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
	// the envelope keys the control-pane fold reads, plus the continuous value key new
	// scorecards prefer over the legacy 0-100 score.
	for _, key := range []string{`"ok":`, `"verdict":`, `"corpus":`, `"x_debt":`, `"grade":`, `"value":`} {
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

// TestDeficitSurplusUnbounded is the load-bearing property: unlike a 0-100 grade, the
// deficit/surplus around a baseline never clamps, so a score twice past a bound reports twice
// the magnitude instead of saturating.
func TestDeficitSurplusUnbounded(t *testing.T) {
	// A score of -50 against the 100 pass-line is 150 below it -- past the -100 floor a raw
	// 0-100 score would hit; the deficit must expose the full 150, not clamp at 100.
	if got := Deficit(-50, 100); got != 150 {
		t.Errorf("Deficit(-50,100)=%g want 150 (unbounded past the floor)", got)
	}
	if got := Surplus(-50, 100); got != 0 {
		t.Errorf("Surplus(-50,100)=%g want 0 (below baseline is not credit)", got)
	}
	// A score of 130 against 100 clears the ceiling by 30 -- headroom a capped score hides.
	if got := Surplus(130, 100); got != 30 {
		t.Errorf("Surplus(130,100)=%g want 30 (unbounded above the ceiling)", got)
	}
	if got := Deficit(130, 100); got != 0 {
		t.Errorf("Deficit(130,100)=%g want 0 (above baseline is not debt)", got)
	}
	// On the baseline: neither debt nor credit.
	if Deficit(100, 100) != 0 || Slack([]KPI{{Score: 100}}, nil) != 0 {
		t.Error("a KPI exactly on its pass-line must contribute zero pressure and zero slack")
	}
}

// TestFoldStampsUnboundedPressure confirms Fold emits a continuous pressure that grows past the
// point a 0-100 grade would saturate, and that a KPI carrying an out-of-band Score drives it.
func TestFoldStampsUnboundedPressure(t *testing.T) {
	// Two KPIs: one perfect (no pressure), one at -100 (200 below the 100 pass-line). Mean
	// weighting -> pressure = 200/... no, pressure SUMS deficits (weighted), not means: 0+200.
	kpis := []KPI{{Key: "a", Score: 100}, {Key: "b", Score: -100, Defects: []string{"d"}}}
	p := Fold("fak-x/1", kpis, "x_debt", nil, Messages{})
	if got, _ := anyFloat(p.Corpus["pressure"]); got != 200 {
		t.Errorf("pressure=%v want 200 (Σ weighted deficit, unbounded past the -100 floor)", p.Corpus["pressure"])
	}
	if p.Corpus["pressure_unit"] != "weighted_score_deficit" {
		t.Errorf("pressure_unit=%v want weighted_score_deficit", p.Corpus["pressure_unit"])
	}
	if got, _ := anyFloat(p.Corpus["slack"]); got != 0 {
		t.Errorf("slack=%v want 0 (no KPI clears the baseline)", p.Corpus["slack"])
	}
	// Legacy score/grade stay intact and Python-readable alongside the new gauge.
	for _, k := range []string{"score", "grade", "value", "x_debt"} {
		if _, ok := p.Corpus[k]; !ok {
			t.Errorf("legacy/continuity key %q dropped from corpus", k)
		}
	}
}

// TestFoldDynamicBaseline confirms PassLine moves the baseline a KPI is measured against, so a
// card can trend against a target other than fixed perfection.
func TestFoldDynamicBaseline(t *testing.T) {
	// Score 80 is 20 below the default 100 pass-line -> pressure 20. Move the baseline to 80
	// and the same score sits exactly on it -> pressure 0, slack 0.
	def := Fold("fak-x/1", []KPI{{Key: "a", Score: 80}}, "x_debt", nil, Messages{})
	if got, _ := anyFloat(def.Corpus["pressure"]); got != 20 {
		t.Errorf("default-baseline pressure=%v want 20", def.Corpus["pressure"])
	}
	moved := Fold("fak-x/1", []KPI{{Key: "a", Score: 80, PassLine: 80}}, "x_debt", nil, Messages{})
	if got, _ := anyFloat(moved.Corpus["pressure"]); got != 0 {
		t.Errorf("moved-baseline pressure=%v want 0 (score on the moved pass-line)", moved.Corpus["pressure"])
	}
	// Above a moved-down baseline the surplus shows up as slack.
	ahead := Fold("fak-x/1", []KPI{{Key: "a", Score: 95, PassLine: 80}}, "x_debt", nil, Messages{})
	if got, _ := anyFloat(ahead.Corpus["slack"]); got != 15 {
		t.Errorf("slack=%v want 15 (95 clears the 80 baseline by 15)", ahead.Corpus["slack"])
	}
}

// TestPressureWeighted confirms pressure uses the SAME Group/Key weights as the composite mean
// -- the "continuous weighting" the gauge is built on.
func TestPressureWeighted(t *testing.T) {
	// group "heavy" weighs 3: its 40-point deficit counts triple; "light" weighs 1.
	kpis := []KPI{{Key: "a", Group: "heavy", Score: 60}, {Key: "b", Group: "light", Score: 90}}
	w := map[string]float64{"heavy": 3, "light": 1}
	// pressure = 3*(100-60) + 1*(100-90) = 120 + 10 = 130
	if got := Pressure(kpis, w); got != 130 {
		t.Errorf("weighted Pressure=%g want 130", got)
	}
	p := Fold("fak-x/1", kpis, "x_debt", w, Messages{})
	if got, _ := anyFloat(p.Corpus["pressure"]); got != 130 {
		t.Errorf("folded weighted pressure=%v want 130", p.Corpus["pressure"])
	}
}

// TestKPIMarginComputed confirms Fold fills each KPI's signed, unbounded Margin (Score-PassLine)
// and marshals it.
func TestKPIMarginComputed(t *testing.T) {
	p := Fold("fak-x/1", []KPI{
		{Key: "below", Score: 70},
		{Key: "above", Score: 120},
		{Key: "moved", Score: 50, PassLine: 40},
	}, "x_debt", nil, Messages{})
	byKey := map[string]float64{}
	for _, k := range p.KPIs {
		byKey[k.Key] = k.Margin
	}
	if byKey["below"] != -30 {
		t.Errorf("below margin=%v want -30", byKey["below"])
	}
	if byKey["above"] != 20 {
		t.Errorf("above margin=%v want 20 (unbounded above 100)", byKey["above"])
	}
	if byKey["moved"] != 10 {
		t.Errorf("moved margin=%v want 10 (against the moved 40 baseline)", byKey["moved"])
	}
	b, err := json.Marshal(p.KPIs)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(b), `"margin":`) {
		t.Errorf("KPI must marshal a margin field: %s", string(b))
	}
}

// TestComparePressureTrend confirms Compare surfaces the unbounded pressure trend as a separate
// line without disturbing the debt line, and reads "eased" when pressure drops.
func TestComparePressureTrend(t *testing.T) {
	p := Fold("fak-x/1", []KPI{{Key: "a", Score: 90, Defects: []string{"d"}}}, "x_debt", nil, Messages{})
	// prior: worse pressure (30) and more debt (3).
	base := map[string]any{"corpus": map[string]any{"x_debt": float64(3), "pressure": float64(30)}}
	out := Compare(p, base, "x_debt")
	if !strings.Contains(out, "x_debt 3 -> 1 (improved by 2)") {
		t.Errorf("debt line disturbed: %q", out)
	}
	// current pressure = 100-90 = 10; eased from 30 by 20.
	if !strings.Contains(out, "pressure 30 -> 10 (eased by 20)") {
		t.Errorf("pressure trend line wrong: %q", out)
	}
	// A baseline predating the pressure layer must degrade to no pressure line, not a crash.
	old := Compare(p, map[string]any{"corpus": map[string]any{"x_debt": float64(3)}}, "x_debt")
	if strings.Contains(old, "pressure ") {
		t.Errorf("no pressure line expected for a pre-pressure baseline: %q", old)
	}
}

func TestPassMark(t *testing.T) {
	if got := PassMark(true); got != "yes" {
		t.Fatalf("PassMark(true) = %q, want yes", got)
	}
	if got := PassMark(false); got != "no" {
		t.Fatalf("PassMark(false) = %q, want no", got)
	}
}

func TestCompletionPercent(t *testing.T) {
	if got := CompletionPercent(1, 3); got != 33.3 {
		t.Fatalf("CompletionPercent = %v", got)
	}
	if got := CompletionPercent(0, 0); got != 100 {
		t.Fatalf("empty CompletionPercent = %v", got)
	}
}
