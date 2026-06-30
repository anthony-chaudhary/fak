package scorecardpane

import (
	"encoding/json"
	"strings"
	"testing"
)

// intp / strp / fltp are small pointer constructors for fixture payloads.
func intp(i int) *int         { return &i }
func strp(s string) *string   { return &s }
func fltp(f float64) *float64 { return &f }

// fixtureMetrics builds a small, deterministic set of metric rows for the fold tests.
func fixtureMetrics() []Metric {
	return []Metric{
		{Key: "code", Label: "code", DebtKey: "code_debt", Debt: intp(15), Grade: strp("B"), OK: false, Verdict: "ACTION"},
		{Key: "slop", Label: "code-slop", DebtKey: "slop_debt", Debt: intp(535), Grade: strp("F"), OK: false, Verdict: "ACTION"},
		{Key: "seo", Label: "seo", DebtKey: "seo_debt", Debt: intp(1), Score: fltp(92.5), OK: false, Verdict: "ACTION"},
		{Key: "readme", Label: "readme-freshness", DebtKey: "readme_debt", Debt: intp(0), OK: true, Verdict: "OK"},
	}
}

func TestMetricFromPayloadExtractsCorpusDebt(t *testing.T) {
	card := Card{Key: "code", Debt: "code_debt", Label: "code"}
	payload := map[string]any{
		"corpus": map[string]any{"code_debt": float64(15), "grade": "B"},
		"ok":     false, "verdict": "ACTION",
	}
	m := MetricFromPayload(card, payload, "")
	if m.Debt == nil || *m.Debt != 15 {
		t.Fatalf("debt: want 15, got %v", m.Debt)
	}
	if m.Grade == nil || *m.Grade != "B" {
		t.Fatalf("grade: want B, got %v", m.Grade)
	}
	if m.Verdict != "ACTION" || m.OK {
		t.Fatalf("ok/verdict mismatch: ok=%v verdict=%q", m.OK, m.Verdict)
	}
}

func TestMetricFromPayloadDocAppealNesting(t *testing.T) {
	// doc-appeal nests its debt under "doc", not "corpus".
	card := Card{Key: "appeal", Debt: "appeal_debt", Label: "doc-appeal"}
	payload := map[string]any{"doc": map[string]any{"appeal_debt": float64(15), "grade": "B"}}
	m := MetricFromPayload(card, payload, "")
	if m.Debt == nil || *m.Debt != 15 {
		t.Fatalf("doc-nested debt: want 15, got %v", m.Debt)
	}
}

func TestMetricFromPayloadErrorRow(t *testing.T) {
	card := Card{Key: "ui_quality", Debt: "ui_quality_debt", Label: "ui-quality"}
	m := MetricFromPayload(card, nil, "non-JSON output (exit 2): boom")
	if m.Debt != nil {
		t.Fatalf("errored metric must carry nil debt, got %v", *m.Debt)
	}
	if m.Verdict != "ERROR" || m.Error == "" {
		t.Fatalf("error row malformed: verdict=%q error=%q", m.Verdict, m.Error)
	}
	// nil-debt metrics serialize as JSON null (the Python contract).
	b, _ := json.Marshal(m)
	if !strings.Contains(string(b), `"debt":null`) {
		t.Fatalf("errored debt must serialize as null: %s", b)
	}
}

func TestFoldSumsPortfolioDebt(t *testing.T) {
	p := Fold(fixtureMetrics(), nil, "/repo", "abc1234")
	if p.TotalDebt != 15+535+1+0 {
		t.Fatalf("total_debt: want 551, got %d", p.TotalDebt)
	}
	// grade_debt: code B(1) + slop F(8) + seo score 92.5 -> A(0) + readme debt 0 -> A(0) = 9.
	if p.GradeDebt != 9 {
		t.Fatalf("grade_debt: want 9, got %d", p.GradeDebt)
	}
	if p.Schema != Schema {
		t.Fatalf("schema: want %q, got %q", Schema, p.Schema)
	}
	if p.Measured != 4 || p.Errored != 0 {
		t.Fatalf("measured/errored: got %d/%d", p.Measured, p.Errored)
	}
	if p.Verdict != "ACTION" || p.Finding != "scorecard_debt" {
		t.Fatalf("nonzero unpinned debt should be scorecard_debt/ACTION, got %s/%s", p.Verdict, p.Finding)
	}
}

func TestFoldUnpinnedTrend(t *testing.T) {
	p := Fold(fixtureMetrics(), nil, "/repo", "abc1234")
	if p.Trend.Direction != "unpinned" {
		t.Fatalf("no baseline must be unpinned, got %q", p.Trend.Direction)
	}
}

func TestFoldRegressionVerdict(t *testing.T) {
	base := &Baseline{
		Schema: BaselineSchema, Commit: "old0000", TotalDebt: 500, GradeDebt: 8,
		Metrics: map[string]int{"code": 15, "slop": 500, "seo": 1, "readme": 0},
	}
	p := Fold(fixtureMetrics(), base, "/repo", "new1111")
	if p.Trend.Direction != "regressed" {
		t.Fatalf("slop 500->535 must regress, got %q", p.Trend.Direction)
	}
	if p.Finding != "scorecard_regressed" || p.OK {
		t.Fatalf("regression should be scorecard_regressed/!ok, got %s ok=%v", p.Finding, p.OK)
	}
	code, msg := CheckGate(p)
	if code != 1 || !strings.Contains(msg, "RATCHET FAIL") {
		t.Fatalf("gate must FAIL on regression: code=%d msg=%q", code, msg)
	}
}

func TestFoldEarlyWarningHiddenUnderGreenPortfolio(t *testing.T) {
	// seo rose 0->1 but the portfolio total FELL (slop 600->535): the ratchet stays
	// green, but the per-metric rise must surface as an early-warning.
	base := &Baseline{
		Schema: BaselineSchema, Commit: "old0000", TotalDebt: 615, GradeDebt: 9,
		Metrics: map[string]int{"code": 15, "slop": 600, "seo": 0, "readme": 0},
	}
	p := Fold(fixtureMetrics(), base, "/repo", "new1111")
	if p.Trend.Direction != "improved" {
		t.Fatalf("portfolio total fell, want improved, got %q", p.Trend.Direction)
	}
	if len(p.EarlyWarning) != 1 || p.EarlyWarning[0].Key != "seo" {
		t.Fatalf("seo rise must surface as early-warning, got %+v", p.EarlyWarning)
	}
	code, msg := CheckGate(p)
	if code != 0 {
		t.Fatalf("gate must stay GREEN under a hidden per-metric rise, got code %d", code)
	}
	if !strings.Contains(msg, "EARLY-WARNING") {
		t.Fatalf("gate message must carry the advisory early-warning: %q", msg)
	}
}

func TestFoldRatchetHoldsAtBaseline(t *testing.T) {
	base := &Baseline{
		Schema: BaselineSchema, Commit: "old0000", TotalDebt: 551, GradeDebt: 9,
		Metrics: map[string]int{"code": 15, "slop": 535, "seo": 1, "readme": 0},
	}
	p := Fold(fixtureMetrics(), base, "/repo", "new1111")
	if p.Trend.Direction != "flat" {
		t.Fatalf("identical debt must be flat, got %q", p.Trend.Direction)
	}
	code, _ := CheckGate(p)
	if code != 0 {
		t.Fatalf("flat ratchet must be green, got code %d", code)
	}
}

func TestFoldUnmeasuredCardFailsGate(t *testing.T) {
	metrics := append(fixtureMetrics(), Metric{
		Key: "ui_quality", Label: "ui-quality", DebtKey: "ui_quality_debt", Debt: nil,
		Verdict: "ERROR", Error: "non-JSON output (exit 1): build failed",
	})
	base := &Baseline{
		Schema: BaselineSchema, Commit: "old0000", TotalDebt: 551, GradeDebt: 9,
		Metrics: map[string]int{"code": 15, "slop": 535, "seo": 1, "readme": 0},
	}
	p := Fold(metrics, base, "/repo", "new1111")
	if p.Errored != 1 || p.Finding != "scorecard_unmeasured" {
		t.Fatalf("an errored card must mark scorecard_unmeasured, got errored=%d finding=%s", p.Errored, p.Finding)
	}
	code, msg := CheckGate(p)
	if code != 1 {
		t.Fatalf("unmeasured card must fail the gate, got code %d", code)
	}
	// ui_quality is go-backed: the build-break hint must fire.
	if !strings.Contains(msg, "does NOT compile") {
		t.Fatalf("go-backed errored card must carry the build-break hint: %q", msg)
	}
}

func TestCheckGateUnpinned(t *testing.T) {
	p := Fold(fixtureMetrics(), nil, "/repo", "abc1234")
	code, msg := CheckGate(p)
	if code != 2 || !strings.Contains(msg, "UNPINNED") {
		t.Fatalf("no baseline must exit 2 UNPINNED, got code=%d msg=%q", code, msg)
	}
}

func TestBaselineDocRoundTrip(t *testing.T) {
	p := Fold(fixtureMetrics(), nil, "/repo", "abc1234")
	doc := BaselineDoc(p)
	if doc.Schema != BaselineSchema {
		t.Fatalf("baseline schema: want %q, got %q", BaselineSchema, doc.Schema)
	}
	if doc.TotalDebt != 551 || doc.GradeDebt != 9 {
		t.Fatalf("baseline totals: got total=%d grade=%d", doc.TotalDebt, doc.GradeDebt)
	}
	if doc.Metrics["slop"] != 535 {
		t.Fatalf("baseline per-metric: slop want 535, got %d", doc.Metrics["slop"])
	}
	// re-folding against the just-pinned baseline must read flat (the ratchet floor).
	p2 := Fold(fixtureMetrics(), &doc, "/repo", "abc1234")
	if p2.Trend.Direction != "flat" {
		t.Fatalf("re-fold against own pin must be flat, got %q", p2.Trend.Direction)
	}
}

func TestDisplayGradePrecedence(t *testing.T) {
	// emitted letter beats score beats debt.
	if g := displayGrade(Metric{Grade: strp("C"), Score: fltp(95), Debt: intp(0)}); g != "C" {
		t.Fatalf("emitted letter must win, got %q", g)
	}
	if g := displayGrade(Metric{Score: fltp(92.5), Debt: intp(900)}); g != "A" {
		t.Fatalf("score must beat debt magnitude, got %q", g)
	}
	if g := displayGrade(Metric{Debt: intp(11)}); g != "F" {
		t.Fatalf("debt-derived F for >10, got %q", g)
	}
	if g := displayGrade(Metric{Debt: intp(0)}); g != "A" {
		t.Fatalf("zero debt derives A, got %q", g)
	}
}

func TestFindIntDeepWalk(t *testing.T) {
	// the debt may sit at the top level or under a nest; a deep walk finds it.
	if got := findInt(map[string]any{"slop_debt": float64(7)}, "slop_debt"); got == nil || *got != 7 {
		t.Fatalf("top-level int: want 7, got %v", got)
	}
	if got := findInt(map[string]any{"x": map[string]any{"y": map[string]any{"k": float64(3)}}}, "k"); got == nil || *got != 3 {
		t.Fatalf("deep int: want 3, got %v", got)
	}
	// a boolean must never be read as an int.
	if got := findInt(map[string]any{"k": true}, "k"); got != nil {
		t.Fatalf("bool must not be read as int, got %v", *got)
	}
}
