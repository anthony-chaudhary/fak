package scorecardpane

import (
	"strings"
	"testing"
)

func TestOutOfControl_Unpinned(t *testing.T) {
	metrics := fixtureMetrics()
	ooc := AssessOutOfControl(metrics, nil, 551, 9, DefaultOutOfControlBounds())
	if ooc.Status != StatusUnpinned || ooc.IsOutOfControl {
		t.Fatalf("unpinned must have status UNPINNED and IsOutOfControl=false, got %+v", ooc)
	}
}

func TestOutOfControl_Stable(t *testing.T) {
	metrics := fixtureMetrics()
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 551,
		GradeDebt: 9,
		Metrics: map[string]int{
			"code":   15,
			"slop":   535,
			"seo":    1,
			"readme": 0,
		},
		GradeWeights: map[string]int{
			"code":   1,
			"slop":   8,
			"seo":    0,
			"readme": 0,
		},
	}

	ooc := AssessOutOfControl(metrics, base, 551, 9, DefaultOutOfControlBounds())
	if ooc.Status != StatusStable || ooc.IsOutOfControl {
		t.Fatalf("flat metrics must be STABLE and not out of control, got status=%s isOutOfControl=%v",
			ooc.Status, ooc.IsOutOfControl)
	}
	if len(ooc.Metrics) != 0 {
		t.Fatalf("expected 0 out-of-control metrics, got %d", len(ooc.Metrics))
	}
}

func TestOutOfControl_NormalFluctuationsIgnored(t *testing.T) {
	// seo 1 -> 2 (+1) or slop 535 -> 537 (+2): small bounded drift must not trigger out of control.
	metrics := []Metric{
		{Key: "code", Label: "code", Debt: intp(15), Grade: strp("B"), OK: false},
		{Key: "slop", Label: "code-slop", Debt: intp(537), Grade: strp("F"), OK: false},
		{Key: "seo", Label: "seo", Debt: intp(2), Grade: strp("A"), OK: false},
		{Key: "readme", Label: "readme-freshness", Debt: intp(0), Grade: strp("A"), OK: true},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 551,
		GradeDebt: 9,
		Metrics: map[string]int{
			"code":   15,
			"slop":   535,
			"seo":    1,
			"readme": 0,
		},
		GradeWeights: map[string]int{
			"code":   1,
			"slop":   8,
			"seo":    0,
			"readme": 0,
		},
	}

	ooc := AssessOutOfControl(metrics, base, 554, 9, DefaultOutOfControlBounds())
	if ooc.IsOutOfControl {
		t.Fatalf("small fluctuations (+1, +2) must NOT be out of control: %+v", ooc)
	}
	if len(ooc.Metrics) != 0 {
		t.Fatalf("small fluctuations must NOT produce out-of-control metrics: %+v", ooc.Metrics)
	}
}

func TestOutOfControl_PerMetricRateOfChangeSurge(t *testing.T) {
	// code debt 10 -> 25 (+150%, delta=15): rapid relative surge + step breach
	codeWeight := 2 // C
	metrics := []Metric{
		{Key: "code", Label: "code", Debt: intp(25), Grade: strp("C"), EffGrade: "C", GradeWeight: &codeWeight, OK: false},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 10,
		GradeDebt: 1,
		Metrics: map[string]int{
			"code": 10,
		},
		GradeWeights: map[string]int{
			"code": 1, // B
		},
	}

	ooc := AssessOutOfControl(metrics, base, 25, 2, DefaultOutOfControlBounds())
	if len(ooc.Metrics) != 1 {
		t.Fatalf("expected 1 out-of-control metric, got %d", len(ooc.Metrics))
	}
	om := ooc.Metrics[0]
	if om.Key != "code" || om.Delta != 15 {
		t.Fatalf("unexpected metric: %+v", om)
	}
	if om.RateOfChange < 1.49 || om.RateOfChange > 1.51 {
		t.Fatalf("rate of change want 1.5, got %f", om.RateOfChange)
	}

	hasRateSurge := false
	hasStepSurge := false
	for _, c := range om.Classifications {
		if c == ClassRateOfChangeSurge {
			hasRateSurge = true
		}
		if c == ClassStepVelocityBreach {
			hasStepSurge = true
		}
	}
	if !hasRateSurge || !hasStepSurge {
		t.Fatalf("expected RATE_OF_CHANGE_SURGE and STEP_VELOCITY_BREACH, got %v", om.Classifications)
	}
}

func TestOutOfControl_PerMetricZeroBaselineExplosion(t *testing.T) {
	// Clean metric (0 debt) explodes to 8 debt in one jump
	docWeight := 4 // D
	metrics := []Metric{
		{Key: "doc", Label: "docs", Debt: intp(8), Grade: strp("D"), EffGrade: "D", GradeWeight: &docWeight, OK: false},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 0,
		GradeDebt: 0,
		Metrics: map[string]int{
			"doc": 0,
		},
		GradeWeights: map[string]int{
			"doc": 0, // A
		},
	}

	ooc := AssessOutOfControl(metrics, base, 8, 4, DefaultOutOfControlBounds())
	if len(ooc.Metrics) != 1 {
		t.Fatalf("expected 1 out-of-control metric, got %d", len(ooc.Metrics))
	}
	om := ooc.Metrics[0]
	if om.Severity != SeverityCritical {
		t.Fatalf("A->D slip with +8 debt should be CRITICAL, got %s", om.Severity)
	}
	hasRateSurge := false
	hasGradeCollapse := false
	for _, c := range om.Classifications {
		if c == ClassRateOfChangeSurge {
			hasRateSurge = true
		}
		if c == ClassGradeCollapse {
			hasGradeCollapse = true
		}
	}
	if !hasRateSurge || !hasGradeCollapse {
		t.Fatalf("expected RATE_OF_CHANGE_SURGE and GRADE_COLLAPSE, got %v", om.Classifications)
	}
}

func TestOutOfControl_PerMetricCeilingBreach(t *testing.T) {
	// readme ceiling is 0; readme debt becomes 5
	w := 1
	metrics := []Metric{
		{Key: "readme", Label: "readme-freshness", Debt: intp(5), Grade: strp("B"), EffGrade: "B", GradeWeight: &w, OK: false},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 0,
		GradeDebt: 0,
		Metrics: map[string]int{
			"readme": 0,
		},
		GradeWeights: map[string]int{
			"readme": 0,
		},
	}

	ooc := AssessOutOfControl(metrics, base, 5, 1, DefaultOutOfControlBounds())
	if len(ooc.Metrics) != 1 {
		t.Fatalf("expected 1 out-of-control metric, got %d", len(ooc.Metrics))
	}
	om := ooc.Metrics[0]
	hasCeilingBreach := false
	for _, c := range om.Classifications {
		if c == ClassCeilingBreach {
			hasCeilingBreach = true
		}
	}
	if !hasCeilingBreach {
		t.Fatalf("expected CEILING_BREACH in classifications, got %v", om.Classifications)
	}
	if om.Severity != SeverityCritical {
		t.Fatalf("readme ceiling breach >= 5 should be CRITICAL, got %s", om.Severity)
	}
}

func TestOutOfControl_PerMetricGradeCollapse(t *testing.T) {
	// Grade slips from A (0) to C (2) -> slipped 2 tiers -> GRADE_COLLAPSE
	cWeight := 2
	metrics := []Metric{
		{Key: "steer", Label: "steerability", Debt: intp(2), Grade: strp("C"), EffGrade: "C", GradeWeight: &cWeight, OK: false},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 0,
		GradeDebt: 0,
		Metrics: map[string]int{
			"steer": 0,
		},
		GradeWeights: map[string]int{
			"steer": 0, // A
		},
	}

	ooc := AssessOutOfControl(metrics, base, 2, 2, DefaultOutOfControlBounds())
	if len(ooc.Metrics) != 1 {
		t.Fatalf("expected 1 out-of-control metric, got %d", len(ooc.Metrics))
	}
	om := ooc.Metrics[0]
	hasGradeCollapse := false
	for _, c := range om.Classifications {
		if c == ClassGradeCollapse {
			hasGradeCollapse = true
		}
	}
	if !hasGradeCollapse {
		t.Fatalf("expected GRADE_COLLAPSE, got %v", om.Classifications)
	}
}

func TestOutOfControl_RepoTotalRateSurge(t *testing.T) {
	// Baseline total debt: 100. Current total debt: 120 (+20%).
	// Bounds threshold: 0.10 (+10%). Repo status must be OUT_OF_CONTROL.
	metrics := []Metric{
		{Key: "slop", Label: "code-slop", Debt: intp(120), Grade: strp("D"), OK: false},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 100,
		GradeDebt: 4,
		Metrics: map[string]int{
			"slop": 100,
		},
		GradeWeights: map[string]int{
			"slop": 4,
		},
	}

	ooc := AssessOutOfControl(metrics, base, 120, 4, DefaultOutOfControlBounds())
	if !ooc.IsOutOfControl || ooc.Status != StatusOutOfControl {
		t.Fatalf("total debt rate surge of +20%% must trigger OUT_OF_CONTROL: %+v", ooc)
	}
	foundReason := false
	for _, r := range ooc.Reasons {
		if strings.Contains(r, "total debt surged by +20.0%") {
			foundReason = true
			break
		}
	}
	if !foundReason {
		t.Fatalf("expected total debt surge in reasons, got %v", ooc.Reasons)
	}
}

func TestOutOfControl_RepoGradeRateSurge(t *testing.T) {
	// Baseline grade debt: 10. Current grade debt: 14 (+40%).
	// Bounds threshold: 0.25 (+25%). Repo status must be OUT_OF_CONTROL.
	metrics := []Metric{
		{Key: "code", Label: "code", Debt: intp(20), Grade: strp("F"), OK: false},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 100,
		GradeDebt: 10,
		Metrics: map[string]int{
			"code": 20,
		},
		GradeWeights: map[string]int{
			"code": 4,
		},
	}

	ooc := AssessOutOfControl(metrics, base, 100, 14, DefaultOutOfControlBounds())
	if !ooc.IsOutOfControl || ooc.Status != StatusOutOfControl {
		t.Fatalf("grade debt surge of +40%% must trigger OUT_OF_CONTROL: %+v", ooc)
	}
	foundReason := false
	for _, r := range ooc.Reasons {
		if strings.Contains(r, "grade severity surged by +40.0%") {
			foundReason = true
			break
		}
	}
	if !foundReason {
		t.Fatalf("expected grade severity surge in reasons, got %v", ooc.Reasons)
	}
}

func TestOutOfControl_DebtContagion(t *testing.T) {
	// 4 out of 10 scorecards worsening simultaneously = 40% contagion rate.
	// Bounds threshold is 25%. Triggers contagion surge and OUT_OF_CONTROL.
	metrics := []Metric{
		{Key: "m1", Label: "m1", Debt: intp(2), OK: false},
		{Key: "m2", Label: "m2", Debt: intp(2), OK: false},
		{Key: "m3", Label: "m3", Debt: intp(2), OK: false},
		{Key: "m4", Label: "m4", Debt: intp(2), OK: false},
		{Key: "m5", Label: "m5", Debt: intp(1), OK: false},
		{Key: "m6", Label: "m6", Debt: intp(1), OK: false},
		{Key: "m7", Label: "m7", Debt: intp(1), OK: false},
		{Key: "m8", Label: "m8", Debt: intp(1), OK: false},
		{Key: "m9", Label: "m9", Debt: intp(1), OK: false},
		{Key: "m10", Label: "m10", Debt: intp(1), OK: false},
	}
	baseMetrics := map[string]int{
		"m1": 1, "m2": 1, "m3": 1, "m4": 1, "m5": 1,
		"m6": 1, "m7": 1, "m8": 1, "m9": 1, "m10": 1,
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 10,
		GradeDebt: 10,
		Metrics:   baseMetrics,
	}

	ooc := AssessOutOfControl(metrics, base, 14, 10, DefaultOutOfControlBounds())
	if !ooc.IsOutOfControl || ooc.Status != StatusOutOfControl {
		t.Fatalf("contagion rate of 40%% must trigger OUT_OF_CONTROL: %+v", ooc)
	}
	if ooc.ContagionRate < 0.39 || ooc.ContagionRate > 0.41 {
		t.Fatalf("contagion rate want 0.40, got %f", ooc.ContagionRate)
	}
	foundReason := false
	for _, r := range ooc.Reasons {
		if strings.Contains(r, "debt contagion breach") {
			foundReason = true
			break
		}
	}
	if !foundReason {
		t.Fatalf("expected debt contagion breach in reasons, got %v", ooc.Reasons)
	}
}

func TestOutOfControl_CriticalMetricSurge(t *testing.T) {
	// 2 metrics with CRITICAL severity breach -> triggers repo OUT_OF_CONTROL
	wF := 8
	metrics := []Metric{
		{Key: "m1", Label: "m1", Debt: intp(50), Grade: strp("F"), EffGrade: "F", GradeWeight: &wF, OK: false},
		{Key: "m2", Label: "m2", Debt: intp(60), Grade: strp("F"), EffGrade: "F", GradeWeight: &wF, OK: false},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 1000,
		GradeDebt: 10,
		Metrics: map[string]int{
			"m1": 0, // 0 -> 50 (+50, >2*stepSurge) => CRITICAL
			"m2": 0, // 0 -> 60 (+60, >2*stepSurge) => CRITICAL
		},
		GradeWeights: map[string]int{
			"m1": 0,
			"m2": 0,
		},
	}

	ooc := AssessOutOfControl(metrics, base, 1110, 26, DefaultOutOfControlBounds())
	if !ooc.IsOutOfControl || ooc.Status != StatusOutOfControl {
		t.Fatalf("2 critical metrics must trigger repo OUT_OF_CONTROL: %+v", ooc)
	}
	foundReason := false
	for _, r := range ooc.Reasons {
		if strings.Contains(r, "metrics in critical runaway") {
			foundReason = true
			break
		}
	}
	if !foundReason {
		t.Fatalf("expected critical runaway in reasons, got %v", ooc.Reasons)
	}
}

func TestOutOfControl_RenderAndCheckGateIntegration(t *testing.T) {
	// Test that Fold integrates OutOfControl, Render displays alert, and CheckGate reports OUT OF CONTROL
	wF := 8
	metrics := []Metric{
		{Key: "doc", Label: "docs", DebtKey: "doc_debt", Debt: intp(40), Grade: strp("F"), EffGrade: "F", GradeWeight: &wF, OK: false, Verdict: "ACTION"},
	}
	base := &Baseline{
		Schema:    BaselineSchema,
		Commit:    "old0000",
		TotalDebt: 5,
		GradeDebt: 0,
		Metrics: map[string]int{
			"doc": 5,
		},
		GradeWeights: map[string]int{
			"doc": 0, // A
		},
	}

	p := Fold(metrics, base, "/repo", "new1111")
	if p.OutOfControl == nil || !p.OutOfControl.IsOutOfControl {
		t.Fatalf("expected OutOfControl in Payload to be out of control, got %+v", p.OutOfControl)
	}

	renderOutput := Render(p)
	if !strings.Contains(renderOutput, "OUT-OF-CONTROL DEBT ALERT") {
		t.Fatalf("render output missing OUT-OF-CONTROL DEBT ALERT block:\n%s", renderOutput)
	}

	code, msg := CheckGate(p)
	if code != 1 {
		t.Fatalf("gate must fail on out of control regression: code=%d", code)
	}
	if !strings.Contains(msg, "OUT OF CONTROL") {
		t.Fatalf("CheckGate message missing OUT OF CONTROL note: %s", msg)
	}
}
