package debtlane

import (
	"os"
	"path/filepath"
	"testing"
)

// TestCalculateProductionGradeClampsOvershoot pins that a lane whose maturity
// exceeds its target ceiling cannot push the grade past 100% or hide another
// lane's debt (the live fak scan reported "Grade A (104.7%)" with 1075 debt pts).
func TestCalculateProductionGradeClampsOvershoot(t *testing.T) {
	lanes := []DebtLane{
		{Lane: "over", Maturity: 10, TargetMaturity: 7, Weight: 1.5, DenominatorContribution: 10.5, RealizedContribution: 15},
		{Lane: "debt", Maturity: 2, TargetMaturity: 8, Weight: 2, DenominatorContribution: 16, RealizedContribution: 4, MaturityGap: 6},
	}
	pg := CalculateProductionGrade(lanes)
	if pg.RealizedPoints != 14.5 {
		t.Fatalf("realized = %.1f, want 14.5 (overshoot clamped to its 10.5 denominator)", pg.RealizedPoints)
	}
	if pg.GradePercent > 100 || pg.DilutionFromWIP < 0 {
		t.Fatalf("grade = %.1f%% dilution = %.1f%%, want <=100%% and non-negative dilution", pg.GradePercent, pg.DilutionFromWIP)
	}
	if want := 54.7; pg.GradePercent != want {
		t.Fatalf("grade = %.1f%%, want %.1f%%", pg.GradePercent, want)
	}
}

// TestPlanWavesSerialSingletonNotStarvedByMaxWaves pins that the
// highest-priority core singleton is planned inside a --max-waves budget
// instead of being appended after every parallel wave.
func TestPlanWavesSerialSingletonNotStarvedByMaxWaves(t *testing.T) {
	lanes := []DebtLane{
		{Lane: "adjudicator", UnitOfWork: "internal/adjudicator", Criticality: CriticalityCore, Weight: 3, Maturity: 7, TargetMaturity: 10, MaturityGap: 3, TotalDebt: 12.2, Interest: Interest{Band: InterestCritical}},
	}
	for _, n := range []string{"a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l"} {
		lanes = append(lanes, DebtLane{Lane: "leaf" + n, UnitOfWork: "internal/leaf" + n, Criticality: CriticalityEnabling, Weight: 2, Maturity: 6, TargetMaturity: 8, MaturityGap: 2, TotalDebt: 4})
	}
	plan := PlanWaves(Report{Lanes: lanes}, WavePlanOptions{WaveSize: 4, MaxWaves: 2, Graph: map[string]map[string]struct{}{}})
	if plan.TotalWaves != 2 {
		t.Fatalf("total waves = %d, want 2", plan.TotalWaves)
	}
	if plan.Waves[0].Safety != WaveSafetySerialSingleton || plan.Waves[0].Lanes[0].Lane != "adjudicator" {
		t.Fatalf("wave-1 = %v (%s), want the top-debt core singleton adjudicator first", plan.Waves[0].LaneNames, plan.Waves[0].Safety)
	}
	if plan.Waves[1].Safety != WaveSafetyDisjointLeaf || plan.Waves[1].WaveSize != 4 {
		t.Fatalf("wave-2 = %v (%s), want a full parallel wave", plan.Waves[1].LaneNames, plan.Waves[1].Safety)
	}
}

// TestScanExpandedSurfaceLanesCarryInterestAndGrade pins that cmd/ (expanded
// surface) lanes get an interest band, a next action, and a denominator share,
// like package lanes. Before, 234 live lanes rendered "0.0% ()" with no next
// action and their debt was outside the production grade.
func TestScanExpandedSurfaceLanesCarryInterestAndGrade(t *testing.T) {
	tmp := t.TempDir()
	cmdDir := filepath.Join(tmp, "cmd", "fixturetool")
	if err := os.MkdirAll(cmdDir, 0o755); err != nil {
		t.Fatal(err)
	}
	src := "package main\n\nfunc main() {}\n"
	if err := os.WriteFile(filepath.Join(cmdDir, "main.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}
	report, err := Scan(Options{WorkspaceRoot: tmp, TargetRepo: "fak", ExpandedBreadth: true, TopN: 50})
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	var found *DebtLane
	for i := range report.Lanes {
		if report.Lanes[i].Lane == "cmd_fixturetool" {
			found = &report.Lanes[i]
		}
	}
	if found == nil {
		t.Fatalf("cmd_fixturetool lane not discovered")
	}
	if found.MaturityGap <= 0 {
		t.Fatalf("fixture lane gap = %.1f, want >0 (untested stub)", found.MaturityGap)
	}
	if found.Interest.Band == "" {
		t.Errorf("interest band empty for expanded-surface lane")
	}
	if found.NextAction == "" {
		t.Errorf("next action empty for expanded-surface lane")
	}
	if found.DenominatorContribution <= 0 {
		t.Errorf("denominator contribution = %.1f, want >0", found.DenominatorContribution)
	}
	if _, ok := report.InterestSummary.Bands[""]; ok {
		t.Errorf("interest summary has an empty band bucket: %v", report.InterestSummary.Bands)
	}
}
