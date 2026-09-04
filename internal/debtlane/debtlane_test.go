package debtlane

import (
	"encoding/json"
	"go/ast"
	"go/parser"
	"go/token"
	"strings"
	"testing"
	"time"
)

func TestNewWorkUnitAtOneTenthIsDebtNow(t *testing.T) {
	// A new work unit `x` (e.g. new core feature or module) is added at 1/10 on the maturity curve.
	// That is DEBT now!
	evidence := Evidence{
		FilesCount:      1,
		CodeLines:       15, // minimal stub/skeleton
		HasCode:         true,
		HasTests:        false,
		Integrated:      false,
		Dogfooded:       false,
		Benchmarked:     false,
		Documented:      false,
		DependentsCount: 0,
	}

	score, rung := EvaluateMaturityCurve(evidence)
	if score != 1.0 {
		t.Fatalf("expected 1.0 (1/10 on maturity curve), got %.1f", score)
	}
	if rung != "stub" {
		t.Fatalf("expected rung 'stub', got %q", rung)
	}

	// For a core module, target ceiling is 10.0 and weight is 3.0.
	bounds := DefaultBoundsAndLimits(CriticalityCore)
	weight := CriticalityWeight(CriticalityCore)
	gap := bounds.TargetCeiling - score
	if gap != 9.0 {
		t.Fatalf("expected maturity gap of 9.0, got %.1f", gap)
	}

	interest := CalculateInterest(CriticalityCore, bounds, evidence, gap)
	principal, carrying, total := CalculateDebt(score, bounds.TargetCeiling, weight, interest, bounds)

	// Principal = 9.0 gap * 3.0 weight = 27.0 debt points!
	if principal != 27.0 {
		t.Fatalf("expected debt principal 27.0, got %.1f", principal)
	}
	if carrying <= 0 {
		t.Fatalf("expected positive carrying cost, got %.1f", carrying)
	}
	if total <= principal {
		t.Fatalf("expected total debt > principal, got total=%.1f principal=%.1f", total, principal)
	}

	// Verify that adding `x` expands the denominator and reflects dilution.
	baselineLanes := []DebtLane{
		{
			Lane:                    "matured_module_a",
			Criticality:             CriticalityCore,
			Weight:                  3.0,
			Maturity:                10.0,
			TargetMaturity:          10.0,
			MaturityGap:             0.0,
			DenominatorContribution: 30.0,
			RealizedContribution:    30.0,
		},
		{
			Lane:                    "matured_module_b",
			Criticality:             CriticalityCore,
			Weight:                  3.0,
			Maturity:                10.0,
			TargetMaturity:          10.0,
			MaturityGap:             0.0,
			DenominatorContribution: 30.0,
			RealizedContribution:    30.0,
		},
	}

	gradeBefore := CalculateProductionGrade(baselineLanes)
	if gradeBefore.GradePercent != 100.0 {
		t.Fatalf("expected 100%% before adding WIP, got %.1f%%", gradeBefore.GradePercent)
	}
	if gradeBefore.DilutionFromWIP != 0.0 {
		t.Fatalf("expected 0%% dilution before WIP, got %.1f%%", gradeBefore.DilutionFromWIP)
	}

	// Now add the new 1/10 WIP unit of work `x`.
	newLanes := append(baselineLanes, DebtLane{
		Lane:                    "new_feature_x",
		Criticality:             CriticalityCore,
		Weight:                  weight,
		Maturity:                score,
		TargetMaturity:          bounds.TargetCeiling,
		MaturityGap:             gap,
		DebtPrincipal:           principal,
		Interest:                interest,
		CarryingCost:            carrying,
		TotalDebt:               total,
		DenominatorContribution: bounds.TargetCeiling * weight, // 30.0 added to denominator!
		RealizedContribution:    score * weight,                // only 3.0 realized!
	})

	gradeAfter := CalculateProductionGrade(newLanes)
	// New denominator = 30 + 30 + 30 = 90.0
	// Realized = 30 + 30 + 3 = 63.0
	// Grade = (63.0 / 90.0) * 100 = 70.0%
	if gradeAfter.DenominatorPoints != 90.0 {
		t.Fatalf("expected denominator 90.0, got %.1f", gradeAfter.DenominatorPoints)
	}
	if gradeAfter.RealizedPoints != 63.0 {
		t.Fatalf("expected realized 63.0, got %.1f", gradeAfter.RealizedPoints)
	}
	if gradeAfter.GradePercent != 70.0 {
		t.Fatalf("expected 70.0%%, got %.1f%%", gradeAfter.GradePercent)
	}
	if gradeAfter.DilutionFromWIP != 30.0 {
		t.Fatalf("expected 30.0%% dilution from WIP, got %.1f%%", gradeAfter.DilutionFromWIP)
	}
}

func TestBoundsAndLimits(t *testing.T) {
	// Some features don't need to progress as fast (e.g. peripheral tools, demos).
	// They have bounded ceilings and lower interest rate caps.
	peripheralBounds := DefaultBoundsAndLimits(CriticalityPeripheral)
	if peripheralBounds.TargetCeiling != 4.0 {
		t.Fatalf("expected peripheral target ceiling 4.0, got %.1f", peripheralBounds.TargetCeiling)
	}
	if peripheralBounds.Pacing != PacingRelaxed {
		t.Fatalf("expected relaxed pacing, got %s", peripheralBounds.Pacing)
	}
	if peripheralBounds.MaxInterestCap > 0.15 {
		t.Fatalf("expected low interest cap for peripheral, got %.2f", peripheralBounds.MaxInterestCap)
	}

	coreBounds := DefaultBoundsAndLimits(CriticalityCore)
	if coreBounds.TargetCeiling != 10.0 {
		t.Fatalf("expected core target ceiling 10.0, got %.1f", coreBounds.TargetCeiling)
	}
	if coreBounds.Pacing != PacingUrgent {
		t.Fatalf("expected urgent pacing, got %s", coreBounds.Pacing)
	}
	if coreBounds.MaxInterestCap <= peripheralBounds.MaxInterestCap {
		t.Fatalf("expected core interest cap > peripheral cap")
	}

	// Test carrying cost cap prevents infinite compounding.
	boundsWithCap := BoundsAndLimits{
		TargetCeiling:   10.0,
		MaxInterestCap:  0.30,
		CarryingCostCap: 5.0, // Hard cap
	}
	interest := Interest{Rate: 0.25}
	principal, carrying, total := CalculateDebt(0.0, 10.0, 10.0, interest, boundsWithCap)
	// Uncapped carrying would be 100 * 0.25 = 25.0, but capped at 5.0.
	if principal != 100.0 {
		t.Fatalf("expected principal 100.0, got %.1f", principal)
	}
	if carrying != 5.0 {
		t.Fatalf("expected carrying cost capped at 5.0, got %.1f", carrying)
	}
	if total != 105.0 {
		t.Fatalf("expected total debt 105.0, got %.1f", total)
	}
}

func TestRelativeInterestRateAndDrivers(t *testing.T) {
	bounds := DefaultBoundsAndLimits(CriticalityCore)

	// Low-risk evidence: peripheral, no dependents.
	evLow := Evidence{
		HasCode:         true,
		HasTests:        true,
		DependentsCount: 0,
	}
	periBounds := DefaultBoundsAndLimits(CriticalityPeripheral)
	intLow := CalculateInterest(CriticalityPeripheral, periBounds, evLow, 2.0)
	if intLow.Band != InterestLow {
		t.Fatalf("expected low band for peripheral, got %s (rate %.3f)", intLow.Band, intLow.Rate)
	}

	// High-risk evidence: core module with 30 dependents and integrated without tests.
	evHigh := Evidence{
		HasCode:         true,
		HasTests:        false,
		Integrated:      true,
		DependentsCount: 30,
	}
	intHigh := CalculateInterest(CriticalityCore, bounds, evHigh, 5.0)
	if intHigh.Band != InterestCritical && intHigh.Band != InterestHigh {
		t.Fatalf("expected high/critical band for high blast radius, got %s (rate %.3f)", intHigh.Band, intHigh.Rate)
	}

	// Verify drivers are recorded.
	drivers := strings.Join(intHigh.Drivers, ",")
	if !strings.Contains(drivers, "core_runtime_criticality") {
		t.Errorf("missing core driver: %s", drivers)
	}
	if !strings.Contains(drivers, "high_blast_radius") {
		t.Errorf("missing blast radius driver: %s", drivers)
	}
	if !strings.Contains(drivers, "integrated_untested_hazard") {
		t.Errorf("missing untested hazard driver: %s", drivers)
	}
}

func TestHermeticScanReport(t *testing.T) {
	fixedTime := time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	opts := Options{
		Clock: func() time.Time { return fixedTime },
		Facts: []DebtLane{
			{
				Lane:        "gateway",
				UnitOfWork:  "internal/gateway",
				Criticality: CriticalityCore,
				Maturity:    8.0,
				Evidence: Evidence{
					HasCode:  true,
					HasTests: true,
				},
			},
			{
				Lane:        "new_experimental",
				UnitOfWork:  "internal/new_experimental",
				Criticality: CriticalityPeripheral,
				Maturity:    1.0,
				Evidence: Evidence{
					HasCode:   true,
					CodeLines: 10,
				},
			},
		},
	}

	report, err := Scan(opts)
	if err != nil {
		t.Fatalf("Scan failed: %v", err)
	}

	if report.Schema != Schema {
		t.Fatalf("expected schema %q, got %q", Schema, report.Schema)
	}
	if report.Verdict != "ACTION" {
		t.Fatalf("expected ACTION verdict, got %q", report.Verdict)
	}
	if report.EvaluatedAt != "2026-09-03T12:00:00Z" {
		t.Fatalf("expected fixed timestamp, got %q", report.EvaluatedAt)
	}
	if len(report.Lanes) != 2 {
		t.Fatalf("expected 2 lanes, got %d", len(report.Lanes))
	}

	// Check JSON serialization round-trip.
	data, err := json.Marshal(report)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	var roundTrip Report
	if err := json.Unmarshal(data, &roundTrip); err != nil {
		t.Fatalf("json.Unmarshal failed: %v", err)
	}
	if roundTrip.Schema != Schema || roundTrip.ProductionGrade.TotalUnits != 2 {
		t.Fatalf("corrupted roundtrip: %+v", roundTrip)
	}

	// Check text and markdown rendering.
	text := Render(report)
	if !strings.Contains(text, "FAK MATURITY DEBT LANES") || !strings.Contains(text, "Production Grade") {
		t.Fatalf("unexpected text render: %s", text)
	}

	md := Markdown(report)
	if !strings.Contains(md, "Maturity Debt Lanes Scorecard") || !strings.Contains(md, "WIP Dilution") {
		t.Fatalf("unexpected markdown render: %s", md)
	}

	// Check compare.
	cmp := Compare(report, report)
	if !strings.Contains(cmp, "DEBT LANES COMPARISON") {
		t.Fatalf("unexpected compare output: %s", cmp)
	}
}

func TestClassifyCriticality(t *testing.T) {
	if ClassifyCriticality("gateway") != CriticalityCore {
		t.Errorf("gateway should be core")
	}
	if ClassifyCriticality("promptmmu") != CriticalityEnabling {
		t.Errorf("promptmmu should be enabling")
	}
	if ClassifyCriticality("leakcheck") != CriticalityStewardship {
		t.Errorf("leakcheck should be stewardship")
	}
	if ClassifyCriticality("terminalbench") != CriticalityPeripheral {
		t.Errorf("terminalbench should be peripheral")
	}
}

func TestRealWorkspaceScan(t *testing.T) {
	report, err := Scan(Options{WorkspaceRoot: "../..", TopN: 5})
	if err != nil {
		t.Fatalf("Scan on real workspace failed: %v", err)
	}
	if report.Schema != Schema {
		t.Errorf("expected schema %q, got %q", Schema, report.Schema)
	}
	if report.ProductionGrade.TotalUnits == 0 {
		t.Errorf("expected units to be discovered, got 0")
	}
	if report.ProductionGrade.DenominatorPoints <= 0 {
		t.Errorf("expected positive denominator points, got %.1f", report.ProductionGrade.DenominatorPoints)
	}
	if len(report.Hotspots) == 0 {
		t.Errorf("expected hotspots to be populated")
	}
}

func TestTreesOverlap(t *testing.T) {
	if !TreesOverlap("internal/foo", "internal/foo") {
		t.Errorf("identical paths should overlap")
	}
	if !TreesOverlap("internal/foo", "internal/foo/sub") {
		t.Errorf("parent and child should overlap")
	}
	if !TreesOverlap("internal/foo/sub", "internal/foo") {
		t.Errorf("child and parent should overlap")
	}
	if TreesOverlap("internal/foo", "internal/foobar") {
		t.Errorf("sibling prefixes with different names should not overlap")
	}
	if TreesOverlap("internal/pkgA", "internal/pkgB") {
		t.Errorf("disjoint packages should not overlap")
	}
	if !TreesOverlap("", "internal/foo") {
		t.Errorf("empty path should overlap conservatively")
	}
}

func TestPlanWavesHermetic(t *testing.T) {
	report := Report{
		Workspace: "/test/workspace",
		ProductionGrade: ProductionGrade{
			DenominatorPoints: 100.0,
			RealizedPoints:    50.0,
			GradePercent:      50.0,
			GradeLetter:       "F",
		},
		Lanes: []DebtLane{
			{
				Lane:           "pkgA",
				UnitOfWork:     "internal/pkgA",
				Criticality:    CriticalityEnabling,
				Weight:         2.0,
				Maturity:       0.0,
				TargetMaturity: 8.0,
				MaturityGap:    8.0,
				TotalDebt:      16.0,
			},
			{
				Lane:           "pkgB",
				UnitOfWork:     "internal/pkgB",
				Criticality:    CriticalityEnabling,
				Weight:         2.0,
				Maturity:       0.0,
				TargetMaturity: 8.0,
				MaturityGap:    8.0,
				TotalDebt:      16.0,
			},
			{
				Lane:           "pkgC",
				UnitOfWork:     "internal/pkgC",
				Criticality:    CriticalityEnabling,
				Weight:         2.0,
				Maturity:       0.0,
				TargetMaturity: 8.0,
				MaturityGap:    8.0,
				TotalDebt:      16.0,
			},
			{
				Lane:           "abi",
				UnitOfWork:     "internal/abi",
				Criticality:    CriticalityCore,
				Weight:         3.0,
				Maturity:       5.0,
				TargetMaturity: 10.0,
				MaturityGap:    5.0,
				TotalDebt:      15.0,
				Interest:       Interest{Band: InterestCritical},
			},
			{
				Lane:           "pkgA_sub",
				UnitOfWork:     "internal/pkgA/sub",
				Criticality:    CriticalityEnabling,
				Weight:         1.0,
				Maturity:       0.0,
				TargetMaturity: 4.0,
				MaturityGap:    4.0,
				TotalDebt:      4.0,
			},
		},
	}

	opts := WavePlanOptions{
		WaveSize:      2,
		ExcludedLanes: []string{"pkgC"},
	}

	plan := PlanWaves(report, opts)
	if plan.Schema != WavePlanSchema {
		t.Fatalf("expected schema %q, got %q", WavePlanSchema, plan.Schema)
	}

	// pkgC should be excluded.
	for _, w := range plan.Waves {
		for _, l := range w.Lanes {
			if l.Lane == "pkgC" {
				t.Fatalf("pkgC should have been excluded")
			}
		}
	}

	// abi should be a serial singleton.
	var foundSerial bool
	for _, w := range plan.Waves {
		if w.Safety == WaveSafetySerialSingleton {
			foundSerial = true
			if len(w.Lanes) != 1 || w.Lanes[0].Lane != "abi" {
				t.Fatalf("expected serial singleton to contain only abi, got %+v", w.Lanes)
			}
		}
	}
	if !foundSerial {
		t.Fatalf("expected serial singleton wave for core critical abi")
	}

	// pkgA and pkgA_sub should NOT be in the same wave because their trees overlap.
	for _, w := range plan.Waves {
		hasPkgA := false
		hasPkgASub := false
		for _, l := range w.Lanes {
			if l.Lane == "pkgA" {
				hasPkgA = true
			}
			if l.Lane == "pkgA_sub" {
				hasPkgASub = true
			}
		}
		if hasPkgA && hasPkgASub {
			t.Fatalf("pkgA and pkgA_sub overlap and must not be in the same wave")
		}
	}

	// Verify text render and markdown render do not crash and contain key phrases.
	text := RenderWaves(plan, report.ProductionGrade)
	if !strings.Contains(text, "CONCURRENT SAFE WAVE PLAN") || !strings.Contains(text, "DISJOINT PARALLEL") {
		t.Fatalf("unexpected text render: %s", text)
	}

	md := MarkdownWaves(plan, report.ProductionGrade)
	if !strings.Contains(md, "Concurrent Safe Debt Retirement Wave Plan") {
		t.Fatalf("unexpected markdown render: %s", md)
	}
}

func TestPlanWavesImportContention(t *testing.T) {
	report := Report{
		ProductionGrade: ProductionGrade{DenominatorPoints: 50.0, RealizedPoints: 10.0},
		Lanes: []DebtLane{
			{Lane: "leafA", UnitOfWork: "internal/leafA", MaturityGap: 2.0, TotalDebt: 5.0, Weight: 2.0},
			{Lane: "leafB", UnitOfWork: "internal/leafB", MaturityGap: 2.0, TotalDebt: 5.0, Weight: 2.0},
		},
	}

	// leafA imports leafB
	graph := map[string]map[string]struct{}{
		"leafA": {"leafB": {}},
	}

	opts := WavePlanOptions{
		WaveSize: 4,
		Graph:    graph,
	}

	plan := PlanWaves(report, opts)
	// Because leafA imports leafB, they must not be placed in the same wave despite having capacity.
	for _, w := range plan.Waves {
		if len(w.Lanes) > 1 {
			t.Fatalf("expected leafA and leafB in separate waves due to import contention, got %d in wave", len(w.Lanes))
		}
	}
	if plan.TotalWaves < 2 {
		t.Fatalf("expected at least 2 waves, got %d", plan.TotalWaves)
	}
}

func TestParseTargetGrade(t *testing.T) {
	cases := []struct {
		input       string
		expectedPct float64
		expectedOK  bool
	}{
		{"80%", 80.0, true},
		{"85.5%", 85.5, true},
		{"90", 90.0, true},
		{"A", 90.0, true},
		{"B", 80.0, true},
		{"C", 70.0, true},
		{"D", 60.0, true},
		{"Grade B", 80.0, true},
		{"grade a", 90.0, true},
		{"0.85", 85.0, true},
		{"", 0.0, false},
		{"invalid", 0.0, false},
	}

	for _, tc := range cases {
		pct, ok := ParseTargetGrade(tc.input)
		if ok != tc.expectedOK {
			t.Errorf("ParseTargetGrade(%q) ok = %v, expected %v", tc.input, ok, tc.expectedOK)
		}
		if ok && pct != tc.expectedPct {
			t.Errorf("ParseTargetGrade(%q) pct = %.1f, expected %.1f", tc.input, pct, tc.expectedPct)
		}
	}
}

func TestPlanWavesTargetGrade(t *testing.T) {
	report := Report{
		ProductionGrade: ProductionGrade{
			DenominatorPoints: 100.0,
			RealizedPoints:    70.0,
			GradePercent:      70.0,
			GradeLetter:       "C",
		},
		Lanes: []DebtLane{
			{Lane: "lane1", UnitOfWork: "internal/lane1", Maturity: 5.0, TargetMaturity: 10.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0}, // +10 realized
			{Lane: "lane2", UnitOfWork: "internal/lane2", Maturity: 5.0, TargetMaturity: 10.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0}, // +10 realized
			{Lane: "lane3", UnitOfWork: "internal/lane3", Maturity: 5.0, TargetMaturity: 10.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0}, // +10 realized
			{Lane: "lane4", UnitOfWork: "internal/lane4", Maturity: 5.0, TargetMaturity: 10.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0}, // +10 realized
		},
	}

	// Target 80% with wave size 1. Lane 1 (+10 pts) brings total from 70 to 80 (80%).
	// Only 1 wave is required to achieve Grade B (80%).
	plan := PlanWaves(report, WavePlanOptions{
		WaveSize:    1,
		TargetGrade: "80%",
	})

	if plan.TotalWaves != 1 {
		t.Fatalf("expected exactly 1 wave to reach target grade 80%%, got %d", plan.TotalWaves)
	}
	if plan.ProjectedGrade != "B" {
		t.Fatalf("expected projected grade B, got %s", plan.ProjectedGrade)
	}
	if plan.ProjectedPercent < 80.0 {
		t.Fatalf("expected projected percent >= 80.0, got %.1f", plan.ProjectedPercent)
	}
}

func TestPlanWavesTargetPoints(t *testing.T) {
	report := Report{
		ProductionGrade: ProductionGrade{
			DenominatorPoints: 100.0,
			RealizedPoints:    50.0,
			GradePercent:      50.0,
			GradeLetter:       "F",
		},
		Lanes: []DebtLane{
			{Lane: "lane1", UnitOfWork: "internal/lane1", Maturity: 0.0, TargetMaturity: 5.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0},
			{Lane: "lane2", UnitOfWork: "internal/lane2", Maturity: 0.0, TargetMaturity: 5.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0},
			{Lane: "lane3", UnitOfWork: "internal/lane3", Maturity: 0.0, TargetMaturity: 5.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0},
		},
	}

	// Target 15 points with wave size 1. Lane 1 gives 10 debt pts, Lane 2 gives 10 debt pts (total 20 >= 15).
	plan := PlanWaves(report, WavePlanOptions{
		WaveSize:     1,
		TargetPoints: 15.0,
	})

	if plan.TotalWaves != 2 {
		t.Fatalf("expected 2 waves to retire 15 points, got %d", plan.TotalWaves)
	}
	if plan.TotalDebtInPlan < 15.0 {
		t.Fatalf("expected total debt in plan >= 15.0, got %.1f", plan.TotalDebtInPlan)
	}
}

func TestPlanWavesAlreadyAtTargetGrade(t *testing.T) {
	report := Report{
		ProductionGrade: ProductionGrade{
			DenominatorPoints: 100.0,
			RealizedPoints:    85.0,
			GradePercent:      85.0,
			GradeLetter:       "B",
		},
		Lanes: []DebtLane{
			{Lane: "lane1", UnitOfWork: "internal/lane1", Maturity: 5.0, TargetMaturity: 10.0, MaturityGap: 5.0, Weight: 2.0, TotalDebt: 10.0},
		},
	}

	plan := PlanWaves(report, WavePlanOptions{
		WaveSize:    2,
		TargetGrade: "80%",
	})

	if plan.TotalWaves != 0 {
		t.Fatalf("expected 0 waves when already at target grade, got %d", plan.TotalWaves)
	}
	if plan.PlannedLanes != 0 {
		t.Fatalf("expected 0 planned lanes, got %d", plan.PlannedLanes)
	}
}

func TestTreesOverlapWindowsSlashes(t *testing.T) {
	if !TreesOverlap(`internal\foo\**`, `internal/foo`) {
		t.Errorf("windows path with glob should overlap normalized path")
	}
	if !TreesOverlap(`internal\foo\sub`, `internal/foo`) {
		t.Errorf("windows child path should overlap parent")
	}
	if TreesOverlap(`internal\foo`, `internal\bar`) {
		t.Errorf("disjoint windows paths should not overlap")
	}
}

func TestAntiGamingTautologicalDocComments(t *testing.T) {
	cases := []struct {
		name string
		text string
		want bool // true = tautological
	}{
		{"Foo", "Foo defines the foo.", true},
		{"Bar", "Bar specifies Bar.", true},
		{"ReasonSupported", "ReasonSupported indicates ReasonSupported.", true},
		{"Model", "Model is a model.", true},
		{"OutcomeSupported", "OutcomeSupported indicates outcome supported.", true},
		{"PackBytes", "PackBytes returns the embedded raw JSON bytes of the test pack fixture.", false},
		{"Exported", "Exported explains the public contract and invariants.", false},
		{"ReasonSupported", "ReasonSupported indicates full support for the metadata and capability envelope.", false},
		{"Config", "Config specifies engine execution hyperparameters and timeout budgets.", false},
	}

	for _, tc := range cases {
		got := isTautologicalDoc(tc.name, tc.text)
		if got != tc.want {
			t.Errorf("isTautologicalDoc(%q, %q) = %v, want %v", tc.name, tc.text, got, tc.want)
		}
	}
}

func TestAntiGamingBenchmarkValidation(t *testing.T) {
	fset := token.NewFileSet()
	src := `package test
import "testing"
func BenchmarkEmpty(b *testing.B) {}
func BenchmarkStub(b *testing.B) {
	_ = 1 + 1
}
func BenchmarkValidFor(b *testing.B) {
	for i := 0; i < b.N; i++ {
		_ = i
	}
}
func BenchmarkValidRun(b *testing.B) {
	b.Run("sub", func(b *testing.B) {})
}
`
	node, err := parser.ParseFile(fset, "sample_bench_test.go", src, 0)
	if err != nil {
		t.Fatal(err)
	}

	for _, decl := range node.Decls {
		fn, ok := decl.(*ast.FuncDecl)
		if !ok {
			continue
		}
		switch fn.Name.Name {
		case "BenchmarkEmpty", "BenchmarkStub":
			if isSubstantiveBenchmark(fn) {
				t.Errorf("%s should NOT be recognized as substantive benchmark", fn.Name.Name)
			}
		case "BenchmarkValidFor", "BenchmarkValidRun":
			if !isSubstantiveBenchmark(fn) {
				t.Errorf("%s should be recognized as substantive benchmark", fn.Name.Name)
			}
		}
	}
}

func TestExcessCommentsDetectedAndPenalized(t *testing.T) {
	// 1. High comment ratio (> 0.35) on substantive code (> 30 lines) is flagged as excess comments.
	evHighRatio := Evidence{
		HasCode:         true,
		CodeLines:       100,
		CommentLines:    45,
		CommentRatio:    0.45,
		ExcessComments:  true,
		HasTests:        true,
		TestFilesCount:  2,
		Integrated:      true,
		Dogfooded:       true,
		Benchmarked:     true,
		DependentsCount: 1,
	}

	bounds := DefaultBoundsAndLimits(CriticalityCore)
	interest := CalculateInterest(CriticalityCore, bounds, evHighRatio, 2.0)

	// Interest should include excess_comment_bloat driver and penalty.
	foundDriver := false
	for _, d := range interest.Drivers {
		if strings.Contains(d, "excess_comment_bloat") && strings.Contains(d, "45.0% comments") {
			foundDriver = true
			break
		}
	}
	if !foundDriver {
		t.Fatalf("expected excess_comment_bloat driver in interest, got: %v", interest.Drivers)
	}

	// 2. EvaluateMaturityCurve applies penalty and blocks advancing to hardened or production_grade.
	score, rung := EvaluateMaturityCurve(evHighRatio)
	if rung == "production_grade" || rung == "hardened" {
		t.Fatalf("excess comments must prevent reaching hardened/production_grade rungs, got rung %q with score %.1f", rung, score)
	}

	// Compare with identical clean evidence without excess comments.
	evClean := evHighRatio
	evClean.CommentLines = 10
	evClean.CommentRatio = 0.10
	evClean.ExcessComments = false

	cleanScore, cleanRung := EvaluateMaturityCurve(evClean)
	if score >= cleanScore {
		t.Fatalf("score with excess comments (%.1f) should be lower than clean score (%.1f)", score, cleanScore)
	}
	if cleanRung != "production_grade" {
		t.Fatalf("clean fully-verified evidence should reach production_grade, got %q", cleanRung)
	}

	// 3. NextActionForGap suggests cleaning comment bloat and formulaic noise.
	action := NextActionForGap("mycore", "internal/mycore", score, 10.0, evHighRatio)
	if !strings.Contains(action, "clean mycore: prune excess comment bloat and formulaic noise") {
		t.Fatalf("expected comment bloat cleanup action, got: %s", action)
	}
	if !strings.Contains(action, "45.0% comment ratio") {
		t.Fatalf("expected action to cite 45.0%% comment ratio, got: %s", action)
	}
}

func TestCommentsDoNotAwardMaturityPoints(t *testing.T) {
	// Adding comments or formulaic doc headers must NOT award maturity points.
	base := Evidence{
		HasCode:           true,
		CodeLines:         100,
		HasTests:          true,
		TestFilesCount:    2,
		Integrated:        true,
		Dogfooded:         true,
		Benchmarked:       true,
		ExportedSymbols:   10,
		DocumentedExports: 0,
		DependentsCount:   1,
	}

	scoreBase, rungBase := EvaluateMaturityCurve(base)

	// Even if someone claims 100% documented exports or contract comments,
	// no artificial bonus is awarded.
	withComments := base
	withComments.DocumentedExports = 10
	withComments.Documented = true
	withComments.HasContractComments = true

	scoreCommented, rungCommented := EvaluateMaturityCurve(withComments)
	if scoreCommented != scoreBase {
		t.Fatalf("comments/docs must NOT increase maturity score: base=%.1f commented=%.1f", scoreBase, scoreCommented)
	}
	if rungCommented != rungBase {
		t.Fatalf("comments/docs must NOT alter maturity rung: base=%q commented=%q", rungBase, rungCommented)
	}
}

func TestFormulaicCommentsDetectedAsExcess(t *testing.T) {
	fset := token.NewFileSet()
	src := `package test

// Invariant: invariant assumption guard fail-closed
var a = 1

// Contract: callers must acquire the lock before mutating state partitions.
var b = 2

// Fail-closed: returns error if context is cancelled
var c = 3
`
	node, err := parser.ParseFile(fset, "sample.go", src, parser.ParseComments)
	if err != nil {
		t.Fatal(err)
	}

	formulaicCount := 0
	hasFiller := false
	for _, cg := range node.Comments {
		isForm, isFill := isFormulaicGamingComment(cg)
		if isForm {
			formulaicCount++
		}
		if isFill {
			hasFiller = true
		}
	}

	if formulaicCount < 3 {
		t.Fatalf("expected all 3 formulaic comments to be detected, got %d", formulaicCount)
	}
	if !hasFiller {
		t.Fatalf("expected keyword-stuffed comment to be detected as filler")
	}
}

func TestMaturityRungGatingRequiresPrerequisites(t *testing.T) {
	// An unintegrated package with tests, benchmarks, docs, contracts
	// MUST NOT climb past tested (ceiling 5.0) or claim dogfooded/integrated rungs.
	unintegrated := Evidence{
		HasCode:             true,
		CodeLines:           250,
		HasTests:            true,
		TestFilesCount:      2,
		Benchmarked:         true,
		Documented:          true,
		ExportedSymbols:     10,
		DocumentedExports:   10,
		HasContractComments: true,
		Integrated:          false,
		Dogfooded:           false,
	}

	score, rung := EvaluateMaturityCurve(unintegrated)
	if score > 5.0 {
		t.Fatalf("unintegrated package score must be clamped at 5.0, got %.1f", score)
	}
	if rung != "tested" {
		t.Fatalf("unintegrated package rung must be 'tested', got %q", rung)
	}

	// Once integrated, climbs to integrated.
	integrated := unintegrated
	integrated.Integrated = true
	scoreInt, rungInt := EvaluateMaturityCurve(integrated)
	if scoreInt < 6.5 {
		t.Fatalf("integrated package score should climb, got %.1f", scoreInt)
	}
	if rungInt != "integrated" {
		t.Fatalf("integrated without dogfooded proof must be rung 'integrated', got %q", rungInt)
	}

	// Once dogfooded, reaches dogfooded or higher rung.
	dogfooded := integrated
	dogfooded.Dogfooded = true
	_, rungDog := EvaluateMaturityCurve(dogfooded)
	if rungDog != "production_grade" && rungDog != "hardened" && rungDog != "benchmarked" && rungDog != "dogfooded" {
		t.Fatalf("dogfooded package should reach at least dogfooded rung, got %q", rungDog)
	}
}
