package debtlane

import (
	"encoding/json"
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
