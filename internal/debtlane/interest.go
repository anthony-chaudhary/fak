package debtlane

import (
	"fmt"
	"math"
)

// DefaultBoundsAndLimits returns the standard bounds, target ceiling, and pacing for a criticality tier.
func DefaultBoundsAndLimits(c Criticality) BoundsAndLimits {
	switch c {
	case CriticalityCore:
		return BoundsAndLimits{
			TargetCeiling:   10.0,
			Pacing:          PacingUrgent,
			MaxInterestCap:  0.35,
			CarryingCostCap: 100.0,
		}
	case CriticalityEnabling:
		return BoundsAndLimits{
			TargetCeiling:   8.0,
			Pacing:          PacingStandard,
			MaxInterestCap:  0.25,
			CarryingCostCap: 50.0,
		}
	case CriticalityStewardship:
		return BoundsAndLimits{
			TargetCeiling:   7.0,
			Pacing:          PacingStandard,
			MaxInterestCap:  0.20,
			CarryingCostCap: 30.0,
		}
	default: // CriticalityPeripheral
		return BoundsAndLimits{
			TargetCeiling:   4.0, // Features that don't need to progress as fast have bounded ceilings!
			Pacing:          PacingRelaxed,
			MaxInterestCap:  0.10,
			CarryingCostCap: 15.0,
		}
	}
}

// CriticalityWeight returns the production denominator multiplier for a criticality tier.
func CriticalityWeight(c Criticality) float64 {
	switch c {
	case CriticalityCore:
		return 3.0
	case CriticalityEnabling:
		return 2.0
	case CriticalityStewardship:
		return 1.5
	default: // CriticalityPeripheral
		return 1.0
	}
}

// CalculateInterest computes the relative interest carrying cost rate and drivers.
func CalculateInterest(c Criticality, bounds BoundsAndLimits, e Evidence, gap float64) Interest {
	if bounds.Pacing == PacingFrozen || gap <= 0 {
		return Interest{
			Band:      InterestLow,
			Rate:      0.0,
			RateLabel: "baseline",
			Drivers:   []string{"quiescent_or_at_target"},
		}
	}

	var baseRate float64
	var drivers []string

	switch c {
	case CriticalityCore:
		baseRate = 0.15
		drivers = append(drivers, "core_runtime_criticality")
	case CriticalityEnabling:
		baseRate = 0.08
		drivers = append(drivers, "enabling_infrastructure")
	case CriticalityStewardship:
		baseRate = 0.05
		drivers = append(drivers, "stewardship_governance")
	default:
		baseRate = 0.02
		drivers = append(drivers, "peripheral_optional_scope")
	}

	rate := baseRate

	// Blast radius: dependent inbound imports.
	if e.DependentsCount > 25 {
		rate += 0.08
		drivers = append(drivers, fmt.Sprintf("high_blast_radius (%d dependents)", e.DependentsCount))
	} else if e.DependentsCount > 5 {
		rate += 0.04
		drivers = append(drivers, fmt.Sprintf("moderate_blast_radius (%d dependents)", e.DependentsCount))
	}

	// Structural risk: integrated into production commands without tests.
	if e.Integrated && !e.HasTests {
		rate += 0.08
		drivers = append(drivers, "integrated_untested_hazard")
	}

	// Excess comments penalty: formulaic noise or comment bloat is bad debt.
	if e.ExcessComments {
		rate += 0.05
		drivers = append(drivers, fmt.Sprintf("excess_comment_bloat (%.1f%% comments)", e.CommentRatio*100))
	}

	// Pacing urgency adjustments.
	switch bounds.Pacing {
	case PacingUrgent:
		rate += 0.04
		drivers = append(drivers, "urgent_pacing_accelerator")
	case PacingRelaxed:
		rate -= 0.02
		drivers = append(drivers, "relaxed_pacing_discount")
	}

	// Bounds: clamp to [0.01, MaxInterestCap].
	if rate < 0.01 {
		rate = 0.01
	}
	if bounds.MaxInterestCap > 0 && rate > bounds.MaxInterestCap {
		rate = bounds.MaxInterestCap
		drivers = append(drivers, fmt.Sprintf("clamped_to_max_cap (%.2f)", bounds.MaxInterestCap))
	}

	rate = math.Round(rate*1000) / 1000

	var band InterestBand
	var label string

	switch {
	case rate > 0.25:
		band = InterestCritical
		label = "compounding"
	case rate > 0.15:
		band = InterestHigh
		label = "accelerating"
	case rate > 0.05:
		band = InterestModerate
		label = "elevated"
	default:
		band = InterestLow
		label = "baseline"
	}

	return Interest{
		Band:      band,
		Rate:      rate,
		RateLabel: label,
		Drivers:   drivers,
	}
}

// CalculateDebt computes the principal, carrying cost, and total debt for a lane.
func CalculateDebt(maturity, target, weight float64, interest Interest, bounds BoundsAndLimits) (principal, carryingCost, totalDebt float64) {
	gap := target - maturity
	if gap <= 0 {
		return 0, 0, 0
	}

	principal = math.Round(gap*weight*100) / 100
	cost := principal * interest.Rate
	if bounds.CarryingCostCap > 0 && cost > bounds.CarryingCostCap {
		cost = bounds.CarryingCostCap
	}
	carryingCost = math.Round(cost*100) / 100
	totalDebt = math.Round((principal+carryingCost)*100) / 100
	return principal, carryingCost, totalDebt
}

// CalculateProductionGrade computes the overall denominator, realized points, and grade.
func CalculateProductionGrade(lanes []DebtLane) ProductionGrade {
	var denominator, realized float64
	var ready, wip int

	for _, l := range lanes {
		denominator += l.DenominatorContribution
		realized += l.RealizedContribution
		if l.MaturityGap <= 0.05 {
			ready++
		} else {
			wip++
		}
	}

	var pct float64
	if denominator > 0 {
		pct = math.Round((realized/denominator)*1000) / 10
	}

	// DilutionFromWIP is the percentage points lost because of incomplete WIP units.
	dilution := 0.0
	if denominator > 0 {
		dilution = math.Round((100.0-pct)*10) / 10
	}

	return ProductionGrade{
		DenominatorPoints:    math.Round(denominator*10) / 10,
		RealizedPoints:       math.Round(realized*10) / 10,
		GradePercent:         pct,
		GradeLetter:          GradeLetter(pct),
		DilutionFromWIP:      dilution,
		TotalUnits:           len(lanes),
		ProductionReadyUnits: ready,
		WIPUnits:             wip,
	}
}

// GradeLetter returns the standard letter grade for a percentage.
func GradeLetter(pct float64) string {
	switch {
	case pct >= 90.0:
		return "A"
	case pct >= 80.0:
		return "B"
	case pct >= 70.0:
		return "C"
	case pct >= 60.0:
		return "D"
	default:
		return "F"
	}
}
