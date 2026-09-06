package debtlane

import (
	"bytes"
	"fmt"
	"strings"
	"text/tabwriter"
)

// Render formats a Report as clear, tabular terminal text.
func Render(r Report) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("=== FAK MATURITY DEBT LANES (%s) ===\n", r.Verdict))
	b.WriteString(fmt.Sprintf("Production Grade: %s (%.1f%%) · Realized: %.1f / Denominator: %.1f pts · WIP Dilution: %.1f%%\n",
		r.ProductionGrade.GradeLetter,
		r.ProductionGrade.GradePercent,
		r.ProductionGrade.RealizedPoints,
		r.ProductionGrade.DenominatorPoints,
		r.ProductionGrade.DilutionFromWIP,
	))
	b.WriteString(fmt.Sprintf("Debt Summary:     %.1f total debt pts (%.1f principal + %.1f carrying cost) across %d active WIP lane(s)\n",
		r.Corpus["debt"],
		r.Corpus["debt_principal"],
		r.Corpus["carrying_cost"],
		r.ProductionGrade.WIPUnits,
	))
	b.WriteString(fmt.Sprintf("Interest Bands:   low: %d · moderate: %d · high: %d · critical: %d (avg rate: %.1f%%, max: %.1f%%)\n",
		r.InterestSummary.Bands[string(InterestLow)],
		r.InterestSummary.Bands[string(InterestModerate)],
		r.InterestSummary.Bands[string(InterestHigh)],
		r.InterestSummary.Bands[string(InterestCritical)],
		r.InterestSummary.AverageRate*100,
		r.InterestSummary.MaxRate*100,
	))
	b.WriteString(fmt.Sprintf("Health Status:    healthy: %d · degraded: %d · critical: %d (avg score: %.2f)\n\n",
		r.HealthSummary.HealthyCount,
		r.HealthSummary.DegradedCount,
		r.HealthSummary.CriticalCount,
		r.HealthSummary.AverageScore,
	))

	if len(r.Hotspots) > 0 {
		b.WriteString("TOP DEBT HOTSPOTS (worst-first carrying cost):\n")
		tw := tabwriter.NewWriter(&b, 2, 4, 2, ' ', 0)
		fmt.Fprintln(tw, "LANE\tHEALTH\tCRITICALITY\tCURVE\tGAP\tWT\tPRINCIPAL\tRATE\tCARRYING\tTOTAL DEBT\tNEXT ACTION")
		for _, h := range r.Hotspots {
			fmt.Fprintf(tw, "%s\t%s\t%s\t%.1f/%.1f (%s)\t%.1f\t%.1f\t%.1f\t%.1f%% (%s)\t%.1f\t%.1f\t%s\n",
				h.Lane,
				h.Health.Status,
				h.Criticality,
				h.Maturity,
				h.TargetMaturity,
				h.MaturityRung,
				h.MaturityGap,
				h.Weight,
				h.DebtPrincipal,
				h.Interest.Rate*100,
				h.Interest.Band,
				h.CarryingCost,
				h.TotalDebt,
				truncProse(h.NextAction, 50),
			)
		}
		tw.Flush()
	}

	b.WriteString("\nNext Action: " + r.NextAction + "\n")
	return b.String()
}

// RenderCrossIndex formats a Report focusing on cross-indexed related items and companions.
func RenderCrossIndex(r Report) string {
	var b strings.Builder
	b.WriteString(fmt.Sprintf("=== FAK DEBT CROSS-INDEX & COMPANIONS (%s) ===\n", r.Verdict))
	b.WriteString(fmt.Sprintf("Total Lanes: %d · Healthy: %d · Degraded: %d · Critical: %d\n\n",
		len(r.Lanes),
		r.HealthSummary.HealthyCount,
		r.HealthSummary.DegradedCount,
		r.HealthSummary.CriticalCount,
	))

	tw := tabwriter.NewWriter(&b, 2, 4, 2, ' ', 0)
	fmt.Fprintln(tw, "LANE\tREPO\tHEALTH\tCOMPANION (REPO:UNIT)\tDEP-IN\tDEP-OUT\tTREES")
	for _, l := range r.Lanes {
		comp := "-"
		if l.Related.CompanionLane != "" {
			comp = fmt.Sprintf("%s:%s", l.Related.CompanionRepo, l.Related.CompanionUnitOfWork)
		}
		trees := "-"
		if len(l.Related.DosTrees) > 0 {
			trees = strings.Join(l.Related.DosTrees, ", ")
		}
		fmt.Fprintf(tw, "%s\t%s\t%s\t%s\t%d\t%d\t%s\n",
			l.Lane,
			l.Repo,
			l.Health.Status,
			comp,
			len(l.Related.Dependents),
			len(l.Related.Dependencies),
			truncProse(trees, 40),
		)
	}
	tw.Flush()
	return b.String()
}

// Markdown formats a Report as GitHub/Jekyll markdown documentation.
func Markdown(r Report) string {
	var b bytes.Buffer
	b.WriteString("---\n")
	b.WriteString("title: \"Maturity Debt Lanes Scorecard\"\n")
	b.WriteString("description: \"Level-setting the production grade denominator across every single unit of work and tracking relative maturity carrying cost.\"\n")
	b.WriteString("---\n\n")

	b.WriteString("# Maturity Debt Lanes Scorecard\n\n")
	b.WriteString(fmt.Sprintf("> **Verdict:** %s · **Production Grade:** %s (%.1f%%) · **Denominator:** %.1f pts · **Debt:** %.1f pts\n\n",
		r.Verdict,
		r.ProductionGrade.GradeLetter,
		r.ProductionGrade.GradePercent,
		r.ProductionGrade.DenominatorPoints,
		r.Corpus["debt"],
	))

	b.WriteString("## Production Grade & WIP Dilution\n\n")
	b.WriteString("Adding new units of work expands the denominator of production grade immediately. Incomplete WIP carries principal debt and relative interest until matured.\n\n")
	b.WriteString(fmt.Sprintf("- **Denominator baseline:** `%.1f` points across `%d` tracked units\n", r.ProductionGrade.DenominatorPoints, r.ProductionGrade.TotalUnits))
	b.WriteString(fmt.Sprintf("- **Realized maturity:** `%.1f` points (ready: `%d`, WIP in debt: `%d`)\n", r.ProductionGrade.RealizedPoints, r.ProductionGrade.ProductionReadyUnits, r.ProductionGrade.WIPUnits))
	b.WriteString(fmt.Sprintf("- **WIP dilution:** `%.1f%%` of production grade is currently diluted by incomplete work\n", r.ProductionGrade.DilutionFromWIP))
	b.WriteString(fmt.Sprintf("- **Total debt points:** `%.1f` (`%.1f` principal + `%.1f` carrying cost)\n\n", r.Corpus["debt"], r.Corpus["debt_principal"], r.Corpus["carrying_cost"]))

	b.WriteString("## Interest Bands & Carrying Cost\n\n")
	b.WriteString("| Band | Carrying Rate | Lanes | Operational Implication |\n")
	b.WriteString("|---|---|---:|---|\n")
	b.WriteString(fmt.Sprintf("| `low` | 0%% – 5%% (baseline) | %d | Peripheral, well-bounded, or relaxed pacing |\n", r.InterestSummary.Bands[string(InterestLow)]))
	b.WriteString(fmt.Sprintf("| `moderate` | 6%% – 15%% (elevated) | %d | Standard enabling modules with modest dependencies |\n", r.InterestSummary.Bands[string(InterestModerate)]))
	b.WriteString(fmt.Sprintf("| `high` | 16%% – 25%% (accelerating) | %d | Core or high-blast-radius leaves with maturity gaps |\n", r.InterestSummary.Bands[string(InterestHigh)]))
	b.WriteString(fmt.Sprintf("| `critical` | > 25%% (compounding) | %d | Critical paths or untested code in production paths |\n\n", r.InterestSummary.Bands[string(InterestCritical)]))

	b.WriteString("## Lane Health & Cross-Indexing\n\n")
	b.WriteString(fmt.Sprintf("- **Healthy lanes:** `%d` (score >= 0.80, passing tests, integrated, clean)\n", r.HealthSummary.HealthyCount))
	b.WriteString(fmt.Sprintf("- **Degraded lanes:** `%d` (missing tests, excess comment bloat, or disconnected)\n", r.HealthSummary.DegradedCount))
	b.WriteString(fmt.Sprintf("- **Critical lanes:** `%d` (compounding interest > 25%% or untested core path)\n", r.HealthSummary.CriticalCount))
	b.WriteString(fmt.Sprintf("- **Fleet health score:** `%.2f / 1.00` average\n\n", r.HealthSummary.AverageScore))

	b.WriteString("## Debt Hotspots (Worst-First)\n\n")
	b.WriteString("| Lane | Health | Criticality | Maturity Curve | Gap | Weight | Principal | Rate (Band) | Carrying Cost | Total Debt | Next Action |\n")
	b.WriteString("|---|---|---|---|---:|---:|---:|---|---:|---:|---|\n")
	for _, h := range r.Hotspots {
		b.WriteString(fmt.Sprintf("| `%s` | `%s` | %s | %.1f / %.1f (`%s`) | %.1f | %.1f | %.1f | %.1f%% (`%s`) | %.1f | **%.1f** | %s |\n",
			h.Lane,
			h.Health.Status,
			h.Criticality,
			h.Maturity,
			h.TargetMaturity,
			h.MaturityRung,
			h.MaturityGap,
			h.Weight,
			h.DebtPrincipal,
			h.Interest.Rate*100,
			h.Interest.Band,
			h.CarryingCost,
			h.TotalDebt,
			h.NextAction,
		))
	}

	b.WriteString("\n---\n*Auto-generated by `fak score debt-lanes --markdown`. Grounded in disk facts and verified evidence.*\n")
	return b.String()
}

// Compare compares the current report against a baseline report.
func Compare(curr, base Report) string {
	var b strings.Builder
	b.WriteString("=== DEBT LANES COMPARISON ===\n")
	b.WriteString(fmt.Sprintf("Production Grade: %s (%.1f%%) vs baseline %s (%.1f%%) [%+.1f%%]\n",
		curr.ProductionGrade.GradeLetter, curr.ProductionGrade.GradePercent,
		base.ProductionGrade.GradeLetter, base.ProductionGrade.GradePercent,
		curr.ProductionGrade.GradePercent-base.ProductionGrade.GradePercent,
	))
	currDen := curr.ProductionGrade.DenominatorPoints
	baseDen := base.ProductionGrade.DenominatorPoints
	b.WriteString(fmt.Sprintf("Denominator:      %.1f pts vs baseline %.1f pts [%+.1f pts]\n",
		currDen, baseDen, currDen-baseDen,
	))
	currDebt := toFloat(curr.Corpus["debt"])
	baseDebt := toFloat(base.Corpus["debt"])
	b.WriteString(fmt.Sprintf("Total Debt:       %.1f pts vs baseline %.1f pts [%+.1f pts]\n",
		currDebt, baseDebt, currDebt-baseDebt,
	))
	currWIP := curr.ProductionGrade.WIPUnits
	baseWIP := base.ProductionGrade.WIPUnits
	b.WriteString(fmt.Sprintf("Active WIP Lanes: %d vs baseline %d [%+d]\n",
		currWIP, baseWIP, currWIP-baseWIP,
	))
	return b.String()
}

func truncProse(s string, limit int) string {
	if len(s) <= limit {
		return s
	}
	return s[:limit-3] + "..."
}

func toFloat(v any) float64 {
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	default:
		return 0.0
	}
}
