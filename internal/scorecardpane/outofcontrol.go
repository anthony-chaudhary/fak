package scorecardpane

import (
	"fmt"
	"sort"
	"strings"
)

// Classification tokens for out-of-control metric movements.
const (
	ClassRateOfChangeSurge  = "RATE_OF_CHANGE_SURGE"
	ClassStepVelocityBreach = "STEP_VELOCITY_BREACH"
	ClassCeilingBreach      = "CEILING_BREACH"
	ClassGradeCollapse      = "GRADE_COLLAPSE"
)

// Severity levels for out-of-control debt conditions.
const (
	SeverityCritical = "CRITICAL"
	SeverityHigh     = "HIGH"
	SeverityWarn     = "WARN"
)

// Portfolio status states.
const (
	StatusStable       = "STABLE"
	StatusElevated     = "ELEVATED"
	StatusOutOfControl = "OUT_OF_CONTROL"
	StatusUnpinned     = "UNPINNED"
)

// OutOfControlMetric classifies one specific scorecard metric whose debt is moving
// out of expected bounds (rapid rate of change, absolute step surge, ceiling breach,
// or grade collapse).
type OutOfControlMetric struct {
	Key             string   `json:"key"`
	Label           string   `json:"label"`
	From            int      `json:"from"`
	To              int      `json:"to"`
	Delta           int      `json:"delta"`
	RateOfChange    float64  `json:"rate_of_change"`
	Severity        string   `json:"severity"`
	Classifications []string `json:"classifications"`
	Reasons         []string `json:"reasons"`
}

// PortfolioOutOfControl evaluates whether repository-wide debt dynamics are moving
// out of expected bounds (overall rate of change, grade-debt surge, or contagion sprawl).
type PortfolioOutOfControl struct {
	Status            string               `json:"status"`
	IsOutOfControl    bool                 `json:"is_out_of_control"`
	TotalRateOfChange float64              `json:"total_rate_of_change"`
	GradeRateOfChange float64              `json:"grade_rate_of_change"`
	ContagionRate     float64              `json:"contagion_rate"`
	WorsenedCount     int                  `json:"worsened_count"`
	MeasuredCount     int                  `json:"measured_count"`
	SeverityBreaches  int                  `json:"severity_breaches"`
	Reasons           []string             `json:"reasons"`
	Metrics           []OutOfControlMetric `json:"metrics,omitempty"`
}

// OutOfControlBounds defines the expected bounds and rate-of-change thresholds
// for both per-metric and portfolio-wide debt dynamics.
type OutOfControlBounds struct {
	// MetricRateOfChangeThreshold is the relative growth threshold (e.g. 0.50 = +50%).
	MetricRateOfChangeThreshold float64 `json:"metric_rate_of_change_threshold"`
	// MetricMinDeltaForRate is the minimum absolute delta required before flagging relative rate.
	MetricMinDeltaForRate int `json:"metric_min_delta_for_rate"`
	// MetricStepSurgeThreshold is the absolute delta in a single measurement considered a velocity surge.
	MetricStepSurgeThreshold int `json:"metric_step_surge_threshold"`
	// MetricGradeDropThreshold is the minimum tier drop (e.g. 2 tiers, A->C) considered a grade collapse.
	MetricGradeDropThreshold int `json:"metric_grade_drop_threshold"`
	// MetricCeilings are optional hard debt ceilings for specific metric keys.
	MetricCeilings map[string]int `json:"metric_ceilings,omitempty"`

	// PortfolioTotalRateThreshold is the maximum allowable repo-wide debt growth rate (e.g. 0.10 = +10%).
	PortfolioTotalRateThreshold float64 `json:"portfolio_total_rate_threshold"`
	// PortfolioGradeRateThreshold is the maximum allowable repo-wide grade debt growth rate (e.g. 0.25 = +25%).
	PortfolioGradeRateThreshold float64 `json:"portfolio_grade_rate_threshold"`
	// ContagionThreshold is the fraction of measured metrics worsening simultaneously (e.g. 0.25 = 25%).
	ContagionThreshold float64 `json:"contagion_threshold"`
	// CriticalMetricCountThreshold is the count of CRITICAL metrics that triggers repo-wide OUT_OF_CONTROL.
	CriticalMetricCountThreshold int `json:"critical_metric_count_threshold"`
}

// DefaultOutOfControlBounds returns calibrated defaults for debt tracking.
func DefaultOutOfControlBounds() OutOfControlBounds {
	return OutOfControlBounds{
		MetricRateOfChangeThreshold: 0.50, // +50% surge
		MetricMinDeltaForRate:       5,    // at least +5 items to avoid false-alarm on 1->2
		MetricStepSurgeThreshold:    15,   // +15 items in one step is an absolute surge
		MetricGradeDropThreshold:    2,    // slipping 2+ tiers (e.g. A->C or B->D)
		MetricCeilings: map[string]int{
			"readme": 0, // front door freshness should never have debt
		},
		PortfolioTotalRateThreshold:  0.10, // +10% repo-wide debt surge
		PortfolioGradeRateThreshold:  0.25, // +25% severity surge
		ContagionThreshold:           0.25, // >=25% of scorecards worsening at once
		CriticalMetricCountThreshold: 2,    // 2+ critical metric runaways triggers repo runaway
	}
}

var gradeRanks = map[string]int{
	"A": 0,
	"B": 1,
	"C": 2,
	"D": 3,
	"F": 4,
}

func gradeTierDistance(from, to string) int {
	fromR, ok1 := gradeRanks[from]
	toR, ok2 := gradeRanks[to]
	if !ok1 || !ok2 {
		return 0
	}
	diff := toR - fromR
	if diff < 0 {
		return 0
	}
	return diff
}

// AssessOutOfControl evaluates both per-metric and repository-wide debt dynamics against bounds.
func AssessOutOfControl(metrics []Metric, baseline *Baseline, totalDebt, gradeDebtTotal int, bounds OutOfControlBounds) PortfolioOutOfControl {
	if baseline == nil || (baseline.TotalDebt == 0 && len(baseline.Metrics) == 0) {
		return PortfolioOutOfControl{
			Status:         StatusUnpinned,
			IsOutOfControl: false,
			Reasons:        []string{"unpinned (no baseline available to measure rate of change)"},
		}
	}

	baseMetrics := baseline.Metrics
	baseVersions := baseline.DetectorVersions
	baseWeights := baseline.GradeWeights

	var outMetrics []OutOfControlMetric
	measuredCount := 0
	worsenedCount := 0
	criticalCount := 0

	for _, m := range metrics {
		if m.Debt == nil {
			continue
		}
		measuredCount++

		prior, ok := baseMetrics[m.Key]
		if !ok {
			continue
		}

		priorVersion := baseVersions[m.Key]
		if priorVersion != "" && m.DetectorVersion != "" && priorVersion != m.DetectorVersion {
			// Version changes cannot be compared for rate of change
			continue
		}

		delta := *m.Debt - prior
		if delta <= 0 {
			// Flat or improved
			continue
		}
		worsenedCount++

		outMetric, ok, isCritical := evaluateOutOfControlMetric(m, prior, delta, baseWeights, bounds)
		if !ok {
			continue
		}
		if isCritical {
			criticalCount++
		}
		outMetrics = append(outMetrics, outMetric)
	}

	// Sort out-of-control metrics worst-first by severity then delta
	sort.SliceStable(outMetrics, func(i, j int) bool {
		if outMetrics[i].Severity != outMetrics[j].Severity {
			return outMetrics[i].Severity == SeverityCritical
		}
		return outMetrics[i].Delta > outMetrics[j].Delta
	})

	// Repo-wide rate of change & contagion
	totalDelta := totalDebt - baseline.TotalDebt
	totalRate := 0.0
	if baseline.TotalDebt > 0 {
		totalRate = float64(totalDelta) / float64(baseline.TotalDebt)
	} else if totalDelta > 0 {
		totalRate = 1.0
	}

	gradeDelta := gradeDebtTotal - baseline.GradeDebt
	gradeRate := 0.0
	if baseline.GradeDebt > 0 {
		gradeRate = float64(gradeDelta) / float64(baseline.GradeDebt)
	} else if gradeDelta > 0 {
		gradeRate = 1.0
	}

	contagionRate := 0.0
	if measuredCount > 0 {
		contagionRate = float64(worsenedCount) / float64(measuredCount)
	}

	totalSurge := totalRate >= bounds.PortfolioTotalRateThreshold && totalDelta >= 5
	gradeSurge := gradeRate >= bounds.PortfolioGradeRateThreshold && gradeDelta >= 3
	contagionSurge := contagionRate >= bounds.ContagionThreshold && worsenedCount >= 3
	criticalSurge := criticalCount >= bounds.CriticalMetricCountThreshold

	var repoReasons []string
	if totalSurge {
		repoReasons = append(repoReasons, fmt.Sprintf("portfolio total debt surged by +%.1f%% (%+d units) exceeding rate-of-change bound (+%.1f%%)",
			totalRate*100, totalDelta, bounds.PortfolioTotalRateThreshold*100))
	}
	if gradeSurge {
		repoReasons = append(repoReasons, fmt.Sprintf("portfolio grade severity surged by +%.1f%% (%+d severity points) exceeding bound (+%.1f%%)",
			gradeRate*100, gradeDelta, bounds.PortfolioGradeRateThreshold*100))
	}
	if contagionSurge {
		repoReasons = append(repoReasons, fmt.Sprintf("debt contagion breach: %d/%d (%.1f%%) scorecards worsening simultaneously exceeding bound (%.1f%%)",
			worsenedCount, measuredCount, contagionRate*100, bounds.ContagionThreshold*100))
	}
	if criticalSurge {
		var critLabels []string
		for _, om := range outMetrics {
			if om.Severity == SeverityCritical {
				critLabels = append(critLabels, om.Label)
			}
		}
		repoReasons = append(repoReasons, fmt.Sprintf("%d metrics in critical runaway (%s)",
			criticalCount, strings.Join(critLabels, ", ")))
	}

	status := StatusStable
	isOutOfControl := false

	if totalSurge || gradeSurge || contagionSurge || criticalSurge {
		status = StatusOutOfControl
		isOutOfControl = true
	} else if len(outMetrics) > 0 || (totalRate > bounds.PortfolioTotalRateThreshold/2 && totalDelta > 0) || (contagionRate > bounds.ContagionThreshold/2 && worsenedCount > 1) {
		status = StatusElevated
		isOutOfControl = false
		if len(outMetrics) > 0 {
			repoReasons = append(repoReasons, fmt.Sprintf("%d metric(s) exceeded expected individual rate/velocity bounds", len(outMetrics)))
		}
	}

	return PortfolioOutOfControl{
		Status:            status,
		IsOutOfControl:    isOutOfControl,
		TotalRateOfChange: totalRate,
		GradeRateOfChange: gradeRate,
		ContagionRate:     contagionRate,
		WorsenedCount:     worsenedCount,
		MeasuredCount:     measuredCount,
		SeverityBreaches:  len(outMetrics),
		Reasons:           repoReasons,
		Metrics:           outMetrics,
	}
}

func evaluateOutOfControlMetric(m Metric, prior, delta int, baseWeights map[string]int, bounds OutOfControlBounds) (OutOfControlMetric, bool, bool) {
	var classifications []string
	var reasons []string

	// 1. Rate of change
	rateOfChange := 0.0
	if prior > 0 {
		rateOfChange = float64(delta) / float64(prior)
		if rateOfChange >= bounds.MetricRateOfChangeThreshold && delta >= bounds.MetricMinDeltaForRate {
			classifications = append(classifications, ClassRateOfChangeSurge)
			reasons = append(reasons, fmt.Sprintf("rate of change +%.1f%% exceeds threshold (+%.1f%%) with delta +%d",
				rateOfChange*100, bounds.MetricRateOfChangeThreshold*100, delta))
		}
	} else if delta >= bounds.MetricMinDeltaForRate {
		rateOfChange = 1.0
		classifications = append(classifications, ClassRateOfChangeSurge)
		reasons = append(reasons, fmt.Sprintf("baseline exploded from 0 to %d (+%d)", *m.Debt, delta))
	}

	// 2. Step velocity breach (absolute surge)
	if delta >= bounds.MetricStepSurgeThreshold {
		classifications = append(classifications, ClassStepVelocityBreach)
		reasons = append(reasons, fmt.Sprintf("debt surge +%d exceeds velocity threshold (%d)",
			delta, bounds.MetricStepSurgeThreshold))
	}

	// 3. Envelope ceiling breach
	if bounds.MetricCeilings != nil {
		if ceiling, hasCeiling := bounds.MetricCeilings[m.Key]; hasCeiling && *m.Debt > ceiling {
			classifications = append(classifications, ClassCeilingBreach)
			reasons = append(reasons, fmt.Sprintf("debt %d exceeds envelope ceiling (%d)",
				*m.Debt, ceiling))
		}
	}

	// 4. Grade collapse (2+ tier slip)
	if baseWeights != nil {
		if priorW, hasW := baseWeights[m.Key]; hasW && m.GradeWeight != nil && *m.GradeWeight > priorW {
			fromGrade := weightLetter(priorW)
			toGrade := m.EffGrade
			if toGrade == "" && m.Grade != nil {
				toGrade = *m.Grade
			}
			drop := gradeTierDistance(fromGrade, toGrade)
			if drop >= bounds.MetricGradeDropThreshold {
				classifications = append(classifications, ClassGradeCollapse)
				reasons = append(reasons, fmt.Sprintf("grade collapsed from %s to %s (slipped %d tiers)",
					fromGrade, toGrade, drop))
			}
		}
	}

	if len(classifications) == 0 {
		return OutOfControlMetric{}, false, false
	}

	// Determine severity
	severity := SeverityHigh
	isCritical := false
	for _, c := range classifications {
		if c == ClassGradeCollapse {
			if priorW, hasW := baseWeights[m.Key]; hasW {
				fromGrade := weightLetter(priorW)
				toGrade := m.EffGrade
				if toGrade == "" && m.Grade != nil {
					toGrade = *m.Grade
				}
				if gradeTierDistance(fromGrade, toGrade) >= 3 {
					isCritical = true
				}
			}
		}
	}
	if rateOfChange >= 2.0 && delta >= 10 {
		isCritical = true
	}
	if delta >= bounds.MetricStepSurgeThreshold*2 {
		isCritical = true
	}
	if bounds.MetricCeilings != nil {
		if ceiling, hasCeiling := bounds.MetricCeilings[m.Key]; hasCeiling {
			if ceiling == 0 && *m.Debt >= bounds.MetricMinDeltaForRate {
				isCritical = true
			} else if ceiling > 0 && *m.Debt >= int(float64(ceiling)*1.5) {
				isCritical = true
			}
		}
	}

	if isCritical {
		severity = SeverityCritical
	}

	return OutOfControlMetric{
		Key:             m.Key,
		Label:           m.Label,
		From:            prior,
		To:              *m.Debt,
		Delta:           delta,
		RateOfChange:    rateOfChange,
		Severity:        severity,
		Classifications: classifications,
		Reasons:         reasons,
	}, true, isCritical
}
