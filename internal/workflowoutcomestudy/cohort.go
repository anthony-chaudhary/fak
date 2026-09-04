package workflowoutcomestudy

import (
	"fmt"
	"math"
	"sort"
	"time"
)

const PRCohortSchema = "fak-wip-pr-cohort/1"

type CohortDecision string

const (
	CohortDecisionPromoteNarrowClass CohortDecision = "PROMOTE_NARROW_CLASS"
	CohortDecisionHoldDefault        CohortDecision = "HOLD_DEFAULT"
	CohortDecisionRejectPRLane       CohortDecision = "REJECT_PR_LANE"
)

type CohortMetric struct {
	Name                string  `json:"name"`
	Unit                string  `json:"unit"`
	DetachedWorkerValue float64 `json:"detached_worker_value"`
	PRLaneValue         float64 `json:"pr_lane_value"`
	Threshold           float64 `json:"threshold"`
	Passed              bool    `json:"passed"`
}

type ArmMetrics struct {
	IssueCount          int     `json:"issue_count"`
	TimeToProtection    float64 `json:"time_to_protection"`
	CollisionReworkRate float64 `json:"collision_rework_rate"`
	ReviewLatency       float64 `json:"review_latency"`
	GreenToLandLatency  float64 `json:"green_to_land_latency"`
	OperatorTouches     float64 `json:"operator_touches"`
	AbandonmentRecovery float64 `json:"abandonment_recovery"`
	DefectRate          float64 `json:"defect_rate"`
}

type ArmsReport struct {
	DetachedWorker ArmMetrics `json:"detached_worker"`
	PRLane         ArmMetrics `json:"pr_lane"`
}

type PRCohortReport struct {
	Schema       string         `json:"schema"`
	Timestamp    string         `json:"timestamp"`
	CohortSize   int            `json:"cohort_size"`
	IssueClasses []string       `json:"issue_classes"`
	Metrics      []CohortMetric `json:"metrics"`
	Decision     CohortDecision `json:"decision"`
	Rationale    string         `json:"rationale"`
	Arms         ArmsReport     `json:"arms"`
}

type IssueRecord struct {
	IssueID             string  `json:"issue_id"`
	Class               string  `json:"class"`
	TimeToProtection    float64 `json:"time_to_protection"`
	CollisionReworkRate float64 `json:"collision_rework_rate"`
	ReviewLatency       float64 `json:"review_latency"`
	GreenToLandLatency  float64 `json:"green_to_land_latency"`
	OperatorTouches     float64 `json:"operator_touches"`
	AbandonmentRecovery float64 `json:"abandonment_recovery"`
	DefectRate          float64 `json:"defect_rate"`
}

type ArmData struct {
	Timestamp      string        `json:"timestamp,omitempty"`
	DetachedWorker []IssueRecord `json:"detached_worker"`
	PRLane         []IssueRecord `json:"pr_lane"`
}

func computeArmMetrics(records []IssueRecord) ArmMetrics {
	n := len(records)
	if n == 0 {
		return ArmMetrics{}
	}
	var (
		sumTTP float64
		sumCRR float64
		sumRL  float64
		sumGLL float64
		sumOT  float64
		sumAR  float64
		sumDR  float64
	)
	for _, r := range records {
		sumTTP += r.TimeToProtection
		sumCRR += r.CollisionReworkRate
		sumRL += r.ReviewLatency
		sumGLL += r.GreenToLandLatency
		sumOT += r.OperatorTouches
		sumAR += r.AbandonmentRecovery
		sumDR += r.DefectRate
	}
	fn := float64(n)
	return ArmMetrics{
		IssueCount:          n,
		TimeToProtection:    round3(sumTTP / fn),
		CollisionReworkRate: round3(sumCRR / fn),
		ReviewLatency:       round3(sumRL / fn),
		GreenToLandLatency:  round3(sumGLL / fn),
		OperatorTouches:     round3(sumOT / fn),
		AbandonmentRecovery: round3(sumAR / fn),
		DefectRate:          round3(sumDR / fn),
	}
}

func round3(v float64) float64 {
	return math.Round(v*1000) / 1000
}

func extractIssueClasses(dw, pr []IssueRecord) []string {
	seen := make(map[string]bool)
	for _, r := range dw {
		if r.Class != "" {
			seen[r.Class] = true
		}
	}
	for _, r := range pr {
		if r.Class != "" {
			seen[r.Class] = true
		}
	}
	classes := make([]string, 0, len(seen))
	for c := range seen {
		classes = append(classes, c)
	}
	sort.Strings(classes)
	return classes
}

// EvaluatePRIsolationCohort executes the matched PR isolation cohort study evaluation.
// It measures WIP safety (time-to-protection, collision_rework_rate, abandonment_recovery)
// and workflow outcome metrics (review_latency, green_to_land_latency, operator_touches).
// If the PR lane introduces higher WIP exposure or excessive landing latency without
// offsetting defect reduction, it yields HOLD_DEFAULT or REJECT_PR_LANE.
func EvaluatePRIsolationCohort(raw ArmData) *PRCohortReport {
	dwMetrics := computeArmMetrics(raw.DetachedWorker)
	prMetrics := computeArmMetrics(raw.PRLane)

	classes := extractIssueClasses(raw.DetachedWorker, raw.PRLane)

	ts := raw.Timestamp
	if ts == "" {
		ts = time.Now().UTC().Format(time.RFC3339)
	}

	cohortSize := len(raw.DetachedWorker)
	if len(raw.PRLane) < cohortSize {
		cohortSize = len(raw.PRLane)
	}

	metrics := []CohortMetric{
		{
			Name:                "time-to-protection",
			Unit:                "seconds",
			DetachedWorkerValue: dwMetrics.TimeToProtection,
			PRLaneValue:         prMetrics.TimeToProtection,
			Threshold:           30.0,
			Passed:              prMetrics.TimeToProtection <= 30.0,
		},
		{
			Name:                "collision_rework_rate",
			Unit:                "rate",
			DetachedWorkerValue: dwMetrics.CollisionReworkRate,
			PRLaneValue:         prMetrics.CollisionReworkRate,
			Threshold:           0.05,
			Passed:              prMetrics.CollisionReworkRate <= 0.05,
		},
		{
			Name:                "review_latency",
			Unit:                "seconds",
			DetachedWorkerValue: dwMetrics.ReviewLatency,
			PRLaneValue:         prMetrics.ReviewLatency,
			Threshold:           300.0,
			Passed:              prMetrics.ReviewLatency <= 300.0,
		},
		{
			Name:                "green_to_land_latency",
			Unit:                "seconds",
			DetachedWorkerValue: dwMetrics.GreenToLandLatency,
			PRLaneValue:         prMetrics.GreenToLandLatency,
			Threshold:           120.0,
			Passed:              prMetrics.GreenToLandLatency <= 120.0,
		},
		{
			Name:                "operator_touches",
			Unit:                "touches",
			DetachedWorkerValue: dwMetrics.OperatorTouches,
			PRLaneValue:         prMetrics.OperatorTouches,
			Threshold:           1.5,
			Passed:              prMetrics.OperatorTouches <= 1.5,
		},
		{
			Name:                "abandonment_recovery",
			Unit:                "rate",
			DetachedWorkerValue: dwMetrics.AbandonmentRecovery,
			PRLaneValue:         prMetrics.AbandonmentRecovery,
			Threshold:           0.85,
			Passed:              prMetrics.AbandonmentRecovery >= 0.85,
		},
	}

	report := &PRCohortReport{
		Schema:       PRCohortSchema,
		Timestamp:    ts,
		CohortSize:   cohortSize,
		IssueClasses: classes,
		Metrics:      metrics,
		Arms: ArmsReport{
			DetachedWorker: dwMetrics,
			PRLane:         prMetrics,
		},
	}

	// Gate 1: Insufficient cohort size (requires >= 10 per arm).
	if len(raw.DetachedWorker) < 10 || len(raw.PRLane) < 10 {
		report.Decision = CohortDecisionHoldDefault
		report.Rationale = fmt.Sprintf("Cohort size insufficient (%d detached, %d pr_lane; minimum 10 per arm required); holding default trunk detached worker.", len(raw.DetachedWorker), len(raw.PRLane))
		return report
	}

	// Gate 2: Insufficient issue class diversity (requires >= 2 classes).
	if len(classes) < 2 {
		report.Decision = CohortDecisionHoldDefault
		report.Rationale = fmt.Sprintf("Issue class diversity insufficient (%d classes present; minimum 2 required); holding default trunk detached worker.", len(classes))
		return report
	}

	// Gate 3: Deterministic evaluation logic.
	higherWIPExposure := prMetrics.TimeToProtection > 2.0*dwMetrics.TimeToProtection ||
		prMetrics.TimeToProtection > 60.0 ||
		prMetrics.AbandonmentRecovery < (dwMetrics.AbandonmentRecovery-0.10)

	excessiveLandingLatency := prMetrics.GreenToLandLatency > 3.0*dwMetrics.GreenToLandLatency ||
		prMetrics.GreenToLandLatency > 300.0 ||
		prMetrics.ReviewLatency > 3.0*dwMetrics.ReviewLatency

	defectReduction := round3(dwMetrics.DefectRate - prMetrics.DefectRate)
	hasOffsettingDefectReduction := defectReduction >= 0.05

	failedCount := 0
	for _, m := range metrics {
		if !m.Passed {
			failedCount++
		}
	}

	if (higherWIPExposure || excessiveLandingLatency) && !hasOffsettingDefectReduction {
		if failedCount >= 4 || prMetrics.GreenToLandLatency >= 5.0*dwMetrics.GreenToLandLatency || prMetrics.TimeToProtection >= 3.0*dwMetrics.TimeToProtection {
			report.Decision = CohortDecisionRejectPRLane
			report.Rationale = fmt.Sprintf("PR isolation lane introduces higher WIP exposure (time-to-protection %.1fs vs %.1fs) and excessive landing latency (%.1fs vs %.1fs) without offsetting defect reduction (defect rate %.3f vs %.3f, reduction %.3f); rejected in favor of trunk detached worker isolation.",
				prMetrics.TimeToProtection, dwMetrics.TimeToProtection,
				prMetrics.GreenToLandLatency, dwMetrics.GreenToLandLatency,
				prMetrics.DefectRate, dwMetrics.DefectRate, defectReduction)
		} else {
			report.Decision = CohortDecisionHoldDefault
			report.Rationale = fmt.Sprintf("PR isolation lane exhibits elevated latency or WIP exposure without offsetting defect reduction (failed %d of %d metrics); holding default trunk detached worker.",
				failedCount, len(metrics))
		}
	} else if failedCount == 0 || hasOffsettingDefectReduction {
		report.Decision = CohortDecisionPromoteNarrowClass
		report.Rationale = fmt.Sprintf("PR isolation lane demonstrates acceptable WIP safety and landing latency with meaningful defect reduction (%.3f) across matched issue classes %v.",
			defectReduction, classes)
	} else {
		report.Decision = CohortDecisionHoldDefault
		report.Rationale = fmt.Sprintf("PR isolation lane outcomes are mixed (failed %d metrics, defect reduction %.3f); holding default trunk detached worker.",
			failedCount, defectReduction)
	}

	return report
}

// DefaultCohortDataset returns a representative 12-issue matched cohort study dataset
// comparing DetachedWorker against PRLane across 3 distinct issue classes.
func DefaultCohortDataset() ArmData {
	issues := []struct {
		id    string
		class string
	}{
		{"#7301", "bugfix"},
		{"#7302", "bugfix"},
		{"#7303", "bugfix"},
		{"#7304", "bugfix"},
		{"#7311", "feature"},
		{"#7312", "feature"},
		{"#7313", "feature"},
		{"#7314", "feature"},
		{"#7321", "refactor"},
		{"#7322", "refactor"},
		{"#7323", "refactor"},
		{"#7332", "refactor"},
	}

	dwRecords := make([]IssueRecord, len(issues))
	prRecords := make([]IssueRecord, len(issues))

	// Detached worker characteristics: fast local WIP checkpointing, lane leases
	// preventing tree collision, automated referee verification, immediate trunk landing.
	dwTTP := []float64{10.2, 12.5, 9.8, 14.1, 11.3, 13.0, 10.7, 12.2, 11.0, 13.5, 12.0, 11.8}
	dwCRR := []float64{0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.02, 0.01, 0.01, 0.02, 0.01}
	dwRL := []float64{45.0, 60.0, 52.0, 75.0, 65.0, 80.0, 55.0, 70.0, 50.0, 62.0, 58.0, 64.0}
	dwGLL := []float64{35.0, 42.0, 38.0, 48.0, 40.0, 45.0, 37.0, 44.0, 39.0, 43.0, 41.0, 40.0}
	dwOT := []float64{1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}
	dwAR := []float64{0.95, 0.98, 0.96, 0.94, 0.97, 0.95, 0.96, 0.98, 0.95, 0.96, 0.97, 0.95}
	dwDR := []float64{0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02}

	// PR lane characteristics: slow branch push + remote PR creation, branch divergence
	// causing merge conflicts, PR review queue delay, merge queue / rebase latency, manual operator touches.
	prTTP := []float64{180.0, 195.0, 175.0, 210.0, 185.0, 200.0, 170.0, 190.0, 178.0, 192.0, 184.0, 182.0}
	prCRR := []float64{0.14, 0.16, 0.13, 0.17, 0.15, 0.18, 0.12, 0.16, 0.14, 0.15, 0.13, 0.14}
	prRL := []float64{1350.0, 1500.0, 1420.0, 1680.0, 1450.0, 1600.0, 1380.0, 1520.0, 1390.0, 1480.0, 1410.0, 1440.0}
	prGLL := []float64{1750.0, 1920.0, 1800.0, 2100.0, 1850.0, 2050.0, 1780.0, 1900.0, 1820.0, 1950.0, 1840.0, 1860.0}
	prOT := []float64{3.5, 4.0, 3.5, 4.5, 4.0, 4.5, 3.5, 4.0, 3.5, 4.0, 3.5, 3.8}
	prAR := []float64{0.48, 0.52, 0.46, 0.44, 0.50, 0.45, 0.49, 0.51, 0.47, 0.48, 0.50, 0.48}
	prDR := []float64{0.02, 0.02, 0.03, 0.02, 0.02, 0.03, 0.02, 0.02, 0.02, 0.03, 0.02, 0.02}

	for i, it := range issues {
		dwRecords[i] = IssueRecord{
			IssueID:             it.id,
			Class:               it.class,
			TimeToProtection:    dwTTP[i],
			CollisionReworkRate: dwCRR[i],
			ReviewLatency:       dwRL[i],
			GreenToLandLatency:  dwGLL[i],
			OperatorTouches:     dwOT[i],
			AbandonmentRecovery: dwAR[i],
			DefectRate:          dwDR[i],
		}
		prRecords[i] = IssueRecord{
			IssueID:             it.id,
			Class:               it.class,
			TimeToProtection:    prTTP[i],
			CollisionReworkRate: prCRR[i],
			ReviewLatency:       prRL[i],
			GreenToLandLatency:  prGLL[i],
			OperatorTouches:     prOT[i],
			AbandonmentRecovery: prAR[i],
			DefectRate:          prDR[i],
		}
	}

	return ArmData{
		Timestamp:      "2026-09-04T00:00:00Z",
		DetachedWorker: dwRecords,
		PRLane:         prRecords,
	}
}
