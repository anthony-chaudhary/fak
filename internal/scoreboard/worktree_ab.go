package scoreboard

import (
	"fmt"
	"math"
	"strings"
)

const WorktreeABSchema = "fak-worktree-ab/1"

type DeliveryOutcome string

const (
	OutcomeAccepted   DeliveryOutcome = "ACCEPTED"
	OutcomeRejected   DeliveryOutcome = "REJECTED"
	OutcomeDuplicate  DeliveryOutcome = "DUPLICATE"
	OutcomeUnverified DeliveryOutcome = "UNVERIFIED"
)

type DeliveryLifecycleRecord struct {
	IssueID              int             `json:"issue_id"`
	Outcome              DeliveryOutcome `json:"outcome"`
	SetupDuration        float64         `json:"setup_duration"`
	ExecutionDuration    float64         `json:"execution_duration"`
	LandingDuration      float64         `json:"landing_duration"`
	VerificationDuration float64         `json:"verification_duration"`
	TotalElapsed         float64         `json:"total_elapsed"`
	Spend                float64         `json:"spend"`
	SpendUnknown         bool            `json:"spend_unknown"`
}

type AcceptedDeliveryRecord = DeliveryLifecycleRecord

type AcceptedDeliveryAccounting struct {
	TotalDeliveries        int     `json:"total_deliveries"`
	AcceptedDeliveries     int     `json:"accepted_deliveries"`
	RejectedDeliveries     int     `json:"rejected_deliveries"`
	DuplicateDeliveries    int     `json:"duplicate_deliveries"`
	UnverifiedDeliveries   int     `json:"unverified_deliveries"`
	TotalElapsedSeconds    float64 `json:"total_elapsed_seconds"`
	AcceptedPerElapsedHour float64 `json:"accepted_per_elapsed_hour"`
	Verified               bool    `json:"verified"`
	Spend                  float64 `json:"spend"`
	SpendUnknown           bool    `json:"spend_unknown"`
	Status                 string  `json:"status"`
}

func normalizeDeliveryOutcome(o DeliveryOutcome) DeliveryOutcome {
	switch strings.ToUpper(strings.TrimSpace(string(o))) {
	case "ACCEPTED":
		return OutcomeAccepted
	case "REJECTED":
		return OutcomeRejected
	case "DUPLICATE":
		return OutcomeDuplicate
	case "UNVERIFIED":
		return OutcomeUnverified
	default:
		return OutcomeUnverified
	}
}

// AccountAcceptedDeliveries folds lifecycle records into accepted-delivery accounting.
// It counts distinct verified landed deliveries (OutcomeAccepted), excludes rejected,
// duplicate, and unverified outcomes from the accepted count, derives throughput
// across the full elapsed window rather than summing overlapping phases, and marks
// incomplete/unverified when acceptance or boundary evidence is missing.
func AccountAcceptedDeliveries(records []DeliveryLifecycleRecord, totalWindowSeconds float64) AcceptedDeliveryAccounting {
	acc := AcceptedDeliveryAccounting{
		TotalDeliveries:     len(records),
		TotalElapsedSeconds: totalWindowSeconds,
	}

	seenAccepted := make(map[int]bool)
	var incompleteBoundary bool
	if totalWindowSeconds <= 0 || math.IsNaN(totalWindowSeconds) || math.IsInf(totalWindowSeconds, 0) {
		incompleteBoundary = true
	}

	for _, r := range records {
		if !math.IsNaN(r.Spend) && !math.IsInf(r.Spend, 0) {
			acc.Spend += r.Spend
		}
		if r.SpendUnknown || math.IsNaN(r.Spend) || math.IsInf(r.Spend, 0) {
			acc.SpendUnknown = true
		}

		if r.TotalElapsed < 0 || r.SetupDuration < 0 || r.ExecutionDuration < 0 || r.LandingDuration < 0 || r.VerificationDuration < 0 ||
			math.IsNaN(r.TotalElapsed) || math.IsNaN(r.SetupDuration) || math.IsNaN(r.ExecutionDuration) || math.IsNaN(r.LandingDuration) || math.IsNaN(r.VerificationDuration) {
			incompleteBoundary = true
		}

		norm := normalizeDeliveryOutcome(r.Outcome)
		switch norm {
		case OutcomeAccepted:
			if seenAccepted[r.IssueID] {
				acc.DuplicateDeliveries++
			} else {
				seenAccepted[r.IssueID] = true
				acc.AcceptedDeliveries++
			}
		case OutcomeRejected:
			acc.RejectedDeliveries++
		case OutcomeDuplicate:
			acc.DuplicateDeliveries++
		case OutcomeUnverified:
			acc.UnverifiedDeliveries++
		}
	}

	if totalWindowSeconds > 0 && !math.IsNaN(totalWindowSeconds) && !math.IsInf(totalWindowSeconds, 0) {
		acc.AcceptedPerElapsedHour = float64(acc.AcceptedDeliveries) * 3600.0 / totalWindowSeconds
	}

	if incompleteBoundary || len(records) == 0 || acc.AcceptedDeliveries == 0 {
		acc.Status = "INCOMPLETE"
		acc.Verified = false
	} else {
		acc.Status = "COMPLETE"
		acc.Verified = true
	}

	return acc
}

type WorktreeABArm struct {
	Name            string                     `json:"name"`
	Worktree        bool                       `json:"worktree"`
	Resolved        int                        `json:"resolved"`
	DurationSeconds float64                    `json:"duration_seconds"`
	PoisonIncidents int                        `json:"poison_incidents"`
	PeakConcurrency int                        `json:"peak_concurrency"`
	WaveID          string                     `json:"wave_id"`
	HostID          string                     `json:"host_id,omitempty"`
	DeliveryRecords []AcceptedDeliveryRecord   `json:"delivery_records,omitempty"`
	Accounting      AcceptedDeliveryAccounting `json:"accounting,omitempty"`
}

func (a WorktreeABArm) IssuesPerHour() float64 {
	if a.DurationSeconds <= 0 {
		return 0
	}
	if a.Accounting.Status != "" {
		return a.Accounting.AcceptedPerElapsedHour
	}
	return float64(a.Resolved) * 3600 / a.DurationSeconds
}

type WorktreeABReport struct {
	Schema             string                     `json:"schema"`
	Baseline           WorktreeABArm              `json:"baseline"`
	Isolated           WorktreeABArm              `json:"isolated"`
	Verdict            string                     `json:"verdict"`
	TrunkAccounting    AcceptedDeliveryAccounting `json:"trunk_accounting,omitempty"`
	WorktreeAccounting AcceptedDeliveryAccounting `json:"worktree_accounting,omitempty"`
}

// WorktreeABComparison is an alias for WorktreeABReport.
type WorktreeABComparison = WorktreeABReport

func FoldWorktreeAB(baseline, isolated WorktreeABArm) WorktreeABReport {
	baseline.Name, baseline.Worktree = "baseline", false
	isolated.Name, isolated.Worktree = "isolated", true

	if len(baseline.DeliveryRecords) > 0 && baseline.Accounting.Status == "" {
		baseline.Accounting = AccountAcceptedDeliveries(baseline.DeliveryRecords, baseline.DurationSeconds)
	}
	if baseline.Resolved == 0 && baseline.Accounting.AcceptedDeliveries > 0 {
		baseline.Resolved = baseline.Accounting.AcceptedDeliveries
	}

	if len(isolated.DeliveryRecords) > 0 && isolated.Accounting.Status == "" {
		isolated.Accounting = AccountAcceptedDeliveries(isolated.DeliveryRecords, isolated.DurationSeconds)
	}
	if isolated.Resolved == 0 && isolated.Accounting.AcceptedDeliveries > 0 {
		isolated.Resolved = isolated.Accounting.AcceptedDeliveries
	}

	verdict := "NOT_PROVEN"
	if baseline.DurationSeconds > 0 && isolated.DurationSeconds > 0 && isolated.PoisonIncidents == 0 {
		verdict = "ISOLATION_POISON_FREE"
	}
	return WorktreeABReport{
		Schema:             WorktreeABSchema,
		Baseline:           baseline,
		Isolated:           isolated,
		Verdict:            verdict,
		TrunkAccounting:    baseline.Accounting,
		WorktreeAccounting: isolated.Accounting,
	}
}

// CompareWorktreeAB folds trunk and worktree arms and returns the comparison report.
func CompareWorktreeAB(trunk, worktree WorktreeABArm) (WorktreeABComparison, error) {
	return FoldWorktreeAB(trunk, worktree), nil
}

func WorktreeABUpdate(r WorktreeABReport) Update {
	hasBothAccounting := len(r.Baseline.DeliveryRecords) > 0 && len(r.Isolated.DeliveryRecords) > 0 &&
		r.Baseline.Accounting.Status != "" && r.Isolated.Accounting.Status != ""

	line := func(a WorktreeABArm) string {
		if hasBothAccounting || (len(a.DeliveryRecords) > 0 && a.Accounting.Status != "") {
			return fmt.Sprintf("%s: %.2f issues/h (%d accepted, %s), %d poison, %.1fs, peak %d",
				a.Name, a.IssuesPerHour(), a.Accounting.AcceptedDeliveries, a.Accounting.Status, a.PoisonIncidents, a.DurationSeconds, a.PeakConcurrency)
		}
		return fmt.Sprintf("%s: %.2f issues/h, %d poison, %.1fs, peak %d",
			a.Name, a.IssuesPerHour(), a.PoisonIncidents, a.DurationSeconds, a.PeakConcurrency)
	}
	return Update{Title: "Dispatch worktree A/B", Verdict: r.Verdict, Lines: []string{line(r.Baseline), line(r.Isolated)}}
}

func WorktreeABEquivalentWave(a, b WorktreeABArm) bool {
	resA := a.Resolved
	if resA == 0 && a.Accounting.AcceptedDeliveries > 0 {
		resA = a.Accounting.AcceptedDeliveries
	}
	resB := b.Resolved
	if resB == 0 && b.Accounting.AcceptedDeliveries > 0 {
		resB = b.Accounting.AcceptedDeliveries
	}
	return a.WaveID != "" && a.WaveID == b.WaveID && resA == resB && resA > 0 &&
		(a.HostID == "" || b.HostID == "" || a.HostID == b.HostID) && !math.IsNaN(a.DurationSeconds) && !math.IsNaN(b.DurationSeconds)
}
