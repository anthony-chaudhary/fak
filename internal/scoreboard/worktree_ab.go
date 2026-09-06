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
	if !isFinitePositive(totalWindowSeconds) {
		incompleteBoundary = true
	}

	for _, r := range records {
		if r.IssueID <= 0 {
			incompleteBoundary = true
		}

		if !math.IsNaN(r.Spend) && !math.IsInf(r.Spend, 0) {
			if r.Spend >= 0 {
				acc.Spend += r.Spend
			} else {
				incompleteBoundary = true
			}
		}
		if r.SpendUnknown || math.IsNaN(r.Spend) || math.IsInf(r.Spend, 0) || r.Spend < 0 {
			acc.SpendUnknown = true
		}

		if r.TotalElapsed < 0 || r.SetupDuration < 0 || r.ExecutionDuration < 0 || r.LandingDuration < 0 || r.VerificationDuration < 0 ||
			math.IsNaN(r.TotalElapsed) || math.IsNaN(r.SetupDuration) || math.IsNaN(r.ExecutionDuration) || math.IsNaN(r.LandingDuration) || math.IsNaN(r.VerificationDuration) ||
			math.IsInf(r.TotalElapsed, 0) || math.IsInf(r.SetupDuration, 0) || math.IsInf(r.ExecutionDuration, 0) || math.IsInf(r.LandingDuration, 0) || math.IsInf(r.VerificationDuration, 0) {
			incompleteBoundary = true
		}

		if r.TotalElapsed > 0 &&
			isFiniteNonNegative(r.SetupDuration) &&
			isFiniteNonNegative(r.ExecutionDuration) &&
			isFiniteNonNegative(r.LandingDuration) &&
			isFiniteNonNegative(r.VerificationDuration) &&
			r.SetupDuration+r.ExecutionDuration+r.LandingDuration+r.VerificationDuration > r.TotalElapsed {
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

	if isFinitePositive(totalWindowSeconds) {
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
	Name             string                     `json:"name"`
	Worktree         bool                       `json:"worktree"`
	Resolved         int                        `json:"resolved"`
	DurationSeconds  float64                    `json:"duration_seconds"`
	PoisonIncidents  int                        `json:"poison_incidents"`
	PeakConcurrency  int                        `json:"peak_concurrency"`
	WaveID           string                     `json:"wave_id"`
	HostID           string                     `json:"host_id,omitempty"`
	DeliveryRecords  []AcceptedDeliveryRecord   `json:"delivery_records,omitempty"`
	LifecycleRecords []AcceptedDeliveryRecord   `json:"lifecycle_records,omitempty"`
	Accounting       AcceptedDeliveryAccounting `json:"accounting,omitempty"`
}

func (a WorktreeABArm) IssuesPerHour() float64 {
	if !isFinitePositive(a.DurationSeconds) {
		return 0
	}
	if a.Accounting.Status != "" {
		st := strings.TrimSpace(a.Accounting.Status)
		if strings.EqualFold(st, "INCOMPLETE") ||
			strings.EqualFold(st, "UNVERIFIED") ||
			(!a.Accounting.Verified && !strings.EqualFold(st, "VERIFIED")) {
			return 0
		}
		if !isFiniteNonNegative(a.Accounting.AcceptedPerElapsedHour) {
			return 0
		}
		return a.Accounting.AcceptedPerElapsedHour
	}
	if a.Resolved <= 0 {
		return 0
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

func isFinitePositive(v float64) bool {
	return v > 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

func isFiniteNonNegative(v float64) bool {
	return v >= 0 && !math.IsNaN(v) && !math.IsInf(v, 0)
}

// WorktreeABComparableArms reports whether two arms represent comparable,
// non-empty measured arms (valid wave matching, non-zero completed work,
// finite positive durations, and host comparability).
func WorktreeABComparableArms(a, b WorktreeABArm) bool {
	return WorktreeABEquivalentWave(a, b)
}

func FoldWorktreeAB(baseline, isolated WorktreeABArm) WorktreeABReport {
	baseline.Name, baseline.Worktree = "baseline", false
	isolated.Name, isolated.Worktree = "isolated", true

	if len(baseline.DeliveryRecords) == 0 && len(baseline.LifecycleRecords) > 0 {
		baseline.DeliveryRecords = baseline.LifecycleRecords
	}
	if len(isolated.DeliveryRecords) == 0 && len(isolated.LifecycleRecords) > 0 {
		isolated.DeliveryRecords = isolated.LifecycleRecords
	}
	if len(baseline.LifecycleRecords) == 0 && len(baseline.DeliveryRecords) > 0 {
		baseline.LifecycleRecords = baseline.DeliveryRecords
	}
	if len(isolated.LifecycleRecords) == 0 && len(isolated.DeliveryRecords) > 0 {
		isolated.LifecycleRecords = isolated.DeliveryRecords
	}

	if len(baseline.DeliveryRecords) > 0 && baseline.Accounting.Status == "" {
		baseline.Accounting = AccountAcceptedDeliveries(baseline.DeliveryRecords, baseline.DurationSeconds)
	}
	if baseline.Accounting.Status != "" {
		baseline.Resolved = baseline.Accounting.AcceptedDeliveries
	} else if baseline.Resolved == 0 && baseline.Accounting.AcceptedDeliveries > 0 {
		baseline.Resolved = baseline.Accounting.AcceptedDeliveries
	}

	if len(isolated.DeliveryRecords) > 0 && isolated.Accounting.Status == "" {
		isolated.Accounting = AccountAcceptedDeliveries(isolated.DeliveryRecords, isolated.DurationSeconds)
	}
	if isolated.Accounting.Status != "" {
		isolated.Resolved = isolated.Accounting.AcceptedDeliveries
	} else if isolated.Resolved == 0 && isolated.Accounting.AcceptedDeliveries > 0 {
		isolated.Resolved = isolated.Accounting.AcceptedDeliveries
	}

	verdict := "NOT_PROVEN"
	if isolated.PoisonIncidents == 0 && WorktreeABComparableArms(baseline, isolated) {
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
	line := func(a WorktreeABArm) string {
		if a.Accounting.Status != "" {
			return fmt.Sprintf("%s: %.2f issues/h (%d accepted, %s), %d poison, %.1fs, peak %d",
				a.Name, a.IssuesPerHour(), a.Accounting.AcceptedDeliveries, a.Accounting.Status, a.PoisonIncidents, a.DurationSeconds, a.PeakConcurrency)
		}
		return fmt.Sprintf("%s: %.2f issues/h, %d poison, %.1fs, peak %d",
			a.Name, a.IssuesPerHour(), a.PoisonIncidents, a.DurationSeconds, a.PeakConcurrency)
	}
	return Update{Title: "Dispatch worktree A/B", Verdict: r.Verdict, Lines: []string{line(r.Baseline), line(r.Isolated)}}
}

func WorktreeABEquivalentWave(a, b WorktreeABArm) bool {
	if !isFinitePositive(a.DurationSeconds) || !isFinitePositive(b.DurationSeconds) {
		return false
	}
	waveA := strings.TrimSpace(a.WaveID)
	waveB := strings.TrimSpace(b.WaveID)
	if waveA == "" || waveA != waveB {
		return false
	}
	hostA := strings.TrimSpace(a.HostID)
	hostB := strings.TrimSpace(b.HostID)
	if hostA != "" && hostB != "" && !strings.EqualFold(hostA, hostB) {
		return false
	}

	if len(a.DeliveryRecords) == 0 && len(a.LifecycleRecords) > 0 {
		a.DeliveryRecords = a.LifecycleRecords
	}
	if len(b.DeliveryRecords) == 0 && len(b.LifecycleRecords) > 0 {
		b.DeliveryRecords = b.LifecycleRecords
	}
	if len(a.LifecycleRecords) == 0 && len(a.DeliveryRecords) > 0 {
		a.LifecycleRecords = a.DeliveryRecords
	}
	if len(b.LifecycleRecords) == 0 && len(b.DeliveryRecords) > 0 {
		b.LifecycleRecords = b.DeliveryRecords
	}
	if len(a.DeliveryRecords) > 0 && a.Accounting.Status == "" {
		a.Accounting = AccountAcceptedDeliveries(a.DeliveryRecords, a.DurationSeconds)
	}
	if len(b.DeliveryRecords) > 0 && b.Accounting.Status == "" {
		b.Accounting = AccountAcceptedDeliveries(b.DeliveryRecords, b.DurationSeconds)
	}

	if (len(a.DeliveryRecords) > 0) != (len(b.DeliveryRecords) > 0) {
		return false
	}
	if (a.Accounting.Status != "") != (b.Accounting.Status != "") {
		return false
	}

	if len(a.DeliveryRecords) > 0 && (a.Accounting.Status != "COMPLETE" || !a.Accounting.Verified) {
		return false
	}
	if len(b.DeliveryRecords) > 0 && (b.Accounting.Status != "COMPLETE" || !b.Accounting.Verified) {
		return false
	}

	resA := a.Resolved
	if a.Accounting.Status != "" {
		if a.Accounting.Status == "INCOMPLETE" || !a.Accounting.Verified {
			return false
		}
		if !strings.EqualFold(a.Accounting.Status, b.Accounting.Status) {
			return false
		}
		if !isFiniteNonNegative(a.Accounting.TotalElapsedSeconds) ||
			!isFiniteNonNegative(a.Accounting.AcceptedPerElapsedHour) ||
			!isFiniteNonNegative(a.Accounting.Spend) ||
			a.Accounting.AcceptedDeliveries <= 0 ||
			a.Accounting.TotalDeliveries < 0 ||
			a.Accounting.RejectedDeliveries < 0 ||
			a.Accounting.DuplicateDeliveries < 0 ||
			a.Accounting.UnverifiedDeliveries < 0 {
			return false
		}
		resA = a.Accounting.AcceptedDeliveries
	} else if resA == 0 && a.Accounting.AcceptedDeliveries > 0 {
		resA = a.Accounting.AcceptedDeliveries
	}

	resB := b.Resolved
	if b.Accounting.Status != "" {
		if b.Accounting.Status == "INCOMPLETE" || !b.Accounting.Verified {
			return false
		}
		if !isFiniteNonNegative(b.Accounting.TotalElapsedSeconds) ||
			!isFiniteNonNegative(b.Accounting.AcceptedPerElapsedHour) ||
			!isFiniteNonNegative(b.Accounting.Spend) ||
			b.Accounting.AcceptedDeliveries <= 0 ||
			b.Accounting.TotalDeliveries < 0 ||
			b.Accounting.RejectedDeliveries < 0 ||
			b.Accounting.DuplicateDeliveries < 0 ||
			b.Accounting.UnverifiedDeliveries < 0 {
			return false
		}
		resB = b.Accounting.AcceptedDeliveries
	} else if resB == 0 && b.Accounting.AcceptedDeliveries > 0 {
		resB = b.Accounting.AcceptedDeliveries
	}

	if resA <= 0 || resB <= 0 || resA != resB {
		return false
	}
	if a.PoisonIncidents < 0 || b.PoisonIncidents < 0 {
		return false
	}
	if a.PeakConcurrency < 0 || b.PeakConcurrency < 0 {
		return false
	}

	return true
}
