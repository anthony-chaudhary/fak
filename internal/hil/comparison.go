package hil

import (
	"fmt"
	"time"
)

// AuditComparison evaluates a proposed head-to-head comparison against the repository's
// hardware-in-the-loop and simulated results discipline.
//
// Invariants enforced:
//  1. Software simulations, analytical rooflines, and trace models are strictly early
//     indicators / hypotheses.
//  2. Any reported head-to-head comparison between engines, models, or optimizations MUST
//     be backed by real physical silicon measurements on both candidate and baseline.
//  3. A comparison citing a simulated arm is NEVER allowed to declare an achieved win,
//     close a performance issue, or be cited without an explicit [SIMULATED] fence.
func AuditComparison(headline string, candidate, baseline ComparisonArm, lowerIsBetter bool) ComparisonAudit {
	now := time.Now().UTC()
	audit := ComparisonAudit{
		Schema:    ComparisonSchema,
		Timestamp: now,
		Headline:  headline,
		Candidate: candidate,
		Baseline:  baseline,
	}

	// Calculate speedup ratio
	if lowerIsBetter {
		if candidate.Value > 0 {
			audit.Speedup = baseline.Value / candidate.Value
		}
	} else {
		if baseline.Value > 0 {
			audit.Speedup = candidate.Value / baseline.Value
		}
	}

	// Invariant Check 1: Missing or invalid metrics
	if candidate.Value <= 0 || baseline.Value <= 0 || candidate.Metric == "" || baseline.Metric == "" {
		audit.Verdict = VerdictInvalidComparison
		audit.AllowedAsAchievedWin = false
		audit.IsEarlyIndicator = false
		audit.Reason = "Comparison has missing or non-positive metric values."
		audit.EnforcementAction = "Reject comparison until valid positive metric values and units are provided."
		return audit
	}

	// Invariant Check 2: Physical hardware verification on both arms
	candPhysical := candidate.IsPhysicalSilicon && (candidate.EvidenceType == "hardware_measurement" || candidate.EvidenceType == "physical_silicon")
	basePhysical := baseline.IsPhysicalSilicon && (baseline.EvidenceType == "hardware_measurement" || baseline.EvidenceType == "physical_silicon")

	if !candPhysical || !basePhysical {
		audit.Verdict = VerdictEarlyIndicatorOnly
		audit.AllowedAsAchievedWin = false
		audit.IsEarlyIndicator = true

		var simArms []string
		if !candPhysical {
			simArms = append(simArms, fmt.Sprintf("candidate %q (%s)", candidate.Name, candidate.EvidenceType))
		}
		if !basePhysical {
			simArms = append(simArms, fmt.Sprintf("baseline %q (%s)", baseline.Name, baseline.EvidenceType))
		}

		audit.Reason = fmt.Sprintf(
			"Software simulation detected in comparison arm(s): %v. Under repository discipline, simulations/models are early indicators only and cannot declare an achieved win or final speedup.",
			simArms,
		)
		audit.EnforcementAction = "Gate as EARLY_INDICATOR_ONLY. Require physical on-device execution receipts on sanctioned fleet hardware before claiming competitive win or closing performance issues."
		return audit
	}

	// Both arms physically measured on real hardware
	audit.Verdict = VerdictRealHardwareVerified
	audit.AllowedAsAchievedWin = true
	audit.IsEarlyIndicator = false
	audit.Reason = fmt.Sprintf(
		"Both arms verified on physical silicon (candidate on %s, baseline on %s) with %d and %d samples respectively.",
		candidate.HardwareTarget,
		baseline.HardwareTarget,
		candidate.SampleCount,
		baseline.SampleCount,
	)
	audit.EnforcementAction = "Admissible for canonical benchmark comparison and issue closure."
	return audit
}
