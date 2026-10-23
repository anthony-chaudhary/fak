package hil

import (
	"math"
	"strings"
	"time"
)

// AuditComparison checks the compatibility of reported values before computing a
// descriptive ratio. Compatibility is not evidence of physical execution or a
// matched workload. Producer flags and receipt references cannot certify either.
// Modeled evidence remains an early indicator; other evidence needs independent
// physical proof before an achieved win or performance-issue closure is allowed.
func AuditComparison(headline string, candidate, baseline ComparisonArm, lowerIsBetter bool) ComparisonAudit {
	audit := ComparisonAudit{
		Schema:            ComparisonSchema,
		Timestamp:         time.Now().UTC(),
		Headline:          headline,
		Candidate:         candidate,
		Baseline:          baseline,
		Verdict:           VerdictInvalidComparison,
		EnforcementAction: "Reject comparison until finite positive values, matching metrics and units, positive sample counts, and compatible explicit scopes are provided.",
	}

	if !finitePositiveComparisonValue(candidate.Value) || !finitePositiveComparisonValue(baseline.Value) {
		audit.Reason = "Both metric values must be finite and positive."
		return audit
	}
	if strings.TrimSpace(candidate.Metric) == "" || strings.TrimSpace(baseline.Metric) == "" || candidate.Metric != baseline.Metric {
		audit.Reason = "Both arms must name the same nonempty metric."
		return audit
	}
	if strings.TrimSpace(candidate.Unit) == "" || strings.TrimSpace(baseline.Unit) == "" || candidate.Unit != baseline.Unit {
		audit.Reason = "Both arms must use the same nonempty unit."
		return audit
	}
	if candidate.SampleCount <= 0 || baseline.SampleCount <= 0 {
		audit.Reason = "Both arms require a positive sample count."
		return audit
	}
	if candidate.Scope != baseline.Scope {
		audit.Reason = "Both arms must declare the same comparison scope."
		return audit
	}

	switch candidate.Scope {
	case "":
		// Legacy callers may still report early evidence, but an unspecified
		// scope cannot silently become an engine or hardware speedup.
	case ComparisonEngine, ComparisonHardware:
		if strings.TrimSpace(candidate.HardwareTarget) == "" || strings.TrimSpace(baseline.HardwareTarget) == "" {
			audit.Reason = "An explicit comparison scope requires both hardware targets."
			return audit
		}
		if candidate.Scope == ComparisonEngine && candidate.HardwareTarget != baseline.HardwareTarget {
			audit.Reason = "An engine comparison requires matching hardware targets; declare hardware scope to compare different targets."
			return audit
		}
		ratio := candidate.Value / baseline.Value
		if lowerIsBetter {
			ratio = baseline.Value / candidate.Value
		}
		if !finitePositiveComparisonValue(ratio) {
			audit.Reason = "The comparison ratio is not finite and positive."
			return audit
		}
		audit.Scope = candidate.Scope
		audit.Comparable = true
		audit.Speedup = ratio
	default:
		audit.Reason = "Unknown comparison scope; use engine or hardware."
		return audit
	}

	if modeledComparisonEvidence(candidate.EvidenceType) || modeledComparisonEvidence(baseline.EvidenceType) {
		audit.Verdict = VerdictEarlyIndicatorOnly
		audit.IsEarlyIndicator = true
		audit.Reason = "Modeled or other non-hardware evidence is an early indicator, not an achieved speedup."
		audit.EnforcementAction = "Keep an explicit [SIMULATED] fence. Do not claim an achieved win or close performance issues without independently verified, matched physical execution receipts."
	} else {
		// No independently verified receipt is bound to these arms. In
		// particular, IsPhysicalSilicon and a nonempty ReceiptRef are claims,
		// not authority. Ordinary CPU/software measurements are not simulations.
		audit.Verdict = VerdictNeedsProof
		audit.Reason = "Independent physical execution proof is not bound to both comparison arms."
		if strings.TrimSpace(candidate.ReceiptRef) == "" || strings.TrimSpace(baseline.ReceiptRef) == "" {
			audit.Reason += " One or both receipt references are missing."
		}
		audit.EnforcementAction = "Do not claim an achieved win or close performance issues. Require independently verified, matched physical execution receipts; producer flags and reference strings are not proof."
	}
	if !audit.Comparable {
		audit.Reason += " No ratio is available until both arms declare an explicit comparison scope."
	}
	return audit
}

func finitePositiveComparisonValue(value float64) bool {
	return value > 0 && !math.IsNaN(value) && !math.IsInf(value, 0)
}

func modeledComparisonEvidence(evidenceType string) bool {
	switch strings.TrimSpace(evidenceType) {
	case "structural_count", "analytical_bound", "trace_sim", "cycle_sim", "learned_estimate", "calibrated_sim":
		return true
	default:
		return false
	}
}
