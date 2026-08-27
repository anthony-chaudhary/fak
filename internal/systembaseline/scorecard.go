package systembaseline

import "github.com/anthony-chaudhary/fak/pkg/scorecard"

const (
	ScorecardSchema  = "fak-system-baseline-health/1"
	ScorecardDebtKey = "system_baseline_debt"
)

// HealthScorecard grades witnessed system-baseline outcomes. It keeps each
// operational signal named so an operator can distinguish adoption, refusal,
// and malformed-evidence regressions without reopening raw reports.
func HealthScorecard(reports []Report) scorecard.Payload {
	counts := CountOutcomes(reports)
	total := counts.Success + counts.Refusal + counts.Error

	adoptionScore := 100.0
	adoptionDefects := []string(nil)
	if total == 0 {
		adoptionScore = 0
		adoptionDefects = []string{"no system-baseline attestation evidence"}
	}

	integrityScore := 100.0
	integrityDefects := []string(nil)
	if total > 0 && counts.Error > 0 {
		integrityScore = 100 * (1 - float64(counts.Error)/float64(total))
		integrityDefects = []string{"invalid or malformed system-baseline evidence present"}
	}

	policyScore := 100.0
	policyDefects := []string(nil)
	if total > 0 && counts.Refusal > 0 {
		policyScore = 100 * (1 - float64(counts.Refusal)/float64(total))
		policyDefects = []string{"system-baseline policy refusals require investigation"}
	}

	kpis := []scorecard.KPI{
		{Key: "adoption", Group: "usage", Score: adoptionScore, Detail: "attestations observed", Defects: adoptionDefects},
		{Key: "integrity", Group: "health", Score: integrityScore, Detail: "structurally valid evidence", Defects: integrityDefects},
		{Key: "policy", Group: "health", Score: policyScore, Detail: "policy-clean attestations", Defects: policyDefects},
	}
	return scorecard.Fold(ScorecardSchema, kpis, ScorecardDebtKey, nil, scorecard.Messages{
		Finding:         "system-baseline attestation health needs action",
		FindingClean:    "system-baseline attestation health is clean",
		NextAction:      "inspect the named outcome evidence and rerun the benchmark",
		NextActionClean: "continue collecting system-baseline attestations",
		ExtraCorpus: map[string]any{
			"evidence": map[string]int{
				"total":   total,
				"success": counts.Success,
				"refusal": counts.Refusal,
				"error":   counts.Error,
			},
		},
	})
}
