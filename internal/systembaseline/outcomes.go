package systembaseline

// OutcomeCounts summarizes attestation results for operational readouts. Clean
// attestations are successes, investigate verdicts are policy refusals, and
// invalid or malformed reports are errors.
type OutcomeCounts struct {
	Success int `json:"success"`
	Refusal int `json:"refusal"`
	Error   int `json:"error"`
}

// CountOutcomes folds reports into the three operator-facing outcome classes.
// Validation runs before verdict classification so corrupted evidence cannot be
// reported as a successful or policy-refused attestation.
func CountOutcomes(reports []Report) OutcomeCounts {
	var counts OutcomeCounts
	for _, report := range reports {
		if err := report.Validate(); err != nil {
			counts.Error++
			continue
		}
		switch report.Verdict {
		case VerdictClean:
			counts.Success++
		case VerdictInvestigate:
			counts.Refusal++
		default:
			counts.Error++
		}
	}
	return counts
}
