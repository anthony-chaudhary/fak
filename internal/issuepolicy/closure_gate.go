package issuepolicy

import "strings"

const (
	ClosureNotRequested = "not_requested"
	ClosureEligible     = "eligible"
	ClosureRefused      = "refused"
)

type ClosureReadout struct {
	Status            string   `json:"status"`
	ClaimedStandard   string   `json:"claimed_standard,omitempty"`
	WitnessedStandard string   `json:"witnessed_standard,omitempty"`
	ProductionCredit  bool     `json:"production_credit"`
	Reasons           []string `json:"reasons,omitempty"`
	Repair            []string `json:"repair,omitempty"`
}

// closureGate grades an explicit ship/close request. An unqualified request
// ("complete", "done", or "close") means production; non-production closure
// must name its maturity and never earns production numerator credit.
func closureGate(c Candidate, project ProjectWorkReadout, witness WitnessGrade, envelope OperatingEnvelopeReadout, scale ScaleEvidenceReadout) ClosureReadout {
	claim := strings.TrimSpace(c.ClosureClaim)
	witnessed := strings.TrimSpace(c.ClosureWitnessStandard)
	if claim == "" && witnessed == "" {
		return ClosureReadout{Status: ClosureNotRequested}
	}
	out := ClosureReadout{Status: ClosureEligible, ClaimedStandard: closureStandard(claim), WitnessedStandard: normalizeCompletionStandard(witnessed)}
	if out.ClaimedStandard == "" {
		out.ClaimedStandard = "production"
	}
	if out.WitnessedStandard == "" {
		out.Reasons = append(out.Reasons, ReasonClosureWitnessMissing)
		out.Repair = append(out.Repair, "add ## Closure witness standard naming the maturity independently witnessed")
	} else if out.WitnessedStandard != out.ClaimedStandard {
		out.Reasons = append(out.Reasons, ReasonClosureWitnessMismatch)
		out.Repair = append(out.Repair, "make the closure claim match the witnessed completion standard")
	}
	if witness.Grade != WitnessGradeStrong {
		out.Reasons = appendUnique(out.Reasons, ReasonClosureWitnessMissing)
		out.Repair = append(out.Repair, "name an independent test, read-back, or commit audit in ## Witness")
	}
	if out.ClaimedStandard == "production" {
		productionGap := project.Status != ProjectWorkValid || project.CompletionStandard != "production"
		productionGap = productionGap || (envelope.Required && envelope.Status != EnvelopeMet)
		productionGap = productionGap || len(scale.Invalid) > 0 || len(scale.MissingStages) > 0
		if productionGap {
			out.Reasons = appendUnique(out.Reasons, ReasonClosureProductionGap)
			out.Repair = append(out.Repair, "satisfy the production contract, operating envelope, and required scale stages before closure")
		}
	}
	if len(out.Reasons) > 0 {
		out.Status = ClosureRefused
		return out
	}
	out.ProductionCredit = out.ClaimedStandard == "production"
	return out
}

func closureStandard(raw string) string {
	lower := strings.ToLower(strings.TrimSpace(raw))
	if lower == "" || lower == "complete" || lower == "completed" || lower == "done" || lower == "close" || lower == "closed" {
		return "production"
	}
	return normalizeCompletionStandard(lower)
}
