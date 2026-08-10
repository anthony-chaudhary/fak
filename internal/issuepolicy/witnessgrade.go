package issuepolicy

import "strings"

const (
	WitnessGradeStrong    = "strong"
	WitnessGradeWeak      = "weak"
	WitnessGradeForgeable = "forgeable"
	WitnessGradeMissing   = "missing"

	FlagWitnessMissing   = "WITNESS_MISSING"
	FlagWitnessSelfClaim = "WITNESS_SELF_CLAIM"
	FlagWitnessNoOracle  = "WITNESS_NO_INDEPENDENT_ORACLE"
)

// WitnessGrade is the advisory-first pre-dispatch forgeability readout for a
// ticket's done condition. StrictWitness promotes any non-strong grade to a hold.
type WitnessGrade struct {
	Grade         string   `json:"grade"`
	StrictWitness bool     `json:"strict_witness"`
	Flags         []string `json:"flags,omitempty"`
	Evidence      []string `json:"evidence,omitempty"`
}

func witnessGrade(c Candidate, strict bool) WitnessGrade {
	done := strings.TrimSpace(c.DoneCondition)
	witness := strings.TrimSpace(c.Witness)
	out := WitnessGrade{StrictWitness: strict}
	if done == "" || witness == "" {
		out.Grade = WitnessGradeMissing
		out.Flags = []string{FlagWitnessMissing}
		return out
	}
	lower := strings.ToLower(witness)
	selfClaim := containsAny(lower,
		"worker reports", "agent reports", "agent says", "worker says",
		"self-report", "declare done", "claims completion", "confirms it completed")
	oracle := containsAny(lower,
		"go test", "make test", "make ci", "fak ", "dos ", "git show", "git diff",
		"assert", "fixture", "captured", "render", "json", "exit code", "read-back",
		"readback", "commit-audit", "screenshot", "ledger", "query", "status")
	if selfClaim && !oracle {
		out.Grade = WitnessGradeForgeable
		out.Flags = []string{FlagWitnessSelfClaim, FlagWitnessNoOracle}
		return out
	}
	if !oracle {
		out.Grade = WitnessGradeWeak
		out.Flags = []string{FlagWitnessNoOracle}
		return out
	}
	out.Grade = WitnessGradeStrong
	out.Evidence = []string{"independent_oracle"}
	return out
}

func containsAny(s string, needles ...string) bool {
	for _, needle := range needles {
		if strings.Contains(s, needle) {
			return true
		}
	}
	return false
}
