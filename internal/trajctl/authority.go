package trajctl

import "fmt"

func (r WitnessRung) valid() bool {
	switch r {
	case W0, W1, W2, W3:
		return true
	default:
		return false
	}
}

// Authority is what an intervention's payload is permitted to do.
type Authority string

const (
	AuthorityAsk    Authority = "ask"
	AuthorityVerify Authority = "verify"
	AuthorityAssert Authority = "assert"
)

// ReasonLessonOverclaims is the workspace's closed dos_check_reason token for
// an intervention that claims more authority than its witnesses earned.
const ReasonLessonOverclaims = "LESSON_OVERCLAIMS"

// AuthorityRequest carries both evidence boundaries. SupervisorRung is
// separate because independently witnessing an effect does not prove that the
// supervisor issuing the intervention can itself read that effect.
type AuthorityRequest struct {
	Requested       Authority
	EvidenceWitness WitnessRung
	SupervisorRung  WitnessRung
}

// AuthorityStamp is the contract-approved authority attached to an admitted
// intervention or lesson.
type AuthorityStamp struct {
	Authority       Authority   `json:"authority"`
	EvidenceWitness WitnessRung `json:"evidence_witness"`
	SupervisorRung  WitnessRung `json:"supervisor_rung"`
}

// AuthorityOutcome records fail-closed refusal without free-text reason drift.
type AuthorityOutcome struct {
	Refused bool   `json:"refused"`
	Reason  string `json:"reason,omitempty"`
}

// StampAuthority applies authority = min(evidence witness, supervisor witness).
// W3 may assert a fact, W1-W2 may only instruct verification, and W0 may only
// ask. Requests above that ceiling are refused rather than silently rewriting
// an assertion into a weaker speech act.
func StampAuthority(req AuthorityRequest) (AuthorityStamp, AuthorityOutcome, error) {
	if !req.EvidenceWitness.valid() || !req.SupervisorRung.valid() {
		return AuthorityStamp{}, AuthorityOutcome{}, fmt.Errorf("trajctl: invalid witness rung")
	}
	if !validAuthority(req.Requested) {
		return AuthorityStamp{}, AuthorityOutcome{}, fmt.Errorf("trajctl: invalid authority %q", req.Requested)
	}

	rung := req.EvidenceWitness
	if rungOrdinal(req.SupervisorRung) < rungOrdinal(rung) {
		rung = req.SupervisorRung
	}
	ceiling := authorityForRung(rung)
	stamp := AuthorityStamp{
		Authority:       ceiling,
		EvidenceWitness: req.EvidenceWitness,
		SupervisorRung:  req.SupervisorRung,
	}
	if authorityRank(req.Requested) > authorityRank(ceiling) {
		return stamp, AuthorityOutcome{Refused: true, Reason: ReasonLessonOverclaims}, nil
	}
	stamp.Authority = req.Requested
	return stamp, AuthorityOutcome{}, nil
}

func rungOrdinal(r WitnessRung) int {
	switch r {
	case W3:
		return 3
	case W2:
		return 2
	case W1:
		return 1
	default:
		return 0
	}
}

func authorityForRung(r WitnessRung) Authority {
	switch r {
	case W3:
		return AuthorityAssert
	case W1, W2:
		return AuthorityVerify
	default:
		return AuthorityAsk
	}
}

func validAuthority(a Authority) bool {
	return a == AuthorityAsk || a == AuthorityVerify || a == AuthorityAssert
}

func authorityRank(a Authority) int {
	switch a {
	case AuthorityAssert:
		return 2
	case AuthorityVerify:
		return 1
	default:
		return 0
	}
}
