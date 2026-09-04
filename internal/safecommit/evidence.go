package safecommit

// EvidenceSchema is the wire version for commit completion evidence. Result.Verified predates
// this object; consumers must prefer Evidence whenever it is present and may retain the legacy
// boolean only for receipts that do not carry this schema.
const EvidenceSchema = "fak-safecommit-evidence/1"

type CompletionClass string

const (
	CompletionVerifiedDelivery CompletionClass = "verified-delivery"
	CompletionRecordOnly       CompletionClass = "record-only"
)

type EvidenceOutcome string

const (
	EvidencePassed      EvidenceOutcome = "passed"
	EvidenceFailed      EvidenceOutcome = "failed"
	EvidenceSkipped     EvidenceOutcome = "skipped"
	EvidenceUnrun       EvidenceOutcome = "unrun"
	EvidenceNotRequired EvidenceOutcome = "not-required"
)

type EvidenceAxis struct {
	Outcome  EvidenceOutcome `json:"outcome"`
	Required bool            `json:"required"`
}

// CommitEvidence separates recording from delivery claims. Result.Verified is derived from
// these axes for versioned receipts and is not duplicated here.
type CommitEvidence struct {
	Schema          string          `json:"schema"`
	CompletionClass CompletionClass `json:"completion_class"`
	Recorded        EvidenceAxis    `json:"recorded"`
	DiffWitnessed   EvidenceAxis    `json:"diff_witnessed"`
	Compiled        EvidenceAxis    `json:"compiled"`
	Tested          EvidenceAxis    `json:"tested"`
	Pushed          EvidenceAxis    `json:"pushed"`
	ClosureBound    EvidenceAxis    `json:"closure_bound"`
}

type EvidenceContract struct {
	CompletionClass CompletionClass
	RequirePush     bool
	RequireClosure  bool
	ClosureBound    bool
}

// FinalizeEvidence captures the incoming legacy Verified value as the executor's diff witness,
// then replaces it with the aggregate derived from the declared completion contract.
func FinalizeEvidence(res Result, contract EvidenceContract) Result {
	class := contract.CompletionClass
	if class == "" {
		class = CompletionVerifiedDelivery
	}
	requiresDelivery := class == CompletionVerifiedDelivery
	compiled, tested := buildEvidence(res.BuildCheck)
	diffWitnessed := res.Verified
	if res.DOSWitness != nil && res.DOSWitness.Ran {
		diffWitnessed = res.DOSWitness.Verdict == "OK" && res.DOSWitness.Witness == "diff-witnessed"
	}
	evidence := CommitEvidence{
		Schema:          EvidenceSchema,
		CompletionClass: class,
		Recorded:        evidenceBool(res.Committed, true),
		DiffWitnessed:   evidenceBool(diffWitnessed, true),
		Compiled:        EvidenceAxis{Outcome: compiled, Required: requiresDelivery && compiled != EvidenceNotRequired},
		Tested:          EvidenceAxis{Outcome: tested, Required: requiresDelivery && tested != EvidenceNotRequired},
		Pushed:          evidenceOptionalEffect(res.Pushed, contract.RequirePush, res.Reason == ReasonPushRejected),
		ClosureBound:    evidenceOptionalEffect(contract.ClosureBound, contract.RequireClosure, false),
	}
	res.Evidence = &evidence
	res.Verified = evidence.Verified()
	requalifyCommitVelocity(&res)
	return res
}

func (e CommitEvidence) Verified() bool {
	for _, axis := range []EvidenceAxis{e.Recorded, e.DiffWitnessed, e.Compiled, e.Tested, e.Pushed, e.ClosureBound} {
		if axis.Required && axis.Outcome != EvidencePassed {
			return false
		}
	}
	return true
}

// DeliveryVerified qualifies grades, velocity, and closure automation. Schema-less results keep
// the historical executor contract; record-only never becomes compile/test-verified delivery.
func (res Result) DeliveryVerified() bool {
	if res.Evidence == nil {
		return res.Verified
	}
	return res.Evidence.CompletionClass == CompletionVerifiedDelivery && res.Evidence.Verified()
}

func (res Result) RecordOnlyVerified() bool {
	return res.Evidence != nil && res.Evidence.CompletionClass == CompletionRecordOnly && res.Evidence.Verified()
}

func evidenceBool(ok, required bool) EvidenceAxis {
	outcome := EvidenceFailed
	if ok {
		outcome = EvidencePassed
	}
	return EvidenceAxis{Outcome: outcome, Required: required}
}

func evidenceOptionalEffect(ok, required, failed bool) EvidenceAxis {
	outcome := EvidenceUnrun
	if ok {
		outcome = EvidencePassed
	} else if failed {
		outcome = EvidenceFailed
	}
	return EvidenceAxis{Outcome: outcome, Required: required}
}

func buildEvidence(check *BuildCheckResult) (compiled, tested EvidenceOutcome) {
	if check == nil {
		return EvidenceUnrun, EvidenceUnrun
	}
	if check.CompileEvidence != "" || check.TestEvidence != "" {
		return defaultEvidence(check.CompileEvidence), defaultEvidence(check.TestEvidence)
	}
	switch check.Outcome {
	case BuildCheckPassed:
		return EvidencePassed, EvidencePassed
	case BuildCheckFailed, BuildCheckHeadRed:
		return EvidenceFailed, EvidenceUnrun
	case BuildCheckSkippedTimeout, BuildCheckSkippedInfra:
		return EvidenceSkipped, EvidenceUnrun
	case BuildCheckNotApplicable:
		return EvidenceNotRequired, EvidenceNotRequired
	default:
		return EvidenceUnrun, EvidenceUnrun
	}
}

func defaultEvidence(outcome EvidenceOutcome) EvidenceOutcome {
	if outcome == "" {
		return EvidenceUnrun
	}
	return outcome
}
