package loopgate

const (
	// ReasonSkippedReproPhase is returned when a bug fix claims done without
	// witnessed reproduction proof.
	ReasonSkippedReproPhase = "SKIPPED_REPRO_PHASE"

	// ReasonFixMissing is returned when a bug fix provides reproduction
	// proof but lacks verified implementation fix proof.
	ReasonFixMissing = "FIX_MISSING"

	// ReasonImplementationMissing is returned when a non-bug fix lacks
	// implementation proof.
	ReasonImplementationMissing = "IMPLEMENTATION_MISSING"
)

// PhaseAwareDoneSpec declares the verification evidence for a done claim,
// distinguishing bug fixes which require sequential reproduction and fix proofs
// from general implementation tasks.
type PhaseAwareDoneSpec struct {
	IsBugFix      bool
	HasReproProof bool
	HasFixProof   bool
	ReproEvidence string
	FixEvidence   string
}

// EvaluatePhaseDone evaluates a done claim's lifecycle phases.
// For bug fixes, it enforces the sequential red-green lifecycle: reproduction
// proof must be established before implementation fix proof is accepted.
func EvaluatePhaseDone(spec PhaseAwareDoneSpec) (Verdict, string, string) {
	if spec.IsBugFix {
		if !spec.HasReproProof {
			return VerdictRefused, ReasonSkippedReproPhase, "reproduction phase proof was skipped: bug fixes must prove the failing test before claiming done"
		}
		if !spec.HasFixProof {
			return VerdictNotYet, ReasonFixMissing, "reproduction test witnessed, but implementation fix proof is missing or unverified"
		}
		return VerdictWitnessed, "", "reproduction test failure and implementation fix both verified"
	}

	if spec.HasFixProof {
		return VerdictWitnessed, "", "implementation verified"
	}
	return VerdictNotYet, ReasonImplementationMissing, "implementation proof missing"
}
