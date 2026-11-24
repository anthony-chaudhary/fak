package routing

// RequestScoped reports whether a failure of this class is a property of the
// REQUEST rather than of the engine that answered: the engine was up and
// answered deterministically (a payload it will not accept, a prompt it cannot
// fit, a request it cannot finish before the client's deadline). A request-scoped failure must never count against the engine's health
// or eligibility, even when the walk fails over past it: the next request may
// fit, and an agent sending oversized or unsupported requests must not be able
// to talk the router into reporting a healthy engine as failing.
func (f FailureClass) RequestScoped() bool {
	switch f {
	case FailureBadPayload, FailureContextOverflow, FailureDeadlineInfeasible:
		return true
	default:
		return false
	}
}

// AdmissionRefused reports whether a failure of this class is a healthy
// engine's deliberate admission refusal (DEADLINE_INFEASIBLE): the engine is
// up and asked to be retried after its Retry-After. Beyond RequestScoped's
// health neutrality it carries two more obligations a walk must honor: the
// attempt yields NO credential verdict (the key was not at fault, so it is
// neither rested nor proven), and the refusal does not license spilling the
// request to a different serving tier (a Halo refusal is not a reason to pay a
// cloud rung) unless the route opts in. The severity verdict stays here; the
// walk only obeys it.
func (f FailureClass) AdmissionRefused() bool {
	return f == FailureDeadlineInfeasible
}

// substituteRungDetail marks a Failure re-decided by AtSubstituteRung, so the
// audit trail says why a PAYLOAD_REJECTED moved on instead of ending the request.
const substituteRungDetail = "substitute rung"

// AtSubstituteRung re-decides a failure observed at a SUBSTITUTE rung: a target
// that serves a different model than the request named (a small local engine
// standing in as fallback capacity for a hosted model). PAYLOAD_REJECTED is
// FATAL because "every provider would answer the same way" - which holds only
// for an engine that IS what the caller asked for. A substitute has its own,
// usually narrower, context window and feature set, so its rejection indicts
// the substitution, not the caller: the request moves to the next rung, where a
// faithful engine's 400 is still fatal.
//
// Only PAYLOAD_REJECTED+FATAL changes; every other Failure is returned as is.
// The re-decided pair keeps its class (the upstream DID reject the payload) and
// authorizes no same-target retry (RetryBudget is 0 for the class): the same
// payload at the same substitute fails the same way.
func (f Failure) AtSubstituteRung() Failure {
	if f.Class != FailureBadPayload || f.Action != ActionFatal {
		return f
	}
	f.Action = ActionFailoverNext
	f.Retries = 0
	if f.Detail == "" {
		f.Detail = substituteRungDetail
	} else {
		f.Detail += "; " + substituteRungDetail
	}
	return f
}

// legalNonDefaultPair reports whether a Failure whose action differs from its
// class default is one of the documented, fabric-produced exceptions: the
// fail-closed UNKNOWN+FATAL (no failure evidence at all) and the substitute-rung
// PAYLOAD_REJECTED+FAILOVER_NEXT (AtSubstituteRung).
func legalNonDefaultPair(f Failure) bool {
	switch {
	case f.Class == FailureUnknown && f.Action == ActionFatal:
		return true
	case f.Class == FailureBadPayload && f.Action == ActionFailoverNext:
		return true
	default:
		return false
	}
}
