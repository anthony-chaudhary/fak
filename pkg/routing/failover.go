package routing

import (
	"fmt"
)

// Failover is the deterministic auto-failover decision: after one inference
// failure, decide where the request goes next, driven by the closed failure
// class and ordered by the degradation ladder.
//
// It is pure — no I/O, no goroutines, no retry loop. The caller owns the retry
// budget and calls again; Failover decides *where*, never *how many times*.
//
// Semantics, in order:
//
//  1. The chain and the policy are validated first, and the failure class and
//     action must be known members of their closed vocabularies: a malformed
//     ladder, a malformed policy, or an unknown class/action is a caller bug
//     and fails loud. The class->action CONTRACT is then enforced — a Failure
//     must carry the action and retry budget its class defaults to, so an
//     externally-built Failure can never forge a mismatched pair. The ONE legal
//     exception is the fail-closed pair Class==UNKNOWN && Action==FATAL: the
//     nil-error / no-evidence caller-bug shape, where there is no failure
//     evidence to act on, so FATAL is correct even though UNKNOWN's default
//     action is FAILOVER_NEXT. The second is PAYLOAD_REJECTED &&
//     FAILOVER_NEXT, produced only by Failure.AtSubstituteRung for a rung that
//     serves a different model than the request named.
//  2. A FATAL action (a caller bug or caller intent, e.g. PAYLOAD_REJECTED or
//     CANCELED) does not re-route at all: the request is neither retried nor
//     failed over, and the Unrouted receipt carries ReasonFailureFatal.
//  3. Otherwise the DEGRADATION LADDER owns the preference during failover.
//     The policy's soft surface preference is deliberately not applied here —
//     failover is about survival down a known ladder, not about optimization —
//     but Route's own hard fences (require-local, unknown surface) still hold
//     because they live inside Route. For ActionFailoverNext the failed engine
//     is excluded entirely; for ActionRetrySame it is kept at the FRONT (retry
//     it first) with the rest of the set in ladder order behind it. If the
//     failed engine is absent from the set, retry-same behaves as failover-next.
//  4. Route is called with the reordered set and its decision is returned,
//     annotated with failover and degradation audit lines.
func Failover(req TaskRequest, failedEngineID string, failure Failure, engines []Engine, chain DegradationChain, policy Policy) (Assignment, []Rejection, Unrouted, error) {
	if err := chain.Validate(); err != nil {
		return Assignment{}, nil, Unrouted{}, err
	}
	if err := policy.Validate(); err != nil {
		return Assignment{}, nil, Unrouted{}, err
	}
	if !failure.Class.IsKnown() {
		return Assignment{}, nil, Unrouted{}, fmt.Errorf("routing: unknown failure class %q", failure.Class)
	}
	if !failure.Action.IsKnown() {
		return Assignment{}, nil, Unrouted{}, fmt.Errorf("routing: unknown failure action %q", failure.Action)
	}
	// Enforce the class->action contract, with the documented fail-closed
	// exception, so a forged or mismatched pair fails loud here instead of
	// short-circuiting FATAL on a class whose default action is not FATAL.
	if failure.Action != failure.Class.DefaultAction() && !legalNonDefaultPair(failure) {
		return Assignment{}, nil, Unrouted{}, fmt.Errorf(
			"routing: failure action %q does not match class %q default %q",
			failure.Action, failure.Class, failure.Class.DefaultAction())
	}

	failoverAudit := []string{
		fmt.Sprintf("failover: failure class=%s action=%s failed_engine=%s",
			failure.Class, failure.Action, failedEngineID),
	}
	if failure.Detail != "" {
		failoverAudit = append(failoverAudit, fmt.Sprintf("failover: evidence %s", failure.Detail))
	}
	failoverAudit = append(failoverAudit, req.Override.auditLines()...)

	effectiveAction := failure.Action
	if req.Override.NoFailover {
		lane := ""
		if req.Candidate != nil {
			lane = req.Candidate.Lane
		}
		audit := append([]string{}, failoverAudit...)
		audit = append(audit, "failover: override no_failover; request not retried and not failed over")
		return Assignment{}, nil, Unrouted{
			TicketID:        req.TicketID(),
			Lane:            lane,
			Role:            req.Role,
			Reason:          ReasonFailureFatal,
			AssignmentAudit: audit,
		}, nil
	}
	if req.Override.NoRetrySame && effectiveAction == ActionRetrySame {
		effectiveAction = ActionFailoverNext
		failoverAudit = append(failoverAudit, "failover: override no_retry_same; RETRY_SAME becomes FAILOVER_NEXT")
	}

	if effectiveAction == ActionFatal {
		lane := ""
		if req.Candidate != nil {
			lane = req.Candidate.Lane
		}
		audit := append([]string{}, failoverAudit...)
		audit = append(audit, fmt.Sprintf(
			"failover: class %s is fatal; request not retried and not failed over", failure.Class))
		return Assignment{}, nil, Unrouted{
			TicketID:        req.TicketID(),
			Lane:            lane,
			Role:            req.Role,
			Reason:          ReasonFailureFatal,
			AssignmentAudit: audit,
		}, nil
	}

	// Build the failover candidate set. Retry-same keeps the failed engine at
	// the front; failover-next drops it. A failed engine absent from the set
	// makes the two actions equivalent, which is the documented fallback.
	ordered, retained, excluded := failoverEngineOrder(failedEngineID, effectiveAction, engines, chain)
	failoverAudit = append(failoverAudit,
		fmt.Sprintf("failover: %d engines retained, %d excluded", len(retained), len(excluded)))
	failoverAudit = append(failoverAudit,
		fmt.Sprintf("failover: ladder order governed by chain %s", chain.Version))

	assignments, unrouted, err := Route([]TaskRequest{req}, ordered)
	if err != nil {
		return Assignment{}, nil, Unrouted{}, err
	}

	if len(assignments) == 1 {
		a := assignments[0]
		a.AssignmentAudit = append(a.AssignmentAudit, failoverAudit...)
		a.Reason = fmt.Sprintf("%s; failover class=%s action=%s", a.Reason, failure.Class, failure.Action)

		if idx, rung, ok := chain.rungFor(a.EngineID, engines); ok && idx > 0 {
			line := fmt.Sprintf("degraded to rung %q", rung.Name)
			if idx == len(chain.Rungs)-1 {
				line += " (maintenance interface)"
			}
			a.AssignmentAudit = append(a.AssignmentAudit, line)
			a.Reason = fmt.Sprintf("%s; %s", a.Reason, line)
		}
		return a, a.Rejected, Unrouted{}, nil
	}

	u := unrouted[0]
	u.AssignmentAudit = append(u.AssignmentAudit, failoverAudit...)
	return Assignment{}, u.Rejected, u, nil
}

// failoverEngineOrder produces the set Route will see during failover, plus
// the retained and excluded ID lists for the audit. Failover-next removes the
// failed engine entirely and runs the rest through the ladder; retry-same pins
// the failed engine to the FRONT and orders the rest through the ladder behind
// it. If the failed engine is not in the set, retry-same degrades to
// failover-next. The input slice is never mutated.
func failoverEngineOrder(failedEngineID string, action Action, engines []Engine, chain DegradationChain) ([]Engine, []string, []string) {
	present := false
	for _, e := range engines {
		if e.ID == failedEngineID {
			present = true
			break
		}
	}
	retrySame := action == ActionRetrySame && present

	remaining := make([]Engine, 0, len(engines))
	for _, e := range engines {
		if e.ID == failedEngineID {
			continue
		}
		remaining = append(remaining, e)
	}

	orderedTail := chain.Order(remaining)
	out := make([]Engine, 0, len(engines))
	if retrySame {
		for _, e := range engines {
			if e.ID == failedEngineID {
				out = append(out, e)
				break
			}
		}
	}
	out = append(out, orderedTail...)

	retained := make([]string, 0, len(out))
	for _, e := range out {
		retained = append(retained, e.ID)
	}
	var excluded []string
	if present && !retrySame {
		excluded = append(excluded, failedEngineID)
	}
	return out, retained, excluded
}

// rungFor finds the engine with the given ID in the original engine set and
// returns the index and Rung of the first ladder rung that accepts it. The
// bool is false when the engine is absent, so a caller never reports a
// degradation for an engine it cannot name. It is pure and never mutates.
//
// First-match is what makes the ladder's LAST rung the catch-all: it is the
// only rung allowed to carry no constraint, so matchesRung accepts any engine
// there and an engine that filled none of the constrained rungs resolves to
// the maintenance slot. With the default chain an unconstrained non-final rung
// is refused by Validate precisely so this promise cannot be swallowed.
func (c DegradationChain) rungFor(engineID string, engines []Engine) (int, Rung, bool) {
	for _, e := range engines {
		if e.ID != engineID {
			continue
		}
		for i, r := range c.Rungs {
			if matchesRung(e, r) {
				return i, r, true
			}
		}
		return len(c.Rungs), Rung{}, false
	}
	return len(c.Rungs), Rung{}, false
}
