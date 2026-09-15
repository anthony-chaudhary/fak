package agent

// infra_reprompt.go — the outer turn loop's RECOVERABLE-infrastructure-failure arm.
//
// A model turn can fail for reasons that are NOT the task's fault and NOT terminal:
// the upstream momentarily overloaded (503/529), rate-limited (429), or the retry
// loop spent a bounded budget on a transient glitch without a 200. Historically every
// one of those hard-stopped the arm. This arm instead RE-PROMPTS the model once with a
// bounded, closed-token continuation so the turn can resume, while still stopping on
// genuinely terminal failures (a 404 for an unknown model, a deterministic dial
// misconfiguration, a cancelled/deadline context).
//
// The classification is LOSSY BY DESIGN: the continuation text and the metrics carry
// only a closed token (never err.Error() or an upstream body), so no upstream text
// crosses the trust boundary through this path.

import (
	"context"
	"errors"
)

// infraClass is the closed verdict of classifying one completion error: whether the
// loop may re-prompt, or must stop.
type infraClass uint8

const (
	// infraTerminal: no automated re-prompt can help — stop the arm and surface the error.
	infraTerminal infraClass = iota
	// infraReprompt: a recoverable infrastructure failure — re-prompt the model, bounded
	// by the run's infra-reprompt budget.
	infraReprompt
)

// classifyCompleteError maps a Complete failure to the loop's re-prompt-vs-stop verdict
// plus a CLOSED reason token. It reuses the existing classifiers (deterministicTransportError,
// classifyUpstream) rather than re-deriving the taxonomy, so there is one source of truth.
// The reason is a closed token — never the raw error string or an upstream body.
//
// RetryCeilingError is checked BEFORE UpstreamStatusError deliberately: a ceiling error
// Unwraps to its classified *UpstreamStatusError cause (retry_ceiling.go:73), so an
// errors.As(&se) probe would otherwise always match the inner status and leave the
// RETRY_CEILING branch unreachable for every real ceiling error.
func classifyCompleteError(err error) (infraClass, string) {
	if err == nil {
		return infraTerminal, ""
	}
	if deterministicTransportError(err) {
		return infraTerminal, "UPSTREAM_UNREACHABLE"
	}
	var rc *RetryCeilingError
	if errors.As(err, &rc) {
		return infraReprompt, "RETRY_CEILING"
	}
	var se *UpstreamStatusError
	if errors.As(err, &se) {
		switch classifyUpstream(se.Status, []byte(se.Body), nil) {
		case RemedyBackoff, RemedyRefreshToken, RemedyFailoverAccount, RemedySwitchModel, RemedyStreamRequired:
			reason := se.LimitReason
			if reason == "" {
				reason = "UPSTREAM_STATUS"
			}
			return infraReprompt, reason
		default:
			return infraTerminal, "UPSTREAM_TERMINAL"
		}
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return infraTerminal, "CONTEXT_DONE"
	}
	return infraReprompt, "TRANSPORT_EXHAUSTED"
}

// infraContinuationText renders the bounded continuation spliced as a user turn after a
// recoverable infrastructure failure. It names the CLOSED reason token (never the upstream
// body/error string), tells the model the failure was transient and to continue from where
// it left off, and warns that the harness will stop if it recurs.
func infraContinuationText(reason string) string {
	if reason == "" {
		reason = "INFRA_HICCUP"
	}
	return "[INFRA_REPROMPT] The harness hit a transient infrastructure failure (" + reason +
		") on the previous turn; that was NOT a problem with your work. Continue from where you left off:" +
		" do not apologise, do not repeat an identical tool call, and proceed with the task." +
		" If this failure recurs the harness will stop the arm."
}
