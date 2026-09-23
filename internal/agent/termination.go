package agent

import (
	"context"
	"errors"
	"strings"
)

// Termination describes a client-safe, closed classification of a failed owned turn.
type Termination struct {
	Cause    string `json:"cause"`
	Evidence string `json:"evidence"`
}

const (
	TerminationCanceled     = "canceled"
	TerminationRateLimited  = "rate_limited"
	TerminationContextLimit = "context_limit"
	TerminationAuth         = "authentication"
	TerminationRefused      = "refused"
	TerminationProvider     = "provider_error"
	TerminationUnknown      = "unknown"
)

// ClassifyTermination deliberately uses only stable error signals and bounded,
// redacted evidence. The full error remains available to server logs.
func ClassifyTermination(err error) Termination {
	if err == nil {
		return Termination{}
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return Termination{Cause: TerminationCanceled, Evidence: "request context ended"}
	}
	s := strings.ToLower(err.Error())
	switch {
	case containsAny(s, "rate limit", "rate_limit", "too many requests", "status 429", "http 429"):
		return Termination{Cause: TerminationRateLimited, Evidence: "provider reported rate limiting"}
	case containsAny(s, "context window", "context length", "maximum context", "too many tokens", "token limit"):
		return Termination{Cause: TerminationContextLimit, Evidence: "request exceeded the model context limit"}
	case containsAny(s, "401 unauthorized", "status 401", "http 401", "status code 401", "403 forbidden", "status 403", "http 403", "status code 403", "missing_credentials", "invalid_credentials"):
		return Termination{Cause: TerminationAuth, Evidence: "model endpoint rejected the configured credentials"}
	case containsAny(s, "policy_block", "policy block", "refused", "denied by", "stop gate", "guard blocked"):
		return Termination{Cause: TerminationRefused, Evidence: "fak refused the turn"}
	case containsAny(s, "upstream", "provider", "status 5", "http 5"):
		return Termination{Cause: TerminationProvider, Evidence: "provider request failed"}
	default:
		return Termination{Cause: TerminationUnknown, Evidence: "unclassified turn failure"}
	}
}
func containsAny(s string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(s, n) {
			return true
		}
	}
	return false
}
