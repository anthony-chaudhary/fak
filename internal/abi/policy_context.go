package abi

import "context"

// Posture selects the policy default-deny behavior after provable refusal checks have passed.
type Posture uint8

const (
	// PostureFailClosed keeps the default floor: anything not affirmatively allowed is denied.
	PostureFailClosed Posture = iota
	// PostureAdmitAndLog downgrades low-risk read-shaped DEFAULT_DENY decisions to ALLOW with forensic metadata.
	PostureAdmitAndLog
	// PostureDefaultOpen permits tools by default after provable refusal checks pass.
	PostureDefaultOpen
)

func (p Posture) String() string {
	switch p {
	case PostureFailClosed:
		return "fail_closed"
	case PostureAdmitAndLog:
		return "admit_and_log"
	case PostureDefaultOpen:
		return "default_open"
	default:
		return "unknown"
	}
}

// PolicyContext carries execution-time capability floor posture and safe sink configuration.
type PolicyContext struct {
	Posture   Posture
	Profile   string
	SafeSinks map[string]bool
}

type policyContextKey struct{}

// ContextWithPolicy returns a derived context carrying the given PolicyContext.
func ContextWithPolicy(ctx context.Context, pc PolicyContext) context.Context {
	return context.WithValue(ctx, policyContextKey{}, pc)
}

// PolicyFromContext extracts the PolicyContext from ctx if present.
func PolicyFromContext(ctx context.Context) (PolicyContext, bool) {
	if ctx == nil {
		return PolicyContext{}, false
	}
	pc, ok := ctx.Value(policyContextKey{}).(PolicyContext)
	return pc, ok
}
