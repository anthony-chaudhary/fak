package routing

import (
	"fmt"
	"strings"
)

// ManualOverride is the agent-controllable bypass for one routing decision.
//
// It exists so a headless agent can say "hit THIS model, THIS engine, THIS
// surface, with AT LEAST this context window, and handle returns THIS way"
// without reconfiguring the fleet — the Ops-style manual bypass
// (`fak-sync ops api --fast --no-census --plane X` names what it skips; this
// names what it pins). Every field is optional; every SET field NARROWS and is
// named in the assignment audit, never silently applied.
//
// Narrow-only contract (mirrors PinnedEngine): an override that names a target
// no engine satisfies leaves the request UNROUTED (ReasonNoEligibleEngine +
// the specific mismatch reason on each rejection) rather than falling back to
// another engine. The fabric never STORES an override — it is an input to one
// decision, not routing state.
type ManualOverride struct {
	// Engine pins the decision to the named engine only. Same semantics as
	// TaskRequest.PinnedEngine; when both are set they must agree (a conflict
	// is a caller bug and fails loud) so a pin can never silently win over
	// another pin.
	Engine string `json:"engine,omitempty"`
	// Model pins the decision to engines serving this exact model handle.
	// Empty means any model.
	Model string `json:"model,omitempty"`
	// MinContextWindow requires engines with at least this many context tokens.
	// Zero means no constraint. Engines with unknown capacity
	// (ContextWindow == 0) fail closed when a window is required: an unknown
	// capacity cannot prove it fits, so it is rejected as CONTEXT_TOO_SMALL
	// rather than admitted on hope.
	MinContextWindow int64 `json:"min_context_window,omitempty"`
	// Surface pins the decision to one execution surface. Empty means any.
	Surface Surface `json:"surface,omitempty"`
	// NoRetrySame is return-handling control: a failure class that says
	// RETRY_SAME is treated as FAILOVER_NEXT (the failed engine is excluded
	// rather than retried first). Named in the failover audit.
	NoRetrySame bool `json:"no_retry_same,omitempty"`
	// NoFailover is return-handling control: any non-fatal failure is treated
	// as FATAL (the request is left unrouted with ReasonFailureFatal rather
	// than moved to the next engine). Named in the failover audit. It decouples
	// the caller from a degraded fleet when retrying elsewhere is worse than
	// stopping (e.g. a write that must not execute twice).
	NoFailover bool `json:"no_failover,omitempty"`
}

// IsEmpty reports whether the override names nothing (the common path).
func (o ManualOverride) IsEmpty() bool {
	return o.Engine == "" && o.Model == "" && o.MinContextWindow == 0 && o.Surface == "" && !o.NoRetrySame && !o.NoFailover
}

// Validate fails loud on a malformed override: an unknown surface or a
// negative context window is a caller bug, not a routable constraint.
func (o ManualOverride) Validate() error {
	if o.Surface != "" && !o.Surface.IsKnown() {
		return fmt.Errorf("routing: override surface %q is not known", o.Surface)
	}
	if o.MinContextWindow < 0 {
		return fmt.Errorf("routing: override min_context_window %d is negative", o.MinContextWindow)
	}
	return nil
}

// auditLines renders the Ops-style named-override lines for the decision audit:
// every set field appears, so a bypassed axis is never a silent zero.
func (o ManualOverride) auditLines() []string {
	if o.IsEmpty() {
		return nil
	}
	var out []string
	if o.Engine != "" {
		out = append(out, fmt.Sprintf("override engine=%s (narrow-only; unrouted when absent)", o.Engine))
	}
	if o.Model != "" {
		out = append(out, fmt.Sprintf("override model=%s (exact match; unrouted when unserved)", o.Model))
	}
	if o.MinContextWindow > 0 {
		out = append(out, fmt.Sprintf("override min_context_window=%d (engines below are CONTEXT_TOO_SMALL)", o.MinContextWindow))
	}
	if o.Surface != "" {
		out = append(out, fmt.Sprintf("override surface=%s", o.Surface))
	}
	if o.NoRetrySame {
		out = append(out, "override no_retry_same (RETRY_SAME becomes FAILOVER_NEXT)")
	}
	if o.NoFailover {
		out = append(out, "override no_failover (non-fatal failures stay unrouted)")
	}
	return out
}

// effectivePinnedEngine resolves the narrow-only engine pin from the legacy
// PinnedEngine field and the override. A conflict fails loud: two different
// pins is a caller bug, not a precedence question.
func effectivePinnedEngine(pinned string, o ManualOverride) (string, error) {
	pinned = strings.TrimSpace(pinned)
	engine := strings.TrimSpace(o.Engine)
	switch {
	case pinned != "" && engine != "" && pinned != engine:
		return "", fmt.Errorf("routing: conflicting engine pins %q (request) vs %q (override)", pinned, engine)
	case engine != "":
		return engine, nil
	default:
		return pinned, nil
	}
}
