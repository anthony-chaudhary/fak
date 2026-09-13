package gateway

import "context"

// subagent_depth.go — the gateway-side half of the subagent fan-out depth cap
// (issue #12631, the policy rule is internal/policy/subagent_depth.go). The rule
// itself is policy-bound and lives in internal/policy; what was missing is the
// SESSION-DEPTH carrier at the adjudication seam: the proxy sees only a stream of
// tool_use calls, not the spawn tree, so a child at depth N could propose another
// `task` call and nothing at the seam said "stop". This file adds that carrier.
//
// The carrier is a context value, the same shape as WithPrincipal (tool_routing.go):
// a host embedding the gateway stamps the depth of the session whose turn it is
// adjudicating, and adjudicateProposed (adjudicate_proposed.go) reads it back to
// evaluate SubagentDepthRule.AdmitChildOf. Absent value => depth 0 (the root), so
// the default deployment — which stamps nothing — behaves exactly as before: a root
// coordinator may spawn a child, and that child's own turn is stamped depth 1.

// sessionDepthContextKey scopes the session-depth value. A struct key (not a
// string) so no unrelated context value can collide with it.
type sessionDepthContextKey struct{}

// WithSessionDepth returns a context carrying the caller's session depth, where the
// root session is 0 and a subagent it spawns is 1. A negative depth is normalized to
// 0 (the root) rather than refused: the depth carrier is descriptive provenance, and
// the actual cap decision is made — and fail-closed — by AdmitChildOf at the seam.
// Exported so a host embedding the gateway can stamp the depth from its own spawn
// tree before calling the adjudication surface (mirrors WithPrincipal).
func WithSessionDepth(ctx context.Context, depth int) context.Context {
	if depth < 0 {
		depth = 0
	}
	return context.WithValue(ctx, sessionDepthContextKey{}, depth)
}

// sessionDepthFrom returns the request-scoped session depth, or 0 (the root) if none
// was stamped. Safe on a nil context.
func sessionDepthFrom(ctx context.Context) int {
	if ctx == nil {
		return 0
	}
	depth, _ := ctx.Value(sessionDepthContextKey{}).(int)
	if depth < 0 {
		return 0
	}
	return depth
}
