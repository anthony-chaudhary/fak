package policy

import (
	"fmt"
	"sort"
)

// LoopDepthRule caps the recurrent depth (loop count) a looped model may run
// for one request, so depth becomes a policy-bound knob instead of whatever the
// caller asks for. MaxLoopDepth is the default cap for every scope; Scopes
// overrides it per tool scope (for example a cheap classifier scope held to 2
// loops while a reasoning scope keeps the default).
//
// A nil rule declares no cap: a looped model is already bounded by its own
// trained maximum, so unlike subagent recursion there is no runaway to fence by
// default.
type LoopDepthRule struct {
	MaxLoopDepth int            `json:"max_loop_depth,omitempty"`
	Scopes       map[string]int `json:"scopes,omitempty"`
}

// CompileLoopDepth validates a declared rule at policy load so a nonsensical
// cap fails loud there rather than silently clamping every request to zero.
func CompileLoopDepth(r *LoopDepthRule) (*LoopDepthRule, error) {
	if r == nil {
		return nil, nil
	}
	if r.MaxLoopDepth < 0 {
		return nil, fmt.Errorf("loop_depth: max_loop_depth must be at least 1 (or omitted), got %d", r.MaxLoopDepth)
	}
	if r.MaxLoopDepth == 0 && len(r.Scopes) == 0 {
		return nil, fmt.Errorf("loop_depth: declares no cap (set max_loop_depth or at least one scope)")
	}
	out := &LoopDepthRule{MaxLoopDepth: r.MaxLoopDepth}
	if len(r.Scopes) > 0 {
		names := make([]string, 0, len(r.Scopes))
		for name := range r.Scopes {
			names = append(names, name)
		}
		sort.Strings(names)
		out.Scopes = make(map[string]int, len(names))
		for _, name := range names {
			if name == "" {
				return nil, fmt.Errorf("loop_depth: scope name must be non-empty")
			}
			// A zero cap would forbid the forward pass itself; a scope that must
			// not run a looped model is denied by tool policy, not by depth.
			if d := r.Scopes[name]; d < 1 {
				return nil, fmt.Errorf("loop_depth: scope %q max_loop_depth must be at least 1, got %d", name, d)
			}
			out.Scopes[name] = r.Scopes[name]
		}
	}
	return out, nil
}

// CapFor returns the loop-depth cap in force for scope and whether one exists.
// A scope override wins over the default; a nil rule caps nothing.
func (r *LoopDepthRule) CapFor(scope string) (int, bool) {
	if r == nil {
		return 0, false
	}
	if d, ok := r.Scopes[scope]; ok {
		return d, true
	}
	if r.MaxLoopDepth > 0 {
		return r.MaxLoopDepth, true
	}
	return 0, false
}

// ClampLoopDepth resolves a requested loop depth against the cap for scope and
// reports whether the policy lowered it. A non-positive request means "model
// default", which runs full trained depth; under a cap that resolves to the cap
// and counts as clamped, since policy cannot prove the default fits. Without a
// cap the request passes through unchanged.
func (r *LoopDepthRule) ClampLoopDepth(scope string, requested int) (granted int, clamped bool) {
	limit, ok := r.CapFor(scope)
	if !ok {
		return requested, false
	}
	if requested <= 0 || requested > limit {
		return limit, true
	}
	return requested, false
}
