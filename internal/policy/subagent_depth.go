// The subagent fan-out depth cap (issue #2603, epic #2000): the declarative,
// policy-bound form of the harness's subagent recursion bound. eve subagents
// may themselves spawn subagents, so an unbounded fan-out tree is a
// resource-exhaustion / runaway vector — a child that spawns a child that
// spawns a child, with no floor that says "stop". fak treats the depth cap as a
// POLICY-BOUND value an operator dials from the manifest (the same idiom as the
// isolation dial and the rate-limit cap: manifest/runtime-only, NOT an
// adjudicator.Policy field), so the bound travels with the deployment's other
// launch policy rather than living as a compiled-in magic number.
//
// Resolution is fail-closed by construction, exactly like IsolationRule:
//
//   - an ABSENT block (nil rule) does NOT mean "no cap" — it resolves to
//     DefaultMaxSubagentDepth, so a deployment that declares nothing still
//     bounds its subagent recursion. The manifest TIGHTENS or adjusts the
//     bound; it can never silently remove it.
//   - a child at a depth ABOVE the cap is refused (ErrSubagentDepthExceeded);
//   - a non-positive child depth is a caller bug (a subagent is nested at least
//     one level below the root), refused rather than silently admitted.
package policy

import (
	"errors"
	"fmt"
)

// DefaultMaxSubagentDepth is fak's conservative fallback subagent-recursion
// bound, in force whenever the manifest declares no subagent_depth block. It
// exists so the cap is ALWAYS enforced (fail-closed to a bound) rather than
// defaulting to unbounded recursion; a deployment sets subagent_depth.max_depth
// to match the depth cap its own harness declares.
const DefaultMaxSubagentDepth = 8

// ErrSubagentDepthExceeded is the typed refusal a launch adapter surfaces when a
// subagent spawn would exceed the policy-bound depth cap. Callers match it with
// errors.Is so the refusal is a first-class, checkable value — not a string.
var ErrSubagentDepthExceeded = errors.New("policy: subagent fan-out depth cap exceeded")

// SubagentDepthRule is the manifest's `subagent_depth` block: the maximum depth
// at which a subagent may be admitted, where the root session is depth 0 and a
// subagent it spawns is depth 1 (a subagent of THAT is depth 2, and so on). A
// present block is validated at load (see compileSubagentDepth); MaxDepth must
// be at least 1 — a cap of 0 would forbid every subagent, which a launch policy
// expresses by not brokering subagents at all, not by a zero cap.
type SubagentDepthRule struct {
	MaxDepth int `json:"max_depth"`
}

// compileSubagentDepth validates a declared subagent_depth block at policy LOAD
// (absent => nil, the DefaultMaxSubagentDepth fallback applies), so a
// nonsensical cap fails loud here rather than at spawn time.
func compileSubagentDepth(r *SubagentDepthRule) (*SubagentDepthRule, error) {
	if r == nil {
		return nil, nil // absent => the fail-closed default cap applies
	}
	if r.MaxDepth < 1 {
		return nil, fmt.Errorf("subagent_depth: max_depth must be at least 1, got %d (a launch policy that admits no subagents brokers none, it does not set a zero cap)", r.MaxDepth)
	}
	return &SubagentDepthRule{MaxDepth: r.MaxDepth}, nil
}

// Cap returns the resolved maximum subagent depth: the declared MaxDepth, or
// DefaultMaxSubagentDepth when the rule is absent (nil). It is safe on a nil
// receiver — an unconfigured deployment still reports the fail-closed default.
func (r *SubagentDepthRule) Cap() int {
	if r == nil {
		return DefaultMaxSubagentDepth
	}
	return r.MaxDepth
}

// AdmitDepth reports whether a subagent that would occupy childDepth may be
// admitted under the resolved cap. It is the single decision point a launch
// adapter routes a subagent spawn through:
//
//   - childDepth < 1        -> a caller bug (a subagent is nested below the
//                              root; depth 0 is the root itself), refused;
//   - childDepth > Cap()    -> ErrSubagentDepthExceeded (the runaway floor);
//   - otherwise             -> nil (admit).
//
// It is safe on a nil receiver (the DefaultMaxSubagentDepth cap applies), so a
// deployment that declares no subagent_depth block is still bounded.
func (r *SubagentDepthRule) AdmitDepth(childDepth int) error {
	if childDepth < 1 {
		return fmt.Errorf("%w: invalid child depth %d (a subagent is nested at least one level below the root; depth 0 is the root)", ErrSubagentDepthExceeded, childDepth)
	}
	cap := r.Cap()
	if childDepth > cap {
		return fmt.Errorf("%w: child depth %d exceeds cap %d", ErrSubagentDepthExceeded, childDepth, cap)
	}
	return nil
}

// AdmitChildOf is the caller-shaped convenience over AdmitDepth: given the
// depth of the PARENT that wants to spawn a subagent, it decides whether the
// child (at parentDepth+1) may be admitted. A negative parentDepth is refused.
func (r *SubagentDepthRule) AdmitChildOf(parentDepth int) error {
	if parentDepth < 0 {
		return fmt.Errorf("%w: invalid parent depth %d", ErrSubagentDepthExceeded, parentDepth)
	}
	return r.AdmitDepth(parentDepth + 1)
}
