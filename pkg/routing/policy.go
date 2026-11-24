package routing

import (
	"fmt"
	"sort"
	"strings"
)

// PolicySchema identifies the placement policy manifest shape.
const PolicySchema = "routing-policy/1"

// Policy is a deterministic placement policy: pure data plus rule evaluation.
// It encodes the proven routing trade-offs as ordered rules. It contains no ML,
// no learned weights, and no runtime mutation — given the same request and
// engines, it always returns the same decision.
//
// Policy is advisory over engine order: it does not override the router's
// declared-order selection, but it supplies the role/tier preference and the
// hard fences (require-local, never-drain, never-exceed-capacity) that Route
// enforces. Evaluate reports, per engine, whether the policy admits it and why.
type Policy struct {
	// Version is the policy version tag.
	Version string `json:"version"`
	// DefaultSurface is the surface used by the default rule when no explicit
	// rule matches and no hard fence applies.
	DefaultSurface Surface `json:"default_surface"`
	// Rules are evaluated in priority order (higher Priority first, declared
	// order on tie).
	Rules []PolicyRule `json:"rules"`
}

// PolicyRule is one deterministic placement rule. A rule matches on role
// and/or surface and maps to a preferred surface. The hard fences are always
// applied before any rule: a drained or unhealthy engine, an exceeded capacity,
// or a require-local request against a hosted surface is never admitted.
type PolicyRule struct {
	// Name identifies the rule in the audit trail.
	Name string `json:"name"`
	// Priority orders evaluation: higher evaluates first; declared order
	// breaks ties.
	Priority int `json:"priority,omitempty"`
	// Roles the rule applies to. Empty matches any role.
	Roles []string `json:"roles,omitempty"`
	// Surface the rule prefers. Empty means the rule contributes a preference
	// only through its name.
	Surface Surface `json:"surface"`
}

// DefaultPolicy returns the standard deterministic routing policy. It mirrors
// the spirit of fak/internal/executionroute's DefaultPlacementManifest without
// importing it (private code may not import fak/internal/*):
//
//   - small auxiliary/screening/scout roles -> local surface
//   - primary heavy work -> local surface
//   - massive/cloud work -> hosted surface
//   - judge/verifier -> local surface (keep the trust plane on-box)
//
// The hard fences — never a drained or unhealthy engine, never over capacity,
// never hosted for a require-local task — are enforced by Route and re-checked
// by Policy.Admits, so they hold regardless of which rule would otherwise win.
func DefaultPolicy() Policy {
	return Policy{
		Version:        PolicySchema,
		DefaultSurface: SurfaceLocal,
		Rules: []PolicyRule{
			{
				Name:     "scout-local-small",
				Priority: 100,
				Roles:    []string{"scout", "balancer"},
				Surface:  SurfaceLocal,
			},
			{
				Name:     "judge-local-trust",
				Priority: 90,
				Roles:    []string{"judge", "reviewer", "verifier"},
				Surface:  SurfaceLocal,
			},
			{
				Name:     "hosted-massive-cloud",
				Priority: 80,
				Roles:    []string{"worker"},
				Surface:  SurfaceHosted,
			},
			{
				Name:     "local-primary-heavy",
				Priority: 60,
				Roles:    []string{"worker"},
				Surface:  SurfaceLocal,
			},
		},
	}
}

// normalizedRules returns the policy's rules in deterministic evaluation order:
// priority descending, declared order on tie. It never mutates the policy.
func (p Policy) normalizedRules() []PolicyRule {
	indexed := make([]PolicyRule, len(p.Rules))
	copy(indexed, p.Rules)
	sort.SliceStable(indexed, func(i, j int) bool {
		return indexed[i].Priority > indexed[j].Priority
	})
	return indexed
}

// preferredSurface returns the surface the policy prefers for a request, and
// the rule that chose it, or the default surface when no rule matches. It is
// pure and order-independent with respect to the engine set.
func (p Policy) preferredSurface(req TaskRequest) (Surface, string) {
	for _, r := range p.normalizedRules() {
		if len(r.Roles) > 0 {
			matched := false
			for _, role := range r.Roles {
				if role == req.Role {
					matched = true
					break
				}
			}
			if !matched {
				continue
			}
		}
		s := r.Surface
		if s == "" {
			s = p.DefaultSurface
		}
		if req.RequireLocal && s.Hosted() {
			continue
		}
		return s, r.Name
	}
	return p.DefaultSurface, "default"
}

// Admits reports whether the policy admits an engine for a request, with a
// closed reason when it does not. It re-applies the hard fences so a caller
// can consult the policy directly without duplicating them. A non-empty reason
// is always a member of the closed rejection vocabulary.
func (p Policy) Admits(req TaskRequest, e Engine) (bool, string) {
	if req.RequireLocal && e.Surface.Hosted() {
		return false, ReasonHostedRefused
	}
	if !e.Healthy {
		return false, ReasonUnhealthy
	}
	if e.Draining {
		return false, ReasonDraining
	}
	if e.CapacityHeadroom <= 0 {
		return false, ReasonNoHeadroom
	}
	return true, ""
}

// PolicyDecision is the policy's deterministic recommendation for one request:
// the preferred surface, the rule that chose it, and the engines admitted and
// refused under the hard fences. It is pure data for audit.
type PolicyDecision struct {
	// PreferredSurface is the surface the policy prefers for the request.
	PreferredSurface Surface `json:"preferred_surface"`
	// Rule is the name of the rule that chose the preferred surface.
	Rule string `json:"rule"`
	// Admitted lists the IDs of engines the policy admits.
	Admitted []string `json:"admitted,omitempty"`
	// Refused lists engines the policy refuses, with closed reasons.
	Refused []Rejection `json:"refused,omitempty"`
}

// Evaluate applies the policy to a request over an engine set and returns its
// deterministic recommendation. It is a pure query: it changes no state and
// selects nothing. Route remains the authority for the final engine choice.
func (p Policy) Evaluate(req TaskRequest, engines []Engine) PolicyDecision {
	surface, rule := p.preferredSurface(req)
	decision := PolicyDecision{PreferredSurface: surface, Rule: rule}
	for _, e := range engines {
		ok, reason := p.Admits(req, e)
		if ok {
			decision.Admitted = append(decision.Admitted, e.ID)
			continue
		}
		decision.Refused = append(decision.Refused, Rejection{EngineID: e.ID, Reason: reason})
	}
	return decision
}

// Validate checks the policy's closed-vocabulary invariants: every rule surface
// is known and every rule has a name. It returns a wrapped error naming the
// first violation, so a malformed manifest fails loud before it routes.
func (p Policy) Validate() error {
	if !p.DefaultSurface.IsKnown() {
		return fmt.Errorf("routing: policy default surface %q is not known", p.DefaultSurface)
	}
	for i, r := range p.Rules {
		if strings.TrimSpace(r.Name) == "" {
			return fmt.Errorf("routing: policy rule %d has no name", i)
		}
		if r.Surface != "" && !r.Surface.IsKnown() {
			return fmt.Errorf("routing: policy rule %q surface %q is not known", r.Name, r.Surface)
		}
	}
	return nil
}

// policyRank returns the declared position of e within engines, or -1 when e is
// not part of the set. It is how a policy-limited call restores the underlying
// declared order after filtering.
func policyRank(engines []Engine, id string) int {
	for i, e := range engines {
		if e.ID == id {
			return i
		}
	}
	return -1
}

// policyOrdered returns engines reordered so that every engine on the policy's
// preferred surface precedes engines on any other surface, preserving declared
// order within each group. It is pure and deterministic. Engines refused by the
// policy's hard fences are dropped, so the result is exactly the eligible set.
//
// This is what makes Policy a consumed seam rather than an inert manifest: it
// is the bridge between "the control plane published this preference" and "the
// fabric actually routed on it". Without it, a policy rule could be edited and
// change no routing outcome — a presence-without-invokability defect.
func (p Policy) policyOrdered(req TaskRequest, engines []Engine) []Engine {
	preferred, _ := p.preferredSurface(req)
	var onPreferred, other []Engine
	for _, e := range engines {
		if ok, _ := p.Admits(req, e); !ok {
			continue
		}
		surface := e.Surface
		if surface == "" {
			surface = SurfaceLocal
		}
		if surface == preferred {
			onPreferred = append(onPreferred, e)
			continue
		}
		other = append(other, e)
	}
	out := make([]Engine, 0, len(onPreferred)+len(other))
	out = append(out, onPreferred...)
	out = append(out, other...)
	return out
}

// RouteWithPolicy routes each request using the fabric's selection rule but with
// the engine set filtered and ordered by the policy: the policy's preferred
// surface is tried first (preserving declared order within it), and engines the
// policy refuses are removed before selection. The rejections returned for a
// request therefore include both the policy refusals and the fabric's
// first-eligible-wins skips.
//
// Route remains the policy-free primitive; RouteWithPolicy is the composed
// entrypoint a control-plane consumer should call when it has published a
// policy. The two share the same deterministic first-eligible-wins rule and the
// same closed reason vocabulary, so a decision made either way is replayable.
func RouteWithPolicy(requests []TaskRequest, engines []Engine, policy Policy) ([]Assignment, []Unrouted, error) {
	if err := policy.Validate(); err != nil {
		return nil, nil, err
	}
	assignments := make([]Assignment, 0, len(requests))
	unrouted := make([]Unrouted, 0)
	for _, req := range requests {
		preferred, rule := policy.preferredSurface(req)

		// Record the policy's own hard-fence refusals before selection strips
		// them, so the audit names every engine that was refused and why. A
		// refused engine must never silently vanish from the decision.
		var policyRejected []Rejection
		var policyAudit []string
		for _, e := range engines {
			ok, reason := policy.Admits(req, e)
			if ok {
				continue
			}
			policyRejected = append(policyRejected, Rejection{EngineID: e.ID, Reason: reason})
			policyAudit = append(policyAudit, fmt.Sprintf(
				"policy %s reject engine %s (%s): %s", policy.Version, e.ID, engineSurfaceLabel(e), reason))
		}

		ordered := policy.policyOrdered(req, engines)
		reqAssignments, reqUnrouted, err := Route([]TaskRequest{req}, ordered)
		if err != nil {
			return nil, nil, err
		}
		policyLine := fmt.Sprintf("policy %s preferred surface=%s rule=%s", policy.Version, preferred, rule)
		if len(reqAssignments) == 1 {
			a := reqAssignments[0]
			a.Reason = fmt.Sprintf("%s; policy surface=%s rule=%s", a.Reason, preferred, rule)
			a.Rejected = append(policyRejected, a.Rejected...)
			a.AssignmentAudit = append(a.AssignmentAudit, policyLine)
			a.AssignmentAudit = append(a.AssignmentAudit, policyAudit...)
			assignments = append(assignments, a)
			continue
		}
		u := reqUnrouted[0]
		u.Rejected = append(policyRejected, u.Rejected...)
		u.AssignmentAudit = append(u.AssignmentAudit, policyLine)
		u.AssignmentAudit = append(u.AssignmentAudit, policyAudit...)
		unrouted = append(unrouted, u)
	}
	return assignments, unrouted, nil
}

// RouteRequestWithPolicy is the single-request convenience form of
// RouteWithPolicy.
func RouteRequestWithPolicy(req TaskRequest, engines []Engine, policy Policy) (Assignment, []Rejection, error) {
	assignments, unrouted, err := RouteWithPolicy([]TaskRequest{req}, engines, policy)
	if err != nil {
		return Assignment{}, nil, err
	}
	if len(assignments) == 1 {
		return assignments[0], assignments[0].Rejected, nil
	}
	if len(unrouted) == 1 {
		return Assignment{}, unrouted[0].Rejected, nil
	}
	return Assignment{}, nil, fmt.Errorf("routing: no decision produced for request")
}
