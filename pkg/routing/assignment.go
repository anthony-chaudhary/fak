package routing

import (
	"fmt"
)

// AssignmentSchema identifies the assignment receipt emitted by Route.
const AssignmentSchema = "routing-assignment/1"

// Reason* are the closed rejection vocabulary for task routing: each engine
// skipped ahead of the winner records exactly one stable token.
const (
	// ReasonUnknownEngine marks an engine whose surface or identity is not a
	// known execution target.
	ReasonUnknownEngine = "UNKNOWN_ENGINE"
	// ReasonUnhealthy marks an engine reported not healthy.
	ReasonUnhealthy = "UNHEALTHY"
	// ReasonDraining marks an engine that is draining and takes no new work.
	ReasonDraining = "DRAINING"
	// ReasonRoleMismatch marks an engine that does not serve the required role.
	ReasonRoleMismatch = "ROLE_MISMATCH"
	// ReasonMissingTag marks an engine missing one or more required tags.
	ReasonMissingTag = "MISSING_TAG"
	// ReasonSurfaceExcluded marks an engine on a surface the task excluded.
	ReasonSurfaceExcluded = "SURFACE_EXCLUDED"
	// ReasonHostedRefused marks a hosted engine disqualified by require-local
	// intent, a hard fence.
	ReasonHostedRefused = "HOSTED_REFUSED"
	// ReasonNoHeadroom marks an engine whose capacity headroom is exhausted.
	ReasonNoHeadroom = "NO_HEADROOM"
	// ReasonNoEligibleEngine is the Unrouted reason when no engine was eligible;
	// it is a backlog fact, never a failure.
	ReasonNoEligibleEngine = "NO_ELIGIBLE_ENGINE"
	// ReasonUnknownRole marks a request naming a role outside the closed vocabulary.
	ReasonUnknownRole = "UNKNOWN_ROLE"
	// ReasonFailureFatal marks a request whose failure classified FATAL; only
	// Failover emits it.
	ReasonFailureFatal = "FAILURE_FATAL"
	// ReasonPostureDisabled marks an engine the published fleet posture removed
	// before the router ran; the posture-applying consumer
	// stamps it so operators can see which engines were removed.
	ReasonPostureDisabled = "POSTURE_DISABLED"
	// ReasonModelMismatch marks an engine whose model differs from an explicit
	// ManualOverride.Model; without an override, model never disqualifies.
	ReasonModelMismatch = "MODEL_MISMATCH"
	// ReasonContextTooSmall marks an engine whose ContextWindow is below (or
	// unknown against) ManualOverride.MinContextWindow.
	ReasonContextTooSmall = "CONTEXT_TOO_SMALL"
)

var reasons = []string{
	ReasonUnknownEngine,
	ReasonUnhealthy,
	ReasonDraining,
	ReasonRoleMismatch,
	ReasonMissingTag,
	ReasonSurfaceExcluded,
	ReasonHostedRefused,
	ReasonNoHeadroom,
	ReasonNoEligibleEngine,
	ReasonUnknownRole,
	ReasonFailureFatal,
	ReasonPostureDisabled,
	ReasonModelMismatch,
	ReasonContextTooSmall,
}

// Reasons returns a copy of every representable rejection reason.
func Reasons() []string {
	out := make([]string, len(reasons))
	copy(out, reasons)
	return out
}

// IsKnownReason reports whether r is a member of the closed rejection vocabulary.
func IsKnownReason(r string) bool {
	for _, known := range reasons {
		if r == known {
			return true
		}
	}
	return false
}

// roles is the closed role vocabulary; a role outside it is refused as
// ReasonUnknownRole rather than silently routed.
var roles = []string{"worker", "scout", "judge", "reviewer", "verifier", "balancer"}

// KnownRoles returns a copy of every representable role.
func KnownRoles() []string {
	out := make([]string, len(roles))
	copy(out, roles)
	return out
}

// IsKnownRole reports whether role is a member of the closed role vocabulary.
func IsKnownRole(role string) bool {
	for _, known := range roles {
		if role == known {
			return true
		}
	}
	return false
}

// TaskRequest is one unit of work to be placed on exactly one engine; its fields
// narrow the eligible engine set.
type TaskRequest struct {
	// Candidate is the task being routed. A nil candidate is unrouted as
	// ReasonUnknownEngine.
	Candidate *TaskCandidate `json:"candidate,omitempty"`
	// Role is the required archetype ("worker", "scout", "judge", ...).
	Role string `json:"role,omitempty"`
	// RequireTags lists capability tags every eligible engine must carry.
	RequireTags []string `json:"require_tags,omitempty"`
	// ExcludeSurfaces lists surfaces the task forbids.
	ExcludeSurfaces []Surface `json:"exclude_surfaces,omitempty"`
	// RequireLocal is a hard fence: never assign a hosted engine.
	RequireLocal bool `json:"require_local,omitempty"`
	// PinnedEngine narrows the candidates to the named engine; a pin naming an
	// engine outside the set leaves the request unrouted rather than falling back.
	// It is an input to one decision and never stored.
	PinnedEngine string `json:"pinned_engine,omitempty"`
	// Override is the manual bypass; every set field narrows and is named in the
	// audit. Prefer it over PinnedEngine; when both pin an engine they must agree.
	Override ManualOverride `json:"override,omitempty"`
}

// TicketID returns the candidate's ID, or "" when there is no candidate.
func (r TaskRequest) TicketID() string {
	if r.Candidate == nil {
		return ""
	}
	return r.Candidate.ID
}

// Rejection records one engine skipped ahead of the winner and its closed reason.
type Rejection struct {
	// EngineID is the rejected engine's ID.
	EngineID string `json:"engine_id"`
	// Reason is the closed rejection reason.
	Reason string `json:"reason"`
}

// Assignment is the routing decision for one request: the chosen engine, why it
// won, the engines rejected ahead of it, and an audit trail. It is inert; admission
// is the control plane's decision.
type Assignment struct {
	// Schema identifies the receipt shape.
	Schema string `json:"schema"`
	// TicketID is the routed task identity.
	TicketID string `json:"ticket_id"`
	// EngineID is the winning engine's ID.
	EngineID string `json:"engine_id"`
	// Surface is the winning engine's surface.
	Surface Surface `json:"surface"`
	// Model is the winning engine's model handle.
	Model string `json:"model,omitempty"`
	// Role is the role the assignment satisfies.
	Role string `json:"role,omitempty"`
	// Reason explains why this engine won, deterministically.
	Reason string `json:"reason"`
	// Rejected lists every engine skipped ahead of the winner, in consideration order.
	Rejected []Rejection `json:"rejected,omitempty"`
	// AssignmentAudit is the step-by-step trail of the decision.
	AssignmentAudit []string `json:"assignment_audit,omitempty"`
}

// Unrouted records a request that could not be placed; it is never a failure, and
// the task stays in the backlog. Reason is ReasonNoEligibleEngine,
// ReasonUnknownRole or ReasonUnknownEngine.
type Unrouted struct {
	// TicketID is the unrouted task identity ("" when there is no candidate).
	TicketID string `json:"ticket_id"`
	// Lane is the task's declared lane, for backlog legibility.
	Lane string `json:"lane,omitempty"`
	// Role is the role the request asked for.
	Role string `json:"role,omitempty"`
	// Reason is the closed reason the request could not be placed.
	Reason string `json:"reason"`
	// Rejected records every engine considered and why each was rejected.
	Rejected []Rejection `json:"rejected,omitempty"`
	// AssignmentAudit is the step-by-step trail of the refusal.
	AssignmentAudit []string `json:"assignment_audit,omitempty"`
}

// routeContext carries per-request routing state.
type routeContext struct {
	request    TaskRequest
	excludeSet map[Surface]bool
	audit      []string
}

// evaluateEngine returns the closed rejection reason for an engine, or "" when it
// is eligible. It is the single source of truth for eligibility, and the audit
// line derives from it. Override checks run after the hard fences and narrow only.
func (c *routeContext) evaluateEngine(e Engine) string {
	surface := e.Surface
	if surface == "" {
		surface = SurfaceLocal
	}
	if !surface.IsKnown() {
		return ReasonUnknownEngine
	}
	if c.request.RequireLocal && surface.Hosted() {
		return ReasonHostedRefused
	}
	if c.excludeSet[surface] {
		return ReasonSurfaceExcluded
	}
	ov := c.request.Override
	if ov.Surface != "" && surface != ov.Surface {
		return ReasonSurfaceExcluded
	}
	if !e.Healthy {
		return ReasonUnhealthy
	}
	if e.Draining {
		return ReasonDraining
	}
	if c.request.Role != "" && !e.HasRole(c.request.Role) {
		return ReasonRoleMismatch
	}
	if ov.Model != "" && e.Model != ov.Model {
		return ReasonModelMismatch
	}
	if ov.MinContextWindow > 0 && e.ContextWindow < ov.MinContextWindow {
		return ReasonContextTooSmall
	}
	if !e.HasTag(c.request.RequireTags) {
		return ReasonMissingTag
	}
	if e.CapacityHeadroom <= 0 {
		return ReasonNoHeadroom
	}
	return ""
}

func rejectionLine(e Engine, reason string) string {
	return fmt.Sprintf("reject engine %s (%s): %s", e.ID, engineSurfaceLabel(e), reason)
}

// engineSurfaceLabel defaults an unset surface to "local", as evaluateEngine does.
func engineSurfaceLabel(e Engine) string {
	if e.Surface == "" {
		return string(SurfaceLocal)
	}
	return string(e.Surface)
}

// Route places each request on the first eligible engine in declared order,
// returning an Assignment per placed request and an Unrouted entry per request
// that could not be placed. It errors only on caller bugs: a malformed engine set
// (empty or duplicate ID), a malformed ManualOverride, or conflicting pins.
// Priority and CapacityHeadroom are order inputs owned by the caller.
func Route(requests []TaskRequest, engines []Engine) ([]Assignment, []Unrouted, error) {
	seen := make(map[string]bool, len(engines))
	for _, e := range engines {
		if e.ID == "" {
			return nil, nil, fmt.Errorf("routing: engine with empty id")
		}
		if seen[e.ID] {
			return nil, nil, fmt.Errorf("routing: duplicate engine id %q", e.ID)
		}
		seen[e.ID] = true
	}
	for _, req := range requests {
		if err := req.Override.Validate(); err != nil {
			return nil, nil, err
		}
	}

	assignments := make([]Assignment, 0, len(requests))
	unrouted := make([]Unrouted, 0)

	for _, req := range requests {
		ticketID := req.TicketID()
		lane := ""
		if req.Candidate != nil {
			lane = req.Candidate.Lane
		}

		audit := make([]string, 0, len(engines)+2)
		if req.Candidate == nil {
			audit = append(audit, "unrouted: no candidate identity")
			unrouted = append(unrouted, Unrouted{
				TicketID:        ticketID,
				Lane:            lane,
				Role:            req.Role,
				Reason:          ReasonUnknownEngine,
				AssignmentAudit: audit,
			})
			continue
		}
		if req.Role != "" && !IsKnownRole(req.Role) {
			audit = append(audit, fmt.Sprintf("unrouted: unknown role %q", req.Role))
			unrouted = append(unrouted, Unrouted{
				TicketID:        ticketID,
				Lane:            lane,
				Role:            req.Role,
				Reason:          ReasonUnknownRole,
				AssignmentAudit: audit,
			})
			continue
		}

		excludeSet := make(map[Surface]bool, len(req.ExcludeSurfaces))
		for _, s := range req.ExcludeSurfaces {
			excludeSet[s] = true
		}

		// Applied here so every entry point (Route, RouteWithPolicy) honors the pin
		// identically; a pin outside the set narrows to nothing, never to another engine.
		pinned, err := effectivePinnedEngine(req.PinnedEngine, req.Override)
		if err != nil {
			return nil, nil, err
		}
		candidates := engines
		if pinned != "" {
			candidates = make([]Engine, 0, 1)
			for _, e := range engines {
				if e.ID == pinned {
					candidates = append(candidates, e)
				}
			}
		}

		ctx := &routeContext{
			request:    req,
			excludeSet: excludeSet,
			audit: append(audit, fmt.Sprintf(
				"route ticket %s role=%q engines=%d require_local=%t",
				ticketID, req.Role, len(candidates), req.RequireLocal)),
		}
		ctx.audit = append(ctx.audit, req.Override.auditLines()...)
		if pinned != "" {
			ctx.audit = append(ctx.audit, fmt.Sprintf(
				"pinned_engine=%s narrows candidates to %d", pinned, len(candidates)))
		}

		rejected := make([]Rejection, 0, len(candidates))
		var chosen *Engine
		for i := range candidates {
			e := candidates[i]
			reason := ctx.evaluateEngine(e)
			if reason == "" {
				chosen = &candidates[i]
				ctx.audit = append(ctx.audit, fmt.Sprintf(
					"select engine %s (%s) first eligible in declared order",
					e.ID, engineSurfaceLabel(e)))
				break
			}
			rejected = append(rejected, Rejection{EngineID: e.ID, Reason: reason})
			ctx.audit = append(ctx.audit, rejectionLine(e, reason))
		}

		if chosen == nil {
			ctx.audit = append(ctx.audit, "unrouted: no eligible engine; task stays in backlog")
			unrouted = append(unrouted, Unrouted{
				TicketID:        ticketID,
				Lane:            lane,
				Role:            req.Role,
				Reason:          ReasonNoEligibleEngine,
				Rejected:        rejected,
				AssignmentAudit: ctx.audit,
			})
			continue
		}

		assignments = append(assignments, Assignment{
			Schema:          AssignmentSchema,
			TicketID:        ticketID,
			EngineID:        chosen.ID,
			Surface:         chosen.Surface,
			Model:           chosen.Model,
			Role:            req.Role,
			Reason:          ctx.winnerReason(*chosen),
			Rejected:        rejected,
			AssignmentAudit: ctx.audit,
		})
	}

	return assignments, unrouted, nil
}

func (c *routeContext) winnerReason(e Engine) string {
	parts := fmt.Sprintf("first eligible engine in declared order (surface=%s priority=%d headroom=%.2f)",
		engineSurfaceLabel(e), e.Priority, e.CapacityHeadroom)
	if c.request.RequireLocal {
		parts += "; require_local satisfied"
	}
	return parts
}
