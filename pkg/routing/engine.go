package routing

// Surface identifies the execution surface an engine runs on. It is the coarse
// "where" of an engine, before any specific host or model is named.
type Surface string

const (
	// SurfaceLocal is an in-process or on-box engine on the operator's machine.
	SurfaceLocal Surface = "local"
	// SurfaceRemote is an engine reached on another machine on the LAN.
	SurfaceRemote Surface = "remote"
	// SurfaceHosted is a third-party hosted engine with external egress.
	SurfaceHosted Surface = "hosted"
	// SurfaceAppliance is a dedicated on-premises appliance (e.g. a local GPU box).
	SurfaceAppliance Surface = "appliance"
)

// surfaces is the single source of truth for the closed surface vocabulary, in
// declaration order. Surfaces returns a copy.
var surfaces = []Surface{
	SurfaceLocal,
	SurfaceRemote,
	SurfaceHosted,
	SurfaceAppliance,
}

// Surfaces returns every representable surface (for a witness/registry test
// that the vocabulary stays closed).
func Surfaces() []Surface {
	out := make([]Surface, len(surfaces))
	copy(out, surfaces)
	return out
}

// IsKnown reports whether s is a member of the closed surface vocabulary. An
// unknown surface is a refusal, not an invitation to invent a rule at runtime.
func (s Surface) IsKnown() bool {
	for _, known := range surfaces {
		if s == known {
			return true
		}
	}
	return false
}

// Hosted reports whether s is an external-egress surface. It is the predicate
// behind the require-local hard fence: a hosted engine can never serve a task
// that demands local execution.
func (s Surface) Hosted() bool {
	return s == SurfaceHosted
}

// Engine is one candidate execution target: a surface plus the model it serves,
// the roles it can fill, and its live health/capacity state. It is pure data:
// Route reads it and never mutates it.
type Engine struct {
	// ID is the stable identity of the engine, unique within one routing call.
	ID string `json:"id"`
	// Surface is the execution surface the engine runs on.
	Surface Surface `json:"surface"`
	// Model is the model or engine handle exposed at this target.
	Model string `json:"model,omitempty"`
	// Roles lists the archetypes this engine can serve ("worker", "scout",
	// "judge", ...). An empty Roles list means "any role".
	Roles []string `json:"roles,omitempty"`
	// Healthy is false when the engine is known-impaired. A healthy engine is
	// required for every assignment.
	Healthy bool `json:"healthy"`
	// Draining is true when the engine is being retired: it accepts no new work.
	Draining bool `json:"draining,omitempty"`
	// CapacityHeadroom is the free fraction of capacity, 0..1. Zero headroom
	// means the engine admits no further work.
	CapacityHeadroom float64 `json:"capacity_headroom,omitempty"`
	// Tags are free-form capability labels a task may require.
	Tags []string `json:"tags,omitempty"`
	// Priority orders engines when more than one is eligible: the higher
	// Priority wins. On an exact tie, declared order is preserved.
	Priority int `json:"priority,omitempty"`
	// ContextWindow is the engine's context capacity in tokens. Zero means
	// unknown/unadvertised: the engine fits no explicit MinContextWindow
	// override (fail-closed), but participates normally when no window is
	// required. Agents override it per-decision via ManualOverride; operators
	// describe it per-engine via the route table / CLI.
	ContextWindow int64 `json:"context_window,omitempty"`
}

// HasRole reports whether the engine can serve the given role. An engine that
// lists no roles serves any role.
func (e Engine) HasRole(role string) bool {
	if len(e.Roles) == 0 {
		return true
	}
	for _, r := range e.Roles {
		if r == role {
			return true
		}
	}
	return false
}

// HasTag reports whether the engine carries every one of the required tags.
// Comparison is exact; an empty requirement is satisfied by any engine.
func (e Engine) HasTag(required []string) bool {
	if len(required) == 0 {
		return true
	}
	have := make(map[string]bool, len(e.Tags))
	for _, t := range e.Tags {
		have[t] = true
	}
	for _, r := range required {
		if !have[r] {
			return false
		}
	}
	return true
}
