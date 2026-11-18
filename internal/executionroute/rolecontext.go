package executionroute

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
)

// rolecontext.go — the ROLE-KEYED CONTEXT-WINDOW PROFILE registry.
//
// The delegation axis and the context-window axis were, until now, two
// independent spines that never met. executionroute owns WHO does the work (the
// closed Role set: scout / worker / judge / primary) and how much of each
// consumption axis a role may spend (Budget). ctxplan owns HOW MUCH CONTEXT a
// turn materializes (the four-area Layout: base / current / recent / deep, each
// an AreaPolicy{MaxSpans, MaxTokens, Precision}). Nothing bound them, so a
// role's context window was whatever the caller happened to pass per call — an
// ad-hoc budget, not a predefined type.
//
// This file is that binding, and only that binding. A RoleContextProfile names a
// role's context window as a first-class, inspectable TYPE: a named Layout, the
// resident token budget it is planned against, and the precision posture that
// says how aggressively the window trades exact bytes for recoverable pointers.
// The profiles are a fixed, compile-time table — one entry per member of the
// CLOSED Role set — so the window a role gets is a property of the role, not of
// the call site.
//
// Why the profiles differ (the axis the operator named: "predefined types of
// context Windows like the way you have the implementer and the judge"):
//
//   - scout probes cheaply and must not pay for history it will not read, so it
//     gets a NARROW window: few recent spans, a shallow deep probe, and a
//     pointer-heavy posture that keeps the window small by eliding cold spans
//     into recoverable pointers.
//   - worker is the implementer: it needs the exact task statement and the exact
//     current file state, plus enough recent tail to stay coherent, but it is
//     bounded — a wide deep-history probe would let it wander.
//   - judge adjudicates: it must see the workers' outputs AND the evidence they
//     were judged against, so it gets the WIDEST deep probe and an exact
//     posture on the base (the rubric/contract must be verbatim, never a
//     pointer).
//   - primary drives the turn: it needs the broadest recent tail to hold the
//     conversation together, but its deep history is planned rather than pinned,
//     because the driver delegates depth to the other roles.
//
// The table is deliberately NOT a free-text map: adding a role is a new constant
// plus a knownRole arm plus a row here, and TestRoleContextProfileRegistryIsClosed
// fails if a role ships without a profile or a profile ships without a role.

// RoleContextProfile is one role's predefined context-window TYPE: the named
// four-area Layout the role's turns are planned against, the resident token
// budget that Layout is planned within, and the precision posture that names how
// the window trades exact bytes for recoverable pointers.
//
// ResidentTokens is the role's default resident budget — the tokens fak tries to
// materialize for a turn of this role. It is a TARGET, never the provider's hard
// cap: the cap is a ceiling the window must stay under, and the ctxplan envelope
// path (ctxplan.EnvelopeForModel) still governs the model-level ceiling. A zero
// ResidentTokens means "no role-specific target; defer to the model envelope".
type RoleContextProfile struct {
	// Name is the stable, human-readable identity of this window type. It is the
	// string a trace, a receipt, or an operator reads to know which predefined
	// window a role ran under, so it is part of the contract and must not be
	// derived from the Layout at read time.
	Name string `json:"name"`
	// Role is the closed delegation role this profile is keyed to.
	Role Role `json:"role"`
	// Layout is the four-area context profile this role's turns are planned
	// against. It is a value, not a pointer: a resolved profile is a snapshot the
	// caller cannot mutate back into the registry.
	Layout ctxplan.Layout `json:"layout"`
	// ResidentTokens is the role's default resident token target (see the type
	// doc). Zero defers to the model envelope.
	ResidentTokens int `json:"resident_tokens,omitempty"`
	// PrecisionPosture names the window's overall exactness stance in one word
	// (see the Posture* constants). It is the summary an operator reads; the
	// per-area Precision fields in Layout are the mechanism.
	PrecisionPosture string `json:"precision_posture"`
	// Rationale is why THIS role gets THIS window. It is required: a profile with
	// no stated reason is an unexplained budget, which is the state this file
	// exists to end.
	Rationale string `json:"rationale"`
}

// PrecisionPosture is the one-word summary of how a role's window trades exact
// bytes for recoverable pointers. It is a CLOSED set: the per-area precisions in
// Layout are the mechanism, and the posture is the label a reader uses to compare
// two roles' windows at a glance.
const (
	// PostureNarrow is a small, pointer-heavy window: few spans, cold history
	// elided into recoverable pointers. For roles that probe and must stay cheap.
	PostureNarrow = "narrow"
	// PostureBounded is a moderate window with exact task/current state and a
	// planned (not pinned) history tail. For roles that do bounded work.
	PostureBounded = "bounded"
	// PostureWide is the broadest window with an exact base: the role must see the
	// evidence verbatim, not as a pointer. For roles that adjudicate.
	PostureWide = "wide"
	// PostureDriving is a broad recent tail with planned deep history: the role
	// holds the conversation together and delegates depth to the others.
	PostureDriving = "driving"
)

// knownPosture reports whether p is one of the closed PrecisionPosture set.
func knownPosture(p string) bool {
	switch p {
	case PostureNarrow, PostureBounded, PostureWide, PostureDriving:
		return true
	}
	return false
}

// DefaultRoleContextProfiles returns the predefined context-window TYPE for every
// member of the closed Role set, in declared role order (scout, worker, judge,
// primary). The table is fixed and compile-time: it is the definition of "a
// role's context window is a predefined type", so it has no append path and no
// caller-supplied key.
//
// The returned slice is a fresh copy of the table, so a caller cannot mutate the
// registry through it.
func DefaultRoleContextProfiles() []RoleContextProfile {
	return []RoleContextProfile{
		{
			Name: "scout-probe-narrow",
			Role: RoleScout,
			// Narrow: a short recent tail, a shallow deep probe, and a pointer
			// posture on both history areas so cold spans cost a pointer, not bytes.
			Layout: ctxplan.Layout{
				Base:    ctxplan.AreaPolicy{MaxSpans: 4, Precision: ctxplan.PrecisionExact},
				Current: ctxplan.AreaPolicy{MaxSpans: 1, Precision: ctxplan.PrecisionExact},
				Recent:  ctxplan.AreaPolicy{MaxSpans: 8, Precision: ctxplan.PrecisionPointer},
				Deep:    ctxplan.AreaPolicy{MaxSpans: 8, Precision: ctxplan.PrecisionPointer},
			},
			ResidentTokens:   2000,
			PrecisionPosture: PostureNarrow,
			Rationale:        "a scout classifies before the expensive roles commit, so it must not pay for history it will not read; cold spans become recoverable pointers rather than resident bytes",
		},
		{
			Name: "worker-implementer-bounded",
			Role: RoleWorker,
			// Bounded: exact task statement and exact current state, a moderate
			// planned recent tail, and a shallow planned deep probe. The worker is
			// the implementer, so its window is wide enough to act and narrow
			// enough that it cannot wander into unrelated history.
			Layout: ctxplan.Layout{
				Base:    ctxplan.AreaPolicy{MaxSpans: 8, Precision: ctxplan.PrecisionExact},
				Current: ctxplan.AreaPolicy{MaxSpans: 1, Precision: ctxplan.PrecisionExact},
				Recent:  ctxplan.AreaPolicy{MaxSpans: 16, Precision: ctxplan.PrecisionPlanned},
				Deep:    ctxplan.AreaPolicy{MaxSpans: 16, Precision: ctxplan.PrecisionPlanned},
			},
			ResidentTokens:   8000,
			PrecisionPosture: PostureBounded,
			Rationale:        "the implementer needs the exact task and exact current file state plus enough recent tail to stay coherent, but a wide deep-history probe would let a bounded doer wander",
		},
		{
			Name: "judge-adjudicator-wide",
			Role: RoleJudge,
			// Wide: the broadest deep probe, and an EXACT base — the rubric or
			// contract the judge scores against must be verbatim, never a pointer.
			Layout: ctxplan.Layout{
				Base:    ctxplan.AreaPolicy{MaxSpans: 16, Precision: ctxplan.PrecisionExact},
				Current: ctxplan.AreaPolicy{MaxSpans: 1, Precision: ctxplan.PrecisionExact},
				Recent:  ctxplan.AreaPolicy{MaxSpans: 32, Precision: ctxplan.PrecisionPlanned},
				Deep:    ctxplan.AreaPolicy{MaxSpans: 64, Precision: ctxplan.PrecisionPlanned},
			},
			ResidentTokens:   16000,
			PrecisionPosture: PostureWide,
			Rationale:        "the adjudicator must see the workers' outputs AND the evidence they were judged against, so it gets the widest deep probe and an exact base: a rubric read as a pointer is not a rubric",
		},
		{
			Name: "primary-driver-driving",
			Role: RolePrimary,
			// Driving: the broadest recent tail (the driver holds the conversation
			// together) with planned deep history, because the driver delegates
			// depth to the other roles rather than reading it itself.
			Layout: ctxplan.Layout{
				Base:    ctxplan.AreaPolicy{MaxSpans: 8, Precision: ctxplan.PrecisionExact},
				Current: ctxplan.AreaPolicy{MaxSpans: 1, Precision: ctxplan.PrecisionExact},
				Recent:  ctxplan.AreaPolicy{MaxSpans: 48, Precision: ctxplan.PrecisionPlanned},
				Deep:    ctxplan.AreaPolicy{MaxSpans: 32, Precision: ctxplan.PrecisionPlanned},
			},
			ResidentTokens:   16000,
			PrecisionPosture: PostureDriving,
			Rationale:        "the driver owns the turn and must hold the conversation together, so it gets the broadest recent tail, but its deep history stays planned because it delegates depth to the other roles",
		},
	}
}

// ContextProfileForRole resolves the predefined context-window profile for role
// r. The second return is false when r is not a member of the closed Role set or
// has no registered profile — callers MUST treat false as a refusal, never as a
// zero Layout, because a zero Layout silently plans against ctxplan defaults and
// would make an unprofiled role look profiled.
func ContextProfileForRole(r Role) (RoleContextProfile, bool) {
	for _, p := range DefaultRoleContextProfiles() {
		if p.Role == r {
			return p, true
		}
	}
	return RoleContextProfile{}, false
}

// resolveContextProfile resolves the context-window TYPE a declared RolePlan runs
// under: the plan's explicit ContextProfile override when set, else the role's
// registered default. It is fail-closed in both directions — an override naming
// an unregistered profile is an error, and a role with no registered default is
// an error — so a role can never run under an implicit zero Layout.
func resolveContextProfile(p RolePlan) (RoleContextProfile, error) {
	if p.ContextProfile == "" {
		profile, ok := ContextProfileForRole(p.Role)
		if !ok {
			return RoleContextProfile{}, fmt.Errorf("execution route: role %q has no registered context profile", p.Role)
		}
		return profile, nil
	}
	for _, profile := range DefaultRoleContextProfiles() {
		if profile.Name == p.ContextProfile {
			return profile, nil
		}
	}
	return RoleContextProfile{}, fmt.Errorf("execution route: role %q names unknown context profile %q", p.Role, p.ContextProfile)
}

// ValidateRoleContextProfiles checks the registry's two closure properties: every
// member of the closed Role set has exactly one profile, and every profile names
// a member of the closed Role set. It also rejects a profile that is missing its
// name, rationale, or a posture outside the closed set. It returns the first
// violation, so a caller (or a test) gets a named defect rather than a bool.
func ValidateRoleContextProfiles(profiles []RoleContextProfile) error {
	seen := make(map[Role]bool, len(profiles))
	for _, p := range profiles {
		if !knownRole(p.Role) {
			return fmt.Errorf("role context profile %q names unknown role %q", p.Name, p.Role)
		}
		if seen[p.Role] {
			return fmt.Errorf("role %q is declared more than once in the context profile registry", p.Role)
		}
		seen[p.Role] = true
		if p.Name == "" {
			return fmt.Errorf("role %q has a context profile with no name", p.Role)
		}
		if p.Rationale == "" {
			return fmt.Errorf("role context profile %q has no rationale", p.Name)
		}
		if !knownPosture(p.PrecisionPosture) {
			return fmt.Errorf("role context profile %q has unknown precision posture %q", p.Name, p.PrecisionPosture)
		}
	}
	for _, r := range []Role{RoleScout, RoleWorker, RoleJudge, RolePrimary} {
		if !seen[r] {
			return fmt.Errorf("role %q has no context profile; every closed role must declare one", r)
		}
	}
	return nil
}
