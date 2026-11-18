package executionroute

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/ctxplan"
	"github.com/anthony-chaudhary/fak/internal/modelroute"
)

// rolecontext_test.go — the witness for the role-keyed context-window profile
// registry. The claim under test is that a role's context window is a predefined
// TYPE: every member of the closed Role set resolves to a named profile, the
// profiles are genuinely distinct along the axis the operator named (the
// implementer and the judge do NOT get the same window), and the binding is
// fail-closed so an unprofiled role can never run under an implicit zero Layout.

// TestRoleContextProfileRegistryIsClosed proves the registry's two closure
// properties: every member of the closed Role set has exactly one profile, and
// every profile names a member of the closed Role set. A new role shipped
// without a profile — or a profile left behind by a removed role — fails here
// rather than silently resolving to a zero Layout at runtime.
func TestRoleContextProfileRegistryIsClosed(t *testing.T) {
	profiles := DefaultRoleContextProfiles()
	if err := ValidateRoleContextProfiles(profiles); err != nil {
		t.Fatalf("default role context profile registry is not closed: %v", err)
	}
	// The closed Role set is the authority; the registry must cover exactly it.
	for _, r := range []Role{RoleScout, RoleWorker, RoleJudge, RolePrimary} {
		if !knownRole(r) {
			t.Fatalf("fixture drift: %q is no longer a member of the closed Role set", r)
		}
		if _, ok := ContextProfileForRole(r); !ok {
			t.Errorf("role %q has no registered context profile", r)
		}
	}
	if len(profiles) != 4 {
		t.Errorf("registry has %d profiles; the closed Role set has 4 members", len(profiles))
	}
}

// TestRoleContextProfilePerRoleDistinct proves the profiles are distinct TYPES,
// not one window relabelled four times: each role resolves to its own name, its
// own precision posture, and its own Layout. This is the operator's axis — "the
// implementer and the judge" must not share a window.
func TestRoleContextProfilePerRoleDistinct(t *testing.T) {
	names := map[string]Role{}
	postures := map[string]Role{}
	for _, r := range []Role{RoleScout, RoleWorker, RoleJudge, RolePrimary} {
		p, ok := ContextProfileForRole(r)
		if !ok {
			t.Fatalf("role %q has no registered context profile", r)
		}
		if p.Role != r {
			t.Errorf("profile %q is keyed to role %q, resolved for %q", p.Name, p.Role, r)
		}
		if p.Name == "" || p.Rationale == "" {
			t.Errorf("role %q profile must carry a name and a rationale", r)
		}
		if !knownPosture(p.PrecisionPosture) {
			t.Errorf("role %q profile has unknown posture %q", r, p.PrecisionPosture)
		}
		if prev, dup := names[p.Name]; dup {
			t.Errorf("roles %q and %q share profile name %q", prev, r, p.Name)
		}
		names[p.Name] = r
		if prev, dup := postures[p.PrecisionPosture]; dup {
			t.Errorf("roles %q and %q share precision posture %q", prev, r, p.PrecisionPosture)
		}
		postures[p.PrecisionPosture] = r
	}

	// The implementer and the judge are the two roles the operator named. They
	// must differ in the mechanism, not only in the label: the judge's deep probe
	// must be strictly wider than the worker's, and the judge's base must be
	// exact (a rubric read as a pointer is not a rubric).
	worker, _ := ContextProfileForRole(RoleWorker)
	judge, _ := ContextProfileForRole(RoleJudge)
	if judge.Layout.Deep.MaxSpans <= worker.Layout.Deep.MaxSpans {
		t.Errorf("judge deep probe %d must be strictly wider than worker's %d",
			judge.Layout.Deep.MaxSpans, worker.Layout.Deep.MaxSpans)
	}
	if judge.Layout.Base.Precision != ctxplan.PrecisionExact {
		t.Errorf("judge base precision = %q; the adjudicated evidence must be exact", judge.Layout.Base.Precision)
	}
	if judge.ResidentTokens <= worker.ResidentTokens {
		t.Errorf("judge resident target %d must exceed worker's %d", judge.ResidentTokens, worker.ResidentTokens)
	}

	// The scout is the cheap probe: it must be the narrowest window, and its
	// history areas must be pointers rather than resident bytes.
	scout, _ := ContextProfileForRole(RoleScout)
	if scout.PrecisionPosture != PostureNarrow {
		t.Errorf("scout posture = %q, want %q", scout.PrecisionPosture, PostureNarrow)
	}
	if scout.Layout.Recent.Precision != ctxplan.PrecisionPointer || scout.Layout.Deep.Precision != ctxplan.PrecisionPointer {
		t.Errorf("scout history areas must be pointer-precision, got recent=%q deep=%q",
			scout.Layout.Recent.Precision, scout.Layout.Deep.Precision)
	}
	for _, other := range []RoleContextProfile{worker, judge} {
		if scout.Layout.Deep.MaxSpans >= other.Layout.Deep.MaxSpans {
			t.Errorf("scout deep probe %d must be narrower than %s's %d",
				scout.Layout.Deep.MaxSpans, other.Name, other.Layout.Deep.MaxSpans)
		}
	}
}

// TestRouteRolesResolvesContextProfile proves the binding is live on the routing
// path: a resolved RoleDecision carries the role's predefined window, so the
// window a role ran under is auditable from the envelope rather than implicit in
// the call site.
func TestRouteRolesResolvesContextProfile(t *testing.T) {
	env, err := RouteRoles(multiRolePlan(), modelroute.DefaultManifest(), nil)
	if err != nil {
		t.Fatalf("RouteRoles: %v", err)
	}
	for _, dec := range env.Roles {
		want, ok := ContextProfileForRole(dec.Role)
		if !ok {
			t.Fatalf("role %q has no registered context profile", dec.Role)
		}
		if dec.ContextProfile.Name != want.Name {
			t.Errorf("role %q resolved profile %q, want %q", dec.Role, dec.ContextProfile.Name, want.Name)
		}
		if dec.ContextProfile.Layout.Deep.MaxSpans != want.Layout.Deep.MaxSpans {
			t.Errorf("role %q resolved deep probe %d, want %d",
				dec.Role, dec.ContextProfile.Layout.Deep.MaxSpans, want.Layout.Deep.MaxSpans)
		}
	}
}

// TestRouteRolesHonorsContextProfileOverride proves a plan may name a different
// registered window for a role, and that the override is what lands in the
// decision — the registry is a default, not a cage.
func TestRouteRolesHonorsContextProfileOverride(t *testing.T) {
	set := multiRolePlan()
	for i := range set.Plans {
		if set.Plans[i].Role == RoleWorker {
			set.Plans[i].ContextProfile = "judge-adjudicator-wide"
		}
	}
	env, err := RouteRoles(set, modelroute.DefaultManifest(), nil)
	if err != nil {
		t.Fatalf("RouteRoles: %v", err)
	}
	dec, ok := env.Role(RoleWorker)
	if !ok {
		t.Fatal("worker role missing from envelope")
	}
	if dec.ContextProfile.Name != "judge-adjudicator-wide" {
		t.Errorf("worker override resolved %q, want %q", dec.ContextProfile.Name, "judge-adjudicator-wide")
	}
}

// TestRouteRolesRejectsUnknownContextProfile proves the binding is FAIL-CLOSED:
// a plan naming an unregistered window is refused, never silently downgraded to
// the role's default or to a zero Layout.
func TestRouteRolesRejectsUnknownContextProfile(t *testing.T) {
	set := multiRolePlan()
	set.Plans[0].ContextProfile = "no-such-window-type"
	if _, err := RouteRoles(set, modelroute.DefaultManifest(), nil); err == nil {
		t.Fatal("RouteRoles accepted an unregistered context profile; the binding must be fail-closed")
	}
}

// TestValidateRoleContextProfilesRejectsDefects proves the validator names each
// defect class rather than returning a bare bool: an unknown role, a duplicate
// role, a missing name, a missing rationale, an unknown posture, and a missing
// closed-set member are each refused.
func TestValidateRoleContextProfilesRejectsDefects(t *testing.T) {
	base := DefaultRoleContextProfiles()
	cases := []struct {
		name   string
		mutate func([]RoleContextProfile) []RoleContextProfile
	}{
		{"unknown role", func(p []RoleContextProfile) []RoleContextProfile {
			p[0].Role = Role("ghost")
			return p
		}},
		{"duplicate role", func(p []RoleContextProfile) []RoleContextProfile {
			p[1].Role = p[0].Role
			return p
		}},
		{"missing name", func(p []RoleContextProfile) []RoleContextProfile {
			p[0].Name = ""
			return p
		}},
		{"missing rationale", func(p []RoleContextProfile) []RoleContextProfile {
			p[0].Rationale = ""
			return p
		}},
		{"unknown posture", func(p []RoleContextProfile) []RoleContextProfile {
			p[0].PrecisionPosture = "vibes"
			return p
		}},
		{"missing closed-set member", func(p []RoleContextProfile) []RoleContextProfile {
			return p[:len(p)-1]
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			profiles := append([]RoleContextProfile(nil), base...)
			if err := ValidateRoleContextProfiles(tc.mutate(profiles)); err == nil {
				t.Errorf("validator accepted a registry with %s", tc.name)
			}
		})
	}
}
