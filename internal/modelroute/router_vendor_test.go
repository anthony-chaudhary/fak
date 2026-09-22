package modelroute

import "testing"

// TestRouterVendorRegistration pins router.com as a first-class vendor kind: its
// default endpoint, membership in the closed kind set, a minimal roster passing
// Validate, and the resolved Target's wire/locality facts.
func TestRouterVendorRegistration(t *testing.T) {
	if got := KindBaseURL(KindRouter); got != RouterOpenAIBaseURL {
		t.Fatalf("KindBaseURL(KindRouter) = %q, want %q", got, RouterOpenAIBaseURL)
	}
	if !knownKind(KindRouter) {
		t.Fatalf("knownKind(KindRouter) = false, want true")
	}
	if KindRouter != ProviderKind(RouterProviderKey) {
		t.Fatalf("KindRouter = %q, want %q", KindRouter, RouterProviderKey)
	}

	roster := Roster{
		Version:  RosterVersion,
		Accounts: []Account{{ID: "router", Kind: KindRouter, CredEnv: RouterAPIKeyEnv}},
		Bindings: []Binding{{Model: "router", Account: "router"}},
	}
	if err := roster.Validate(); err != nil {
		t.Fatalf("Validate() = %v, want nil", err)
	}

	target, err := roster.Resolve("router")
	if err != nil {
		t.Fatalf("Resolve(router) = %v", err)
	}
	if target.Kind != KindRouter {
		t.Fatalf("Target.Kind = %q, want %q", target.Kind, KindRouter)
	}
	if target.BaseURL != RouterOpenAIBaseURL {
		t.Fatalf("Target.BaseURL = %q, want %q", target.BaseURL, RouterOpenAIBaseURL)
	}
	if !target.Remote() {
		t.Fatalf("Target.Remote() = false, want true")
	}
	if target.Zone() != ZoneVendor {
		t.Fatalf("Target.Zone() = %q, want %q", target.Zone(), ZoneVendor)
	}
}

// TestDefaultRosterRegistersRouter proves the shipped default roster carries the
// router vendor account, not just the closed kind set.
func TestDefaultRosterRegistersRouter(t *testing.T) {
	roster := DefaultRoster()
	if err := roster.Validate(); err != nil {
		t.Fatalf("DefaultRoster().Validate() = %v", err)
	}
	found := false
	for _, a := range roster.Accounts {
		if a.Kind == KindRouter {
			found = true
			if a.CredEnv != RouterAPIKeyEnv {
				t.Fatalf("router account cred_env = %q, want %q", a.CredEnv, RouterAPIKeyEnv)
			}
		}
	}
	if !found {
		t.Fatalf("DefaultRoster() has no KindRouter account")
	}
}
