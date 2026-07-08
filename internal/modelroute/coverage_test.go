package modelroute

import (
	"reflect"
	"testing"
)

// TestManifestModelIDs checks the routed-id enumeration is distinct, sorted, and
// spans members + scouts across the rules and the default plan.
func TestManifestModelIDs(t *testing.T) {
	m := Manifest{
		Default: Plan{Members: []Member{{Model: "large"}}},
		Rules: []Rule{
			{Name: "guard", Plan: Plan{Members: []Member{{Model: "guard-a"}, {Model: "guard-b"}}, Scout: "scout-x"}},
			{Name: "cheap", Plan: Plan{Members: []Member{{Model: "small"}}}},
			{Name: "dup", Plan: Plan{Members: []Member{{Model: "small"}}}}, // repeat collapses
		},
	}
	got := m.ModelIDs()
	want := []string{"guard-a", "guard-b", "large", "scout-x", "small"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ModelIDs = %v, want %v", got, want)
	}
}

// TestRosterCoverUnboundAndDefault checks the three dispositions: an explicit
// binding is bound, an unbound id with no default is UNBOUND (fail-closed), and the
// same roster with a default serves the rest via-default.
func TestRosterCoverUnboundAndDefault(t *testing.T) {
	ids := []string{"large", "scout-x", "small"}
	acct := Account{ID: "a", Kind: KindOpenAI, CredEnv: "OPENAI_API_KEY"}

	noDefault := Roster{Accounts: []Account{acct}, Bindings: []Binding{{Model: "small", Account: "a"}}}
	cov := noDefault.Cover(ids)
	if cov.Bound != 1 || cov.Default != 0 || cov.Unbound != 2 {
		t.Fatalf("no-default cover = %+v, want bound=1 default=0 unbound=2", cov)
	}
	if cov.Rows[2].Model != "small" || cov.Rows[2].Status != CoverageBound || cov.Rows[2].Account != "a" {
		t.Fatalf("small row = %+v, want bound to a", cov.Rows[2])
	}
	if cov.Rows[0].Status != CoverageUnbound {
		t.Fatalf("large row = %+v, want unbound", cov.Rows[0])
	}

	withDefault := noDefault
	withDefault.Default = "a"
	cov2 := withDefault.Cover(ids)
	if cov2.Bound != 1 || cov2.Default != 2 || cov2.Unbound != 0 {
		t.Fatalf("with-default cover = %+v, want bound=1 default=2 unbound=0", cov2)
	}
	if cov2.Rows[0].Status != CoverageViaDefault || cov2.Rows[0].Account != "a" {
		t.Fatalf("large row = %+v, want via-default -> a", cov2.Rows[0])
	}
}

// TestRosterCoverDefaultRosterCoversDefaultManifest is the product invariant: the
// starter roster `fak route --accounts-dump` emits fully covers the built-in routing
// manifest, so a user who edits neither sees no unbound routes.
func TestRosterCoverDefaultRosterCoversDefaultManifest(t *testing.T) {
	cov := DefaultRoster().Cover(DefaultManifest().ModelIDs())
	if cov.Unbound != 0 {
		t.Fatalf("the default roster must cover the default manifest, got %d unbound: %+v", cov.Unbound, cov.Rows)
	}
}
