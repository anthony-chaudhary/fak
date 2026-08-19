package goalregistry

import (
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestStableIdentityAcrossEditsAliasesAndNameCollision(t *testing.T) {
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json"), Now: func() time.Time { return now }}
	p := Provenance{Actor: "operator", Authority: "operator-declared", Witness: "ticket-1"}
	g1, err := s.Create("Improve observability", "safe summary", p, nil)
	if err != nil {
		t.Fatal(err)
	}
	g2, err := s.Create("Improve observability", "same title, distinct intent", p, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g1.GoalID == g2.GoalID || !strings.HasPrefix(g1.GoalID, "goal_") {
		t.Fatalf("opaque IDs not distinct: %q %q", g1.GoalID, g2.GoalID)
	}
	updated, err := s.Update(g1.GoalID, "Observe the fleet", "edited", Active)
	if err != nil {
		t.Fatal(err)
	}
	if updated.GoalID != g1.GoalID {
		t.Fatalf("title edit changed identity: %q -> %q", g1.GoalID, updated.GoalID)
	}
	for _, alias := range []struct{ ns, id string }{{"fak:trajctl", "objective-7"}, {"claude:goal", "g-1"}, {"codex:goal", "g-1"}, {"github:issue", "anthony-chaudhary/fak#6663"}, {"dos:unit", "unit-9"}} {
		if _, err := s.Bind(g1.GoalID, alias.ns, alias.id, "", p); err != nil {
			t.Fatalf("bind %s: %v", alias.ns, err)
		}
	}
	shown, bindings, err := s.Show(g1.GoalID)
	if err != nil {
		t.Fatal(err)
	}
	if shown.Title != "Observe the fleet" || len(bindings) != 5 {
		t.Fatalf("show = %+v bindings=%d", shown, len(bindings))
	}
}

func TestBindingCollisionRefusesSilentMerge(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json")}
	p := Provenance{Actor: "operator", Authority: "operator-declared"}
	a, _ := s.Create("A", "", p, nil)
	b, _ := s.Create("B", "", p, nil)
	if _, err := s.Bind(a.GoalID, "github:issue", "repo#1", "", p); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Bind(b.GoalID, "github:issue", "repo#1", "", p); err == nil || !strings.Contains(err.Error(), "collision") {
		t.Fatalf("want collision, got %v", err)
	}
}

func TestRelationsAreNotExecutionParentageAndLifecycleIsExplicit(t *testing.T) {
	now := time.Date(2026, 8, 19, 12, 0, 0, 0, time.UTC)
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json"), Now: func() time.Time { return now }}
	p := Provenance{Actor: "operator", Authority: "independent-witness", Witness: "sha256:abc"}
	g, err := s.Create("Child intent", "", p, []Relation{{Kind: "derived_from", GoalID: "goal_parent"}})
	if err != nil {
		t.Fatal(err)
	}
	now = now.Add(time.Hour)
	g, err = s.Update(g.GoalID, g.Title, g.Summary, Achieved)
	if err != nil {
		t.Fatal(err)
	}
	if g.Lifecycle != Achieved || !g.UpdatedAt.Equal(now) || g.Relations[0].Kind != "derived_from" {
		t.Fatalf("updated goal = %+v", g)
	}
}

func TestResolveExplicitHarnessBindings(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json"), Now: func() time.Time { return time.Unix(1700000000, 0).UTC() }}
	g, err := s.Create("Observe fleet", "", Provenance{Actor: "operator", Authority: "user"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, b := range []struct{ namespace, externalID string }{
		{"claude:goal", "claude-goal-7"}, {"codex:goal", "codex-goal-9"},
		{"fak:trajctl", "objective-11"}, {"github:issue", "6662"}, {"dos:unit", "unit-4"},
	} {
		if _, err := s.Bind(g.GoalID, b.namespace, b.externalID, "", Provenance{Actor: "adapter", Authority: "harness"}); err != nil {
			t.Fatal(err)
		}
		got, binding, err := s.Resolve(b.namespace, b.externalID, "")
		if err != nil {
			t.Fatalf("resolve %s: %v", b.namespace, err)
		}
		if got.GoalID != g.GoalID || binding.GoalID != g.GoalID {
			t.Fatalf("resolve %s = %#v %#v", b.namespace, got, binding)
		}
	}
	if _, _, err := s.Resolve("codex:goal", "unbound", ""); err == nil {
		t.Fatal("unbound identity resolved")
	}
}

func TestResolveRequiresRevisionWhenBindingHistoryIsAmbiguous(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json")}
	g, err := s.Create("Rev", "", Provenance{Actor: "operator", Authority: "user"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, revision := range []string{"r1", "r2"} {
		if _, err := s.Bind(g.GoalID, "codex:goal", "thread-goal", revision, Provenance{Actor: "adapter", Authority: "harness"}); err != nil {
			t.Fatal(err)
		}
	}
	if _, _, err := s.Resolve("codex:goal", "thread-goal", ""); err == nil || !strings.Contains(err.Error(), "ambiguous") {
		t.Fatalf("ambiguity = %v", err)
	}
	got, _, err := s.Resolve("codex:goal", "thread-goal", "r2")
	if err != nil || got.GoalID != g.GoalID {
		t.Fatalf("revision resolve = %#v, %v", got, err)
	}
}
