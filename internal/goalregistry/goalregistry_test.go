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
	aliases := []struct{ ns, id string }{{"fak:trajctl", "objective-7"}, {"claude:goal", "g-1"}, {"codex:goal", "g-1"}, {"github:issue", "anthony-chaudhary/fak#6663"}, {"dos:unit", "unit-9"}}
	for _, alias := range aliases {
		if _, err := s.Bind(g1.GoalID, alias.ns, alias.id, "", p); err != nil {
			t.Fatalf("bind %s: %v", alias.ns, err)
		}
	}
	shown, bindings, err := s.Show(g1.GoalID)
	if err != nil {
		t.Fatal(err)
	}
	if shown.Title != "Observe the fleet" || len(bindings) != len(aliases) {
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
	g, err = s.Transition(g.GoalID, Achieved, OutcomeEvidence{Class: IndependentWitness, Author: "judge", Reference: "commit:abc"})
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

func TestLifecycleRequiresTypedWitnessAndPreservesHistory(t *testing.T) {
	now := time.Unix(1700000000, 0).UTC()
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json"), Now: func() time.Time { return now }}
	g, err := s.Create("Retry succeeds", "", Provenance{Actor: "operator", Authority: "user"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if g.EvidencePolicy != DefaultEvidencePolicy {
		t.Fatalf("policy=%q", g.EvidencePolicy)
	}
	for _, class := range []EvidenceClass{HarnessAssertion, AgentAssertion, OperatorDeclaration} {
		if _, err := s.Transition(g.GoalID, Achieved, OutcomeEvidence{Class: class, Author: "claimant", Reference: "run:failed-then-retry"}); err == nil {
			t.Fatalf("%s terminalized goal", class)
		}
	}
	g, err = s.Transition(g.GoalID, Achieved, OutcomeEvidence{Class: IndependentWitness, Author: "judge", Reference: "commit:success"})
	if err != nil {
		t.Fatal(err)
	}
	if g.Lifecycle != Achieved {
		t.Fatalf("lifecycle=%s", g.Lifecycle)
	}
	if _, err := s.Transition(g.GoalID, Abandoned, OutcomeEvidence{Class: IndependentWitness, Author: "judge", Reference: "report:conflict"}); err == nil || !strings.Contains(err.Error(), "conflict") {
		t.Fatalf("conflict=%v", err)
	}
	if _, err := s.Reopen(g.GoalID, "operator", "decision:continue"); err != nil {
		t.Fatal(err)
	}
	if _, err := s.Transition(g.GoalID, Superseded, OutcomeEvidence{Class: IndependentWitness, Author: "judge", Reference: "goal:replacement"}); err != nil {
		t.Fatal(err)
	}
	evidence, err := s.OutcomeEvidence(g.GoalID)
	if err != nil {
		t.Fatal(err)
	}
	if len(evidence) != 3 || evidence[0].Lifecycle != Achieved || evidence[1].Lifecycle != Active || evidence[2].Lifecycle != Superseded {
		t.Fatalf("evidence=%#v", evidence)
	}
}

func TestGenericUpdateCannotTerminalizeGoal(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json")}
	g, err := s.Create("No self certification", "", Provenance{Actor: "operator", Authority: "user"}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := s.Update(g.GoalID, g.Title, g.Summary, Achieved); err == nil {
		t.Fatal("generic update terminalized goal")
	}
}

func TestReopenLifecycleTransitionTable(t *testing.T) {
	now := time.Date(2026, 9, 8, 12, 0, 0, 0, time.UTC)
	p := Provenance{Actor: "operator", Authority: "operator-declared"}

	cases := []struct {
		state       Lifecycle
		terminal    bool
		setup       func(s Store, g Goal) (Goal, error)
		allowReopen bool
	}{
		{
			state:       Active,
			terminal:    false,
			allowReopen: false,
			setup: func(s Store, g Goal) (Goal, error) {
				return g, nil
			},
		},
		{
			state:       Paused,
			terminal:    false,
			allowReopen: false,
			setup: func(s Store, g Goal) (Goal, error) {
				return s.Update(g.GoalID, g.Title, g.Summary, Paused)
			},
		},
		{
			state:       Achieved,
			terminal:    true,
			allowReopen: true,
			setup: func(s Store, g Goal) (Goal, error) {
				return s.Transition(g.GoalID, Achieved, OutcomeEvidence{
					Class:     IndependentWitness,
					Author:    "judge",
					Reference: "test:achieved",
				})
			},
		},
		{
			state:       Abandoned,
			terminal:    true,
			allowReopen: true,
			setup: func(s Store, g Goal) (Goal, error) {
				return s.Transition(g.GoalID, Abandoned, OutcomeEvidence{
					Class:     IndependentWitness,
					Author:    "judge",
					Reference: "test:abandoned",
				})
			},
		},
		{
			state:       Superseded,
			terminal:    true,
			allowReopen: true,
			setup: func(s Store, g Goal) (Goal, error) {
				return s.Transition(g.GoalID, Superseded, OutcomeEvidence{
					Class:     IndependentWitness,
					Author:    "judge",
					Reference: "test:superseded",
				})
			},
		},
		{
			state:       Blocked,
			terminal:    true,
			allowReopen: true,
			setup: func(s Store, g Goal) (Goal, error) {
				return s.Transition(g.GoalID, Blocked, OutcomeEvidence{
					Class:     IndependentWitness,
					Author:    "judge",
					Reference: "test:blocked",
				})
			},
		},
	}

	for _, tc := range cases {
		tc := tc
		t.Run(string(tc.state), func(t *testing.T) {
			s := Store{
				Path: filepath.Join(t.TempDir(), "goals.json"),
				Now:  func() time.Time { return now },
			}
			g, err := s.Create("Test goal", "summary", p, nil)
			if err != nil {
				t.Fatalf("create failed: %v", err)
			}
			g, err = tc.setup(s, g)
			if err != nil {
				t.Fatalf("setup %s failed: %v", tc.state, err)
			}
			if g.Lifecycle != tc.state {
				t.Fatalf("setup produced lifecycle %s; want %s", g.Lifecycle, tc.state)
			}

			initialEvidence, err := s.OutcomeEvidence(g.GoalID)
			if err != nil {
				t.Fatalf("get initial evidence: %v", err)
			}

			reopened, err := s.Reopen(g.GoalID, "operator", "reopen-ref")
			if tc.allowReopen {
				if err != nil {
					t.Fatalf("reopen from terminal state %s failed: %v", tc.state, err)
				}
				if reopened.Lifecycle != Active {
					t.Fatalf("reopened lifecycle = %s; want %s", reopened.Lifecycle, Active)
				}
				evidence, err := s.OutcomeEvidence(g.GoalID)
				if err != nil {
					t.Fatalf("get outcome evidence: %v", err)
				}
				if len(evidence) != len(initialEvidence)+1 {
					t.Fatalf("evidence count = %d; want %d", len(evidence), len(initialEvidence)+1)
				}
				last := evidence[len(evidence)-1]
				if last.Lifecycle != Active || last.Class != OperatorDeclaration || last.Author != "operator" || last.Reference != "reopen-ref" {
					t.Fatalf("reopen evidence = %+v", last)
				}
			} else {
				if err == nil {
					t.Fatalf("reopen from non-terminal state %s succeeded unexpectedly", tc.state)
				}
				// Verify refusals leave state and evidence unchanged
				evidence, err := s.OutcomeEvidence(g.GoalID)
				if err != nil {
					t.Fatalf("get outcome evidence after refusal: %v", err)
				}
				if len(evidence) != len(initialEvidence) {
					t.Fatalf("refused reopen mutated evidence: before=%d after=%d", len(initialEvidence), len(evidence))
				}
				cur, _, err := s.Show(g.GoalID)
				if err != nil {
					t.Fatalf("show goal: %v", err)
				}
				if cur.Lifecycle != tc.state {
					t.Fatalf("refused reopen mutated lifecycle: got %s; want %s", cur.Lifecycle, tc.state)
				}
			}
		})
	}
}

func TestBindingCanonicalTuple(t *testing.T) {
	s := Store{Path: filepath.Join(t.TempDir(), "goals.json")}
	p := Provenance{Actor: "operator", Authority: "operator-declared"}
	g1, err := s.Create("Goal 1", "first goal", p, nil)
	if err != nil {
		t.Fatalf("create g1: %v", err)
	}
	g2, err := s.Create("Goal 2", "second goal", p, nil)
	if err != nil {
		t.Fatalf("create g2: %v", err)
	}

	// 1. Initial bind with revision
	b1, err := s.Bind(g1.GoalID, "fak:trajctl", "obj-1", "rev-1", p)
	if err != nil {
		t.Fatalf("initial bind: %v", err)
	}
	if b1.Revision != "rev-1" {
		t.Fatalf("b1 revision = %q; want rev-1", b1.Revision)
	}

	// 2. Whitespace-padded revision on same goal must be idempotent (not create duplicate or error)
	b1Retry, err := s.Bind(" "+g1.GoalID+" ", " fak:trajctl ", " obj-1 ", " rev-1 \t\n", p)
	if err != nil {
		t.Fatalf("whitespace idempotent bind: %v", err)
	}
	if b1Retry.GoalID != g1.GoalID || b1Retry.Revision != "rev-1" {
		t.Fatalf("b1Retry mismatch: %+v", b1Retry)
	}
	_, bindings, err := s.Show(g1.GoalID)
	if err != nil {
		t.Fatalf("show g1: %v", err)
	}
	if len(bindings) != 1 {
		t.Fatalf("expected exactly 1 binding after idempotent bind, got %d", len(bindings))
	}

	// 3. Whitespace-padded revision on different goal must collide
	_, err = s.Bind(g2.GoalID, " fak:trajctl ", " obj-1 ", "  rev-1  ", p)
	if err == nil || !strings.Contains(err.Error(), "collision") {
		t.Fatalf("want collision error for whitespace-equivalent revision on distinct goal, got %v", err)
	}

	// 4. Resolve with whitespace-padded revision
	resGoal, resBinding, err := s.Resolve("  fak:trajctl\t", "\nobj-1 ", " rev-1 ")
	if err != nil {
		t.Fatalf("resolve with whitespace: %v", err)
	}
	if resGoal.GoalID != g1.GoalID || resBinding.Revision != "rev-1" {
		t.Fatalf("resolve mismatch: goal=%s binding=%+v", resGoal.GoalID, resBinding)
	}

	// 5. Unbind with whitespace-padded tuple
	err = s.Unbind(" "+g1.GoalID+" ", " fak:trajctl ", " obj-1 ", " rev-1 ")
	if err != nil {
		t.Fatalf("unbind with whitespace: %v", err)
	}
	_, bindingsAfterUnbind, err := s.Show(g1.GoalID)
	if err != nil {
		t.Fatalf("show g1 after unbind: %v", err)
	}
	if len(bindingsAfterUnbind) != 0 {
		t.Fatalf("expected 0 bindings after unbind, got %d", len(bindingsAfterUnbind))
	}
}
