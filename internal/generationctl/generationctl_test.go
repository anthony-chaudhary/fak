package generationctl

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/streamrules"
)

func redirectRules(t *testing.T) []streamrules.Rule {
	t.Helper()
	return []streamrules.Rule{{Name: "no-shell-delete", Tool: "shell", Scope: streamrules.ScopeNamedTool, Pattern: `(?i)remove-item`, Interrupt: true, SubstituteAction: "Inspect the target with the read-only inventory tool."}}
}

func TestLiveRedirectPreservesAcceptedPrefixAcrossComputeHandoff(t *testing.T) {
	rules := redirectRules(t)
	c, err := New("traj-42", "planner", Compute{Worker: "w1", Model: "fast", Device: "cpu"}, rules)
	if err != nil {
		t.Fatal(err)
	}
	if err := c.Accept("I will inspect the workspace. "); err != nil {
		t.Fatal(err)
	}

	key := streamrules.StreamKey{ToolCallID: "call-1", ToolName: "shell", Scope: streamrules.ScopeNamedTool}
	if tr, err := c.ObserveToolDelta(key, `{"command":"Remove-`); err != nil || tr.Directive.Kind != Continue {
		t.Fatalf("first delta = %#v, %v", tr, err)
	}
	tr, err := c.ObserveToolDelta(key, `Item -Recurse C:\\work"}`)
	if err != nil {
		t.Fatal(err)
	}
	if tr.Directive.Kind != Redirect || tr.Checkpoint == nil {
		t.Fatalf("redirect = %#v", tr)
	}
	if got, want := tr.Checkpoint.Accepted, "I will inspect the workspace. "; got != want {
		t.Fatalf("accepted prefix = %q, want %q", got, want)
	}
	if err := c.Accept("must fail"); err == nil {
		t.Fatal("closed epoch accepted output")
	}

	next, err := Resume(*tr.Checkpoint, "safety-micro-agent", Compute{Worker: "gpu-7", Model: "deep", Device: "L4"}, rules)
	if err != nil {
		t.Fatal(err)
	}
	if next.Epoch().TrajectoryID != "traj-42" || next.Epoch().Number != 2 {
		t.Fatalf("next epoch = %#v", next.Epoch())
	}
	if err := next.Accept("Inventory complete."); err != nil {
		t.Fatal(err)
	}
	if got := next.Checkpoint().Accepted; got != "I will inspect the workspace. Inventory complete." {
		t.Fatalf("resumed prefix = %q", got)
	}
}

func TestDirectiveVocabularyCreatesSteeringPoints(t *testing.T) {
	for _, kind := range []DirectiveKind{Redirect, Fork, Yield, Stop} {
		c, err := New("traj", "owner", Compute{}, nil)
		if err != nil {
			t.Fatal(err)
		}
		tr, err := c.Steer(Directive{Kind: kind, Reason: "operator"})
		if err != nil || tr.Checkpoint == nil {
			t.Fatalf("%s: %#v %v", kind, tr, err)
		}
	}
}
