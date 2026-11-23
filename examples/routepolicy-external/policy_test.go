package external

import (
	"testing"

	"github.com/anthony-chaudhary/fak/pkg/routepolicy"
	"github.com/anthony-chaudhary/fak/pkg/routepolicy/routepolicytest"
)

const manifestJSON = `{
  "version": "fak-route/v1",
  "default": {"members": [{"model": "default", "role": "primary"}]},
  "rules": [
    {"name": "hard", "match": {"min_complexity": "high"}, "plan": {"members": [{"model": "large"}]}},
    {"name": "guard-writes", "match": {"aspect": "tool_call", "tool": "write_*"},
     "plan": {"members": [{"model": "guard-a"}, {"model": "guard-b"}], "reduce": "vote"}}
  ]
}`

// TestTakeOursAlone routes with fak's reference policy using only pkg/routepolicy.
func TestTakeOursAlone(t *testing.T) {
	d, err := RouteWithFak([]byte(manifestJSON), routepolicy.Subject{Aspect: routepolicy.AspectToolCall, Tool: "write_file"})
	if err != nil {
		t.Fatal(err)
	}
	if !d.Matched || d.RuleName != "guard-writes" || !d.Plan.IsEnsemble() {
		t.Fatalf("decision = %+v, want the guard-writes ensemble", d)
	}
	res, err := routepolicy.Combine(d.Plan.Reduce, []routepolicy.Vote{
		{Member: d.Plan.Members[0], Output: "deny"},
		{Member: d.Plan.Members[1], Output: "deny"},
	})
	if err != nil || res.Output != "deny" {
		t.Fatalf("Combine = %+v, %v; want deny", res, err)
	}

	d, err = RouteWithFak([]byte(manifestJSON), routepolicy.Subject{Aspect: routepolicy.AspectRequest})
	if err != nil {
		t.Fatal(err)
	}
	if d.Matched || d.Plan.Primary() != "default" {
		t.Fatalf("decision = %+v, want the fail-closed default", d)
	}

	if _, err := RouteWithFak([]byte(`{"default":{"members":[]}}`), routepolicy.Subject{}); err == nil {
		t.Fatal("a manifest with an empty default plan must be refused")
	}
}

// TestBringYourOwnPassesConformance runs fak's conformance kit against an
// independent Policy implementation.
func TestBringYourOwnPassesConformance(t *testing.T) {
	routepolicytest.Run(t, func(m routepolicy.Manifest) routepolicy.Policy { return NewIndexedPolicy(m) })
}

// TestBringYourOwnAgreesWithFak checks the independent policy against the
// reference on the example manifest as well.
func TestBringYourOwnAgreesWithFak(t *testing.T) {
	m, err := routepolicy.ParseManifest([]byte(manifestJSON))
	if err != nil {
		t.Fatal(err)
	}
	var ours, theirs routepolicy.Policy = NewIndexedPolicy(m), m
	for _, s := range []routepolicy.Subject{
		{Aspect: routepolicy.AspectToolCall, Tool: "write_file", Complexity: routepolicy.ComplexityHigh},
		{Aspect: routepolicy.AspectToolCall, Tool: "read_file"},
		{Aspect: routepolicy.AspectStep, Complexity: routepolicy.ComplexityHigh},
	} {
		a, b := ours.Route(s), theirs.Route(s)
		if a.RuleName != b.RuleName || a.Matched != b.Matched || a.Plan.Primary() != b.Plan.Primary() {
			t.Errorf("subject %+v: ours %+v, fak %+v", s, a, b)
		}
	}
}
