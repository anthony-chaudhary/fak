package routing

import (
	"reflect"
	"testing"
)

// TestDefaultPolicyIsValid proves the default policy satisfies its own closed
// invariants and is deterministic in rule order.
func TestDefaultPolicyIsValid(t *testing.T) {
	p := DefaultPolicy()
	if err := p.Validate(); err != nil {
		t.Fatalf("DefaultPolicy().Validate() = %v, want nil", err)
	}
	if p.Version != PolicySchema {
		t.Errorf("policy version = %q, want %q", p.Version, PolicySchema)
	}
	if !p.DefaultSurface.IsKnown() {
		t.Errorf("default surface %q is not known", p.DefaultSurface)
	}
}

// TestPolicyPreferredSurfaceTable covers the proven trade-offs encoded in the
// default policy.
func TestPolicyPreferredSurfaceTable(t *testing.T) {
	cases := []struct {
		name     string
		req      TaskRequest
		want     Surface
		wantRule string
	}{
		{
			name:     "scout routes local",
			req:      TaskRequest{Candidate: candidate("T1", "docs"), Role: "scout"},
			want:     SurfaceLocal,
			wantRule: "scout-local-small",
		},
		{
			name:     "judge routes local for trust",
			req:      TaskRequest{Candidate: candidate("T2", "docs"), Role: "judge"},
			want:     SurfaceLocal,
			wantRule: "judge-local-trust",
		},
		{
			name:     "worker prefers hosted (massive/cloud)",
			req:      TaskRequest{Candidate: candidate("T3", "docs"), Role: "worker"},
			want:     SurfaceHosted,
			wantRule: "hosted-massive-cloud",
		},
		{
			name:     "require-local worker falls through to local heavy",
			req:      TaskRequest{Candidate: candidate("T4", "docs"), Role: "worker", RequireLocal: true},
			want:     SurfaceLocal,
			wantRule: "local-primary-heavy",
		},
		{
			name:     "worker with unmet tag still prefers hosted by role",
			req:      TaskRequest{Candidate: candidate("T5", "docs"), Role: "worker", RequireTags: []string{"x"}},
			want:     SurfaceHosted,
			wantRule: "hosted-massive-cloud",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got, rule := DefaultPolicy().preferredSurface(tc.req)
			if got != tc.want {
				t.Errorf("preferredSurface = %q, want %q", got, tc.want)
			}
			if rule != tc.wantRule {
				t.Errorf("rule = %q, want %q", rule, tc.wantRule)
			}
		})
	}
}

// TestPolicyHardFences proves the hard fences hold regardless of the preferred
// surface, and that every refusal carries a closed reason.
func TestPolicyHardFences(t *testing.T) {
	p := DefaultPolicy()
	req := TaskRequest{Candidate: candidate("T1", "docs"), Role: "worker", RequireLocal: true}

	if ok, reason := p.Admits(req, Engine{ID: "cloud", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 1}); ok {
		t.Error("policy admitted a hosted engine for a require-local request")
	} else if reason != ReasonHostedRefused {
		t.Errorf("refusal reason = %q, want %q", reason, ReasonHostedRefused)
	}

	if ok, reason := p.Admits(TaskRequest{Role: "worker"}, Engine{ID: "sick", Surface: SurfaceLocal, Healthy: false}); ok {
		t.Error("policy admitted an unhealthy engine")
	} else if reason != ReasonUnhealthy {
		t.Errorf("refusal reason = %q, want %q", reason, ReasonUnhealthy)
	}

	if ok, reason := p.Admits(TaskRequest{Role: "worker"}, Engine{ID: "drain", Surface: SurfaceLocal, Healthy: true, Draining: true}); ok {
		t.Error("policy admitted a draining engine")
	} else if reason != ReasonDraining {
		t.Errorf("refusal reason = %q, want %q", reason, ReasonDraining)
	}

	if ok, reason := p.Admits(TaskRequest{Role: "worker"}, Engine{ID: "full", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 0}); ok {
		t.Error("policy admitted an engine with no headroom")
	} else if reason != ReasonNoHeadroom {
		t.Errorf("refusal reason = %q, want %q", reason, ReasonNoHeadroom)
	}

	if ok, reason := p.Admits(TaskRequest{Role: "worker"}, workerEngine("ok")); !ok {
		t.Errorf("policy refused a healthy engine: %s", reason)
	}
}

// TestPolicyEvaluate is the deterministic audit witness: admitted and refused
// sets are computed purely from the request and engine set.
func TestPolicyEvaluate(t *testing.T) {
	p := DefaultPolicy()
	req := TaskRequest{Candidate: candidate("T1", "docs"), Role: "worker"}
	engines := []Engine{
		workerEngine("local-ok"),
		{ID: "cloud", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1},
		{ID: "sick", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: false},
	}

	d1 := p.Evaluate(req, engines)
	d2 := p.Evaluate(req, engines)
	if !reflect.DeepEqual(d1, d2) {
		t.Fatalf("Evaluate is not deterministic:\n got %+v\nwant %+v", d1, d2)
	}
	if len(d1.Admitted) != 2 {
		t.Errorf("Admitted = %v, want 2 eligible engines", d1.Admitted)
	}
	if !reflect.DeepEqual(d1.Refused, []Rejection{{EngineID: "sick", Reason: ReasonUnhealthy}}) {
		t.Errorf("Refused = %+v, want sick/UNHEALTHY", d1.Refused)
	}
	for _, r := range d1.Refused {
		if !IsKnownReason(r.Reason) {
			t.Errorf("refusal reason %q is outside the closed vocabulary", r.Reason)
		}
	}
}

// TestPolicyValidateRejectsUnknownSurface proves a malformed policy manifest
// fails loud instead of routing on an unknown surface.
func TestPolicyValidateRejectsUnknownSurface(t *testing.T) {
	bad := Policy{DefaultSurface: Surface("quantum")}
	if err := bad.Validate(); err == nil {
		t.Error("Validate() with unknown default surface should fail, got nil")
	}

	bad = Policy{
		DefaultSurface: SurfaceLocal,
		Rules:          []PolicyRule{{Name: "r", Surface: Surface("quantum")}},
	}
	if err := bad.Validate(); err == nil {
		t.Error("Validate() with unknown rule surface should fail, got nil")
	}

	bad = Policy{DefaultSurface: SurfaceLocal, Rules: []PolicyRule{{Name: ""}}}
	if err := bad.Validate(); err == nil {
		t.Error("Validate() with unnamed rule should fail, got nil")
	}
}

// TestPolicyIsConsumedByRouting is the invoked-vs-present witness: it proves the
// published policy actually changes the routing OUTCOME, not merely that a
// manifest exists. Two policies that differ only in preferred surface must
// produce different winning engines for the same request and engine set.
//
// Per docs/PRESENCE-VS-INVOKABILITY.md, "Policy.Validate passes" is a presence
// witness. This test is the reachability witness.
func TestPolicyIsConsumedByRouting(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T1", "docs"), Role: "worker"}
	engines := []Engine{
		{ID: "local-a", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1, Roles: []string{"worker"}},
		{ID: "hosted-b", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 1, Roles: []string{"worker"}},
	}

	localFirst := Policy{
		Version:        "test-local-first",
		DefaultSurface: SurfaceLocal,
		Rules:          []PolicyRule{{Name: "worker-local", Priority: 10, Roles: []string{"worker"}, Surface: SurfaceLocal}},
	}
	hostedFirst := Policy{
		Version:        "test-hosted-first",
		DefaultSurface: SurfaceHosted,
		Rules:          []PolicyRule{{Name: "worker-hosted", Priority: 10, Roles: []string{"worker"}, Surface: SurfaceHosted}},
	}

	a1, _, err := RouteWithPolicy([]TaskRequest{req}, engines, localFirst)
	if err != nil {
		t.Fatalf("RouteWithPolicy(localFirst) error = %v", err)
	}
	a2, _, err := RouteWithPolicy([]TaskRequest{req}, engines, hostedFirst)
	if err != nil {
		t.Fatalf("RouteWithPolicy(hostedFirst) error = %v", err)
	}
	if len(a1) != 1 || len(a2) != 1 {
		t.Fatalf("got %d and %d assignments, want 1 each", len(a1), len(a2))
	}
	if a1[0].EngineID != "local-a" {
		t.Errorf("local-first policy routed engine = %q, want local-a", a1[0].EngineID)
	}
	if a2[0].EngineID != "hosted-b" {
		t.Errorf("hosted-first policy routed engine = %q, want hosted-b", a2[0].EngineID)
	}
	if a1[0].EngineID == a2[0].EngineID {
		t.Fatal("policy had no effect on the routing outcome — the manifest is an unconsumed seam")
	}
}

// TestRouteWithPolicyHonoursRequireLocalFence proves the hard fence survives the
// policy composition: even a hosted-first policy can never place a require-local
// task on a hosted engine.
func TestRouteWithPolicyHonoursRequireLocalFence(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T1", "docs"), Role: "worker", RequireLocal: true}
	engines := []Engine{
		{ID: "hosted-b", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 1, Roles: []string{"worker"}},
	}
	_, unrouted, err := RouteWithPolicy([]TaskRequest{req}, engines, DefaultPolicy())
	if err != nil {
		t.Fatalf("RouteWithPolicy error = %v", err)
	}
	if len(unrouted) != 1 {
		t.Fatalf("got %d unrouted, want 1 (hosted must be refused under require_local)", len(unrouted))
	}
	if unrouted[0].Reason != ReasonNoEligibleEngine {
		t.Errorf("unrouted reason = %q, want %q", unrouted[0].Reason, ReasonNoEligibleEngine)
	}
	found := false
	for _, r := range unrouted[0].Rejected {
		if r.Reason == ReasonHostedRefused {
			found = true
		}
	}
	if !found {
		t.Errorf("expected a %s rejection, got %+v", ReasonHostedRefused, unrouted[0].Rejected)
	}
}

// TestRouteWithPolicyRejectsInvalidPolicy proves a malformed policy fails loud
// before it routes, rather than routing on partial rules.
func TestRouteWithPolicyRejectsInvalidPolicy(t *testing.T) {
	bad := Policy{DefaultSurface: Surface("quantum")}
	if _, _, err := RouteWithPolicy(nil, nil, bad); err == nil {
		t.Error("RouteWithPolicy with invalid policy should fail, got nil")
	}
}

// TestRouteWithPolicyDeterministic proves the composed path is replayable.
func TestRouteWithPolicyDeterministic(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T1", "compute"), Role: "worker"}
	engines := []Engine{
		{ID: "local-a", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1, Roles: []string{"worker"}},
		{ID: "hosted-b", Surface: SurfaceHosted, Healthy: true, CapacityHeadroom: 1, Roles: []string{"worker"}},
	}
	first, _, err := RouteWithPolicy([]TaskRequest{req}, engines, DefaultPolicy())
	if err != nil {
		t.Fatalf("error = %v", err)
	}
	for i := 0; i < 5; i++ {
		got, _, err := RouteWithPolicy([]TaskRequest{req}, engines, DefaultPolicy())
		if err != nil {
			t.Fatalf("iteration %d error = %v", i, err)
		}
		if !reflect.DeepEqual(first, got) {
			t.Fatalf("iteration %d changed the decision:\nfirst=%+v\ngot=%+v", i, first, got)
		}
	}
}

// TestRouteRequestWithPolicyConvenience covers the single-request form.
func TestRouteRequestWithPolicyConvenience(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T1", "docs"), Role: "scout"}
	engines := []Engine{
		{ID: "local-a", Surface: SurfaceLocal, Healthy: true, CapacityHeadroom: 1, Roles: []string{"scout"}},
	}
	a, rejected, err := RouteRequestWithPolicy(req, engines, DefaultPolicy())
	if err != nil {
		t.Fatalf("RouteRequestWithPolicy error = %v", err)
	}
	if a.EngineID != "local-a" {
		t.Errorf("engine = %q, want local-a", a.EngineID)
	}
	if len(rejected) != 0 {
		t.Errorf("rejected = %+v, want none", rejected)
	}
}
