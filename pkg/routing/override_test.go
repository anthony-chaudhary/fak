package routing

import (
	"strings"
	"testing"
)

func overrideTestEngines() []Engine {
	return []Engine{
		{ID: "big-hosted", Surface: SurfaceHosted, Model: "deepseek-ai/DeepSeek-V4.1-Flash", Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1.0, ContextWindow: 1048576},
		{ID: "local-small", Surface: SurfaceLocal, Model: "fak/Qwen3.8-27B-UD-Q2_K_XL", Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1.0, ContextWindow: 20480},
		{ID: "local-unknown-win", Surface: SurfaceLocal, Model: "fak/other", Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1.0},
	}
}

func overrideTestReq() TaskRequest {
	return TaskRequest{Candidate: &TaskCandidate{ID: "T-OVERRIDE", Lane: "routing"}, Role: "worker"}
}

// TestOverrideModelPinNarrowsOnly proves an agent can hit a specific model:
// the pin selects the matching engine, and a pin naming no served model leaves
// the request unrouted (never silently placed elsewhere) with MODEL_MISMATCH
// on every rejection and the override named in the audit.
func TestOverrideModelPinNarrowsOnly(t *testing.T) {
	engines := overrideTestEngines()
	req := overrideTestReq()
	req.Override.Model = "fak/Qwen3.8-27B-UD-Q2_K_XL"
	assignments, unrouted, err := Route([]TaskRequest{req}, engines)
	if err != nil {
		t.Fatalf("Route with model override: %v", err)
	}
	if len(assignments) != 1 || assignments[0].EngineID != "local-small" {
		t.Fatalf("model override routed to %v (unrouted %v), want local-small", assignments, unrouted)
	}
	if !containsAudit(assignments[0].AssignmentAudit, "override model=fak/Qwen3.8-27B-UD-Q2_K_XL") {
		t.Errorf("model override not named in audit: %v", assignments[0].AssignmentAudit)
	}

	req.Override.Model = "no-such-model/xyz"
	assignments, unrouted, err = Route([]TaskRequest{req}, engines)
	if err != nil {
		t.Fatalf("Route with unserved model override: %v", err)
	}
	if len(assignments) != 0 || len(unrouted) != 1 {
		t.Fatalf("unserved model override must unroute, got %v / %v", assignments, unrouted)
	}
	if unrouted[0].Reason != ReasonNoEligibleEngine {
		t.Errorf("unrouted reason = %q, want %q", unrouted[0].Reason, ReasonNoEligibleEngine)
	}
	for _, r := range unrouted[0].Rejected {
		if r.Reason != ReasonModelMismatch {
			t.Errorf("rejection for %s = %q, want %q", r.EngineID, r.Reason, ReasonModelMismatch)
		}
	}
}

// TestOverrideContextWindowRequiresCapacity proves a context-window override is
// invokable: only engines advertising at least the window are eligible, unknown
// capacity fails closed, and the override is named in the audit.
func TestOverrideContextWindowRequiresCapacity(t *testing.T) {
	engines := overrideTestEngines()
	req := overrideTestReq()
	req.Override.MinContextWindow = 1000000
	assignments, unrouted, err := Route([]TaskRequest{req}, engines)
	if err != nil {
		t.Fatalf("Route with context override: %v", err)
	}
	if len(assignments) != 1 || assignments[0].EngineID != "big-hosted" {
		t.Fatalf("context override routed to %v (unrouted %v), want big-hosted", assignments, unrouted)
	}
	if !containsAudit(assignments[0].AssignmentAudit, "override min_context_window=1000000") {
		t.Errorf("context override not named in audit: %v", assignments[0].AssignmentAudit)
	}

	// Unknown-capacity engine must fail closed when a window is required.
	small := []Engine{{ID: "unknown", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1.0}}
	req2 := overrideTestReq()
	req2.Override.MinContextWindow = 8192
	_, unrouted, err = Route([]TaskRequest{req2}, small)
	if err != nil {
		t.Fatalf("Route unknown capacity: %v", err)
	}
	if len(unrouted) != 1 || len(unrouted[0].Rejected) != 1 || unrouted[0].Rejected[0].Reason != ReasonContextTooSmall {
		t.Fatalf("unknown capacity must reject CONTEXT_TOO_SMALL, got %+v", unrouted)
	}

	// No override: unknown capacity participates normally.
	req3 := overrideTestReq()
	assignments, _, err = Route([]TaskRequest{req3}, small)
	if err != nil || len(assignments) != 1 {
		t.Fatalf("no override must admit unknown capacity, got %v err %v", assignments, err)
	}
}

// TestOverrideSurfaceAndEnginePin proves surface and engine pins narrow with
// named audits, and conflicting pins fail loud (caller bug, not precedence).
func TestOverrideSurfaceAndEnginePin(t *testing.T) {
	engines := overrideTestEngines()
	req := overrideTestReq()
	req.Override.Surface = SurfaceLocal
	assignments, _, err := Route([]TaskRequest{req}, engines)
	if err != nil || len(assignments) != 1 || assignments[0].EngineID != "local-small" {
		t.Fatalf("surface override routed to %v err %v, want local-small first", assignments, err)
	}

	req = overrideTestReq()
	req.Override.Engine = "local-small"
	assignments, _, err = Route([]TaskRequest{req}, engines)
	if err != nil || len(assignments) != 1 || assignments[0].EngineID != "local-small" {
		t.Fatalf("engine override routed to %v err %v, want local-small", assignments, err)
	}

	req = overrideTestReq()
	req.PinnedEngine = "big-hosted"
	req.Override.Engine = "local-small"
	if _, _, err := Route([]TaskRequest{req}, engines); err == nil {
		t.Fatal("conflicting engine pins must fail loud, got nil error")
	}

	if err := (ManualOverride{Surface: "bogus"}).Validate(); err == nil {
		t.Error("unknown override surface must fail validation")
	}
	if err := (ManualOverride{MinContextWindow: -1}).Validate(); err == nil {
		t.Error("negative override window must fail validation")
	}
}

// TestOverrideReturnHandlingControl proves return-handling flags decouple the
// caller from fleet behavior: NoRetrySame turns RETRY_SAME into FAILOVER_NEXT
// (failed engine excluded), NoFailover leaves the request unrouted FATAL.
func TestOverrideReturnHandlingControl(t *testing.T) {
	engines := []Engine{
		{ID: "a", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1.0},
		{ID: "b", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1.0},
	}
	chain := DegradationChain{Version: "test", Rungs: []Rung{{Name: "r1", Tags: []string{"x"}}, {Name: "maintenance", Surface: SurfaceLocal}}}
	policy := Policy{Version: PolicySchema, DefaultSurface: SurfaceLocal}
	retrySame := Failure{Class: FailureRateLimited, Action: ActionRetrySame}

	req := overrideTestReq()
	a, _, _, err := Failover(req, "a", retrySame, engines, chain, policy)
	if err != nil || a.EngineID != "a" {
		t.Fatalf("plain RETRY_SAME must keep failed engine first, got %q err %v", a.EngineID, err)
	}

	req.Override.NoRetrySame = true
	a, _, _, err = Failover(req, "a", retrySame, engines, chain, policy)
	if err != nil || a.EngineID != "b" {
		t.Fatalf("NoRetrySame must exclude failed engine, got %q err %v", a.EngineID, err)
	}
	if !containsAudit(a.AssignmentAudit, "no_retry_same") {
		t.Errorf("NoRetrySame not named in audit: %v", a.AssignmentAudit)
	}

	req = overrideTestReq()
	req.Override.NoFailover = true
	_, _, u, err := Failover(req, "a", retrySame, engines, chain, policy)
	if err != nil || u.Reason != ReasonFailureFatal {
		t.Fatalf("NoFailover must leave FATAL unrouted, got %+v err %v", u, err)
	}
	if !containsAudit(u.AssignmentAudit, "no_failover") {
		t.Errorf("NoFailover not named in audit: %v", u.AssignmentAudit)
	}
}

// TestOverrideReasonsAreClosed witnesses the two new tokens are members of the
// closed rejection vocabulary (callers can assert on them; no prose parsing).
func TestOverrideReasonsAreClosed(t *testing.T) {
	if !IsKnownReason(ReasonModelMismatch) || !IsKnownReason(ReasonContextTooSmall) {
		t.Fatal("override rejection tokens must be in the closed vocabulary")
	}
}

func containsAudit(audit []string, sub string) bool {
	for _, line := range audit {
		if strings.Contains(line, sub) {
			return true
		}
	}
	return false
}
