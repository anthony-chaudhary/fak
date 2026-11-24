package routing

import (
	"strings"
	"testing"
)

// failoverEngines is the standard failover fixture: a primary engine on the
// hosted surface and a small local fallback, both healthy unless a case says
// otherwise. They carry the default chain's capability tags ("primary",
// "small") so each resolves to its intended rung rather than the catch-all.
func failoverEngines() []Engine {
	return []Engine{
		{ID: "primary", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1, Tags: []string{"primary"}},
		{ID: "fallback", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1, Tags: []string{"small"}},
	}
}

// auditContains reports whether any assignment/refusal audit line contains all
// the given substrings, so a test can assert on the emitted evidence.
func auditContains(audit []string, subs ...string) bool {
	for _, line := range audit {
		all := true
		for _, sub := range subs {
			if !strings.Contains(line, sub) {
				all = false
				break
			}
		}
		if all {
			return true
		}
	}
	return false
}

// TestFailoverAutoNextIsTheLoadBearingWitness is the LOAD-BEARING auto-failover
// witness: after the primary fails 503, Failover assigns the FALLBACK engine,
// never the primary, and the audit names the class, the action, the exclusion,
// and the ladder chain.
func TestFailoverAutoNextIsTheLoadBearingWitness(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T1", "docs"), Role: "worker"}
	engines := failoverEngines()

	a, rejected, u, err := Failover(req, "primary", Classify(503, ""), engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() error = %v", err)
	}
	if a.EngineID != "fallback" {
		t.Fatalf("assigned engine = %q, want %q (primary failed)", a.EngineID, "fallback")
	}
	if !a.Surface.IsKnown() {
		t.Errorf("assigned surface %q is outside the closed vocabulary", a.Surface)
	}
	if !auditContains(a.AssignmentAudit, "UPSTREAM_UNAVAILABLE", "FAILOVER_NEXT") {
		t.Errorf("audit does not name class/action: %v", a.AssignmentAudit)
	}
	if !auditContains(a.AssignmentAudit, "chain "+ChainSchema) {
		t.Errorf("audit does not name the ladder chain: %v", a.AssignmentAudit)
	}

	// The failed engine must be EXCLUDED from the failover set, not merely lose
	// to declared order. Three independent, falsifiable witnesses: the winner is
	// not the failed engine, the failover audit records exactly one exclusion,
	// and the evidence line names which engine was excluded. Any of these may
	// fail on its own if the implementation regresses.
	if a.EngineID == "primary" {
		t.Errorf("assigned engine = %q, want the failed engine excluded", a.EngineID)
	}
	if !auditContains(a.AssignmentAudit, "1 excluded") {
		t.Errorf("audit does not record the failed engine's exclusion: %v", a.AssignmentAudit)
	}
	if !auditContains(a.AssignmentAudit, "failed_engine=primary") {
		t.Errorf("audit does not name the excluded failed engine: %v", a.AssignmentAudit)
	}
	for _, r := range rejected {
		if r.EngineID == "primary" {
			t.Errorf("failed primary appeared in Route rejections %+v; it should have been removed before selection", rejected)
		}
	}

	if u.Reason != "" {
		t.Errorf("unrouted reason = %q, want empty on an assignment", u.Reason)
	}
	if a.Schema != AssignmentSchema {
		t.Errorf("schema = %q, want %q", a.Schema, AssignmentSchema)
	}
}

// TestFailoverContextOverflowNeverRetriesSame is the LOAD-BEARING witness that a
// context overflow moves on: a 400 whose body names the context length is a
// CONTEXT_OVERFLOW/FAILOVER_NEXT, so Failover must never re-assign the failed
// primary and must assign the next engine.
func TestFailoverContextOverflowNeverRetriesSame(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T2", "docs"), Role: "worker"}
	engines := failoverEngines()
	failure := Classify(400, "request too large: maximum context length is 8192 tokens")
	if failure.Class != FailureContextOverflow {
		t.Fatalf("precondition: Classify gave %q, want %q", failure.Class, FailureContextOverflow)
	}

	a, _, _, err := Failover(req, "primary", failure, engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() error = %v", err)
	}
	if a.EngineID == "primary" {
		t.Fatal("Failover re-assigned the failed primary on a CONTEXT_OVERFLOW; the same input cannot fit twice")
	}
	if a.EngineID != "fallback" {
		t.Errorf("assigned engine = %q, want %q", a.EngineID, "fallback")
	}
	if !auditContains(a.AssignmentAudit, "CONTEXT_OVERFLOW", "FAILOVER_NEXT") {
		t.Errorf("audit does not name CONTEXT_OVERFLOW/FAILOVER_NEXT: %v", a.AssignmentAudit)
	}
}

// TestFailoverRetrySamePinsFailedEngineFirst witnesses the retry-same action: a
// 429 is transient, so Failover RE-ASSIGNS the failed primary (it is pinned to
// the front) and the audit names RATE_LIMITED/RETRY_SAME.
func TestFailoverRetrySamePinsFailedEngineFirst(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T3", "docs"), Role: "worker"}
	engines := failoverEngines()
	failure := Classify(429, "")
	if failure.Action != ActionRetrySame {
		t.Fatalf("precondition: Classify(429,\"\").Action = %q, want %q", failure.Action, ActionRetrySame)
	}

	a, _, _, err := Failover(req, "primary", failure, engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() error = %v", err)
	}
	if a.EngineID != "primary" {
		t.Errorf("assigned engine = %q, want %q (retry-same pins the failed engine first)", a.EngineID, "primary")
	}
	if !auditContains(a.AssignmentAudit, "RATE_LIMITED", "RETRY_SAME") {
		t.Errorf("audit does not name RATE_LIMITED/RETRY_SAME: %v", a.AssignmentAudit)
	}
}

// TestFailoverMaintenanceInterfaceWhenBigModelsDown is the LOAD-BEARING
// survival witness: with the big primary and fallback both unhealthy, a 503 on
// the primary degrades all the way to the local maintenance engine, and the
// assignment audit says the maintenance interface answered. This is what keeps
// maintenance work served when the big models are all down.
//
// The primary and fallback carry their rung tags ("primary", "small"); the
// maintenance engine is deliberately untagged, so it is the catch-all final
// rung. If the default chain's non-final rungs were unconstrained, the
// maintenance engine would resolve to rung 1 and this line could never fire.
func TestFailoverMaintenanceInterfaceWhenBigModelsDown(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T4", "docs"), Role: "worker"}
	engines := []Engine{
		{ID: "primary", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: false, Tags: []string{"primary"}},
		{ID: "fallback", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: false, Tags: []string{"small"}},
		{ID: "maintenance", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1},
	}

	a, _, _, err := Failover(req, "primary", Classify(503, ""), engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() error = %v", err)
	}
	if a.EngineID != "maintenance" {
		t.Fatalf("assigned engine = %q, want %q", a.EngineID, "maintenance")
	}
	if a.Surface != SurfaceLocal {
		t.Errorf("maintenance assignment surface = %q, want %q", a.Surface, SurfaceLocal)
	}
	if !auditContains(a.AssignmentAudit, "degraded to rung \"maintenance\" (maintenance interface)") {
		t.Errorf("audit does not name the maintenance interface: %v", a.AssignmentAudit)
	}
}

// TestFailoverFatalDoesNotReroute witnesses the FATAL short-circuit: a
// PAYLOAD_REJECTED (caller bug) returns NO assignment, an Unrouted carrying
// ReasonFailureFatal, and an audit saying the request is neither retried nor
// failed over.
func TestFailoverFatalDoesNotReroute(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T5", "docs"), Role: "worker"}
	engines := failoverEngines()
	failure := Classify(400, "invalid json payload")
	if failure.Class != FailureBadPayload || failure.Action != ActionFatal {
		t.Fatalf("precondition: Classify gave %q/%q, want %q/%q", failure.Class, failure.Action, FailureBadPayload, ActionFatal)
	}

	a, _, u, err := Failover(req, "primary", failure, engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() error = %v", err)
	}
	if a.EngineID != "" {
		t.Errorf("assignment engine = %q, want empty (fatal does not re-route)", a.EngineID)
	}
	if u.Reason != ReasonFailureFatal {
		t.Errorf("unrouted reason = %q, want %q", u.Reason, ReasonFailureFatal)
	}
	if !IsKnownReason(u.Reason) {
		t.Errorf("unrouted reason %q is outside the closed vocabulary", u.Reason)
	}
	if !auditContains(u.AssignmentAudit, "fatal") {
		t.Errorf("audit does not say the failure was fatal: %v", u.AssignmentAudit)
	}
	if !auditContains(u.AssignmentAudit, "not retried") {
		t.Errorf("audit does not say the request was not retried: %v", u.AssignmentAudit)
	}
	if !auditContains(u.AssignmentAudit, "PAYLOAD_REJECTED", "FATAL") {
		t.Errorf("audit does not name the class/action: %v", u.AssignmentAudit)
	}
}

// TestFailoverFailsLoud proves Failover refuses a malformed chain, a malformed
// policy, or an unknown failure class before it routes.
func TestFailoverFailsLoud(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T6", "docs"), Role: "worker"}
	engines := failoverEngines()

	shortChain := DegradationChain{Version: "x", Rungs: []Rung{{Name: "only"}}}
	if _, _, _, err := Failover(req, "primary", Classify(503, ""), engines, shortChain, DefaultPolicy()); err == nil {
		t.Error("Failover with a 1-rung chain should fail, got nil")
	}

	badPolicy := Policy{DefaultSurface: Surface("quantum")}
	if _, _, _, err := Failover(req, "primary", Classify(503, ""), engines, DefaultDegradationChain(), badPolicy); err == nil {
		t.Error("Failover with an unknown policy surface should fail, got nil")
	}

	badFailure := Failure{Class: FailureClass("MADE_UP"), Action: ActionFailoverNext}
	if _, _, _, err := Failover(req, "primary", badFailure, engines, DefaultDegradationChain(), DefaultPolicy()); err == nil {
		t.Error("Failover with an unknown failure class should fail, got nil")
	}
}

// TestFailoverLadderChangesOutcome is the LOAD-BEARING invoked-vs-present
// witness: two healthy eligible engines declared A then B, but the ladder's
// "primary" rung matches only B, so Failover must assign B. If the ladder were
// inert, declared order would win and A would be chosen.
func TestFailoverLadderChangesOutcome(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T7", "docs"), Role: "worker"}
	engines := []Engine{
		{ID: "A", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1},
		{ID: "B", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: true, CapacityHeadroom: 1, Tags: []string{"primary"}},
	}
	chain := DegradationChain{
		Version: "test-ladder",
		Rungs: []Rung{
			{Name: "primary", Tags: []string{"primary"}},
			{Name: "maintenance", Surface: SurfaceLocal},
		},
	}
	if err := chain.Validate(); err != nil {
		t.Fatalf("test chain invalid: %v", err)
	}

	a, _, _, err := Failover(req, "failed-engine", Classify(503, ""), engines, chain, DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() error = %v", err)
	}
	if a.EngineID != "B" {
		t.Fatalf("assigned engine = %q, want %q (ladder's primary rung matches only B)", a.EngineID, "B")
	}
	if a.EngineID == "A" {
		t.Fatal("declared order won: the degradation ladder is an inert seam")
	}
}

// TestFailoverUnroutedKeepsAudit witnesses the degraded Unrouted path: with all
// engines unhealthy after a 503, Failover returns an Unrouted with a closed
// reason whose audit still names the failure class and the failover decision.
func TestFailoverUnroutedKeepsAudit(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T8", "docs"), Role: "worker"}
	engines := []Engine{
		{ID: "primary", Surface: SurfaceHosted, Roles: []string{"worker"}, Healthy: false},
		{ID: "fallback", Surface: SurfaceLocal, Roles: []string{"worker"}, Healthy: false},
	}

	a, _, u, err := Failover(req, "primary", Classify(503, ""), engines, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() error = %v", err)
	}
	if a.EngineID != "" {
		t.Errorf("assignment engine = %q, want empty when nothing is eligible", a.EngineID)
	}
	if u.Reason == "" {
		t.Fatal("unrouted reason is empty")
	}
	if !IsKnownReason(u.Reason) {
		t.Errorf("unrouted reason %q is outside the closed vocabulary", u.Reason)
	}
	if !auditContains(u.AssignmentAudit, "UPSTREAM_UNAVAILABLE", "FAILOVER_NEXT") {
		t.Errorf("unrouted audit does not name the failure class/action: %v", u.AssignmentAudit)
	}
	if !auditContains(u.AssignmentAudit, "chain "+ChainSchema) {
		t.Errorf("unrouted audit does not name the ladder chain: %v", u.AssignmentAudit)
	}
	if !auditContains(u.AssignmentAudit, "retained", "excluded") {
		t.Errorf("unrouted audit does not name the retained/excluded counts: %v", u.AssignmentAudit)
	}
}

// TestFailoverEmptyEngineSetIsUnrouted is the fail-closed witness: an empty
// engine set is not a crash, it is an Unrouted with a closed reason. A caller
// with no engines left still gets a coherent backlog fact.
func TestFailoverEmptyEngineSetIsUnrouted(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T9", "docs"), Role: "worker"}

	a, _, u, err := Failover(req, "primary", Classify(503, ""), nil, DefaultDegradationChain(), DefaultPolicy())
	if err != nil {
		t.Fatalf("Failover() with no engines error = %v, want nil (unrouted, not error)", err)
	}
	if a.EngineID != "" {
		t.Errorf("assigned engine = %q, want empty with no engines", a.EngineID)
	}
	if u.Reason != ReasonNoEligibleEngine {
		t.Errorf("unrouted reason = %q, want %q", u.Reason, ReasonNoEligibleEngine)
	}
	if !IsKnownReason(u.Reason) {
		t.Errorf("unrouted reason %q is outside the closed vocabulary", u.Reason)
	}
	if !auditContains(u.AssignmentAudit, "UPSTREAM_UNAVAILABLE", "FAILOVER_NEXT") {
		t.Errorf("unrouted audit does not name class/action: %v", u.AssignmentAudit)
	}
}

// TestFailoverAbsentFailedEngineWitnessesBothActions proves the documented
// fallback when the named failed engine is not in the set: FAILOVER_NEXT uses
// the set as-is (nothing to exclude), and RETRY_SAME degrades to failover-next
// (it cannot pin an engine it does not have), so both keep every engine and
// exclude none.
func TestFailoverAbsentFailedEngineWitnessesBothActions(t *testing.T) {
	req := TaskRequest{Candidate: candidate("T10", "docs"), Role: "worker"}
	engines := failoverEngines()

	t.Run("failover-next absent failed engine uses set as-is", func(t *testing.T) {
		a, _, _, err := Failover(req, "ghost", Classify(503, ""), engines, DefaultDegradationChain(), DefaultPolicy())
		if err != nil {
			t.Fatalf("Failover() error = %v", err)
		}
		if a.EngineID != "primary" {
			t.Errorf("assigned engine = %q, want %q (nothing excluded, declared/ladder order wins)", a.EngineID, "primary")
		}
		if !auditContains(a.AssignmentAudit, "2 engines retained, 0 excluded") {
			t.Errorf("audit does not record the as-is set: %v", a.AssignmentAudit)
		}
	})

	t.Run("retry-same absent failed engine degrades to failover-next", func(t *testing.T) {
		failure := Classify(429, "")
		if failure.Action != ActionRetrySame {
			t.Fatalf("precondition: Classify(429).Action = %q, want %q", failure.Action, ActionRetrySame)
		}
		a, _, _, err := Failover(req, "ghost", failure, engines, DefaultDegradationChain(), DefaultPolicy())
		if err != nil {
			t.Fatalf("Failover() error = %v", err)
		}
		if a.EngineID != "primary" {
			t.Errorf("assigned engine = %q, want %q (absent engine cannot be pinned)", a.EngineID, "primary")
		}
		if !auditContains(a.AssignmentAudit, "2 engines retained, 0 excluded") {
			t.Errorf("audit does not record the degraded retry-same as failover-next: %v", a.AssignmentAudit)
		}
		if !auditContains(a.AssignmentAudit, "RATE_LIMITED", "RETRY_SAME") {
			t.Errorf("audit does not name the original retry-same class/action: %v", a.AssignmentAudit)
		}
	})
}
