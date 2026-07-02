package gateway

import (
	"context"
	"testing"
	"time"
)

func TestDebugVarsMirrorsActiveSessionAssumptions(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{
			{
				TraceID: "live",
				Run:     "running",
				Assumptions: []SessionAssumption{
					{Key: "deployment-target", Statement: "staging", Source: "user_stated", Confidence: 0.95, Expiry: "session"},
				},
			},
			{
				TraceID: "done",
				Run:     "stopped",
				Assumptions: []SessionAssumption{
					{Key: "old", Statement: "terminal row", Source: "inferred", Confidence: 0.5, Expiry: "none"},
				},
			},
		}
	}

	vars := srv.debugVars(time.Now())
	if len(vars.Assumptions) != 1 {
		t.Fatalf("debug assumptions = %+v, want exactly the live session row", vars.Assumptions)
	}
	got := vars.Assumptions[0]
	if got.TraceID != "live" || got.Key != "deployment-target" || got.Source != "user_stated" || got.Expiry != "session" {
		t.Fatalf("debug assumption row = %+v, want live session provenance", got)
	}
}

// TestDebugVarsMirrorsLiveSessions pins the /debug/vars sessions block: every
// non-stopped session becomes one row carrying its sub-agent lineage
// (parent/generation), remaining budget axes, live wall-clock, and assumption
// count — and a stopped session is dropped, matching the assumptions mirror.
func TestDebugVarsMirrorsLiveSessions(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = func(context.Context) []SessionState {
		return []SessionState{
			{
				TraceID: "main",
				Run:     "running",
				Budget:  SessionBudget{TurnsLeft: 7, TokensLeft: 380_000, ContextTokensLeft: 120_000},
				Time:    SessionTime{Bounded: true, ElapsedSeconds: 95, RemainingSeconds: 505, LimitSeconds: 600},
				Assumptions: []SessionAssumption{
					{Key: "deployment-target", Statement: "staging", Source: "user_stated"},
				},
			},
			{
				TraceID:     "sub-1",
				Run:         "running",
				ParentTrace: "main",
				Generation:  1,
				Budget:      SessionBudget{TurnsLeft: 3, TokensLeft: 50_000},
			},
			{TraceID: "done", Run: "stopped"},
		}
	}

	vars := srv.debugVars(time.Now())
	if len(vars.Sessions) != 2 {
		t.Fatalf("debug sessions = %+v, want the two live rows (stopped dropped)", vars.Sessions)
	}
	main := vars.Sessions[0]
	if main.TraceID != "main" || main.Run != "running" || main.ParentTrace != "" || main.Generation != 0 {
		t.Fatalf("main session row = %+v, want a root (no parent, generation 0)", main)
	}
	if main.TurnsLeft != 7 || main.TokensLeft != 380_000 || main.ContextTokensLeft != 120_000 {
		t.Fatalf("main session budget = %+v, want the registry's remaining axes", main)
	}
	if main.ElapsedSeconds != 95 {
		t.Fatalf("main session elapsed = %d, want the live wall-clock 95", main.ElapsedSeconds)
	}
	if main.Assumptions != 1 {
		t.Fatalf("main session assumptions = %d, want the count 1", main.Assumptions)
	}
	sub := vars.Sessions[1]
	if sub.TraceID != "sub-1" || sub.ParentTrace != "main" || sub.Generation != 1 {
		t.Fatalf("sub-agent row = %+v, want lineage parent=main generation=1", sub)
	}
}

// TestDebugVarsSessionsNilWithoutRegistry pins the fail-quiet contract: a gateway with
// no session registry wired (listSessions == nil) omits the block rather than
// fabricating an empty-but-present one.
func TestDebugVarsSessionsNilWithoutRegistry(t *testing.T) {
	srv := newTestServer(t)
	srv.listSessions = nil
	if got := srv.debugVars(time.Now()).Sessions; got != nil {
		t.Fatalf("sessions with no registry = %+v, want nil (block omitted)", got)
	}
}
