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
