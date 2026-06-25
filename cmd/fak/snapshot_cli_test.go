package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/snapshot"
)

// TestFleetWireDumpRestore proves over-the-wire fleet offload end to end against a fake
// gateway (the #620 routes): dump captures every live session's drive into a fleet
// snapshot, and restore re-establishes each axis on the target — with a terminal session
// counted as a skip (a 409 the live table returns), not a fatal error, because a wire
// restore cannot resurrect a session the target already drove terminal.
func TestFleetWireDumpRestore(t *testing.T) {
	var mu sync.Mutex
	posts := map[string]bool{}

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet && r.URL.Path == "/v1/fak/sessions" {
			_ = json.NewEncoder(w).Encode(gateway.SessionListResponse{
				Count: 2,
				Sessions: []gateway.SessionState{
					{TraceID: "a", Run: "throttled", Budget: gateway.SessionBudget{TurnsLeft: 2, TokensLeft: 1000}, Priority: 5, Pace: gateway.SessionPace{MaxTokensPerTurn: 256, MinTurnGapMs: 50}, Reason: "operator-offload", Rev: 3},
					{TraceID: "b", Run: "stopped", Budget: gateway.SessionBudget{TurnsLeft: 0, TokensLeft: 0}, Reason: "BUDGET_TURNS_EXHAUSTED", Rev: 9},
				},
			})
			return
		}
		// POST /v1/fak/session/{id}/{verb}
		rest := strings.TrimPrefix(r.URL.Path, "/v1/fak/session/")
		parts := strings.SplitN(rest, "/", 2)
		if len(parts) == 2 {
			mu.Lock()
			posts[parts[0]+"/"+parts[1]] = true
			mu.Unlock()
			// The target session "b" is already terminal: setting its run is refused 409.
			if parts[0] == "b" && parts[1] == "run" {
				http.Error(w, `{"error":{"message":"terminal"}}`, http.StatusConflict)
				return
			}
			_ = json.NewEncoder(w).Encode(gateway.SessionState{TraceID: parts[0], Run: "running"})
			return
		}
		http.Error(w, "bad path", http.StatusNotFound)
	}))
	defer srv.Close()

	c := newFleetClient(srv.URL, "")
	dir := t.TempDir()
	f := filepath.Join(dir, "fleet.snap")

	// Dump the live fleet.
	n, err := dumpFleetWire(c, "fleet-test", f, 1)
	if err != nil {
		t.Fatalf("dumpFleetWire: %v", err)
	}
	if n != 2 {
		t.Fatalf("dumped %d sessions, want 2", n)
	}

	// The snapshot round-trips with the right drive (incl. the stopped session).
	snap, err := snapshot.ReadFile(f)
	if err != nil {
		t.Fatalf("ReadFile: %v", err)
	}
	var body snapshot.FleetBody
	if err := snap.Into(&body); err != nil {
		t.Fatalf("Into: %v", err)
	}
	if len(body.Sessions) != 2 {
		t.Fatalf("snapshot has %d sessions, want 2", len(body.Sessions))
	}
	byID := map[string]session.State{}
	for _, s := range body.Sessions {
		byID[s.TraceID] = s
	}
	if a := byID["a"]; a.Run != session.Throttled || a.Budget.TokensLeft != 1000 || a.Priority != 5 {
		t.Fatalf("session a dumped wrong: %+v", a)
	}
	if b := byID["b"]; b.Run != session.Stopped || b.Reason != "BUDGET_TURNS_EXHAUSTED" || b.Rev != 9 {
		t.Fatalf("session b (stopped) dumped wrong: %+v", b)
	}

	// Restore onto the (same fake) gateway: a restores fully; b's run is refused (409),
	// so b is a skip, not a fatal error.
	restored, skipped, err := restoreFleetWire(c, f, io.Discard)
	if err != nil {
		t.Fatalf("restoreFleetWire: %v", err)
	}
	if restored != 1 || skipped != 1 {
		t.Fatalf("restore counts = (restored=%d skipped=%d), want (1,1)", restored, skipped)
	}
	// Every axis was POSTed for both sessions (the run for b is what 409'd).
	for _, want := range []string{"a/budget", "a/pace", "a/priority", "a/run", "b/budget", "b/pace", "b/priority", "b/run"} {
		if !posts[want] {
			t.Fatalf("expected POST %q was not made", want)
		}
	}
}
