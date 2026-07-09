package gateway

import (
	"testing"
	"time"
)

// TestSessionFleetRoundTripsThroughDebugVars is the witness for the cross-machine fleet
// status area: a provider the host sets comes back on /debug/vars, an unset provider omits
// the block, a provider reporting ok=false omits it, and a provider that reports ok=true
// but zero machines is also omitted (never an empty "machines=0" object).
func TestSessionFleetRoundTripsThroughDebugVars(t *testing.T) {
	srv := newTestServer(t)
	if got := srv.debugVars(time.Now()).Fleet; got != nil {
		t.Fatalf("fleet before SetSessionFleetProvider = %+v, want nil (omitted)", got)
	}

	// A provider that reports ok=false must omit the block.
	srv.SetSessionFleetProvider(func() (SessionFleet, bool) { return SessionFleet{}, false })
	if got := srv.debugVars(time.Now()).Fleet; got != nil {
		t.Fatalf("fleet for a not-ok provider = %+v, want nil (omitted)", got)
	}

	// A provider reporting ok=true but zero machines is folded to omitted by sessionFleet,
	// so a cold operator never shows "machines=0".
	srv.SetSessionFleetProvider(func() (SessionFleet, bool) { return SessionFleet{Verdict: "EMPTY"}, true })
	if got := srv.debugVars(time.Now()).Fleet; got != nil {
		t.Fatalf("fleet for a zero-machine provider = %+v, want nil (omitted)", got)
	}

	want := SessionFleet{
		Verdict:           "ACTION",
		Machines:          3,
		Stale:             1,
		Action:            1,
		Sessions:          5,
		AuthBlocked:       1,
		VersionMismatches: 2,
		Rows: []SessionFleetMachine{
			{ID: "win-box", State: "OK", AgeMin: 0.5, Sessions: 2, Version: "v1.2.3"},
			{ID: "lab-box", State: "STALE", AgeMin: 92, Sessions: 1},
		},
	}
	srv.SetSessionFleetProvider(func() (SessionFleet, bool) { return want, true })
	got := srv.debugVars(time.Now()).Fleet
	if got == nil {
		t.Fatal("fleet after SetSessionFleetProvider = nil, want the reported block")
	}
	if got.Verdict != "ACTION" || got.Machines != 3 || got.Stale != 1 || got.Action != 1 {
		t.Fatalf("fleet aggregate = %+v, want ACTION/3 machines/1 stale/1 action", got)
	}
	if got.Sessions != 5 || got.AuthBlocked != 1 || got.VersionMismatches != 2 {
		t.Fatalf("fleet totals = %+v, want 5 sessions/1 auth-blocked/2 version-skew", got)
	}
	if len(got.Rows) != 2 || got.Rows[0].ID != "win-box" || got.Rows[1].State != "STALE" {
		t.Fatalf("fleet.Rows = %+v, want win-box then a stale lab-box", got.Rows)
	}

	// Detaching restores the omitted state.
	srv.SetSessionFleetProvider(nil)
	if got := srv.debugVars(time.Now()).Fleet; got != nil {
		t.Fatalf("fleet after detach = %+v, want nil", got)
	}
}
