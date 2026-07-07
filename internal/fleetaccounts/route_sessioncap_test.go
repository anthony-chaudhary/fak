package fleetaccounts

import (
	"strings"
	"testing"
)

// The single-launch route admits through the per-account session budget the same
// way AllocateWave does: a route-weighted seat already running at its cap spills
// to the next candidate, and when every seat is full the route refuses instead of
// overfilling one account until it limit-walls.

func capRoster() []Account {
	return []Account{
		{Dir: "C:/u/.claude-weighted", Product: "claude", Account: ".claude-weighted", Tag: "weighted",
			Kind: KindWorker, ModelTier: intp(1), Available: boolp(true),
			RouteWeight: intp(10), LiveSessions: intp(DefaultClaudeSessionsPerAccount)},
		{Dir: "C:/u/.claude-idle", Product: "claude", Account: ".claude-idle", Tag: "idle",
			Kind: KindWorker, ModelTier: intp(1), Available: boolp(true), LiveSessions: intp(1)},
	}
}

func TestRouteAccountSessionCapSpill(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
	route := RouteAccount(capRoster(), "implement the feature", "engineering", false, false, "claude", DefaultPolicy())
	if !route.OK || route.Account == nil || route.Account.Tag != "idle" {
		t.Fatalf("route = %+v, want under-cap idle seat over route-weighted seat at cap", route)
	}
	if len(route.BlockedTargetAccounts) != 1 || route.BlockedTargetAccounts[0].Tag != "weighted" ||
		!strings.Contains(route.BlockedTargetAccounts[0].Reason, "at session cap") {
		t.Fatalf("blocked = %+v, want the at-cap weighted seat named", route.BlockedTargetAccounts)
	}
}

func TestRouteAccountSessionCapRefusal(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
	rows := capRoster()
	rows[1].LiveSessions = intp(DefaultClaudeSessionsPerAccount)
	route := RouteAccount(rows, "implement the feature", "engineering", false, false, "claude", DefaultPolicy())
	if route.OK || !strings.Contains(route.Reason, "session cap") {
		t.Fatalf("route = %+v, want refusal naming the session cap", route)
	}
	if len(route.BlockedTargetAccounts) != 2 {
		t.Fatalf("blocked = %+v, want both at-cap seats listed", route.BlockedTargetAccounts)
	}

	// Raising the budget (the documented env knob) re-admits the same roster.
	t.Setenv(SessionsPerAccountEnv, "9")
	relaxed := RouteAccount(rows, "implement the feature", "engineering", false, false, "claude", DefaultPolicy())
	if !relaxed.OK || relaxed.Account.Tag != "weighted" {
		t.Fatalf("route with raised cap = %+v, want route-weighted seat admitted again", relaxed)
	}
}

func TestAllocateWaveFloorsLoadAtRegistryLiveSessions(t *testing.T) {
	t.Setenv(SessionsPerAccountEnv, "")
	rows := capRoster()[1:] // just the idle seat, one live session already running
	rows[0].LiveSessions = intp(3)
	wave := AllocateWave(rows, WaveRequest{Count: 4, TaskText: "implement the feature",
		TaskClass: "engineering", Product: "claude"}, DefaultPolicy())
	if !wave.OK || wave.Granted != 1 || wave.Shortfall != 3 || wave.Lanes[0].SessionSlot != 4 {
		t.Fatalf("wave = %+v, want registry-live sessions to consume 3 of 4 slots", wave)
	}
}
