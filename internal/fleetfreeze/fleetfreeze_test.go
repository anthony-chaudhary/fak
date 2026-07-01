package fleetfreeze

import (
	"strings"
	"testing"
)

// a fixed, non-zero freeze instant so the whole suite is deterministic.
// 2026-06-21T17:00:00Z.
const testFreezeUnix int64 = 1782061200

// TestFrozenHoldsSpawnButAllowsHarvest is the core witness: when frozen the
// spawn class is HELD with an actionable reason, while the progress-harvesting
// classes (witness-close, status-refresh, comment) stay ALLOWED.
func TestFrozenHoldsSpawnButAllowsHarvest(t *testing.T) {
	st := Freeze("draining before rate-limit cooldown", testFreezeUnix)

	spawn := Allowed(st, OpSpawn)
	if spawn.Allow {
		t.Fatalf("frozen: OpSpawn must be HELD, got Allow=true")
	}
	if spawn.Reason == "" {
		t.Fatalf("frozen: held OpSpawn must carry an actionable reason, got empty")
	}
	// the reason must name the freeze so an operator can act on it.
	for _, want := range []string{"held", "frozen", "draining before rate-limit cooldown"} {
		if !strings.Contains(spawn.Reason, want) {
			t.Errorf("frozen: OpSpawn reason %q missing %q", spawn.Reason, want)
		}
	}
	// the reason must state that harvesting continues (the reassurance).
	if !strings.Contains(spawn.Reason, "witness-close") ||
		!strings.Contains(spawn.Reason, "status-refresh") {
		t.Errorf("frozen: OpSpawn reason %q should say close/status still allowed", spawn.Reason)
	}

	// the harvesting classes must remain allowed while frozen.
	for _, op := range []OpClass{OpWitnessClose, OpStatusRefresh, OpComment} {
		d := Allowed(st, op)
		if !d.Allow {
			t.Errorf("frozen: %s must remain ALLOWED (progress harvesting), got held: %s", op, d.Reason)
		}
		if d.Reason != "" {
			t.Errorf("frozen: allowed %s should have empty reason, got %q", op, d.Reason)
		}
	}
}

// TestNotFrozenAllowsEverything: with no freeze, every class proceeds.
func TestNotFrozenAllowsEverything(t *testing.T) {
	for name, st := range map[string]State{
		"zero-value":        {},
		"explicit-unfreeze": Unfreeze(),
	} {
		for _, op := range []OpClass{OpSpawn, OpWitnessClose, OpStatusRefresh, OpComment} {
			d := Allowed(st, op)
			if !d.Allow {
				t.Errorf("%s: %s must be allowed when not frozen, got held: %s", name, op, d.Reason)
			}
			if d.Reason != "" {
				t.Errorf("%s: allowed %s should have empty reason, got %q", name, op, d.Reason)
			}
		}
	}
}

// TestDryRunStringShowsFreezeStateAndReason: the operator dry-run status line
// reflects both states, and the frozen line carries the reason + the
// harvest-continues reassurance.
func TestDryRunStringShowsFreezeStateAndReason(t *testing.T) {
	running := Unfreeze().String()
	if !strings.Contains(running, "RUNNING") {
		t.Errorf("running String() should say RUNNING, got %q", running)
	}
	if strings.Contains(running, "FROZEN") {
		t.Errorf("running String() should not say FROZEN, got %q", running)
	}

	frozen := Freeze("holding for a risky migration", testFreezeUnix).String()
	for _, want := range []string{
		"FROZEN",
		"holding for a risky migration", // the reason
		"2026-06-21T17:00:00Z",          // the since-time (deterministic)
		"HELD",                          // spawns held
		"witness-close",                 // harvest continues
		"status-refresh",
	} {
		if !strings.Contains(frozen, want) {
			t.Errorf("frozen String() %q missing %q", frozen, want)
		}
	}
}

// TestFreezeBackfillsBlankReason: a freeze with no reason still yields an
// actionable held-spawn decision (never an empty reason).
func TestFreezeBackfillsBlankReason(t *testing.T) {
	st := Freeze("", testFreezeUnix)
	if st.Reason == "" {
		t.Fatalf("Freeze(\"\") must backfill a reason, got empty")
	}
	d := Allowed(st, OpSpawn)
	if d.Allow || d.Reason == "" {
		t.Fatalf("blank-reason freeze must still HOLD spawn with a reason, got Allow=%v reason=%q", d.Allow, d.Reason)
	}
}

// TestFreezeRecordsSince: the freeze instant is taken from input, not the
// clock, and surfaces in the state.
func TestFreezeRecordsSince(t *testing.T) {
	st := Freeze("x", testFreezeUnix)
	if !st.Frozen {
		t.Fatalf("Freeze must set Frozen")
	}
	if st.SinceUnix != testFreezeUnix {
		t.Errorf("Freeze must record the supplied instant %d, got %d", testFreezeUnix, st.SinceUnix)
	}
}

// TestZeroSinceReportsUnknown: a freeze recorded without a clock (SinceUnix 0)
// reports "unknown time", not the Unix epoch, so it never misleads.
func TestZeroSinceReportsUnknown(t *testing.T) {
	st := Freeze("no clock available", 0)
	got := Allowed(st, OpSpawn).Reason
	if !strings.Contains(got, "unknown time") {
		t.Errorf("zero-since held reason should say 'unknown time', got %q", got)
	}
	if strings.Contains(got, "1970") {
		t.Errorf("zero-since should not render the Unix epoch, got %q", got)
	}
}

// TestOpClassString: operation classes render their operator-facing names.
func TestOpClassString(t *testing.T) {
	cases := map[OpClass]string{
		OpSpawn:         "spawn",
		OpWitnessClose:  "witness-close",
		OpStatusRefresh: "status-refresh",
		OpComment:       "comment",
	}
	for op, want := range cases {
		if got := op.String(); got != want {
			t.Errorf("OpClass(%d).String() = %q, want %q", int(op), got, want)
		}
	}
	if got := OpClass(99).String(); !strings.Contains(got, "99") {
		t.Errorf("unknown OpClass should render its number, got %q", got)
	}
}
