package fleetaccounts

import (
	"testing"
	"time"
)

// The synthetic-_probe rung (a project=="_probe" session row) is the sibling of the
// probe-ledger rung: both are "a fresh probe just hit the live account". These tests pin
// that the weekly-cap identity hold #2253 ported into the ledger rung applies EQUALLY to
// the synthetic-probe rung, matching the Python's fresh_probe_ok branch. Without it the Go
// port reopened a still-weekly-capped seat that Python holds, so the roster would offer a
// seat that fails on first use. No FLEET_REG_DIR is set: the synthetic rung returns before
// the ledger consult, so these exercise it in isolation.

// probeOKReg builds a registry with one synthetic _probe OK row for .claude-a plus the
// given carried throttle, the exact shape the fresh_probe_ok branch folds.
func probeOKReg(thr map[string]any) Registry {
	return Registry{
		Sessions: []Session{{
			Account: ".claude-a", Project: "_probe", ProbeStatus: "OK",
		}},
		Throttle: map[string]any{".claude-a": thr},
	}
}

func TestSyntheticProbeOKHoldsActiveWeeklyCap(t *testing.T) {
	reg := probeOKReg(map[string]any{
		"reset":  futureResetStr(4 * time.Hour),
		"weekly": futureResetStr(72 * time.Hour),
	})
	st := computeRuntimeStatus(".claude-a", "", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("synthetic-probe OK must not reopen an active weekly cap, got %+v", st)
	}
	if st.BlockKind != "usage" || st.Weekly == "" {
		t.Fatalf("weekly hold shape: kind=%q weekly=%q", st.BlockKind, st.Weekly)
	}
}

// TestSyntheticProbeOKClearsThrottleOnIdentityMismatch is the #2253 refinement on the
// synthetic rung: a fresh OK reopens a weekly-capped seat when the cap provably belonged to
// a DIFFERENT account (the dir was re-logged since the cap was stamped).
func TestSyntheticProbeOKClearsThrottleOnIdentityMismatch(t *testing.T) {
	// The dir is logged in as BBBB now; the carried weekly cap was stamped for AAAA.
	dir := writeAccountConfig(t, "BBBB")
	reg := probeOKReg(map[string]any{
		"reset":        futureResetStr(4 * time.Hour),
		"weekly":       futureResetStr(72 * time.Hour),
		"account_uuid": "AAAA",
	})
	st := computeRuntimeStatus(".claude-a", dir, reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("proven identity mismatch must clear the weekly hold, got %+v", st)
	}
	if st.StatusSource != "probe" {
		t.Fatalf("status_source = %q, want probe", st.StatusSource)
	}
}

// TestSyntheticProbeOKHoldsThrottleOnIdentityMatch is the other verdict direction: the
// cap's stamped identity matches the seat's CURRENT login, so the weekly window still binds
// and the fresh OK does not reopen the seat.
func TestSyntheticProbeOKHoldsThrottleOnIdentityMatch(t *testing.T) {
	dir := writeAccountConfig(t, "AAAA")
	reg := probeOKReg(map[string]any{
		"reset":        futureResetStr(4 * time.Hour),
		"weekly":       futureResetStr(72 * time.Hour),
		"account_uuid": "AAAA",
	})
	st := computeRuntimeStatus(".claude-a", dir, reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("matching identity must keep the active weekly cap, got %+v", st)
	}
	if st.BlockKind != "usage" || st.Weekly == "" {
		t.Fatalf("weekly hold shape: kind=%q weekly=%q", st.BlockKind, st.Weekly)
	}
}

// TestSyntheticProbeOKClearsCarriedDailyThrottle pins that the hold is WEEKLY-only: a
// carried daily-only cap (no weekly leg) never holds the seat, so a fresh OK reopens it and
// the daily reset is surfaced advisory-only. This is the day24 incident the reopen exists
// for, and guards against the hold over-firing on ordinary daily caps.
func TestSyntheticProbeOKClearsCarriedDailyThrottle(t *testing.T) {
	reg := probeOKReg(map[string]any{"reset": futureResetStr(48 * time.Hour)})
	st := computeRuntimeStatus(".claude-a", "", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("daily-only cap must reopen on a fresh OK, got %+v", st)
	}
	if st.StatusSource != "probe" {
		t.Fatalf("status_source = %q, want probe", st.StatusSource)
	}
}
