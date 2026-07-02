package fleetaccounts

import (
	"testing"
	"time"
)

// futureResetStr renders a reset string resetTime parses ("Jan 2, 3:04pm") for an
// instant d from now. The dated form's 180-day rollover keeps a year-boundary +48h
// instant future, so these stay valid whenever the tests run.
func futureResetStr(d time.Duration) string {
	return time.Now().UTC().Add(d).Format("Jan 2, 3:04pm")
}

func TestLedgerOKClearsCarriedThrottle(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"reset": futureResetStr(48 * time.Hour)},
	}}
	st := computeRuntimeStatus(".claude-a", reg)
	if !st.Available || st.Blocked || st.Throttled {
		t.Fatalf("fresh ledger OK should clear the carried throttle, got %+v", st)
	}
	if st.StatusSource != "probe-ledger" {
		t.Fatalf("status_source = %q, want probe-ledger", st.StatusSource)
	}
}

func TestLedgerOKHoldsActiveWeeklyCap(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{
			"reset":  futureResetStr(4 * time.Hour),
			"weekly": futureResetStr(72 * time.Hour),
		},
	}}
	st := computeRuntimeStatus(".claude-a", reg)
	if st.Available || !st.Blocked || !st.Throttled {
		t.Fatalf("fresh OK must not reopen an active weekly cap, got %+v", st)
	}
	if st.BlockKind != "usage" || st.Weekly == "" {
		t.Fatalf("weekly hold shape: kind=%q weekly=%q", st.BlockKind, st.Weekly)
	}
}

func TestLedgerBlockOverridesRegistry(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "LIMIT", time.Now(), `"reset":"3pm"`))
	st := computeRuntimeStatus(".claude-a", Registry{})
	if st.Available || !st.Blocked || st.BlockKind != "usage" || !st.Throttled {
		t.Fatalf("fresh ledger LIMIT should block, got %+v", st)
	}
	if st.StatusSource != "probe-ledger" || st.Reset != "3pm" {
		t.Fatalf("source/reset = %q/%q, want probe-ledger/3pm", st.StatusSource, st.Reset)
	}
}

func TestLedgerIgnoredWithoutRegDir(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", "")
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Throttle: map[string]any{
		".claude-a": map[string]any{"reset": futureResetStr(48 * time.Hour)},
	}}
	st := computeRuntimeStatus(".claude-a", reg)
	if st.Available || !st.Throttled {
		t.Fatalf("without FLEET_REG_DIR the passive fold should keep the throttle, got %+v", st)
	}
	if st.StatusSource != "registry" {
		t.Fatalf("status_source = %q, want registry", st.StatusSource)
	}
}

func TestSyntheticProbeRowWinsOverLedger(t *testing.T) {
	rd := t.TempDir()
	t.Setenv("FLEET_REG_DIR", rd)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	reg := Registry{Sessions: []Session{{
		Account:     ".claude-a",
		Project:     "_probe",
		ProbeStatus: "LIMIT",
		Reason:      "usage limit",
	}}}
	st := computeRuntimeStatus(".claude-a", reg)
	if st.Available || st.BlockKind != "usage" || st.StatusSource != "probe" {
		t.Fatalf("synthetic _probe row should win over the ledger, got %+v", st)
	}
}
