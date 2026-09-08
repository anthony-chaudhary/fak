package main

import (
	"bytes"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
	"github.com/anthony-chaudhary/fak/internal/session"
)

func TestGuardSoftDeadline(t *testing.T) {
	t.Run("Check", TestGuardCheckSoftDeadline)
	t.Run("Normalize", TestNormalizeGuardDeadlineSettings)
	t.Run("EmitWarning", TestGuardEmitSoftDeadlineWarning)
}

func TestGuardCheckSoftDeadline(t *testing.T) {
	t0 := time.Unix(1_700_000_000, 0)

	// Bounded table with 30m limit.
	tbl := session.NewTable()
	tbl.StartTimeBudget("test-trace", 30*time.Minute, t0)

	// At +10m (20m remaining, lead=2m) -> warn=false.
	if warn, rem := guardCheckSoftDeadline(tbl, "test-trace", 2*time.Minute, t0.Add(10*time.Minute)); warn {
		t.Fatalf("at +10m: expected warn=false, got warn=true (rem=%v)", rem)
	}

	// At +28m30s (1m30s remaining, lead=2m) -> warn=true, remaining=90s.
	if warn, rem := guardCheckSoftDeadline(tbl, "test-trace", 2*time.Minute, t0.Add(28*time.Minute+30*time.Second)); !warn || rem != 90*time.Second {
		t.Fatalf("at +28m30s: expected warn=true rem=90s, got warn=%v rem=%v", warn, rem)
	}

	// At +31m (past limit) -> warn=false.
	if warn, rem := guardCheckSoftDeadline(tbl, "test-trace", 2*time.Minute, t0.Add(31*time.Minute)); warn {
		t.Fatalf("at +31m: expected warn=false, got warn=true (rem=%v)", rem)
	}

	// Unbounded table (limit=0) -> warn=false.
	unboundedTbl := session.NewTable()
	unboundedTbl.StartTimeBudget("unbounded-trace", 0, t0)
	if warn, rem := guardCheckSoftDeadline(unboundedTbl, "unbounded-trace", 2*time.Minute, t0.Add(10*time.Minute)); warn {
		t.Fatalf("unbounded table: expected warn=false, got warn=true (rem=%v)", rem)
	}

	// Nil table and empty traceID -> warn=false.
	if warn, _ := guardCheckSoftDeadline(nil, "test-trace", 2*time.Minute, t0); warn {
		t.Fatalf("nil table: expected warn=false, got true")
	}
	if warn, _ := guardCheckSoftDeadline(tbl, "", 2*time.Minute, t0); warn {
		t.Fatalf("empty traceID: expected warn=false, got true")
	}
	if warn, _ := guardCheckSoftDeadline(tbl, "test-trace", 0, t0); warn {
		t.Fatalf("zero lead: expected warn=false, got true")
	}
}

func TestNormalizeGuardDeadlineSettings(t *testing.T) {
	// Tests clamping when softLead >= maxDuration (clamped to maxDuration/2).
	cfg1 := normalizeGuardDeadlineSettings(10*time.Minute, 10*time.Minute, 30*time.Second, 5*time.Second)
	if cfg1.SoftDeadlineLead != 5*time.Minute {
		t.Fatalf("expected softLead clamped to 5m, got %v", cfg1.SoftDeadlineLead)
	}

	cfg2 := normalizeGuardDeadlineSettings(10*time.Minute, 15*time.Minute, 30*time.Second, 5*time.Second)
	if cfg2.SoftDeadlineLead != 5*time.Minute {
		t.Fatalf("expected softLead clamped to 5m, got %v", cfg2.SoftDeadlineLead)
	}

	// Normal valid softLead within bound
	cfg3 := normalizeGuardDeadlineSettings(10*time.Minute, 2*time.Minute, 30*time.Second, 5*time.Second)
	if cfg3.SoftDeadlineLead != 2*time.Minute {
		t.Fatalf("expected softLead 2m, got %v", cfg3.SoftDeadlineLead)
	}

	// Tests maxDuration <= 0 resets lead/grace.
	cfgZero := normalizeGuardDeadlineSettings(0, 2*time.Minute, 30*time.Second, 5*time.Second)
	if cfgZero.MaxDuration != 0 || cfgZero.SoftDeadlineLead != 0 || cfgZero.CommitGracePeriod != 0 {
		t.Fatalf("expected maxDuration<=0 to reset lead and grace, got %+v", cfgZero)
	}

	cfgNegative := normalizeGuardDeadlineSettings(-5*time.Minute, 2*time.Minute, 30*time.Second, 5*time.Second)
	if cfgNegative.MaxDuration != 0 || cfgNegative.SoftDeadlineLead != 0 || cfgNegative.CommitGracePeriod != 0 {
		t.Fatalf("expected negative maxDuration to reset lead and grace, got %+v", cfgNegative)
	}

	// Tests childStopGrace floor.
	cfgFloor1 := normalizeGuardDeadlineSettings(10*time.Minute, 2*time.Minute, 30*time.Second, 500*time.Millisecond)
	if cfgFloor1.ChildStopGrace != 3*time.Second {
		t.Fatalf("expected stopGrace floored to 3s when <1s, got %v", cfgFloor1.ChildStopGrace)
	}

	cfgFloor2 := normalizeGuardDeadlineSettings(10*time.Minute, 2*time.Minute, 30*time.Second, 0)
	if cfgFloor2.ChildStopGrace != 3*time.Second {
		t.Fatalf("expected stopGrace floored to 3s when 0, got %v", cfgFloor2.ChildStopGrace)
	}

	cfgFloor3 := normalizeGuardDeadlineSettings(10*time.Minute, 2*time.Minute, 30*time.Second, 2*time.Second)
	if cfgFloor3.ChildStopGrace != 2*time.Second {
		t.Fatalf("expected stopGrace 2s preserved, got %v", cfgFloor3.ChildStopGrace)
	}

	// Negative commitGrace and softLead
	cfgNegLead := normalizeGuardDeadlineSettings(10*time.Minute, -1*time.Minute, -10*time.Second, 3*time.Second)
	if cfgNegLead.SoftDeadlineLead != 0 || cfgNegLead.CommitGracePeriod != 0 {
		t.Fatalf("expected negative lead/grace clamped to 0, got %+v", cfgNegLead)
	}
}

func TestGuardEmitSoftDeadlineWarning(t *testing.T) {
	var buf1 bytes.Buffer
	var buf2 bytes.Buffer
	j := journal.OpenMemory()

	guardEmitSoftDeadlineWarning(&buf1, &buf2, j, nil, "trace-123", 90*time.Second, 30*time.Minute)

	expectedMsg := "fak guard: WARNING — soft deadline reached (1m30s remaining of 30m0s max-duration); please flush and commit all in-flight work\n"
	if buf1.String() != expectedMsg {
		t.Fatalf("buf1 got %q, want %q", buf1.String(), expectedMsg)
	}
	if buf2.String() != expectedMsg {
		t.Fatalf("buf2 got %q, want %q", buf2.String(), expectedMsg)
	}

	// Test deduplication when w1 == w2
	var bufSame bytes.Buffer
	guardEmitSoftDeadlineWarning(&bufSame, &bufSame, nil, nil, "trace-123", 90*time.Second, 30*time.Minute)
	if bufSame.String() != expectedMsg {
		t.Fatalf("bufSame got %q, want single message %q", bufSame.String(), expectedMsg)
	}

	// Verify journal event
	rows := j.Recent(1)
	if len(rows) != 1 {
		t.Fatalf("expected 1 journal row, got %d", len(rows))
	}
	row := rows[0]
	if row.Kind != "SOFT_DEADLINE_WARNING" {
		t.Fatalf("expected Kind=SOFT_DEADLINE_WARNING, got %q", row.Kind)
	}
	if row.TraceID != "trace-123" {
		t.Fatalf("expected TraceID=trace-123, got %q", row.TraceID)
	}
	if !strings.Contains(row.Reason, "remaining=1m30s limit=30m0s") {
		t.Fatalf("expected Reason to contain remaining and limit, got %q", row.Reason)
	}
}
