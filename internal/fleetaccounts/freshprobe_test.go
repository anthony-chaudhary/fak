package fleetaccounts

import (
	"os"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/accountprobe"
)

// writeProbeLedger writes probe ledger lines under regDir.
func writeProbeLedger(t *testing.T, regDir string, lines ...string) {
	t.Helper()
	if err := os.MkdirAll(regDir, 0o755); err != nil {
		t.Fatal(err)
	}
	body := ""
	for _, l := range lines {
		body += l + "\n"
	}
	if err := os.WriteFile(accountprobe.ProbeLedgerPath(regDir), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func probeLine(t *testing.T, account, status string, when time.Time, extra string) string {
	t.Helper()
	ts := when.UTC().Format(time.RFC3339)
	line := `{"ts":"` + ts + `","account":"` + account + `","status":"` + status + `"`
	if extra != "" {
		line += "," + extra
	}
	return line + "}"
}

func TestFreshProbeFromLedgerNoEntry(t *testing.T) {
	rd := t.TempDir()
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", time.Now(), ""))
	if got := FreshProbeFromLedger(".claude-missing", rd, time.Now().UTC(), 0); got != nil {
		t.Fatalf("FreshProbeFromLedger for missing account = %+v, want nil", got)
	}
}

func TestFreshProbeFromLedgerFreshOK(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", now.Add(-5*time.Minute), ""))
	got := FreshProbeFromLedger(".claude-a", rd, now, 0)
	if got == nil || !got.Available {
		t.Fatalf("fresh OK = %+v, want available", got)
	}
	if got.AgeMin < 4.9 || got.AgeMin > 5.1 {
		t.Fatalf("age_min = %v, want ~5", got.AgeMin)
	}
}

func TestFreshProbeFromLedgerFreshLimitWithReset(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "LIMIT", now.Add(-2*time.Minute), `"reset":"3pm","weekly":"Jul 5"`))
	got := FreshProbeFromLedger(".claude-a", rd, now, 0)
	if got == nil || got.Available {
		t.Fatalf("fresh LIMIT = %+v, want blocked", got)
	}
	if got.BlockKind != "usage" {
		t.Fatalf("block_kind = %q, want usage", got.BlockKind)
	}
	if got.BlockReason != "usage limit; resets 3pm" {
		t.Fatalf("block_reason = %q", got.BlockReason)
	}
	if got.Reset != "3pm" || got.Weekly != "Jul 5" {
		t.Fatalf("reset/weekly = %q/%q, want 3pm/Jul 5", got.Reset, got.Weekly)
	}
}

func TestFreshProbeFromLedgerAuthDefaultReason(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "AUTH", now.Add(-1*time.Minute), ""))
	got := FreshProbeFromLedger(".claude-a", rd, now, 0)
	if got == nil || got.Available {
		t.Fatalf("fresh AUTH = %+v, want blocked", got)
	}
	if got.BlockKind != "auth" || got.BlockReason != "auth block" {
		t.Fatalf("kind/reason = %q/%q, want auth/'auth block'", got.BlockKind, got.BlockReason)
	}
}

func TestFreshProbeFromLedgerStaleIsNil(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	// 25 min old, explicit fresh window 20 -> stale -> nil (independent of any
	// ambient FLEET_PROBE_FRESH_MIN).
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "OK", now.Add(-25*time.Minute), ""))
	if got := FreshProbeFromLedger(".claude-a", rd, now, 20); got != nil {
		t.Fatalf("stale probe = %+v, want nil", got)
	}
}

func TestFreshProbeFromLedgerUnknownStatusIsNil(t *testing.T) {
	rd := t.TempDir()
	now := time.Date(2026, 7, 1, 12, 0, 0, 0, time.UTC)
	writeProbeLedger(t, rd, probeLine(t, ".claude-a", "APIERR", now.Add(-1*time.Minute), ""))
	if got := FreshProbeFromLedger(".claude-a", rd, now, 0); got != nil {
		t.Fatalf("APIERR probe = %+v, want nil (not a clean signal)", got)
	}
}

func TestProbeLedgerFreshMinEnvOverride(t *testing.T) {
	t.Setenv("FLEET_PROBE_FRESH_MIN", "45")
	if got := ProbeLedgerFreshMin(); got != 45 {
		t.Fatalf("ProbeLedgerFreshMin = %v, want 45", got)
	}
}
