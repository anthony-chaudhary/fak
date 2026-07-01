package accountprobe

import (
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeLedger(t *testing.T, rd string, lines ...string) {
	t.Helper()
	if err := os.MkdirAll(rd, 0o755); err != nil {
		t.Fatal(err)
	}
	body := ""
	for _, l := range lines {
		body += l + "\n"
	}
	if err := os.WriteFile(ProbeLedgerPath(rd), []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestProbeLedgerPath(t *testing.T) {
	if got := ProbeLedgerPath("/reg"); got != filepath.Join("/reg", "probe_ledger.jsonl") {
		t.Fatalf("ProbeLedgerPath = %q", got)
	}
	t.Setenv("FLEET_REG_DIR", "/env/reg")
	if got := ProbeLedgerPath(""); got != filepath.Join("/env/reg", "probe_ledger.jsonl") {
		t.Fatalf("ProbeLedgerPath(default) = %q, want env-based", got)
	}
}

func TestReadLedgerSkipsBlankAndMalformed(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd,
		`{"ts":"2026-07-01T10:00:00Z","account":".claude-a","status":"OK"}`,
		``,
		`not json`,
		`{"ts":"2026-07-01T10:05:00Z","account":".claude-b","status":"LIMIT"}`,
	)
	got := ReadLedger(ProbeLedgerPath(rd))
	if len(got) != 2 {
		t.Fatalf("ReadLedger returned %d entries, want 2 (blank+malformed skipped)", len(got))
	}
}

func TestReadLedgerMissingFile(t *testing.T) {
	if got := ReadLedger(ProbeLedgerPath(t.TempDir())); got != nil {
		t.Fatalf("ReadLedger of missing file = %v, want nil", got)
	}
}

func TestLastProbeByAccountLastWins(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd,
		`{"ts":"2026-07-01T10:00:00Z","account":".claude-a","status":"AUTH"}`,
		`{"ts":"2026-07-01T10:10:00Z","account":".claude-a","status":"OK"}`,
		`{"ts":"2026-07-01T10:05:00Z","account":".claude-b","status":"LIMIT","reset":"3pm"}`,
		`{"account":"","status":"OK"}`, // empty account is ignored
	)
	latest := LastProbeByAccount(rd)
	if len(latest) != 2 {
		t.Fatalf("LastProbeByAccount size = %d, want 2", len(latest))
	}
	if latest[".claude-a"].Status != "OK" {
		t.Fatalf(".claude-a status = %q, want OK (last write wins)", latest[".claude-a"].Status)
	}
	if latest[".claude-b"].Reset != "3pm" {
		t.Fatalf(".claude-b reset = %q, want 3pm", latest[".claude-b"].Reset)
	}
}

func TestRecentProbeAgeMin(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd,
		`{"ts":"2026-07-01T10:00:00Z","account":".claude-a","status":"OK"}`,
	)
	now := time.Date(2026, 7, 1, 10, 30, 0, 0, time.UTC)
	age := RecentProbeAgeMin(".claude-a", rd, now)
	if age == nil {
		t.Fatal("age = nil, want ~30")
	}
	if math.Abs(*age-30.0) > 1e-6 {
		t.Fatalf("age = %v, want 30", *age)
	}
	if got := RecentProbeAgeMin(".claude-never", rd, now); got != nil {
		t.Fatalf("age for never-probed = %v, want nil", got)
	}
}

func TestRecentProbeAgeMinUnparseableTS(t *testing.T) {
	rd := t.TempDir()
	writeLedger(t, rd,
		`{"ts":"not-a-time","account":".claude-a","status":"OK"}`,
	)
	if got := RecentProbeAgeMin(".claude-a", rd, time.Now().UTC()); got != nil {
		t.Fatalf("age for unparseable ts = %v, want nil", got)
	}
}
