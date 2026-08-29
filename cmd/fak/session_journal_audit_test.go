package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/sessiondiag"
)

func TestSessionJournalAuditIsFirstClassJSONSurface(t *testing.T) {
	userRoot := t.TempDir()
	regDir := t.TempDir()
	t.Setenv("CODEX_HOME", "")
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	const identity = "d1000001-0000-4000-8000-000000000001"
	if err := os.WriteFile(filepath.Join(regDir, "resume_identity.jsonl"), []byte(
		`{"ts":"2026-08-25T10:00:00Z","uuid":"`+identity+`","trace":"trace-d","provider":"claude","via":"guard-sessionstart","source":"startup"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	transcript := filepath.Join(userRoot, ".claude-a", "projects", "proj", identity+".jsonl")
	if err := os.MkdirAll(filepath.Dir(transcript), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(transcript, []byte(
		`{"uuid":"assistant-post","timestamp":"2026-08-25T10:01:00Z","message":{"role":"assistant","content":"progress"}}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldNow := sessionJournalAuditNow
	t.Cleanup(func() { sessionJournalAuditNow = oldNow })
	sessionJournalAuditNow = func() time.Time { return time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC) }

	var stdout, stderr bytes.Buffer
	code := runSession(&stdout, &stderr, []string{"journal-audit", "--since", "24h", "--reg-dir", regDir, "--home", userRoot, "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	var report sessiondiag.JournalAuditReport
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Schema != sessiondiag.JournalAuditSchema || report.Verdict != sessiondiag.JournalVerdictGreen || report.Counts.Advanced != 1 || len(report.Rows) != 1 || report.Rows[0].IdentityProvenance.Provider != "claude" {
		t.Fatalf("report=%+v", report)
	}
}

func TestSessionJournalAuditHumanVerdictFailsClosed(t *testing.T) {
	userRoot := t.TempDir()
	regDir := t.TempDir()
	t.Setenv("CODEX_HOME", "")
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	oldNow := sessionJournalAuditNow
	t.Cleanup(func() { sessionJournalAuditNow = oldNow })
	sessionJournalAuditNow = func() time.Time { return time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC) }

	var stdout, stderr bytes.Buffer
	code := runSessionJournalAudit(&stdout, &stderr, []string{"--reg-dir", regDir, "--home", userRoot})
	if code != 1 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if got := stdout.String(); !strings.HasPrefix(got, "SESSION JOURNAL AUDIT RED") || !strings.Contains(got, "IDENTITY_JOURNAL_READ_FAILED") {
		t.Fatalf("human verdict=%q", got)
	}
}

func TestSessionJournalAuditPresentWithoutPostLaunchProgressIsNonzeroRed(t *testing.T) {
	userRoot := t.TempDir()
	regDir := t.TempDir()
	t.Setenv("CODEX_HOME", "")
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	const identity = "d1000002-0000-4000-8000-000000000002"
	if err := os.WriteFile(filepath.Join(regDir, "resume_identity.jsonl"), []byte(
		`{"ts":"2026-08-25T10:00:00Z","uuid":"`+identity+`","trace":"trace-stalled","provider":"claude","via":"guard-sessionstart","source":"startup"}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	transcript := filepath.Join(userRoot, ".claude-a", "projects", "proj", identity+".jsonl")
	if err := os.MkdirAll(filepath.Dir(transcript), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(transcript, []byte(
		`{"uuid":"assistant-pre","timestamp":"2026-08-25T09:59:00Z","message":{"role":"assistant","content":"before launch"}}`+"\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	oldNow := sessionJournalAuditNow
	t.Cleanup(func() { sessionJournalAuditNow = oldNow })
	sessionJournalAuditNow = func() time.Time { return time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC) }

	var stdout, stderr bytes.Buffer
	code := runSessionJournalAudit(&stdout, &stderr, []string{"--reg-dir", regDir, "--home", userRoot, "--json"})
	if code != 1 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	if !strings.Contains(stdout.String(), `"status": "present_no_post_launch_progress"`) || !strings.Contains(stdout.String(), `"verdict": "red"`) {
		t.Fatalf("stdout missing RED present-no-progress row: %s", stdout.String())
	}
}

func TestSessionJournalAuditReportsExcludedUnboundStartup(t *testing.T) {
	userRoot := t.TempDir()
	regDir := t.TempDir()
	t.Setenv("CODEX_HOME", "")
	t.Setenv("CLAUDE_CONFIG_DIR", "")
	const identity = "0198f76a-67c2-7d11-a8f5-8f3d82149734"
	rows := strings.Join([]string{
		`{"ts":"2026-08-29T09:00:00Z","uuid":"` + identity + `","trace":"guard-trace-9849","provider":"codex","via":"guard-sessionstart","source":"startup"}`,
		`{"ts":"2026-08-29T10:00:00Z","uuid":"` + identity + `","trace":"model-controlled-trace","provider":"codex","via":"guard-sessionstart","source":"startup"}`,
	}, "\n") + "\n"
	if err := os.WriteFile(filepath.Join(regDir, "resume_identity.jsonl"), []byte(rows), 0o644); err != nil {
		t.Fatal(err)
	}
	oldNow := sessionJournalAuditNow
	t.Cleanup(func() { sessionJournalAuditNow = oldNow })
	sessionJournalAuditNow = func() time.Time { return time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC) }

	var stdout, stderr bytes.Buffer
	code := runSessionJournalAudit(&stdout, &stderr, []string{"--since", "24h", "--reg-dir", regDir, "--home", userRoot})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s stdout=%s", code, stderr.String(), stdout.String())
	}
	got := stdout.String()
	if !strings.Contains(got, "excluded_synthetic_or_unbound=1") || !strings.Contains(got, sessiondiag.JournalStatusSyntheticOrUnbound) {
		t.Fatalf("human verdict=%q", got)
	}
}
