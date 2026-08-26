package sessiondiag

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestJournalAuditAdvancedForClaudeAndCodex(t *testing.T) {
	userRoot := t.TempDir()
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	launchAt := now.Add(-2 * time.Hour)
	const claudeID = "a1000001-0000-4000-8000-000000000001"
	const codexID = "b1000001-0000-4000-8000-000000000001"

	claudePath := filepath.Join(userRoot, ".claude-a", "projects", "C--work-fak", claudeID+".jsonl")
	writeJournalAuditFixture(t, claudePath, strings.Join([]string{
		`{"uuid":"assistant-before","timestamp":"2026-08-25T09:50:00Z","message":{"role":"assistant","content":"baseline"}}`,
		`{"uuid":"assistant-after","timestamp":"2026-08-25T10:01:00Z","message":{"role":"assistant","content":"continued work"}}`,
	}, "\n")+"\n")
	codexPath := filepath.Join(userRoot, ".codex-b", "sessions", "2026", "08", "25", "rollout-exact.jsonl")
	writeJournalAuditFixture(t, codexPath, strings.Join([]string{
		`{"timestamp":"2026-08-25T09:00:00Z","type":"session_meta","payload":{"id":"` + codexID + `"}}`,
		`{"timestamp":"2026-08-25T09:55:00Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-before"}}`,
		`{"timestamp":"2026-08-25T10:02:00Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-after"}}`,
	}, "\n")+"\n")

	report := AuditRecentLaunches(JournalAuditOptions{
		Now: now, Window: 24 * time.Hour, UserHome: userRoot, IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
		Identities: []JournalLaunchIdentity{
			{Identity: claudeID, Trace: "trace-claude", LaunchAt: launchAt, Provider: "claude", Via: "guard-sessionstart"},
			{Identity: codexID, Trace: "trace-codex", LaunchAt: launchAt, Provider: "codex", Via: "guard-sessionstart"},
		},
	})
	if report.Schema != JournalAuditSchema || report.Verdict != JournalVerdictGreen {
		t.Fatalf("report schema/verdict = %q/%q; report=%+v", report.Schema, report.Verdict, report)
	}
	if report.Counts.Identities != 2 || report.Counts.ExactTranscripts != 2 || report.Counts.Advanced != 2 || report.Counts.AuthorityErrors != 0 {
		t.Fatalf("counts=%+v", report.Counts)
	}
	byProvider := map[string]JournalAuditRow{}
	for _, row := range report.Rows {
		byProvider[row.Provider] = row
	}
	if got := byProvider["claude"]; got.Status != JournalStatusAdvanced || got.TranscriptPath != claudePath || got.BaselineCursor == nil || got.BaselineCursor.ID != "assistant-before" || got.PostLaunchCursor == nil || got.PostLaunchCursor.ID != "assistant-after" {
		t.Fatalf("Claude row=%+v", got)
	}
	if got := byProvider["codex"]; got.Status != JournalStatusAdvanced || got.TranscriptPath != codexPath || got.BaselineCursor == nil || got.BaselineCursor.ID != "turn-before" || got.PostLaunchCursor == nil || got.PostLaunchCursor.ID != "turn-after" {
		t.Fatalf("Codex row=%+v", got)
	}
}

func TestJournalAuditDistinguishesMissingAndPresentWithoutProgress(t *testing.T) {
	userRoot := t.TempDir()
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	launchAt := time.Date(2026, 8, 25, 10, 0, 0, 0, time.UTC)
	const missingID = "c1000001-0000-4000-8000-000000000001"
	const claudeID = "c1000002-0000-4000-8000-000000000002"
	const codexID = "c1000003-0000-4000-8000-000000000003"

	writeJournalAuditFixture(t, filepath.Join(userRoot, ".claude-a", "projects", "proj", claudeID+".jsonl"),
		`{"uuid":"old-assistant","timestamp":"2026-08-25T09:59:00Z","message":{"role":"assistant","content":"before launch only"}}`+"\n")
	// A prefix/suffix filename is deliberately not an exact identity join.
	writeJournalAuditFixture(t, filepath.Join(userRoot, ".claude-a", "projects", "proj", missingID+"-other.jsonl"),
		`{"uuid":"not-a-match","timestamp":"2026-08-25T11:00:00Z","message":{"role":"assistant","content":"wrong identity"}}`+"\n")
	writeJournalAuditFixture(t, filepath.Join(userRoot, ".codex-a", "sessions", "rollout-old.jsonl"), strings.Join([]string{
		`{"timestamp":"2026-08-25T09:00:00Z","type":"session_meta","payload":{"session_id":"` + codexID + `"}}`,
		`{"timestamp":"2026-08-25T09:58:00Z","type":"event_msg","payload":{"type":"task_started","turn_id":"old-turn"}}`,
	}, "\n")+"\n")

	report := AuditRecentLaunches(JournalAuditOptions{
		Now: now, Window: 24 * time.Hour, UserHome: userRoot, IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
		Identities: []JournalLaunchIdentity{
			{Identity: missingID, LaunchAt: launchAt, Provider: "codex"},
			{Identity: claudeID, LaunchAt: launchAt}, // legacy row: provider comes only from the exact transcript
			{Identity: codexID, LaunchAt: launchAt, Provider: "codex"},
		},
	})
	if report.Verdict != JournalVerdictRed || report.Counts.Identities != 3 || report.Counts.ExactTranscripts != 2 || report.Counts.Advanced != 0 || report.Counts.MissingTranscript != 1 || report.Counts.PresentNoPostLaunchProgress != 2 {
		t.Fatalf("report=%+v", report)
	}
	status := map[string]string{}
	for _, row := range report.Rows {
		status[row.Identity] = row.Status
	}
	if status[missingID] != JournalStatusMissingTranscript || status[claudeID] != JournalStatusPresentNoPostLaunchProgress || status[codexID] != JournalStatusPresentNoPostLaunchProgress {
		t.Fatalf("statuses=%v", status)
	}
}

func TestJournalAuditFailsClosedOnUnreadableConfiguredAuthority(t *testing.T) {
	userRoot := t.TempDir()
	notDir := filepath.Join(userRoot, "codex-home-is-a-file")
	if err := os.WriteFile(notDir, []byte("not a provider root"), 0o644); err != nil {
		t.Fatal(err)
	}
	report := AuditRecentLaunches(JournalAuditOptions{
		Now: time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC), Window: 24 * time.Hour,
		UserHome: userRoot, CodexHome: notDir, IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
	})
	if report.Verdict != JournalVerdictRed || report.Counts.AuthorityErrors != 1 || len(report.AuthorityErrors) != 1 || report.AuthorityErrors[0].Code != "ROOT_UNAVAILABLE" {
		t.Fatalf("report=%+v", report)
	}
}

func TestJournalAuditFailsClosedOnMalformedMatchedTranscript(t *testing.T) {
	userRoot := t.TempDir()
	now := time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC)
	launchAt := time.Date(2026, 8, 25, 10, 0, 0, 0, time.UTC)
	const identity = "c1000004-0000-4000-8000-000000000004"

	writeJournalAuditFixture(t, filepath.Join(userRoot, ".claude-a", "projects", "proj", identity+".jsonl"), "not-json\n")
	report := AuditRecentLaunches(JournalAuditOptions{
		Now: now, Window: 24 * time.Hour, UserHome: userRoot, IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
		Identities: []JournalLaunchIdentity{{Identity: identity, LaunchAt: launchAt, Provider: "claude"}},
	})
	if report.Counts.AuthorityErrors != 1 || len(report.AuthorityErrors) != 1 || report.AuthorityErrors[0].Code != "READ_FAILED" {
		t.Fatalf("report=%+v", report)
	}
}

func writeJournalAuditFixture(t *testing.T, path, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
