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
		`{"timestamp":"2026-08-25T10:02:00Z","type":"response_item","payload":{"type":"function_call","name":"exec_command"}}`,
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
	if got := byProvider["codex"]; got.Status != JournalStatusAdvanced || got.TranscriptPath != codexPath || got.BaselineCursor != nil || got.PostLaunchCursor == nil || got.PostLaunchCursor.ID != "turn-before" || got.PostLaunchCursor.Kind != "function_call" {
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

func TestJournalAuditCountsInFlightCodexActivityAfterLaunch(t *testing.T) {
	for _, tc := range []struct {
		name string
		row  string
		kind string
	}{
		{name: "function call", row: `{"timestamp":"2026-08-25T10:00:01Z","type":"response_item","payload":{"type":"function_call","name":"exec_command"}}`, kind: "function_call"},
		{name: "function call output", row: `{"timestamp":"2026-08-25T10:00:01Z","type":"response_item","payload":{"type":"function_call_output","output":"ok"}}`, kind: "function_call_output"},
		{name: "assistant output", row: `{"timestamp":"2026-08-25T10:00:01Z","type":"response_item","payload":{"type":"message","role":"assistant","content":[{"type":"output_text","text":"Working"}]}}`, kind: "assistant"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			userRoot := t.TempDir()
			const identity = "c1000010-0000-4000-8000-000000000010"
			writeJournalAuditFixture(t, filepath.Join(userRoot, ".codex", "sessions", "2026", "08", "25", "rollout-"+identity+".jsonl"), strings.Join([]string{
				`{"timestamp":"2026-08-25T09:59:00Z","type":"session_meta","payload":{"id":"` + identity + `"}}`,
				`{"timestamp":"2026-08-25T09:59:30Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-live"}}`,
				tc.row,
			}, "\n")+"\n")

			report := AuditRecentLaunches(JournalAuditOptions{
				Now: time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC), Window: 24 * time.Hour,
				UserHome: userRoot, CodexHome: filepath.Join(userRoot, ".codex"), IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
				Identities: []JournalLaunchIdentity{{Identity: identity, LaunchAt: time.Date(2026, 8, 25, 10, 0, 0, 0, time.UTC), Provider: "codex"}},
			})
			if len(report.Rows) != 1 || report.Rows[0].Status != JournalStatusAdvanced || report.Rows[0].PostLaunchCursor == nil || report.Rows[0].PostLaunchCursor.Kind != tc.kind {
				t.Fatalf("report=%+v", report)
			}
		})
	}
}

func TestJournalAuditRejectsCodexSetupOnlyAfterLaunch(t *testing.T) {
	userRoot := t.TempDir()
	const identity = "c1000011-0000-4000-8000-000000000011"
	writeJournalAuditFixture(t, filepath.Join(userRoot, ".codex", "sessions", "2026", "08", "25", "rollout-"+identity+".jsonl"), strings.Join([]string{
		`{"timestamp":"2026-08-25T09:59:00Z","type":"session_meta","payload":{"id":"` + identity + `"}}`,
		`{"timestamp":"2026-08-25T10:00:01Z","type":"turn_context","payload":{"type":"metadata"}}`,
		`{"timestamp":"2026-08-25T10:00:02Z","type":"event_msg","payload":{"type":"task_started","turn_id":"turn-idle"}}`,
		`{"timestamp":"2026-08-25T10:00:03Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"instructions"}]}}`,
	}, "\n")+"\n")

	report := AuditRecentLaunches(JournalAuditOptions{
		Now: time.Date(2026, 8, 25, 12, 0, 0, 0, time.UTC), Window: 24 * time.Hour,
		UserHome: userRoot, CodexHome: filepath.Join(userRoot, ".codex"), IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
		Identities: []JournalLaunchIdentity{{Identity: identity, LaunchAt: time.Date(2026, 8, 25, 10, 0, 0, 0, time.UTC), Provider: "codex"}},
	})
	if len(report.Rows) != 1 || report.Rows[0].Status != JournalStatusPresentNoPostLaunchProgress || report.Rows[0].PostLaunchCursor != nil {
		t.Fatalf("report=%+v", report)
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

func TestJournalAuditExcludesDuplicateUnboundStartupIdentity(t *testing.T) {
	userRoot := t.TempDir()
	now := time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
	const identity = "0198f76a-67c2-7d11-a8f5-8f3d82149734"
	report := AuditRecentLaunches(JournalAuditOptions{
		Now: now, Window: 24 * time.Hour, UserHome: userRoot, IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
		Identities: []JournalLaunchIdentity{
			{Identity: identity, Trace: "guard-trace-9849", LaunchAt: now.Add(-3 * time.Hour), Provider: "codex", Via: "guard-sessionstart", Source: "startup"},
			{Identity: identity, Trace: "model-controlled-trace", LaunchAt: now.Add(-2 * time.Hour), Provider: "codex", Via: "guard-sessionstart", Source: "startup"},
		},
	})
	if report.Verdict != JournalVerdictGreen || report.Counts.Identities != 1 || report.Counts.ExcludedSyntheticOrUnbound != 1 || report.Counts.MissingTranscript != 0 || len(report.Rows) != 1 {
		t.Fatalf("report=%+v", report)
	}
	if report.Rows[0].Status != JournalStatusSyntheticOrUnbound {
		t.Fatalf("row=%+v", report.Rows[0])
	}
}

func TestJournalAuditKeepsExactNonSyntheticLaunchMissingTranscriptRed(t *testing.T) {
	userRoot := t.TempDir()
	now := time.Date(2026, 8, 29, 12, 0, 0, 0, time.UTC)
	report := AuditRecentLaunches(JournalAuditOptions{
		Now: now, Window: 24 * time.Hour, UserHome: userRoot, IdentityPath: filepath.Join(userRoot, "resume_identity.jsonl"),
		Identities: []JournalLaunchIdentity{{Identity: "real-codex-thread", Trace: "trace-real", LaunchAt: now.Add(-time.Hour), Provider: "codex", Via: "guard-sessionstart", Source: "resume"}},
	})
	if report.Verdict != JournalVerdictRed || report.Counts.MissingTranscript != 1 || report.Counts.ExcludedSyntheticOrUnbound != 0 || report.Rows[0].Status != JournalStatusMissingTranscript {
		t.Fatalf("report=%+v", report)
	}
}
