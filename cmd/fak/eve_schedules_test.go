package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEveSchedulesCLIAndDispatchReceipt(t *testing.T) {
	root := t.TempDir()
	mustWriteSchedule(t, filepath.Join(root, "agent", "agent.ts"), "")
	mustWriteSchedule(t, filepath.Join(root, "agent", "schedules", "daily.md"), "---\ncron: 0 9 * * *\n---\nDaily summary")
	var stdout, stderr bytes.Buffer
	if code := runEveSchedules(&stdout, &stderr, []string{"--json", root}); code != 0 {
		t.Fatalf("schedules code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"ledger_unit_id"`) && !strings.Contains(stdout.String(), `"id": "eve-daily"`) {
		t.Fatalf("missing ledger linkage: %s", stdout.String())
	}
	stdout.Reset()
	stderr.Reset()
	if code := runEveDispatchReceipt(&stdout, &stderr, []string{"--root", root, "--schedule", "daily", "--session", "sess_smoke"}); code != 0 {
		t.Fatalf("receipt code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), `"session_id": "sess_smoke"`) || !strings.Contains(stdout.String(), `"ledger_unit_id": "eve-daily"`) {
		t.Fatalf("receipt = %s", stdout.String())
	}
}

func TestEveSchedulesCLIFailsClosed(t *testing.T) {
	root := t.TempDir()
	mustWriteSchedule(t, filepath.Join(root, "agent", "agent.ts"), "")
	mustWriteSchedule(t, filepath.Join(root, "agent", "subagents", "a", "schedules", "bad.ts"), `defineSchedule({cron:"0 0 * * *",markdown:"x"})`)
	var stdout, stderr bytes.Buffer
	if code := runEveSchedules(&stdout, &stderr, []string{"--json", root}); code != eveInspectFailed {
		t.Fatalf("code=%d output=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stdout.String(), "EVE_SCHEDULE_ROOT_ONLY") {
		t.Fatalf("typed diagnostic absent: %s", stdout.String())
	}
}

func mustWriteSchedule(t *testing.T, p, body string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
		t.Fatal(err)
	}
}
