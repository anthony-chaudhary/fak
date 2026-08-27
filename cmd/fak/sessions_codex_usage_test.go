package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestCodexSubmitUsageLedgerAndWeeklyFold(t *testing.T) {
	t.Setenv(guardActiveEnv, "")
	home := t.TempDir()
	oldNow := codexSubmitUsageNow
	codexSubmitUsageNow = func() time.Time { return time.Date(2026, 8, 26, 12, 0, 0, 0, time.UTC) }
	t.Cleanup(func() { codexSubmitUsageNow = oldNow })

	var stdout, stderr bytes.Buffer
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader("not json"), []string{"--codex-home", home}); code != 0 {
		t.Fatalf("permissive hook code=%d stderr=%s", code, stderr.String())
	}
	t.Setenv(codexLoopHookHardenedEnv, "1")
	stdout.Reset()
	stderr.Reset()
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader("not json"), []string{"--codex-home", home}); code != 0 {
		t.Fatalf("hardened hook code=%d stderr=%s", code, stderr.String())
	}

	path := filepath.Join(home, "fak-ledgers", "codex-userpromptsubmit.jsonl")
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if rows := bytes.Count(raw, []byte{'\n'}); rows != 2 {
		t.Fatalf("ledger rows=%d, want 2: %s", rows, raw)
	}

	stdout.Reset()
	stderr.Reset()
	if code := sessionsCodexLoopHook(&stdout, &stderr, strings.NewReader(""), []string{"--codex-home", home, "--usage-summary"}); code != 0 {
		t.Fatalf("summary code=%d stderr=%s", code, stderr.String())
	}
	var summary codexSubmitUsageSummary
	if err := json.Unmarshal(stdout.Bytes(), &summary); err != nil {
		t.Fatal(err)
	}
	if len(summary.Weeks) != 1 || summary.Weeks[0].Week != "2026-W35" || summary.Weeks[0].Total != 2 {
		t.Fatalf("summary=%+v", summary)
	}
	if summary.Weeks[0].Modes["permissive"] != 1 || summary.Weeks[0].Modes["hardened"] != 1 || summary.Weeks[0].Outcomes["allow"] != 2 {
		t.Fatalf("weekly counts=%+v", summary.Weeks[0])
	}
	t.Logf("Codex UserPromptSubmit weekly usage: %s", stdout.String())
}
