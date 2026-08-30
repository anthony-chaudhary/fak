package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/codexresume"
)

func TestCodexResumeUsage(t *testing.T) {
	var out, err bytes.Buffer
	if c := runCodexResume(&out, &err, nil); c != 2 {
		t.Fatalf("code=%d", c)
	}
	if !strings.Contains(err.String(), "usage: fak codex-resume") {
		t.Fatalf("stderr=%q", err.String())
	}
}

func TestWriteCodexResumeResultRoundTrip(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nested", "result.json")
	want := codexresume.Result{Outcome: codexresume.OutcomeCompletedReclaimed, UsefulWork: true, TaskCompleted: true, ForcedReclaim: true}
	if err := writeCodexResumeResult(path, want); err != nil {
		t.Fatal(err)
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	var got codexresume.Result
	if err := json.Unmarshal(raw, &got); err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("got=%+v want=%+v", got, want)
	}
}

func writeCodexResumeFixture(t *testing.T, home, threadID, functionCallID string) string {
	t.Helper()
	path := filepath.Join(home, "sessions", "2026", "08", "11", "rollout-"+threadID+".jsonl")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	rows := []map[string]any{
		{"type": "session_meta", "payload": map[string]any{"id": threadID, "model_provider": "fak"}},
		{"type": "response_item", "payload": map[string]any{"type": "function_call", "id": functionCallID, "call_id": "call_logical", "name": "shell_command"}},
		{"type": "response_item", "payload": map[string]any{"type": "function_call_output", "id": "fco_fixture", "call_id": "call_logical", "output": "ok"}},
	}
	file, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	encoder := json.NewEncoder(file)
	for _, row := range rows {
		if err := encoder.Encode(row); err != nil {
			_ = file.Close()
			t.Fatal(err)
		}
	}
	if err := file.Close(); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestCodexResumePreflightOnlyReportsCompatibilityWithoutPrompt(t *testing.T) {
	home := t.TempDir()
	threadID := "019ff200-1000-7000-8000-000000000001"
	rollout := writeCodexResumeFixture(t, home, threadID, "call_legacy")
	var out, errOut bytes.Buffer
	code := runCodexResume(&out, &errOut, []string{
		"--json",
		"--preflight-only",
		"--codex-home", home,
		"--rollout", rollout,
		threadID,
	})
	if code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	var got codexresume.Result
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Outcome != codexresume.OutcomeRefused || got.Preflight == nil ||
		got.Preflight.Verdict != codexresume.VerdictIncompatibleHistory ||
		got.LaunchPID != 0 {
		t.Fatalf("result=%+v", got)
	}
}

func TestCodexResumeBulkPreflightEmitsPerThreadResultsAndAggregateExit(t *testing.T) {
	home := t.TempDir()
	compatible := "019ff200-1000-7000-8000-000000000002"
	incompatible := "019ff200-1000-7000-8000-000000000003"
	writeCodexResumeFixture(t, home, compatible, "fc_response_item")
	writeCodexResumeFixture(t, home, incompatible, "call_legacy")

	var out, errOut bytes.Buffer
	code := runCodexResume(&out, &errOut, []string{
		"--json",
		"--preflight-only",
		"--codex-home", home,
		"--thread", compatible,
		"--thread", incompatible,
	})
	if code != 1 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, out.String(), errOut.String())
	}
	var got codexResumeBatchResult
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Schema != "fak-codex-resume-batch/1" || got.Succeeded != 1 || got.Failed != 1 ||
		got.ExitCode != 1 || len(got.Results) != 2 {
		t.Fatalf("batch=%+v", got)
	}
	if got.Results[0].Preflight == nil || got.Results[0].Preflight.Verdict != codexresume.VerdictResumable {
		t.Fatalf("compatible result=%+v", got.Results[0])
	}
	if got.Results[1].Preflight == nil || got.Results[1].Preflight.Verdict != codexresume.VerdictIncompatibleHistory {
		t.Fatalf("incompatible result=%+v", got.Results[1])
	}
}
