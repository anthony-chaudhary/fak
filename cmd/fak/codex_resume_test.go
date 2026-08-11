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
