package devcmd

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestRunCheckpointCapturesProgress(t *testing.T) {
	logPath := filepath.Join(t.TempDir(), "dev-status.jsonl")
	code, out, errOut := runCheckpointForTest([]string{
		"--actor", "worker-issue-123", "--scope", "issue #123", "--state", "progress",
		"--stage-current", "2", "--stage-total", "4", "--stage-name", "implementation",
		"--summary", "Added stale-lease classification", "--evidence", "internal/leases/lease_test.go::TestStale",
		"--next", "Run supervisor integration tests", "--log", logPath, "--json",
	})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut)
	}
	var got struct {
		Actor string `json:"actor"`
		Stage struct {
			Percent int `json:"percent"`
		} `json:"stage"`
	}
	if err := json.Unmarshal([]byte(out), &got); err != nil {
		t.Fatalf("output %q: %v", out, err)
	}
	if got.Actor != "worker-issue-123" || got.Stage.Percent != 50 {
		t.Fatalf("output=%s", out)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatalf("log: %v", err)
	}
}

func TestRunCheckpointRejectsIncompleteProgress(t *testing.T) {
	code, _, _ := runCheckpointForTest([]string{"--actor", "worker", "--scope", "task", "--state", "progress", "--summary", "working", "--next", "continue"})
	if code == 0 {
		t.Fatal("incomplete progress accepted")
	}
}

func runCheckpointForTest(args []string) (int, string, string) {
	var out, errOut bytes.Buffer
	code := RunCheckpoint(&out, &errOut, args)
	return code, out.String(), errOut.String()
}
