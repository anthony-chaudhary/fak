package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
)

func TestAgentCheckpointCommandCapturesProgress(t *testing.T) {
	if os.Getenv("FAK_AGENT_CHECKPOINT_HELPER") == "1" {
		cmdAgentCheckpoint(os.Args[3:])
		return
	}
	logPath := filepath.Join(t.TempDir(), "agent-status.jsonl")
	cmd := exec.Command(os.Args[0], "-test.run=TestAgentCheckpointCommandCapturesProgress", "--",
		"--actor", "worker-issue-123", "--scope", "issue #123", "--state", "progress",
		"--stage-current", "2", "--stage-total", "4", "--stage-name", "implementation",
		"--summary", "Added stale-lease classification", "--evidence", "tests/test_leases.py::test_stale",
		"--next", "Run supervisor integration tests", "--log", logPath, "--json")
	cmd.Env = append(os.Environ(), "FAK_AGENT_CHECKPOINT_HELPER=1")
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("command: %v\n%s", err, out)
	}
	var got struct {
		Actor string `json:"actor"`
		Stage struct {
			Percent int `json:"percent"`
		} `json:"stage"`
	}
	line := out
	if i := bytes.IndexByte(out, '\n'); i >= 0 {
		line = out[:i]
	}
	if err := json.Unmarshal(line, &got); err != nil {
		t.Fatalf("output %q: %v", out, err)
	}
	if got.Actor != "worker-issue-123" || got.Stage.Percent != 50 {
		t.Fatalf("output = %s", out)
	}
	if _, err := os.Stat(logPath); err != nil {
		t.Fatalf("log: %v", err)
	}
}
