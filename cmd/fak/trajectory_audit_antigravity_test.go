package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms
func TestTrajectoryAuditAntigravityRootAndSnapshotFlag(t *testing.T) {
	tmp := t.TempDir()
	absent := filepath.Join(tmp, "absent")
	agRoot := filepath.Join(tmp, "brain")
	logDir := filepath.Join(agRoot, "11111111-2222-3333-4444-555555555555", ".system_generated", "logs")
	if err := os.MkdirAll(logDir, 0o755); err != nil {
		t.Fatal(err)
	}
	line := `{"step_index":0,"source":"MODEL","type":"PLANNER_RESPONSE","status":"DONE","model":"gemini-4-neon-medium","input_tokens":120,"cache_read_tokens":300,"output_tokens":45}` + "\n"
	if err := os.WriteFile(filepath.Join(logDir, "transcript.jsonl"), []byte(line), 0o600); err != nil {
		t.Fatal(err)
	}

	jsonlOut := filepath.Join(tmp, "out.jsonl")
	mdOut := filepath.Join(tmp, "out.md")
	var stdout, stderr bytes.Buffer
	code := runTrajectoryAudit(&stdout, &stderr, []string{
		"--since", "0",
		"--claude-root", absent,
		"--codex-root", absent,
		"--opencode-root", absent,
		"--antigravity-root", agRoot,
		"--jsonl", jsonlOut,
		"--md", mdOut,
	})
	if code != 0 {
		t.Fatalf("runTrajectoryAudit exit=%d stderr=%s", code, stderr.String())
	}
	data, err := os.ReadFile(jsonlOut)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"source":"antigravity"`) {
		t.Fatalf("expected antigravity source row in jsonl, got:\n%s", string(data))
	}

	// --snapshot must reject --antigravity-root.
	stdout.Reset()
	stderr.Reset()
	code = runTrajectoryAudit(&stdout, &stderr, []string{"--snapshot", tmp, "--antigravity-root", agRoot})
	if code == 0 || !strings.Contains(stderr.String(), "TRAJECTORY_SNAPSHOT_REFUSED") {
		t.Fatalf("expected TRAJECTORY_SNAPSHOT_REFUSED, got code=%d stderr=%s", code, stderr.String())
	}
}
