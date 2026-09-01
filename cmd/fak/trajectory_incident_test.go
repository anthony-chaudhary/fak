package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

func TestRunTrajectoryIncidentJSON(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "root.jsonl")
	lines := []string{
		`{"timestamp":"2026-08-31T18:29:00Z","type":"session_meta","payload":{"id":"root","timestamp":"2026-08-31T18:29:00Z","originator":"codex_exec","source":"exec"}}`,
		`{"timestamp":"2026-08-31T18:29:01Z","type":"response_item","payload":{"type":"message","role":"user","content":[{"type":"input_text","text":"launch guard-crash-rsi/test-tag"}]}}`,
	}
	if err := os.WriteFile(path, []byte(strings.Join(lines, "\n")+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	var stdout, stderr bytes.Buffer
	code := runTrajectory(&stdout, &stderr, []string{"incident", "--root", root, "--tag", "guard-crash-rsi/test-tag", "--restart", "2026-09-01T03:25:00Z"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var packet trajectory.IncidentPacket
	if err := json.Unmarshal(stdout.Bytes(), &packet); err != nil {
		t.Fatalf("decode: %v output=%s", err, stdout.String())
	}
	if len(packet.Sessions) != 1 || packet.Sessions[0].SessionID != "root" || packet.Sessions[0].Boundary != "before_restart" {
		t.Fatalf("packet=%#v", packet)
	}
}

func TestRunTrajectoryIncidentRejectsBadBoundary(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runTrajectory(&stdout, &stderr, []string{"incident", "--root", t.TempDir(), "--tag", "tag", "--restart", "not-a-time"})
	if code != 2 || !strings.Contains(stderr.String(), "must be RFC3339") {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}
