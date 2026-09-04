package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFleetCompareCLI(t *testing.T) {
	fixtureJSON := `{
		"agents": [50, 50, 50, 20],
		"turns": [30, 10, 20, 30],
		"shared_saved_mean": [90, 40, 65, 30],
		"cross_uplift_mean": [25, 10, 18, 8]
	}`

	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "cols.json")
	if err := os.WriteFile(path, []byte(fixtureJSON), 0o644); err != nil {
		t.Fatalf("failed to write fixture: %v", err)
	}

	var stdout, stderr bytes.Buffer
	code := runFleetCompare(&stdout, &stderr, []string{"--key", "agents", "--val", "50", "--file", path, "--json"})
	if code != 0 {
		t.Fatalf("runFleetCompare exited %d; stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, `"Xs": [`) || !strings.Contains(out, `"Isolated": [`) {
		t.Fatalf("unexpected output: %s", out)
	}
}
