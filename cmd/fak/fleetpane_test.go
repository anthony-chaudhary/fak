package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/fleetpane"
)

type fleetPaneFakeRunner struct {
	paths map[string]bool
	runs  map[string]fleetpane.RunResult
}

func (f fleetPaneFakeRunner) LookPath(file string) (string, error) {
	if f.paths[file] {
		return file, nil
	}
	return "", errors.New("not found")
}

func (f fleetPaneFakeRunner) Run(ctx context.Context, req fleetpane.RunRequest) fleetpane.RunResult {
	if res, ok := f.runs[strings.Join(req.Args, "\x00")]; ok {
		return res
	}
	return fleetpane.RunResult{ExitCode: 127, Stderr: "not configured", Err: errors.New("not configured")}
}

func TestRunFleetPaneLoopCheckFixtureSmoke(t *testing.T) {
	root := t.TempDir()
	mustWriteFleetPaneFixture(t, filepath.Join(root, "VERSION"), "v1.2.3\n")
	mustWriteFleetPaneFixture(t, filepath.Join(root, "tools", "control_pane.loops.json"), `{
  "loops": {
    "ok": {"enabled": true, "status_cmd": ["okcmd"]}
  }
}`)
	mustMkdirFleetPaneFixture(t, filepath.Join(root, "tools", "_registry", "machines"))

	prevRunner := fleetPaneRunner
	fleetPaneRunner = fleetPaneFakeRunner{
		paths: map[string]bool{"okcmd": true},
		runs:  map[string]fleetpane.RunResult{"okcmd": {ExitCode: 0, Stdout: `{"ok": true, "detail": "green"}`}},
	}
	defer func() { fleetPaneRunner = prevRunner }()

	var stdout, stderr bytes.Buffer
	rc := runFleetPane(&stdout, &stderr, []string{"--root", root, "loop-check", "ok", "--json"})
	if rc != 0 {
		t.Fatalf("runFleetPane rc=%d stderr=%q stdout=%q", rc, stderr.String(), stdout.String())
	}
	var doc struct {
		Schema  string `json:"schema"`
		Verdict string `json:"verdict"`
		Check   struct {
			State  string `json:"state"`
			Detail string `json:"detail"`
		} `json:"check"`
	}
	if err := json.Unmarshal(stdout.Bytes(), &doc); err != nil {
		t.Fatalf("unmarshal output: %v\n%s", err, stdout.String())
	}
	if doc.Schema != "fleet-control-pane.loop-check/1" || doc.Verdict != "OK" || doc.Check.State != "OK" || doc.Check.Detail != "green" {
		t.Fatalf("unexpected doc: %+v", doc)
	}
}

func TestRunFleetPaneUnsupportedMutatingCommand(t *testing.T) {
	root := t.TempDir()
	mustWriteFleetPaneFixture(t, filepath.Join(root, "VERSION"), "v1.2.3\n")
	mustMkdirFleetPaneFixture(t, filepath.Join(root, "tools"))
	var stdout, stderr bytes.Buffer
	rc := runFleetPane(&stdout, &stderr, []string{"--root", root, "commit", "--path", "x"})
	if rc != 2 {
		t.Fatalf("rc=%d, want 2", rc)
	}
	if !strings.Contains(stderr.String(), "unsupported in native read-only subset") {
		t.Fatalf("stderr=%q", stderr.String())
	}
}

func mustMkdirFleetPaneFixture(t *testing.T, path string) {
	t.Helper()
	if err := os.MkdirAll(path, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", path, err)
	}
}

func mustWriteFleetPaneFixture(t *testing.T, path, text string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", filepath.Dir(path), err)
	}
	if err := os.WriteFile(path, []byte(text), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}
