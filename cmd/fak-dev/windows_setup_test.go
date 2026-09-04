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

func TestWindowsSetupCommandIsDispatched(t *testing.T) {
	old := windowsSetupGOOS
	windowsSetupGOOS = "linux"
	defer func() { windowsSetupGOOS = old }()
	var out, errb bytes.Buffer
	if rc := run(&out, &errb, []string{"windows-setup"}); rc != 2 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "Windows only") {
		t.Fatalf("command was not dispatched: %s", errb.String())
	}
}

func TestWindowsSetupDryRunNamesUACAndFleetSpine(t *testing.T) {
	old := windowsSetupGOOS
	windowsSetupGOOS = "windows"
	defer func() { windowsSetupGOOS = old }()
	t.Setenv(FleetGroupEnv, "239.1.2.3")
	t.Setenv(FleetPortEnv, "9876")
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runWindowsSetup(&out, &errb, []string{"--repo", repo}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	for _, want := range []string{"UAC", "fleet spine", "239.1.2.3:9876"} {
		if !strings.Contains(out.String(), want) {
			t.Errorf("output omits %q: %s", want, out.String())
		}
	}
}

func TestWindowsSetupRejectsOtherPlatforms(t *testing.T) {
	old := windowsSetupGOOS
	windowsSetupGOOS = "linux"
	defer func() { windowsSetupGOOS = old }()
	var out, errb bytes.Buffer
	if rc := runWindowsSetup(&out, &errb, nil); rc != 2 {
		t.Fatalf("rc=%d", rc)
	}
	if !strings.Contains(errb.String(), "Windows only") {
		t.Fatalf("stderr=%s", errb.String())
	}
}

func TestWindowsSetupThrottleMinFlag(t *testing.T) {
	old := windowsSetupGOOS
	windowsSetupGOOS = "windows"
	defer func() { windowsSetupGOOS = old }()
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runWindowsSetup(&out, &errb, []string{"--repo", repo, "--throttle-min", "15", "--json"}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	var plan SetupSpec
	if err := json.Unmarshal(out.Bytes(), &plan); err != nil {
		t.Fatalf("failed to unmarshal plan JSON: %v\noutput=%s", err, out.String())
	}
	if plan.ProcThrottleMin != 15 {
		t.Fatalf("plan.ProcThrottleMin = %d, want 15", plan.ProcThrottleMin)
	}
}

func TestWindowsSetupOneDriveWarning(t *testing.T) {
	old := windowsSetupGOOS
	windowsSetupGOOS = "windows"
	defer func() { windowsSetupGOOS = old }()

	oneDriveDir := filepath.Join(t.TempDir(), "OneDriveFake")
	t.Setenv("OneDrive", oneDriveDir)
	repo := filepath.Join(oneDriveDir, "work", "myrepo")
	if err := os.MkdirAll(repo, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runWindowsSetup(&out, &errb, []string{"--repo", repo}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	errStr := errb.String()
	if !strings.Contains(errStr, "inside OneDrive sync path") {
		t.Errorf("stderr omits OneDrive sync path warning: %s", errStr)
	}
	if !strings.Contains(errStr, "filesystem latency") {
		t.Errorf("stderr omits filesystem latency warning: %s", errStr)
	}
}

func TestWindowsSetupStaleTempDirsReported(t *testing.T) {
	old := windowsSetupGOOS
	windowsSetupGOOS = "windows"
	defer func() { windowsSetupGOOS = old }()

	mockTemp := t.TempDir()
	oldTempFunc := windowsSetupTempDir
	windowsSetupTempDir = func() string { return mockTemp }
	defer func() { windowsSetupTempDir = oldTempFunc }()

	staleDir := filepath.Join(mockTemp, "fak-windows-setup-stale1")
	if err := os.MkdirAll(staleDir, 0o755); err != nil {
		t.Fatal(err)
	}
	staleTime := time.Now().Add(-50 * time.Hour)
	if err := os.Chtimes(staleDir, staleTime, staleTime); err != nil {
		t.Fatal(err)
	}

	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}

	var out, errb bytes.Buffer
	if rc := runWindowsSetup(&out, &errb, []string{"--repo", repo}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	if !strings.Contains(errb.String(), "stale fak-* temporary directory") {
		t.Errorf("stderr omits stale temp directory warning: %s", errb.String())
	}
	if !strings.Contains(out.String(), "1 stale temp dirs to reap") {
		t.Errorf("output omits stale temp dirs summary: %s", out.String())
	}
}
