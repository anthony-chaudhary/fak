package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
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
	repo := t.TempDir()
	if err := os.WriteFile(filepath.Join(repo, "go.mod"), []byte("module example"), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, errb bytes.Buffer
	if rc := runWindowsSetup(&out, &errb, []string{"--repo", repo}); rc != 0 {
		t.Fatalf("rc=%d stderr=%s", rc, errb.String())
	}
	for _, want := range []string{"UAC", "fleet spine", "239.255.70.65:4765"} {
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
