package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestRunDevHandoffExecutesSeparateArtifact(t *testing.T) {
	dir := t.TempDir()
	name := "fak-dev"
	body := "#!/bin/sh\nprintf 'child:%s\\n' \"$*\"\nprintf 'child-err\\n' >&2\nexit 7\n"
	if runtime.GOOS == "windows" {
		name += ".cmd"
		body = "@echo off\r\necho child:%*\r\necho child-err 1>&2\r\nexit /b 7\r\n"
	}
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	old := findFakDev
	findFakDev = func() (string, error) { return path, nil }
	t.Cleanup(func() { findFakDev = old })

	var stdout, stderr bytes.Buffer
	got := runDevHandoff(strings.NewReader(""), &stdout, &stderr, []string{"index", "ownership", "--json"})
	if got != 7 {
		t.Fatalf("exit = %d, want child exit 7; stderr=%s", got, stderr.String())
	}
	if !strings.Contains(stdout.String(), "child:index ownership --json") {
		t.Fatalf("stdout did not prove argv handoff: %q", stdout.String())
	}
	if !strings.Contains(stderr.String(), "child-err") {
		t.Fatalf("stderr was not connected: %q", stderr.String())
	}
}

func TestRunDevHandoffMissingArtifactIsActionable(t *testing.T) {
	old := findFakDev
	findFakDev = func() (string, error) { return "", errors.New("not found") }
	t.Cleanup(func() { findFakDev = old })

	var stderr bytes.Buffer
	if got := runDevHandoff(strings.NewReader(""), io.Discard, &stderr, []string{"index"}); got != 2 {
		t.Fatalf("exit = %d, want 2", got)
	}
	for _, want := range []string{"separate 'fak-dev' executable", "go install ./cmd/fak-dev", "go install github.com/anthony-chaudhary/fak/cmd/fak-dev@latest", "fak-dev <command>"} {
		if !strings.Contains(stderr.String(), want) {
			t.Errorf("missing %q in actionable error:\n%s", want, stderr.String())
		}
	}
}

func TestDevAvailabilityMissingIsMachineReadable(t *testing.T) {
	old := findFakDev
	findFakDev = func() (string, error) { return "", errors.New("not found") }
	t.Cleanup(func() { findFakDev = old })

	var stdout, stderr bytes.Buffer
	if got := runDevHandoff(strings.NewReader(""), &stdout, &stderr, []string{"availability", "--json"}); got != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", got, stderr.String())
	}
	var result devAvailability
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatalf("decode availability: %v\n%s", err, stdout.String())
	}
	if result.Schema != "fak-dev-availability/1" || result.Available || result.Source != "missing" {
		t.Fatalf("unexpected result: %+v", result)
	}
	if result.Recovery != "go install ./cmd/fak-dev" {
		t.Fatalf("recovery = %q", result.Recovery)
	}
}

func TestDevAvailabilityReportsResolvedPath(t *testing.T) {
	old := findFakDev
	path := filepath.Join(t.TempDir(), "fak-dev")
	findFakDev = func() (string, error) { return path, nil }
	t.Cleanup(func() { findFakDev = old })

	var stdout, stderr bytes.Buffer
	if got := runDevHandoff(strings.NewReader(""), &stdout, &stderr, []string{"availability", "--json"}); got != 0 {
		t.Fatalf("exit = %d, want 0; stderr=%s", got, stderr.String())
	}
	var result devAvailability
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		t.Fatal(err)
	}
	if !result.Available || result.Source != "path" || result.Path != path {
		t.Fatalf("unexpected result: %+v", result)
	}
}
