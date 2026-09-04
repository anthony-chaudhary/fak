package main

import (
	"bytes"
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/safesync"
)

func TestWipParkCLI(t *testing.T) {
	tmp := t.TempDir()

	origin := filepath.Join(tmp, "origin")
	if err := os.MkdirAll(origin, 0755); err != nil {
		t.Fatal(err)
	}
	execGit(t, origin, "init", "-b", "main")
	execGit(t, origin, "config", "user.name", "test")
	execGit(t, origin, "config", "user.email", "test@example.com")
	writeFileCLI(t, filepath.Join(origin, "file.txt"), "top\nmiddle\nbottom\n")
	execGit(t, origin, "add", ".")
	execGit(t, origin, "commit", "-m", "base")

	clone := filepath.Join(tmp, "clone")
	execGit(t, tmp, "clone", origin, clone)
	execGit(t, clone, "config", "user.name", "test")
	execGit(t, clone, "config", "user.email", "test@example.com")

	// Upstream commit
	writeFileCLI(t, filepath.Join(origin, "file.txt"), "top\nincoming\nmiddle\nbottom\n")
	execGit(t, origin, "add", ".")
	execGit(t, origin, "commit", "-m", "upstream")
	execGit(t, clone, "fetch", "origin")

	// Working tree in clone has local edit
	writeFileCLI(t, filepath.Join(clone, "file.txt"), "top\nincoming\nmiddle\nbottom\nunique\n")

	// 1. Dry run preview via CLI
	var stdout, stderr bytes.Buffer
	code := runWipPark(&stdout, &stderr, []string{
		"--session", "cli-sess",
		"-C", clone,
		"--path", "file.txt",
		"--target", "origin/main",
		"--json",
	})
	if code != 0 {
		t.Fatalf("dry run CLI exit %d, stderr: %s, stdout: %s", code, stderr.String(), stdout.String())
	}
	var receipt safesync.ParkReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("unmarshal receipt: %v\noutput: %s", err, stdout.String())
	}
	if receipt.Status != safesync.ParkStatusDryRun || !receipt.OK {
		t.Fatalf("dry run receipt = %+v, want DRY_RUN and OK", receipt)
	}
	if receipt.Schema != safesync.ParkSchema {
		t.Fatalf("schema = %q, want %q", receipt.Schema, safesync.ParkSchema)
	}

	// 2. Apply via CLI (positional session arg)
	stdout.Reset()
	stderr.Reset()
	code = runWipPark(&stdout, &stderr, []string{
		"cli-sess",
		"-C", clone,
		"--path", "file.txt",
		"--target", "origin/main",
		"--apply",
		"--json",
	})
	if code != 0 {
		t.Fatalf("apply CLI exit %d, stderr: %s", code, stderr.String())
	}
	receipt = safesync.ParkReceipt{}
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatalf("unmarshal receipt: %v\noutput: %s", err, stdout.String())
	}
	if receipt.Status != safesync.ParkStatusRestored || !receipt.OK {
		t.Fatalf("apply receipt = %+v, want RESTORED and OK", receipt)
	}

	content, err := os.ReadFile(filepath.Join(clone, "file.txt"))
	if err != nil {
		t.Fatal(err)
	}
	if string(content) != "top\nincoming\nmiddle\nbottom\nunique\n" {
		t.Fatalf("file.txt = %q, want 'top\\nincoming\\nmiddle\\nbottom\\nunique\\n'", string(content))
	}
}

func TestWipParkCLIErrors(t *testing.T) {
	var stdout, stderr bytes.Buffer
	// Missing session
	code := runWipPark(&stdout, &stderr, []string{"--path", "foo.txt"})
	if code != 2 {
		t.Fatalf("missing session exit code = %d, want 2", code)
	}

	// Missing path
	stdout.Reset()
	stderr.Reset()
	code = runWipPark(&stdout, &stderr, []string{"sess-1"})
	if code != 2 {
		t.Fatalf("missing path exit code = %d, want 2", code)
	}
}

func execGit(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %s in %s: %v\n%s", strings.Join(args, " "), dir, err, out)
	}
}

func writeFileCLI(t *testing.T, path, text string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(text), 0644); err != nil {
		t.Fatal(err)
	}
}
