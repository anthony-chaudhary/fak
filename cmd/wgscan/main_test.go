package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func gitRepo(t *testing.T) string {
	t.Helper()
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not on PATH")
	}
	dir := t.TempDir()
	cmd := exec.Command("git", "init", "-q", dir)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git init: %v: %s", err, out)
	}
	return dir
}

func lines(out string) []string {
	return strings.Split(strings.TrimRight(strings.ReplaceAll(out, "\r\n", "\n"), "\n"), "\n")
}

func TestRunReportsScanErrorOutsideRepo(t *testing.T) {
	var out bytes.Buffer
	if code := run(filepath.Join(t.TempDir(), "absent"), &out); code != 0 {
		t.Fatalf("exit=%d, want 0", code)
	}
	if got := lines(out.String()); len(got) != 1 || !strings.HasPrefix(got[0], "ERR ") {
		t.Fatalf("output=%q", out.String())
	}
}

func TestRunCleanRepoPrintsEverySectionAndOK(t *testing.T) {
	dir := gitRepo(t)
	if err := os.WriteFile(filepath.Join(dir, "clean.go"), []byte("package clean\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if code := run(dir, &out); code != 0 {
		t.Fatalf("exit=%d", code)
	}
	want := []string{
		"== PSInstallers (0) ==",
		"== PSStartProcesses (0) ==",
		"== PySpawns (0) ==",
		"== PyCandidates (0) ==",
		"== GoExecs (0) ==",
		"== GoCandidates (0) ==",
		"OK: true",
	}
	if got := lines(out.String()); strings.Join(got, "\n") != strings.Join(want, "\n") {
		t.Fatalf("got %q, want %q", got, want)
	}
}

func TestRunFlagsPopupSpawnAndReportsNotOK(t *testing.T) {
	dir := gitRepo(t)
	ps := "Start-Process -FilePath notepad.exe\n"
	if err := os.WriteFile(filepath.Join(dir, "spawn.ps1"), []byte(ps), 0o644); err != nil {
		t.Fatal(err)
	}
	var out bytes.Buffer
	if code := run(dir, &out); code != 0 {
		t.Fatalf("exit=%d", code)
	}
	got := lines(out.String())
	if !strings.Contains(out.String(), "== PSStartProcesses (1) ==") {
		t.Fatalf("Start-Process not counted: %q", out.String())
	}
	if last := got[len(got)-1]; last != "OK: false" {
		t.Fatalf("verdict=%q output=%q", last, out.String())
	}
}
