package main

import (
	"bytes"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestStandaloneStrixKnownHostsCallbackPreemptsValidation(t *testing.T) {
	exe := buildStandaloneValidate(t)

	t.Run("strict argument count", func(t *testing.T) {
		for _, argv := range [][]string{
			{"__fak_strix_known_hosts"},
			{"__fak_strix_known_hosts", "127.0.0.1:1"},
			{"__fak_strix_known_hosts", "127.0.0.1:1", strings.Repeat("A", 43), "extra"},
		} {
			stdout, stderr, code := runStandaloneValidate(t, exe, argv...)
			if code != 2 || stdout != "" || stderr != "STRIX_HOST_TRUST_REFUSED\n" {
				t.Fatalf("argv=%q: code=%d stdout=%q stderr=%q", argv, code, stdout, stderr)
			}
		}
	})

	t.Run("broker refusal fails closed before validation", func(t *testing.T) {
		stdout, stderr, code := runStandaloneValidate(t, exe,
			"__fak_strix_known_hosts", "127.0.0.1:1", strings.Repeat("A", 43))
		if code != 1 || stdout != "" || stderr != "STRIX_HOST_TRUST_REFUSED\n" {
			t.Fatalf("code=%d stdout=%q stderr=%q", code, stdout, stderr)
		}
	})
}

func buildStandaloneValidate(t *testing.T) string {
	t.Helper()
	name := "fak-validate"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	exe := filepath.Join(t.TempDir(), name)
	cmd := exec.Command("go", "build", "-o", exe, ".")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("build standalone fak-validate: %v\n%s", err, output)
	}
	return exe
}

func runStandaloneValidate(t *testing.T, exe string, argv ...string) (string, string, int) {
	t.Helper()
	cmd := exec.Command(exe, argv...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	err := cmd.Run()
	if err == nil {
		return stdout.String(), stderr.String(), 0
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		t.Fatalf("run standalone fak-validate: %v", err)
	}
	return stdout.String(), stderr.String(), exitErr.ExitCode()
}
