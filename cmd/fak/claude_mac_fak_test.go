package main

import (
	"bytes"
	"strings"
	"testing"
)

func TestClaudeMacFakDryRunDefaultsToInteractive(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	t.Setenv("API_TIMEOUT_MS", "")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080/v1",
		"--model", "qwen-local",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{
		"provider=existing-fak-gateway",
		"gateway=http://node.example:8080",
		"<redacted from FAK_GATEWAY_KEY>",
		"API_TIMEOUT_MS",
		"1800000",
		"Launch\n  claude",
	} {
		if !strings.Contains(out, want) {
			t.Fatalf("dry-run missing %q:\n%s", want, out)
		}
	}
	if strings.Contains(out, " -p ") || strings.Contains(out, "--output-format json") {
		t.Fatalf("default dry-run should be interactive, not a probe:\n%s", out)
	}
	if strings.Contains(out, "super-secret-test-key") {
		t.Fatalf("dry-run leaked bearer:\n%s", out)
	}
}

func TestClaudeMacFakProbeAddsPromptAndJSONOutput(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--probe",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
	})
	if code != 0 {
		t.Fatalf("runClaudeMacFak code=%d stderr=%s", code, stderr.String())
	}
	out := stdout.String()
	for _, want := range []string{"claude", "-p", "Reply with exactly: OK", "--output-format json"} {
		if !strings.Contains(out, want) {
			t.Fatalf("probe dry-run missing %q:\n%s", want, out)
		}
	}
}

func TestClaudeMacFakProbeInteractiveConflict(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "super-secret-test-key")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--probe",
		"--interactive",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
	})
	if code != 2 {
		t.Fatalf("runClaudeMacFak code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "either --probe or --interactive") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}

func TestClaudeMacFakRequiresKeyWhenFetchDisabled(t *testing.T) {
	t.Setenv("FAK_GATEWAY_KEY", "")
	dir := t.TempDir()

	var stdout, stderr bytes.Buffer
	code := runClaudeMacFak(&stdout, &stderr, []string{
		"--dry-run",
		"--fetch-key=false",
		"--claude-config-dir", dir,
		"--gateway-url", "http://node.example:8080",
		"--model", "qwen-local",
	})
	if code != 2 {
		t.Fatalf("runClaudeMacFak code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "FAK_GATEWAY_KEY is empty") {
		t.Fatalf("stderr = %q", stderr.String())
	}
}
