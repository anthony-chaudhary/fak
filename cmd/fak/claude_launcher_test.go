package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestClaudeLauncherDryRun(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{
		"--dry-run",
		"--no-probe",
		"--gateway-url", "http://127.0.0.1:8080",
		"--model", "qwen38:27b-q4",
		"--api-key", "test-key-123",
	}

	code := runClaude(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runClaude --dry-run failed with code %d, stderr: %s", code, stderr.String())
	}

	errOut := stderr.String()
	stdOut := stdout.String()

	if !strings.Contains(errOut, "fak claude: dry-run - not launching") {
		t.Errorf("expected dry-run header in stderr: %s", errOut)
	}
	if !strings.Contains(errOut, "gateway     = http://127.0.0.1:8080") {
		t.Errorf("expected gateway URL in stderr: %s", errOut)
	}
	if !strings.Contains(errOut, "model       = qwen38:27b-q4") {
		t.Errorf("expected model in stderr: %s", errOut)
	}
	if !strings.Contains(errOut, "ANTHROPIC_BASE_URL=http://127.0.0.1:8080") {
		t.Errorf("expected ANTHROPIC_BASE_URL in stderr: %s", errOut)
	}
	if !strings.Contains(errOut, "ANTHROPIC_API_KEY=test-key-123") {
		t.Errorf("expected ANTHROPIC_API_KEY in stderr: %s", errOut)
	}
	if !strings.Contains(errOut, "ANTHROPIC_MODEL=qwen38:27b-q4") {
		t.Errorf("expected ANTHROPIC_MODEL in stderr: %s", errOut)
	}
	if !strings.Contains(stdOut, "claude") {
		t.Errorf("expected claude command on stdout: %s", stdOut)
	}
}

func TestClaudeLauncherPrintEnv(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{
		"--print-env",
		"--gateway-url", "http://127.0.0.1:8080",
		"--model", "qwen38:27b-q4",
		"--api-key", "test-key-xyz",
	}

	code := runClaude(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runClaude --print-env failed with code %d, stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	expectedLines := []string{
		`export ANTHROPIC_BASE_URL="http://127.0.0.1:8080"`,
		`export ANTHROPIC_API_KEY="test-key-xyz"`,
		`export ANTHROPIC_MODEL="qwen38:27b-q4"`,
		`export ANTHROPIC_DEFAULT_OPUS_MODEL="qwen38:27b-q4"`,
		`export ANTHROPIC_DEFAULT_SONNET_MODEL="qwen38:27b-q4"`,
		`export ANTHROPIC_DEFAULT_HAIKU_MODEL="qwen38:27b-q4"`,
		`export ANTHROPIC_SMALL_FAST_MODEL="qwen38:27b-q4"`,
		`export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"`,
		`export API_TIMEOUT_MS="1800000"`,
	}

	for _, expected := range expectedLines {
		if !strings.Contains(out, expected) {
			t.Errorf("missing expected export line %q in stdout:\n%s", expected, out)
		}
	}
}

func TestClaudeLauncherProbeArgv(t *testing.T) {
	opts := claudeLaunchOptions{
		probePrompt:     "Reply with pong",
		command:         "claude",
		skipPermissions: true,
	}

	argv := buildClaudeLaunchArgv(opts)
	cmdLine := strings.Join(argv, " ")

	expectedTokens := []string{
		"claude",
		"-p Reply with pong",
		"--output-format json",
		"--safe-mode",
		"--no-session-persistence",
		"--dangerously-skip-permissions",
	}

	for _, token := range expectedTokens {
		if !strings.Contains(cmdLine, token) {
			t.Errorf("expected token %q in probe command line: %s", token, cmdLine)
		}
	}
}

func TestClaudeLauncherPassthroughArgs(t *testing.T) {
	opts := claudeLaunchOptions{
		command:         "claude",
		skipPermissions: false,
		passthrough:     []string{"--resume", "ses_abc123", "--verbose"},
	}

	argv := buildClaudeLaunchArgv(opts)
	cmdLine := strings.Join(argv, " ")

	if !strings.Contains(cmdLine, "--resume ses_abc123") {
		t.Errorf("expected passthrough args in command line: %s", cmdLine)
	}
	if !strings.Contains(cmdLine, "--verbose") {
		t.Errorf("expected --verbose in command line: %s", cmdLine)
	}
}

func TestClaudeLauncherHealthProbeAutoModel(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"served-metal-qwen","engine":"inkernel"}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	var capturedArgv []string
	var capturedEnv []string
	origRun := claudeLaunchRun
	claudeLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		capturedArgv = argv
		capturedEnv = env
		return 0
	}
	defer func() { claudeLaunchRun = origRun }()

	var stdout, stderr bytes.Buffer
	args := []string{
		"--gateway-url", ts.URL,
		"--probe", "test health auto model",
	}

	code := runClaude(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runClaude returned %d, stderr: %s", code, stderr.String())
	}

	foundModelEnv := false
	for _, envVar := range capturedEnv {
		if envVar == "ANTHROPIC_MODEL=served-metal-qwen" {
			foundModelEnv = true
			break
		}
	}
	if !foundModelEnv {
		t.Errorf("expected ANTHROPIC_MODEL=served-metal-qwen in child env, got %v", capturedEnv)
	}
	if len(capturedArgv) == 0 || capturedArgv[0] != "claude" {
		t.Errorf("expected claude command, got %v", capturedArgv)
	}
}

func TestClaudeLauncherBackendUnreachableFails(t *testing.T) {
	origProbe := claudeStatusFetcher
	claudeStatusFetcher = func(serverURL string, timeout time.Duration) (*claudeStatusReport, error) {
		return nil, fmt.Errorf("connection refused")
	}
	defer func() { claudeStatusFetcher = origProbe }()

	var stdout, stderr bytes.Buffer
	args := []string{
		"--gateway-url", "http://127.0.0.1:9999",
	}

	code := runClaude(&stdout, &stderr, args)
	if code == 0 {
		t.Fatalf("expected non-zero exit code when gateway is unreachable")
	}
	if !strings.Contains(stderr.String(), "backend is unreachable") {
		t.Errorf("expected 'backend is unreachable' in stderr: %s", stderr.String())
	}
}

func TestClaudeLauncherConfig(t *testing.T) {
	tmp := t.TempDir()

	var stdout, stderr bytes.Buffer
	previewArgs := []string{
		"config",
		"--dir", tmp,
		"--addr", "http://127.0.0.1:8080",
		"--model", "qwen38:27b-q4",
	}

	code := runClaude(&stdout, &stderr, previewArgs)
	if code != 0 {
		t.Fatalf("runClaude config preview failed: %s", stderr.String())
	}

	previewOut := stdout.String()
	if !strings.Contains(previewOut, `"ANTHROPIC_BASE_URL": "http://127.0.0.1:8080"`) {
		t.Errorf("expected ANTHROPIC_BASE_URL in preview: %s", previewOut)
	}
	if !strings.Contains(previewOut, `"ANTHROPIC_MODEL": "qwen38:27b-q4"`) {
		t.Errorf("expected ANTHROPIC_MODEL in preview: %s", previewOut)
	}

	// Now test --write
	stdout.Reset()
	stderr.Reset()
	writeArgs := []string{
		"config",
		"--write",
		"--dir", tmp,
		"--addr", "http://127.0.0.1:8080",
		"--model", "qwen38:27b-q4",
	}

	code = runClaude(&stdout, &stderr, writeArgs)
	if code != 0 {
		t.Fatalf("runClaude config --write failed: %s", stderr.String())
	}

	writtenFile := filepath.Join(tmp, ".claude", "settings.json")
	data, err := os.ReadFile(writtenFile)
	if err != nil {
		t.Fatalf("failed to read written settings file: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("unmarshal written settings: %v", err)
	}
	env := parsed["env"].(map[string]interface{})
	if env["ANTHROPIC_BASE_URL"] != "http://127.0.0.1:8080" {
		t.Errorf("ANTHROPIC_BASE_URL = %v, want http://127.0.0.1:8080", env["ANTHROPIC_BASE_URL"])
	}
	if env["ANTHROPIC_MODEL"] != "qwen38:27b-q4" {
		t.Errorf("ANTHROPIC_MODEL = %v, want qwen38:27b-q4", env["ANTHROPIC_MODEL"])
	}
}
