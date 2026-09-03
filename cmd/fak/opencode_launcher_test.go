package main

import (
	"bytes"
	"strings"
	"testing"
	"time"
)

func TestOpencodeLauncherDryRunBasic(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--dry-run", "--split", "off"}
	code := runOpencode(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "guard") {
		t.Errorf("expected guard in dry-run stdout: %s", out)
	}
	if !strings.Contains(out, "--provider openai") {
		t.Errorf("expected --provider openai in dry-run stdout: %s", out)
	}
	if !strings.Contains(out, "-- opencode") {
		t.Errorf("expected '-- opencode' in dry-run stdout: %s", out)
	}
}

func TestOpencodeLauncherProbeWiring(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--dry-run", "--probe", "say hello from test", "--split", "off", "--pure"}
	code := runOpencode(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "--probe") {
		t.Errorf("expected --probe flag for guard in dry-run stdout: %s", out)
	}
	if !strings.Contains(out, "run \"say hello from test\" --format json") && !strings.Contains(out, "run say hello from test --format json") {
		t.Errorf("expected probe run command in dry-run stdout: %s", out)
	}
	if !strings.Contains(out, "--auto") {
		t.Errorf("expected --auto for probe in dry-run stdout: %s", out)
	}
	if !strings.Contains(out, "--pure") {
		t.Errorf("expected --pure in dry-run stdout: %s", out)
	}
	if !strings.Contains(out, "--dangerously-skip-permissions") {
		t.Errorf("expected --dangerously-skip-permissions for unattended probe in dry-run stdout: %s", out)
	}
}

func TestOpencodeLauncherOptions(t *testing.T) {
	opts := opencodeLaunchOptions{
		splitMode:     "off",
		splitWhere:    "bottom",
		splitInterval: 1 * time.Second,
		policyPath:    "custom-policy.json",
		apiKeyEnv:     "MY_API_KEY",
		baseURL:       "http://127.0.0.1:8001/v1",
		model:         "glm-5.3-flash",
		auditPath:     "my-audit.jsonl",
		quiet:         true,
		localAuto:     true,
		passthrough:   []string{"run", "do task"},
	}
	argv := buildOpencodeLaunchArgv("fak", opts)
	line := strings.Join(argv, " ")
	expectedParts := []string{
		"fak guard",
		"--provider openai",
		"--policy custom-policy.json",
		"--api-key-env MY_API_KEY",
		"--base-url http://127.0.0.1:8001/v1",
		"--model glm-5.3-flash",
		"--audit my-audit.jsonl",
		"--quiet",
		"--local",
		"-- opencode run do task",
	}
	for _, part := range expectedParts {
		if !strings.Contains(line, part) {
			t.Errorf("missing expected part %q in argv line: %s", part, line)
		}
	}
}

func TestOpencodeLauncherSplitValidation(t *testing.T) {
	if err := validateOpencodeLaunchSplit("invalid", "bottom"); err == nil {
		t.Errorf("expected error for invalid split mode")
	}
	if err := validateOpencodeLaunchSplit("auto", "invalid"); err == nil {
		t.Errorf("expected error for invalid split where")
	}
	if err := validateOpencodeLaunchSplit("auto", "bottom"); err != nil {
		t.Errorf("unexpected error for valid split: %v", err)
	}
}
