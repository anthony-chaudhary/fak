package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
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

func TestOpencodeLauncherSynchronizesProjectAssets(t *testing.T) {
	ws := t.TempDir()
	manifestDir := filepath.Join(ws, ".claude")
	if err := os.MkdirAll(filepath.Join(ws, ".claude", "skills", "openskill"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(ws, ".claude", "memory"), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(filepath.Join(ws, ".claude", "goal-prompts"), 0755); err != nil {
		t.Fatal(err)
	}
	manifestJSON := `{
  "schema": "fak-project-assets/1",
  "skills": {
    "canonical_root": ".claude/skills",
    "codex_root": ".agents/skills",
    "include": ["SKILL.md"],
    "exclude": []
  },
  "memories": {
    "canonical_root": ".claude/memory",
    "include": ["*.md"],
    "exclude": []
  },
  "goal_prompts": {
    "canonical_root": ".claude/goal-prompts",
    "include": ["*.md"],
    "exclude": []
  },
  "harnesses": {
    "claude": {"skills": ".claude/skills", "memories": ".claude/memory", "goal_prompts": ".claude/goal-prompts"},
    "codex": {"skills": ".agents/skills", "memories": "cmd", "goal_prompts": ".claude/goal-prompts"},
    "fak-native": {"skills": ".claude/skills", "memories": "cmd", "goal_prompts": ".claude/goal-prompts"},
    "opencode": {"skills": ".agents/skills", "memories": "cmd", "goal_prompts": ".claude/goal-prompts"}
  }
}`
	if err := os.WriteFile(filepath.Join(manifestDir, "project-assets.json"), []byte(manifestJSON), 0644); err != nil {
		t.Fatal(err)
	}
	skillMD := "---\nname: openskill\ndescription: OpenCode test skill\n---\n# Open\n"
	if err := os.WriteFile(filepath.Join(ws, ".claude", "skills", "openskill", "SKILL.md"), []byte(skillMD), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, ".claude", "memory", "base.md"), []byte("memory\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, ".claude", "goal-prompts", "base.md"), []byte("prompt\n"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(ws, "opencode.json"), []byte(`{"snapshot": false}`), 0644); err != nil {
		t.Fatal(err)
	}

	adapterPath := filepath.Join(ws, ".agents", "skills", "openskill", "SKILL.md")
	if _, err := os.Stat(adapterPath); !os.IsNotExist(err) {
		t.Fatalf("expected adapter to not exist before launch")
	}

	origRun := opencodeLaunchRun
	ran := false
	opencodeLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		ran = true
		return 0
	}
	t.Cleanup(func() { opencodeLaunchRun = origRun })

	t.Chdir(ws)
	var stdout, stderr bytes.Buffer
	code := runOpencode(&stdout, &stderr, []string{"--quiet"})
	if code != 0 {
		t.Fatalf("runOpencode failed with code %d, stderr: %s", code, stderr.String())
	}
	if !ran {
		t.Fatal("expected opencodeLaunchRun to be called")
	}
	if _, err := os.Stat(adapterPath); err != nil {
		t.Fatalf("expected adapter to be synchronized, got error: %v", err)
	}
}

func TestOpencodeLauncherVerifiesSnapshotWarning(t *testing.T) {
	ws := t.TempDir()
	if err := os.WriteFile(filepath.Join(ws, "opencode.json"), []byte(`{"snapshot": true}`), 0644); err != nil {
		t.Fatal(err)
	}

	origRun := opencodeLaunchRun
	opencodeLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		return 0
	}
	t.Cleanup(func() { opencodeLaunchRun = origRun })

	t.Chdir(ws)
	var stdout, stderr bytes.Buffer
	code := runOpencode(&stdout, &stderr, nil)
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stderr.String(), "warning:") || !strings.Contains(stderr.String(), "snapshot") {
		t.Fatalf("expected snapshot warning in stderr, got: %s", stderr.String())
	}

	// With --quiet, warning should be suppressed
	stderr.Reset()
	code = runOpencode(&stdout, &stderr, []string{"--quiet"})
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	if strings.Contains(stderr.String(), "warning:") {
		t.Fatalf("expected warning to be suppressed with --quiet, got: %s", stderr.String())
	}
}
