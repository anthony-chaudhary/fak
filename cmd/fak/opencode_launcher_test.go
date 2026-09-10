package main

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
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
	if strings.Contains(out, "--dangerously-skip-permissions") {
		t.Errorf("unexpected retired --dangerously-skip-permissions flag in dry-run stdout: %s", out)
	}
}

func TestOpencodeLauncherSkipPermissionsFalsePreservesNativePrompts(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "test-key")

	var stdout, stderr bytes.Buffer
	code := runOpencode(&stdout, &stderr, []string{
		"--dry-run",
		"--probe", "say hello from test",
		"--skip-permissions=false",
	})
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	if out := stdout.String(); strings.Contains(out, "--auto") || strings.Contains(out, "--dangerously-skip-permissions") {
		t.Fatalf("--skip-permissions=false emitted an OpenCode permission bypass flag: %s", out)
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

func TestOpencodeLauncherModelDefault(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	var stdout, stderr bytes.Buffer
	args := []string{"--dry-run", "--split", "off"}
	code := runOpencode(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if !strings.Contains(out, "--model") {
		t.Errorf("expected --model in dry-run stdout: %s", out)
	}
}

func TestOpencodeLauncherAutoDetectedModelWiring(t *testing.T) {
	t.Run("buildOpencodeLaunchArgv wires detected model", func(t *testing.T) {
		opts := opencodeLaunchOptions{
			splitMode:  "off",
			splitWhere: "bottom",
			model:      "qwen2.5-coder:7b",
		}
		argv := buildOpencodeLaunchArgv("fak", opts)
		foundModel := false
		for i, arg := range argv {
			if arg == "--model" && i+1 < len(argv) && argv[i+1] == "qwen2.5-coder:7b" {
				foundModel = true
				break
			}
		}
		if !foundModel {
			t.Errorf("buildOpencodeLaunchArgv missing '--model qwen2.5-coder:7b', got argv: %v", argv)
		}
	})

	t.Run("buildOpencodeLaunchArgv omits model flag when empty", func(t *testing.T) {
		opts := opencodeLaunchOptions{
			splitMode:  "off",
			splitWhere: "bottom",
			model:      "",
		}
		argv := buildOpencodeLaunchArgv("fak", opts)
		for _, arg := range argv {
			if arg == "--model" {
				t.Errorf("buildOpencodeLaunchArgv unexpectedly included '--model' when model was empty: %v", argv)
			}
		}
	})

	t.Run("runOpencode auto-detects local backend model", func(t *testing.T) {
		ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.URL.Path == "/api/tags" {
				w.Header().Set("Content-Type", "application/json")
				w.WriteHeader(http.StatusOK)
				_, _ = w.Write([]byte(`{"models":[{"name":"qwen2.5-coder:7b"}]}`))
				return
			}
			http.NotFound(w, r)
		}))
		defer ts.Close()

		t.Setenv("OLLAMA_HOST", ts.URL)
		t.Setenv("OPENAI_API_KEY", "")

		var stdout, stderr bytes.Buffer
		args := []string{"--dry-run", "--split", "off"}
		code := runOpencode(&stdout, &stderr, args)
		if code != 0 {
			t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "--model qwen2.5-coder:7b") {
			t.Errorf("expected '--model qwen2.5-coder:7b' in dry-run stdout: %s", out)
		}
		errOut := stderr.String()
		if !strings.Contains(errOut, "auto-connected to local Ollama") || !strings.Contains(errOut, "qwen2.5-coder:7b") {
			t.Errorf("expected auto-connection diagnostics in stderr: %s", errOut)
		}
	})

	t.Run("runOpencode with explicit model flag overrides auto-detection", func(t *testing.T) {
		var stdout, stderr bytes.Buffer
		args := []string{"--dry-run", "--split", "off", "--model", "custom-model:32b"}
		code := runOpencode(&stdout, &stderr, args)
		if code != 0 {
			t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
		}
		out := stdout.String()
		if !strings.Contains(out, "--model custom-model:32b") {
			t.Errorf("expected '--model custom-model:32b' in dry-run stdout: %s", out)
		}
	})
}

func TestOpencodeLauncherHaloFlag(t *testing.T) {
	t.Setenv("OPENAI_API_KEY", "")
	t.Setenv("FAK_HALO_HOST", "")
	t.Setenv("FAK_STRIX_HOST", "")

	for _, flag := range []string{"--halo", "--strix"} {
		t.Run(flag, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			args := []string{"--dry-run", "--split", "off", flag}
			code := runOpencode(&stdout, &stderr, args)
			if code != 0 {
				t.Fatalf("runOpencode %s returned %d, stderr: %s", flag, code, stderr.String())
			}
			out := stdout.String()
			if !strings.Contains(out, "--base-url http://127.0.0.1:8080/v1") {
				t.Errorf("expected '--base-url http://127.0.0.1:8080/v1' in dry-run stdout: %s", out)
			}
			if !strings.Contains(out, "--model qwen-2.5-coder-32b-instruct") {
				t.Errorf("expected '--model qwen-2.5-coder-32b-instruct' in dry-run stdout: %s", out)
			}
			errOut := stderr.String()
			if !strings.Contains(errOut, "targeting local Halo server") {
				t.Errorf("expected targeting diagnostics in stderr: %s", errOut)
			}
		})
	}
}

func TestOpencodeConfigHaloFlag(t *testing.T) {
	tmp := t.TempDir()
	var stdout, stderr bytes.Buffer
	args := []string{"config", "--halo", "--write", "--dir", tmp}
	code := runOpencode(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runOpencode config --halo returned %d, stderr: %s", code, stderr.String())
	}
	data, err := os.ReadFile(filepath.Join(tmp, "opencode.json"))
	if err != nil {
		t.Fatalf("failed to read created config: %v", err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to parse config: %v", err)
	}
	prov := parsed["provider"].(map[string]interface{})
	fak := prov["fak"].(map[string]interface{})
	opts := fak["options"].(map[string]interface{})
	if opts["baseURL"] != "http://127.0.0.1:8080/v1" {
		t.Errorf("expected baseURL http://127.0.0.1:8080/v1, got %v", opts["baseURL"])
	}
	models := fak["models"].(map[string]interface{})
	if models["qwen-2.5-coder-32b-instruct"] == nil {
		t.Errorf("expected qwen-2.5-coder-32b-instruct in models: %v", models)
	}
}

