package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestOpencodeLauncherDryRunBasic(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--dry-run"}
	code := runOpencode(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if strings.Contains(out, " guard ") {
		t.Errorf("direct OpenCode launch unexpectedly routed through guard: %s", out)
	}
	if !strings.Contains(strings.ToLower(out), "opencode") {
		t.Errorf("expected direct OpenCode executable in dry-run stdout: %s", out)
	}
	if strings.Contains(out, "--provider openai") || strings.Contains(out, "-- opencode") {
		t.Errorf("direct OpenCode launch emitted guard-only wrapper arguments: %s", out)
	}
}

func TestOpencodeLauncherProbeWiring(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--dry-run", "--probe", "say hello from test", "--pure"}
	code := runOpencode(&stdout, &stderr, args)
	if code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	out := stdout.String()
	if strings.Contains(out, "--probe") {
		t.Errorf("direct OpenCode launch emitted guard-only --probe: %s", out)
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
	opts := opencodeLaunchOptions{model: "glm-5.3-flash", passthrough: []string{"run", "do task"}}
	argv := buildOpencodeLaunchArgv("opencode", opts)
	want := []string{"opencode", "--model", "fak/glm-5.3-flash", "run", "do task"}
	if strings.Join(argv, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("direct argv = %#v, want %#v", argv, want)
	}
	for _, guardOnly := range []string{"guard", "--provider", "--policy", "--base-url", "--audit", "--"} {
		if argvHas(argv, guardOnly) {
			t.Errorf("direct argv contains guard-only %q: %#v", guardOnly, argv)
		}
	}
}

func TestOpencodeLauncherRejectsGuardOnlyOptions(t *testing.T) {
	for _, args := range [][]string{
		{"--dry-run", "--split", "off"}, {"--dry-run", "--policy", "policy.json"},
		{"--dry-run", "--api-key-env", "OPENAI_API_KEY"}, {"--dry-run", "--base-url", "http://127.0.0.1:8080/v1"},
		{"--dry-run", "--audit", "audit.jsonl"}, {"--dry-run", "--local"},
		{"--dry-run", "--gguf", "model.gguf"}, {"--dry-run", "--metal"}, {"--dry-run", "--backend", "cpu"},
	} {
		var stdout, stderr bytes.Buffer
		if code := runOpencode(&stdout, &stderr, args); code != 2 {
			t.Errorf("runOpencode(%v) code=%d, want 2; stdout=%s stderr=%s", args, code, stdout.String(), stderr.String())
		}
	}
}

func TestOpencodeLauncherPinsChildModelOverProjectAgent(t *testing.T) {
	tests := []struct {
		name      string
		opts      opencodeLaunchOptions
		wantChild []string
	}{
		{
			name: "interactive avoids double provider prefix and preserves passthrough",
			opts: opencodeLaunchOptions{
				splitMode: "off", splitWhere: "bottom",
				model:       "fak/Qwen3.8-27B-UD-Q2_K_XL",
				passthrough: []string{"--agent", "build"},
			},
			wantChild: []string{
				"--model", "fak/Qwen3.8-27B-UD-Q2_K_XL", "--agent", "build",
			},
		},
		{
			name: "probe pins child before run and preserves passthrough",
			opts: opencodeLaunchOptions{
				splitMode: "off", splitWhere: "bottom", model: "Qwen3.8-27B-UD-Q2_K_XL",
				probePrompt: "inspect repo", auto: true, pure: true,
				passthrough: []string{"--log-level", "ERROR"},
			},
			wantChild: []string{
				"--model", "fak/Qwen3.8-27B-UD-Q2_K_XL", "run", "inspect repo",
				"--format", "json", "--auto", "--pure", "--log-level", "ERROR",
			},
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := buildOpencodeLaunchArgv("opencode", tc.opts)
			if len(got) == 0 || got[0] != "opencode" {
				t.Fatalf("direct argv = %#v", got)
			}
			if argvHas(got, "guard") || argvHas(got, "--") {
				t.Fatalf("wrapper token in direct argv: %#v", got)
			}
			child := got[1:]
			if strings.Join(child, "\x00") != strings.Join(tc.wantChild, "\x00") {
				t.Fatalf("child argv = %#v, want %#v", child, tc.wantChild)
			}
			modelFlags := 0
			for _, arg := range child {
				if arg == "--model" {
					modelFlags++
				}
			}
			if modelFlags != 1 {
				t.Fatalf("child argv has %d --model flags, want exactly one: %#v", modelFlags, child)
			}
		})
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
	var stdout, stderr bytes.Buffer
	if code := runOpencode(&stdout, &stderr, []string{"--dry-run"}); code != 0 {
		t.Fatalf("runOpencode returned %d, stderr: %s", code, stderr.String())
	}
	if out := stdout.String(); strings.Contains(out, "guard") || strings.Contains(out, "--model") {
		t.Errorf("bare direct launch acquired a guard/model default: %s", out)
	}
}

func TestOpencodeLauncherAutoDetectedModelWiring(t *testing.T) {
	t.Run("explicit model is wired", func(t *testing.T) {
		argv := buildOpencodeLaunchArgv("opencode", opencodeLaunchOptions{model: "qwen2.5-coder:7b"})
		if !argvHasPair(argv, "--model", "fak/qwen2.5-coder:7b") {
			t.Fatalf("argv=%v", argv)
		}
	})
	t.Run("empty model is omitted", func(t *testing.T) {
		argv := buildOpencodeLaunchArgv("opencode", opencodeLaunchOptions{})
		if argvHas(argv, "--model") {
			t.Fatalf("argv=%v", argv)
		}
	})
}

func TestOpencodeLauncherHaloFlag(t *testing.T) {
	for _, flag := range []string{"--halo", "--strix"} {
		t.Run(flag, func(t *testing.T) {
			var stdout, stderr bytes.Buffer
			if code := runOpencode(&stdout, &stderr, []string{"--dry-run", flag}); code != 2 {
				t.Fatalf("guard-only %s code=%d, want 2; stdout=%s stderr=%s", flag, code, stdout.String(), stderr.String())
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

func TestOpencodeConfigHaloDynamicModelFromEnv(t *testing.T) {
	t.Setenv("FAK_HALO_MODEL", "my-custom-qwen-70b")
	tmp := t.TempDir()
	var stdout, stderr bytes.Buffer
	if code := runOpencode(&stdout, &stderr, []string{"config", "--halo", "--write", "--dir", tmp}); code != 0 {
		t.Fatalf("runOpencode config --halo returned %d, stderr: %s", code, stderr.String())
	}
	data, err := os.ReadFile(filepath.Join(tmp, "opencode.json"))
	if err != nil {
		t.Fatal(err)
	}
	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatal(err)
	}
	providers := parsed["provider"].(map[string]interface{})
	models := providers["fak"].(map[string]interface{})["models"].(map[string]interface{})
	if models["my-custom-qwen-70b"] == nil {
		t.Fatalf("expected dynamic Halo model in config: %v", models)
	}
}
func TestOpencodeConfigHaloDynamicModelFromDir(t *testing.T) {
	tmp := t.TempDir()
	initialConfig := `{"model": "fak/qwen-2.5-coder-7b"}`
	if err := os.WriteFile(filepath.Join(tmp, "opencode.json"), []byte(initialConfig), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FAK_HALO_MODEL", "")
	t.Setenv("FAK_MODEL", "")

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
	if parsed["model"] != "fak/qwen-2.5-coder-7b" {
		t.Errorf("expected model fak/qwen-2.5-coder-7b, got %v", parsed["model"])
	}
	prov := parsed["provider"].(map[string]interface{})
	fak := prov["fak"].(map[string]interface{})
	models := fak["models"].(map[string]interface{})
	if models["qwen-2.5-coder-7b"] == nil {
		t.Errorf("expected qwen-2.5-coder-7b in models: %v", models)
	}
}

func TestOpencodeLauncherDarwinOneTouchMetalSession(t *testing.T) {
	argv := buildOpencodeLaunchArgv("opencode", opencodeLaunchOptions{ggufPath: "default", metal: true})
	for _, guardOnly := range []string{"guard", "--metal", "--gguf", "--backend"} {
		if argvHas(argv, guardOnly) {
			t.Errorf("direct argv contains guard-only %q: %v", guardOnly, argv)
		}
	}
}

func TestOpencodeLauncherBackendStillEmittedWithMetal(t *testing.T) {
	argv := buildOpencodeLaunchArgv("opencode", opencodeLaunchOptions{ggufPath: "default", gpuBackend: "cpu", metal: true})
	for _, guardOnly := range []string{"guard", "--metal", "--gguf", "--backend"} {
		if argvHas(argv, guardOnly) {
			t.Errorf("direct argv contains guard-only %q: %v", guardOnly, argv)
		}
	}
}

func argvHas(argv []string, want string) bool {
	for _, a := range argv {
		if a == want {
			return true
		}
	}
	return false
}

func argvHasPair(argv []string, flag, val string) bool {
	for i, a := range argv {
		if a == flag && i+1 < len(argv) && argv[i+1] == val {
			return true
		}
	}
	return false
}
