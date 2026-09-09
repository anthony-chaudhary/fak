package main

import (
	"bytes"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBuildPiLaunchArgv(t *testing.T) {
	opts := piLaunchOptions{
		command:     "pi",
		provider:    "fak",
		model:       "qwen38:27b-q4",
		probePrompt: "Explain Metal GPU acceleration",
		thinking:    "high",
		tools:       "read,bash,edit,write",
		passthrough: []string{"--no-session"},
	}

	argv := buildPiLaunchArgv(opts)
	want := []string{
		"pi",
		"--provider", "fak",
		"--model", "qwen38:27b-q4",
		"-p", "Explain Metal GPU acceleration",
		"--thinking", "high",
		"--tools", "read,bash,edit,write",
		"--no-session",
	}

	if len(argv) != len(want) {
		t.Fatalf("buildPiLaunchArgv length = %d, want %d: %v", len(argv), len(want), argv)
	}
	for i := range argv {
		if argv[i] != want[i] {
			t.Errorf("argv[%d] = %q, want %q", i, argv[i], want[i])
		}
	}
}

func TestRunPiDryRun(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{"--dry-run", "--model", "qwen38:27b-q4", "--check-backend=false"})
	if code != 0 {
		t.Fatalf("runPi --dry-run returned %d, want 0; stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	errOut := stderr.String()

	if !strings.Contains(out, "pi --provider fak --model qwen38:27b-q4") {
		t.Errorf("expected stdout to contain launch command, got: %s", out)
	}
	if !strings.Contains(errOut, "backend     = http://127.0.0.1:8080/v1 (raw without guard)") {
		t.Errorf("expected stderr to contain raw backend note, got: %s", errOut)
	}
	if !strings.Contains(errOut, "provider    = fak") {
		t.Errorf("expected stderr to contain provider fak, got: %s", errOut)
	}
}

func TestRunPiConfigSubcommand(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{"config", "--addr", "127.0.0.1:8080", "--model", "qwen38:27b-q4"})
	if code != 0 {
		t.Fatalf("runPi config returned %d, want 0; stderr: %s", code, stderr.String())
	}

	out := stdout.String()
	if !strings.Contains(out, "http://127.0.0.1:8080/v1") {
		t.Errorf("expected baseURL in output: %s", out)
	}
	if !strings.Contains(out, "qwen38:27b-q4") {
		t.Errorf("expected model in output: %s", out)
	}
	if !strings.Contains(out, `"openai-completions"`) {
		t.Errorf("expected openai-completions in output: %s", out)
	}
}

func TestRunPiConfigSubcommandWrite(t *testing.T) {
	tmp := t.TempDir()
	targetPath := filepath.Join(tmp, "models.json")

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"config",
		"--write",
		"--path", targetPath,
		"--addr", "127.0.0.1:9000",
		"--model", "qwen38:27b-q4",
	})
	if code != 0 {
		t.Fatalf("runPi config --write returned %d, want 0; stderr: %s", code, stderr.String())
	}

	data, err := os.ReadFile(targetPath)
	if err != nil {
		t.Fatalf("failed to read written file %s: %v", targetPath, err)
	}

	content := string(data)
	if !strings.Contains(content, "http://127.0.0.1:9000/v1") {
		t.Errorf("expected baseURL in written file: %s", content)
	}
	if !strings.Contains(content, "qwen38:27b-q4") {
		t.Errorf("expected model in written file: %s", content)
	}
}

func TestProbePiBackend(t *testing.T) {
	// 1. Healthz endpoint returns model
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/healthz" {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"ok":true,"model":"qwen38:27b-q4","engine":"inkernel"}`))
			return
		}
		http.NotFound(w, r)
	}))
	defer ts.Close()

	model, ok := probePiBackend(ts.URL+"/v1", time.Second)
	if !ok || model != "qwen38:27b-q4" {
		t.Errorf("probePiBackend healthz = (%q, %v), want (\"qwen38:27b-q4\", true)", model, ok)
	}

	// 2. Unreachable endpoint returns false
	_, unreachOk := probePiBackend("http://127.0.0.1:65530/v1", 100*time.Millisecond)
	if unreachOk {
		t.Errorf("probePiBackend unreachable want false, got true")
	}
}

func TestRunPiMockExecution(t *testing.T) {
	origRun := piLaunchRun
	defer func() { piLaunchRun = origRun }()

	var capturedArgv []string
	var capturedEnv []string
	piLaunchRun = func(stdout, stderr io.Writer, argv, env []string) int {
		capturedArgv = argv
		capturedEnv = env
		return 0
	}

	tmp := t.TempDir()
	configPath := filepath.Join(tmp, "models.json")

	var stdout, stderr bytes.Buffer
	code := runPi(&stdout, &stderr, []string{
		"--check-backend=false",
		"--config-path", configPath,
		"--model", "qwen38:27b-q4",
		"--probe", "hello from pi",
	})
	if code != 0 {
		t.Fatalf("runPi mock execution returned %d, want 0; stderr: %s", code, stderr.String())
	}

	// Check argv
	cmdStr := strings.Join(capturedArgv, " ")
	if !strings.Contains(cmdStr, "pi --provider fak --model qwen38:27b-q4 -p hello from pi") {
		t.Errorf("unexpected captured argv: %v", capturedArgv)
	}

	// Check config file was created
	if _, err := os.Stat(configPath); err != nil {
		t.Errorf("expected config file %s to be created: %v", configPath, err)
	}

	// Check env contains PI_CODING_AGENT_DIR
	foundEnv := false
	for _, e := range capturedEnv {
		if strings.HasPrefix(e, "PI_CODING_AGENT_DIR=") {
			foundEnv = true
			break
		}
	}
	if !foundEnv {
		t.Errorf("expected PI_CODING_AGENT_DIR in env: %v", capturedEnv)
	}
}
