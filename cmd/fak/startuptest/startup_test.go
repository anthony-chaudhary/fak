package startuptest

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestAgentDefaultLiveStartsInteractiveBeforeInference(t *testing.T) {
	repoRoot := findRepoRoot(t)
	buildDir := t.TempDir()
	exeName := "fak-startup-test"
	if runtime.GOOS == "windows" {
		exeName += ".exe"
	}
	exePath := filepath.Join(buildDir, exeName)

	buildCtx, cancelBuild := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancelBuild()
	build := exec.CommandContext(buildCtx, "go", "build", "-o", exePath, "./cmd/fak")
	build.Dir = repoRoot
	build.Env = append(os.Environ(), "GOWORK=off")
	buildOutput, err := build.CombinedOutput()
	if buildCtx.Err() != nil {
		t.Fatalf("building checkout CLI timed out: %v\n%s", buildCtx.Err(), buildOutput)
	}
	if err != nil {
		t.Fatalf("building checkout CLI: %v\n%s", err, buildOutput)
	}

	var generations atomic.Int32
	provider := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		generations.Add(1)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(map[string]any{
			"model": "startup-fixture",
			"choices": []map[string]any{{
				"message":       map[string]string{"role": "assistant", "content": "fixture answer"},
				"finish_reason": "stop",
			}},
		})
	}))
	defer provider.Close()

	runDir := t.TempDir()
	runCtx, cancelRun := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancelRun()
	var stdout, stderr bytes.Buffer
	run := exec.CommandContext(runCtx, exePath,
		"agent",
		"--provider", "openai",
		"--base-url", provider.URL+"/v1",
		"--model", "startup-fixture",
		"--api-key-env", "FAK_STARTUP_TEST_EMPTY_KEY",
		"--code-tools=false",
		"--sys-tools=false",
		"--mcp-tools=false",
		"--subagents=false",
		"--skills=false",
		"--memory=false",
		"--max-turns", "1",
	)
	run.Dir = runDir
	run.Env = append(os.Environ(),
		"FAK_CONSOLE_FILE="+filepath.Join(runDir, "console.json"),
		"FAK_STARTUP_TEST_EMPTY_KEY=",
	)
	run.Stdin = strings.NewReader("")
	run.Stdout = &stdout
	run.Stderr = &stderr
	err = run.Run()
	if runCtx.Err() != nil {
		t.Fatalf("checkout CLI timed out waiting at EOF: %v\nstdout:\n%s\nstderr:\n%s", runCtx.Err(), stdout.String(), stderr.String())
	}
	if err != nil {
		t.Fatalf("checkout CLI exited with %v\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}

	if got := stdout.String(); !strings.Contains(got, "native REPL") || !strings.Contains(got, "you> ") {
		t.Fatalf("default live agent did not show the interactive prompt before EOF:\n%s", got)
	}
	if got := generations.Load(); got != 0 {
		t.Fatalf("default live agent generated %d times before user input, want 0", got)
	}
	if _, err := os.Stat(filepath.Join(runDir, "agent-report.json")); !os.IsNotExist(err) {
		t.Fatalf("default live agent created a one-shot report before user input: %v", err)
	}
}

func findRepoRoot(t *testing.T) string {
	t.Helper()
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			t.Fatalf("could not find repository root above %s", dir)
		}
		dir = parent
	}
}
