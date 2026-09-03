package tb4bench

import (
	"context"
	"strings"
	"testing"
	"time"
)

func TestLlamaServerSupervisorLifecycle(t *testing.T) {
	cfg := DefaultLlamaServerConfig("model.gguf", "sha256:dummy", 0)
	supervisor := NewLlamaServerSupervisor(cfg)

	// Start mock HTTP server
	if err := supervisor.StartMock(); err != nil {
		t.Fatalf("failed to start mock llama-server: %v", err)
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Verify health polling succeeds
	if err := supervisor.WaitReady(ctx, 3*time.Second); err != nil {
		t.Fatalf("server failed readiness probe: %v", err)
	}

	// Verify base URL contains port
	url := supervisor.BaseURL()
	if !strings.HasPrefix(url, "http://127.0.0.1:") {
		t.Errorf("expected http://127.0.0.1:<port>, got %s", url)
	}

	// Stop server cleanly
	if err := supervisor.Stop(); err != nil {
		t.Fatalf("failed to stop server: %v", err)
	}
}

func TestBuildLlamaServerArgs(t *testing.T) {
	cfg := DefaultLlamaServerConfig("test-qwen.gguf", "sha256:1234", 8080)
	args := BuildLlamaServerArgs(cfg)

	argStr := strings.Join(args, " ")
	expectedFlags := []string{
		"--model test-qwen.gguf",
		"--port 8080",
		"--temp 0.0",
		"--seed 42",
		"--parallel 1",
		"--ctx-size 32768",
	}

	for _, flag := range expectedFlags {
		if !strings.Contains(argStr, flag) {
			t.Errorf("args missing required flag %q: got %s", flag, argStr)
		}
	}
}
