package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestOpsRunGuardedReceipt(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(prompt, []byte("private prompt\n"), 0600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("OPENCODE_CONFIG_CONTENT", `{"permission":{"*":"deny","read":"allow"},"plugin":["protection"],"small_model":"outside/model","enabled_providers":["outside"]}`)
	t.Setenv("FAK_OPS_TEST_KEY", "fixture-only")
	old := opsRunExecute
	t.Cleanup(func() { opsRunExecute = old })
	for _, tc := range []struct {
		name             string
		complete, failed bool
		exit, want       int
		status           string
	}{
		{"complete", true, false, 0, 0, "succeeded"},
		{"gemini", true, false, 0, 0, "succeeded"},
		{"missing_completion", false, false, 0, 1, "failed"},
		{"tool_error", true, true, 0, 1, "failed"},
		{"child_error", true, false, 7, 7, "failed"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wire := "openai"
			if tc.name == "gemini" {
				wire = "gemini"
			}
			opsRunExecute = func(ctx context.Context, out, errOut io.Writer, argv, env []string, p []byte) (int, bool, bool) {
				if string(p) != "private prompt\n" || strings.Contains(strings.Join(argv, " "), "private prompt") {
					t.Fatal("prompt must travel only on stdin")
				}
				if len(argv) < 10 || argv[1] != "guard" || !strings.Contains(strings.Join(argv, " "), "--provider "+wire+" --split off") {
					t.Fatalf("unguarded argv: %q", argv)
				}
				var cfg map[string]any
				for _, e := range env {
					if strings.HasPrefix(e, "OPENCODE_CONFIG_CONTENT=") {
						if err := json.Unmarshal([]byte(strings.TrimPrefix(e, "OPENCODE_CONFIG_CONTENT=")), &cfg); err != nil {
							t.Fatal(err)
						}
					}
				}
				model := cfg["model"].(string)
				provider, _, _ := strings.Cut(model, "/")
				if !strings.HasPrefix(provider, "fak_ops_") || cfg["small_model"] != model || cfg["enabled_providers"].([]any)[0] != provider {
					t.Fatalf("routing not pinned: %v", cfg)
				}
				if wire == "gemini" {
					p := cfg["provider"].(map[string]any)[provider].(map[string]any)
					opts := p["options"].(map[string]any)
					if p["npm"] != "@ai-sdk/google" || opts["baseURL"] != "{env:GOOGLE_GEMINI_BASE_URL}/v1beta" || opts["apiKey"] != "fak-ops-guard" {
						t.Fatalf("Gemini native route not pinned: %v", p)
					}
				}
				if cfg["permission"].(map[string]any)["*"] != "deny" || cfg["plugin"].([]any)[0] != "protection" {
					t.Fatal("operator protections changed")
				}
				return tc.exit, tc.complete, tc.failed
			}
			receipt := filepath.Join(dir, tc.name+".json")
			var out, errs bytes.Buffer
			got := runOpsRun(&out, &errs, []string{"--prompt-file", prompt, "--receipt", receipt, "--model", "fixture", "--provider", wire, "--api-key-env", "FAK_OPS_TEST_KEY"})
			if got != tc.want {
				t.Fatalf("exit=%d want=%d stderr=%s", got, tc.want, errs.String())
			}
			data, err := os.ReadFile(receipt)
			if err != nil {
				t.Fatal(err)
			}
			var r opsRunReceipt
			if err := json.Unmarshal(data, &r); err != nil {
				t.Fatal(err)
			}
			if r.Status != tc.status || r.ExitCode != tc.want || r.Finished.IsZero() || strings.Contains(string(data), "private prompt") {
				t.Fatalf("invalid receipt: %s", data)
			}
		})
	}
	opsRunExecute = func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool) {
		t.Fatal("aliased receipt launched child")
		return 0, true, false
	}
	if got := runOpsRun(io.Discard, io.Discard, []string{"--prompt-file", prompt, "--receipt", prompt, "--model", "fixture"}); got != 2 {
		t.Fatalf("alias exit=%d", got)
	}
}

func TestOpsRunEventsCompletion(t *testing.T) {
	for _, tc := range []struct {
		events           string
		complete, failed bool
	}{
		{"{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}\n", true, false},
		{"{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}\n{\"type\":\"step_start\"}\n", false, false},
		{"{\"type\":\"tool_use\",\"part\":{\"state\":{\"status\":\"error\"}}}\n{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}", true, true},
		{"{\"type\":\"error\",\"error\":{\"message\":\"provider failed\"}}\n{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}", true, true},
		{"{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}\n{\"type\":", true, true},
	} {
		w := &opsRunEvents{output: io.Discard}
		for _, b := range []byte(tc.events) {
			_, _ = w.Write([]byte{b})
		}
		w.finishLine()
		if w.complete != tc.complete || w.failed != tc.failed {
			t.Fatalf("events=%q complete=%v failed=%v", tc.events, w.complete, w.failed)
		}
	}
}

type opsRunReadyWriter struct {
	ready chan struct{}
	once  sync.Once
}

func (w *opsRunReadyWriter) Write(p []byte) (int, error) {
	if bytes.Contains(p, []byte("ready")) {
		w.once.Do(func() { close(w.ready) })
	}
	return len(p), nil
}

func TestOpsRunCancelChild(t *testing.T) {
	if os.Getenv("FAK_OPS_TEST_CHILD") == "1" {
		_, _ = os.Stdout.WriteString("ready\n")
		time.Sleep(time.Minute)
		os.Exit(0)
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := &opsRunReadyWriter{ready: make(chan struct{})}
	done := make(chan int, 1)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		code, _, _ := executeOpsRun(ctx, w, io.Discard, []string{exe, "-test.run=^TestOpsRunCancelChild$"}, append(os.Environ(), "FAK_OPS_TEST_CHILD=1"), nil)
		done <- code
	}()
	select {
	case <-w.ready:
	case <-time.After(10 * time.Second):
		t.Fatal("child never started")
	}
	cancel()
	select {
	case code := <-done:
		if code == 0 {
			t.Fatal("cancelled child returned success")
		}
	case <-time.After(10 * time.Second):
		t.Fatal("child survived cancellation")
	}
}
