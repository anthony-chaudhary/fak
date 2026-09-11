package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
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
			opsRunExecute = func(ctx context.Context, out, errOut io.Writer, argv, env []string, p []byte) (int, bool, bool, []opsRunLifecycleRecord) {
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
				return tc.exit, tc.complete, tc.failed, nil
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
	opsRunExecute = func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		t.Fatal("aliased receipt launched child")
		return 0, true, false, nil
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
	if os.Getenv("FAK_OPS_TEST_CHILD_EXIT_EARLY") == "1" {
		_, _ = os.Stdout.WriteString("ready\n")
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
	var (
		lcMu sync.Mutex
		lc   []opsRunLifecycleRecord
	)
	go func() {
		code, _, _, l := executeOpsRun(ctx, w, io.Discard, []string{exe, "-test.run=^TestOpsRunCancelChild$"}, append(os.Environ(), "FAK_OPS_TEST_CHILD=1"), nil)
		lcMu.Lock()
		lc = l
		lcMu.Unlock()
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
	lcMu.Lock()
	defer lcMu.Unlock()
	if len(lc) == 0 {
		t.Fatal("expected lifecycle record on cancelled child")
	}
	if lc[0].Reason != "cancelled" {
		t.Fatalf("reason=%q want=cancelled", lc[0].Reason)
	}
	if lc[0].ChildState != "alive" {
		t.Fatalf("child_state=%q want=alive", lc[0].ChildState)
	}
	if lc[0].Signal != "SIGKILL" {
		t.Fatalf("signal=%q want=SIGKILL", lc[0].Signal)
	}
	if lc[0].Error != "" {
		t.Fatalf("unexpected kill error: %s", lc[0].Error)
	}
}

func TestOpsRunReceiptLifecycleCancellation(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(prompt, []byte("sentinel prompt\n"), 0600); err != nil {
		t.Fatal(err)
	}
	receipt := filepath.Join(dir, "receipt.json")
	old := opsRunExecute
	t.Cleanup(func() { opsRunExecute = old })

	opsRunExecute = func(ctx context.Context, stdout, stderr io.Writer, argv, env []string, p []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		return 130, false, false, []opsRunLifecycleRecord{
			{
				Reason:            "cancelled",
				TerminationReason: "cancelled",
				ChildState:        "alive",
				State:             "alive",
				Signal:            "SIGKILL",
				ElapsedMS:         42,
				DurationMS:        42,
				Duration:          "42ms",
			},
		}
	}

	var out, errs bytes.Buffer
	_ = runOpsRun(&out, &errs, []string{"--prompt-file", prompt, "--receipt", receipt, "--model", "fixture", "--timeout", "5s"})
	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var r opsRunReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	if r.Schema != "fak-ops-run/1" {
		t.Fatalf("schema = %q, want fak-ops-run/1", r.Schema)
	}
	if len(r.Lifecycle) != 1 {
		t.Fatalf("lifecycle length = %d, want 1", len(r.Lifecycle))
	}
	rec := r.Lifecycle[0]
	if rec.Reason != "cancelled" || rec.TerminationReason != "cancelled" {
		t.Fatalf("unexpected reason: %+v", rec)
	}
	if rec.ChildState != "alive" || rec.State != "alive" {
		t.Fatalf("unexpected child state: %+v", rec)
	}
	if rec.Signal != "SIGKILL" {
		t.Fatalf("unexpected signal: %+v", rec)
	}
	if rec.ElapsedMS != 42 {
		t.Fatalf("unexpected elapsed_ms: %+v", rec)
	}
	if strings.Contains(string(data), "sentinel prompt") {
		t.Fatal("receipt leaked prompt")
	}
}

func TestOpsRunLifecycleKillFailureAttribution(t *testing.T) {
	if os.Getenv("FAK_OPS_TEST_CHILD") == "1" {
		_, _ = os.Stdout.WriteString("ready\n")
		time.Sleep(time.Minute)
		os.Exit(0)
	}
	oldCancel := opsRunCancelProcess
	t.Cleanup(func() { opsRunCancelProcess = oldCancel })

	simulatedErr := errors.New("ChildProcess.kill: simulated OS failure: Access is denied")
	opsRunCancelProcess = func(cmd *exec.Cmd, origCancel func() error) error {
		if origCancel != nil {
			_ = origCancel()
		}
		return simulatedErr
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := &opsRunReadyWriter{ready: make(chan struct{})}
	done := make(chan int, 1)
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	var (
		lcMu sync.Mutex
		lc   []opsRunLifecycleRecord
	)
	go func() {
		code, _, _, l := executeOpsRun(ctx, w, io.Discard, []string{exe, "-test.run=^TestOpsRunLifecycleKillFailureAttribution$"}, append(os.Environ(), "FAK_OPS_TEST_CHILD=1"), nil)
		lcMu.Lock()
		lc = l
		lcMu.Unlock()
		done <- code
	}()
	select {
	case <-w.ready:
	case <-time.After(10 * time.Second):
		t.Fatal("child never started")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("child survived cancellation")
	}

	lcMu.Lock()
	defer lcMu.Unlock()
	if len(lc) == 0 {
		t.Fatal("expected lifecycle record on killed child")
	}
	if lc[0].Error != simulatedErr.Error() {
		t.Fatalf("error = %q, want %q", lc[0].Error, simulatedErr.Error())
	}
	if lc[0].OSError != simulatedErr.Error() {
		t.Fatalf("os_error = %q, want %q", lc[0].OSError, simulatedErr.Error())
	}
	if lc[0].ChildState != "alive" {
		t.Fatalf("child_state = %q, want alive", lc[0].ChildState)
	}
	if lc[0].Reason != "cancelled" {
		t.Fatalf("reason = %q, want cancelled", lc[0].Reason)
	}
}

func TestOpsRunLifecycleAlreadyExitedChild(t *testing.T) {
	if os.Getenv("FAK_OPS_TEST_CHILD_EXIT_EARLY") == "1" {
		_, _ = os.Stdout.WriteString("ready\n")
		os.Exit(0)
	}
	exe, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}

	// 1. Probe nil cmd
	if state := probeChildProcessState(nil); state != "unknown" {
		t.Fatalf("probe nil Cmd = %q, want unknown", state)
	}
	if state := probeChildProcessState(&exec.Cmd{}); state != "unknown" {
		t.Fatalf("probe nil Process = %q, want unknown", state)
	}

	// 2. Probe executed process that exited
	dummy := exec.Command(exe, "-test.run=^TestOpsRunLifecycleAlreadyExitedChild$")
	dummy.Env = append(os.Environ(), "FAK_OPS_TEST_CHILD_EXIT_EARLY=1")
	_ = dummy.Run()
	if probed := probeChildProcessState(dummy); probed != "exited" {
		t.Fatalf("probe exited process = %q, want exited", probed)
	}

	// 3. Test execution where probe reports exited at cancellation attempt
	oldProbe := opsRunProbeChildState
	t.Cleanup(func() { opsRunProbeChildState = oldProbe })
	opsRunProbeChildState = func(cmd *exec.Cmd) string {
		return "exited"
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	w := &opsRunReadyWriter{ready: make(chan struct{})}
	done := make(chan int, 1)
	var (
		lcMu sync.Mutex
		lc   []opsRunLifecycleRecord
	)
	go func() {
		code, _, _, l := executeOpsRun(ctx, w, io.Discard, []string{exe, "-test.run=^TestOpsRunLifecycleAlreadyExitedChild$"}, append(os.Environ(), "FAK_OPS_TEST_CHILD_EXIT_EARLY=1"), nil)
		lcMu.Lock()
		lc = l
		lcMu.Unlock()
		done <- code
	}()
	select {
	case <-w.ready:
	case <-time.After(10 * time.Second):
		t.Fatal("child never started")
	}
	cancel()
	select {
	case <-done:
	case <-time.After(10 * time.Second):
		t.Fatal("executeOpsRun timed out")
	}

	lcMu.Lock()
	defer lcMu.Unlock()
	if len(lc) == 0 {
		t.Fatal("expected lifecycle record")
	}
	if lc[0].ChildState != "exited" {
		t.Fatalf("child_state = %q, want exited", lc[0].ChildState)
	}
	if lc[0].Reason != "cancelled" {
		t.Fatalf("reason = %q, want cancelled", lc[0].Reason)
	}
}

func TestOpsRunLifecycleReasonsAndContext(t *testing.T) {
	// 1. Timeout
	ctxTimeout, cancelTimeout := context.WithTimeout(context.Background(), time.Nanosecond)
	defer cancelTimeout()
	time.Sleep(5 * time.Millisecond)
	if got := opsRunDetermineTerminationReason(ctxTimeout); got != "timeout" {
		t.Fatalf("reason = %q, want timeout", got)
	}

	// 2. Cancelled
	ctxCancel, cancel := context.WithCancel(context.Background())
	cancel()
	if got := opsRunDetermineTerminationReason(ctxCancel); got != "cancelled" {
		t.Fatalf("reason = %q, want cancelled", got)
	}

	// 3. Shutdown
	ctxShutdown := withSignalChecker(context.Background(), func() bool { return true })
	if got := opsRunDetermineTerminationReason(ctxShutdown); got != "shutdown" {
		t.Fatalf("reason = %q, want shutdown", got)
	}

	// 4. Error
	ctxError := withOpsRunTerminationReason(context.Background(), "error")
	if got := opsRunDetermineTerminationReason(ctxError); got != "error" {
		t.Fatalf("reason = %q, want error", got)
	}
}

func TestOpsRunReceiptTimedOutFallback(t *testing.T) {
	dir := t.TempDir()
	prompt := filepath.Join(dir, "prompt.txt")
	if err := os.WriteFile(prompt, []byte("prompt text\n"), 0600); err != nil {
		t.Fatal(err)
	}
	receipt := filepath.Join(dir, "receipt.json")
	old := opsRunExecute
	t.Cleanup(func() { opsRunExecute = old })

	opsRunExecute = func(ctx context.Context, stdout, stderr io.Writer, argv, env []string, p []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		<-ctx.Done()
		return 124, false, false, nil
	}

	var out, errs bytes.Buffer
	got := runOpsRun(&out, &errs, []string{"--prompt-file", prompt, "--receipt", receipt, "--model", "fixture", "--timeout", "20ms"})
	if got != 124 {
		t.Fatalf("exit = %d, want 124", got)
	}
	data, err := os.ReadFile(receipt)
	if err != nil {
		t.Fatal(err)
	}
	var r opsRunReceipt
	if err := json.Unmarshal(data, &r); err != nil {
		t.Fatal(err)
	}
	if r.Status != "timed_out" || r.ExitCode != 124 {
		t.Fatalf("status = %q, exit_code = %d", r.Status, r.ExitCode)
	}
	if len(r.Lifecycle) == 0 {
		t.Fatal("expected lifecycle record on timeout")
	}
	if r.Lifecycle[0].Reason != "timeout" {
		t.Fatalf("lifecycle reason = %q, want timeout", r.Lifecycle[0].Reason)
	}
	if r.Lifecycle[0].ChildState != "unknown" {
		t.Fatalf("lifecycle child_state = %q, want unknown", r.Lifecycle[0].ChildState)
	}
}

func TestResolvePOSIXOpenCodeBinary(t *testing.T) {
	if runtime.GOOS == "windows" {
		t.Skip("POSIX-only install-location resolver")
	}
	home := t.TempDir()
	if got := resolvePOSIXOpenCodeBinary(home); got != "" {
		t.Fatalf("expected empty resolution for dir without installs, got %q", got)
	}
	official := filepath.Join(home, ".opencode", "bin", "opencode")
	if err := os.MkdirAll(filepath.Dir(official), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(official, []byte("#!/bin/sh\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	if got := resolvePOSIXOpenCodeBinary(home); got != official {
		t.Fatalf("got %q, want %q", got, official)
	}
	// A directory at the candidate path must not resolve.
	if err := os.Remove(official); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(official, 0o755); err != nil {
		t.Fatal(err)
	}
	if got := resolvePOSIXOpenCodeBinary(home); got != "" {
		t.Fatalf("directory candidate must not resolve, got %q", got)
	}
}
