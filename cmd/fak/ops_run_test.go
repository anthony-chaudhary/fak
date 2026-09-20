package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestOpsRunWorkspace(t *testing.T) {
	t.Run("nonexistent_refuses_before_launch", func(t *testing.T) {
		dir := t.TempDir()
		prompt := filepath.Join(dir, "prompt.txt")
		receipt := filepath.Join(dir, "receipt.json")
		if err := os.WriteFile(prompt, []byte("must not launch\n"), 0o600); err != nil {
			t.Fatal(err)
		}

		old := opsRunExecute
		t.Cleanup(func() { opsRunExecute = old })
		var launches atomic.Int32
		opsRunExecute = func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
			launches.Add(1)
			return 0, true, false, nil
		}

		code := runOpsRun(io.Discard, io.Discard, []string{
			"--workspace", filepath.Join(dir, "does-not-exist"),
			"--prompt-file", prompt,
			"--receipt", receipt,
			"--model", "fixture",
			"--provider", "openai",
			"--base-url", "http://127.0.0.1:1/v1",
		})
		if code != 2 {
			t.Fatalf("nonexistent workspace exit=%d, want 2", code)
		}
		if launches.Load() != 0 {
			t.Fatalf("nonexistent workspace launched child %d time(s)", launches.Load())
		}
		if _, err := os.Stat(receipt); !errors.Is(err, os.ErrNotExist) {
			t.Fatalf("nonexistent workspace created receipt: %v", err)
		}
	})

	t.Run("canonical_alias_bound_to_child_and_receipt", func(t *testing.T) {
		workspace := t.TempDir()
		if err := os.Mkdir(filepath.Join(workspace, ".git"), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(workspace, "dos.toml"), []byte("[lanes]\n[lanes.trees]\ncmd = [\"cmd/**\"]\n"), 0o644); err != nil {
			t.Fatal(err)
		}
		workspaceAlias := filepath.Join(t.TempDir(), "workspace-alias")
		if err := os.Symlink(workspace, workspaceAlias); err != nil {
			aliasChild := filepath.Join(workspace, "alias-child")
			if err := os.Mkdir(aliasChild, 0o755); err != nil {
				t.Fatal(err)
			}
			workspaceAlias = workspace + string(os.PathSeparator) + "alias-child" + string(os.PathSeparator) + ".."
		}
		coordinator, err := os.Getwd()
		if err != nil {
			t.Fatal(err)
		}

		helperDir := t.TempDir()
		helperSource := filepath.Join(helperDir, "main.go")
		helper := filepath.Join(helperDir, "workspace-child")
		if runtime.GOOS == "windows" {
			helper += ".exe"
		}
		const source = `package main
import (
	"fmt"
	"os"
)
func main() {
	wd, err := os.Getwd()
	if err != nil { panic(err) }
	if err := os.WriteFile(os.Getenv("FAK_OPS_WORKSPACE_MARKER"), []byte(wd), 0600); err != nil { panic(err) }
	fmt.Println("{\"type\":\"step_finish\",\"part\":{\"reason\":\"stop\"}}")
}
`
		if err := os.WriteFile(helperSource, []byte(source), 0o600); err != nil {
			t.Fatal(err)
		}
		build := exec.Command("go", "build", "-o", helper, helperSource)
		if out, err := build.CombinedOutput(); err != nil {
			t.Fatalf("build workspace child: %v\n%s", err, out)
		}

		gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			_, _ = io.WriteString(w, `data: {"id":"preflight","object":"chat.completion.chunk","model":"fixture","choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_preflight","type":"function","function":{"name":"fak_inference_preflight","arguments":"{\"ok\":true}"}}]},"finish_reason":"tool_calls"}]}`+"\n\n")
			_, _ = io.WriteString(w, "data: [DONE]\n\n")
		}))
		defer gateway.Close()

		marker := filepath.Join(t.TempDir(), "cwd.txt")
		prompt := filepath.Join(t.TempDir(), "prompt.txt")
		receipt := filepath.Join(t.TempDir(), "receipt.json")
		if err := os.WriteFile(prompt, []byte("check workspace\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		t.Setenv("FAK_OPS_WORKSPACE_MARKER", marker)
		t.Setenv(guardE2EHelperEnv, strings.Join([]string{"--provider", "openai", "--split", "off", "--model", "fixture", "--base-url", gateway.URL + "/v1", "--", helper}, " "))

		var stderr bytes.Buffer
		code := runOpsRun(io.Discard, &stderr, []string{
			"--workspace", workspaceAlias,
			"--prompt-file", prompt,
			"--receipt", receipt,
			"--model", "fixture",
			"--provider", "openai",
			"--base-url", gateway.URL + "/v1",
		})
		if code != 0 {
			t.Fatalf("ops run exit=%d, want 0: %s", code, stderr.String())
		}
		rawCWD, err := os.ReadFile(marker)
		if err != nil {
			t.Fatalf("read child cwd marker: %v", err)
		}
		gotCWD, err := filepath.Abs(string(rawCWD))
		if err != nil {
			t.Fatal(err)
		}
		wantCWD, err := filepath.Abs(workspace)
		if err != nil {
			t.Fatal(err)
		}
		if !strings.EqualFold(filepath.Clean(gotCWD), filepath.Clean(wantCWD)) {
			t.Fatalf("child cwd=%q, want isolated workspace %q (coordinator cwd=%q)", gotCWD, wantCWD, coordinator)
		}
		data, err := os.ReadFile(receipt)
		if err != nil {
			t.Fatal(err)
		}
		var got struct {
			Workspace string `json:"workspace"`
		}
		if err := json.Unmarshal(data, &got); err != nil {
			t.Fatal(err)
		}
		if !strings.EqualFold(filepath.Clean(got.Workspace), filepath.Clean(wantCWD)) {
			t.Fatalf("receipt workspace=%q, want %q", got.Workspace, wantCWD)
		}
	})
}

func TestOpsRunInferencePreflight(t *testing.T) {
	for _, tc := range []struct {
		name       string
		statusCode int
		malformed  bool
		timeout    bool
		wantLaunch bool
		invokes    int
	}{
		{name: "unauthorized_prevents_child_launch", statusCode: http.StatusUnauthorized, invokes: 1},
		{name: "rate_limit_prevents_child_launch", statusCode: http.StatusTooManyRequests, invokes: 1},
		{name: "unavailable_prevents_child_launch", statusCode: http.StatusServiceUnavailable, invokes: 1},
		{name: "malformed_stream_prevents_child_launch", statusCode: http.StatusOK, malformed: true, invokes: 1},
		{name: "timeout_prevents_child_launch", statusCode: http.StatusOK, timeout: true, invokes: 1},
		{name: "qualified_gateway_proceeds_and_reuses_probe", statusCode: http.StatusOK, wantLaunch: true, invokes: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var probes atomic.Int32
			gateway := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				probes.Add(1)
				if r.Method != http.MethodPost || r.URL.Path != "/v1/chat/completions" {
					t.Errorf("probe = %s %s, want POST /v1/chat/completions", r.Method, r.URL.Path)
				}
				var req struct {
					Model  string `json:"model"`
					Stream bool   `json:"stream"`
					Tools  []struct {
						Type     string `json:"type"`
						Function struct {
							Name string `json:"name"`
						} `json:"function"`
					} `json:"tools"`
				}
				if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
					t.Errorf("decode probe request: %v", err)
				}
				if req.Model != "fixture" || !req.Stream || len(req.Tools) != 1 || req.Tools[0].Type != "function" || req.Tools[0].Function.Name == "" {
					t.Errorf("probe request did not bind intended model + streamed tool-call witness: %+v", req)
				}
				if tc.statusCode != http.StatusOK {
					http.Error(w, "unavailable", tc.statusCode)
					return
				}
				if tc.timeout {
					time.Sleep(100 * time.Millisecond)
				}
				w.Header().Set("Content-Type", "text/event-stream")
				if tc.malformed {
					_, _ = io.WriteString(w, "data: {not-json}\n\n")
					return
				}
				chunk, err := json.Marshal(map[string]any{
					"id": "preflight", "object": "chat.completion.chunk", "model": "fixture",
					"choices": []any{map[string]any{
						"delta": map[string]any{"tool_calls": []any{map[string]any{
							"index": 0, "id": "call_preflight", "type": "function",
							"function": map[string]any{"name": req.Tools[0].Function.Name, "arguments": "{\"ok\":true}"},
						}}},
						"finish_reason": "tool_calls",
					}},
				})
				if err != nil {
					t.Fatal(err)
				}
				_, _ = io.WriteString(w, "data: "+string(chunk)+"\n\n")
				_, _ = io.WriteString(w, "data: [DONE]\n\n")
			}))
			defer gateway.Close()

			dir := t.TempDir()
			prompt := filepath.Join(dir, "prompt.txt")
			if err := os.WriteFile(prompt, []byte("private prompt\n"), 0600); err != nil {
				t.Fatal(err)
			}

			old := opsRunExecute
			t.Cleanup(func() { opsRunExecute = old })
			var launches atomic.Int32
			opsRunExecute = func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
				launches.Add(1)
				return 0, true, false, nil
			}

			var code int
			for i := 0; i < tc.invokes; i++ {
				receipt := filepath.Join(dir, fmt.Sprintf("receipt-%d.json", i))
				args := []string{"--prompt-file", prompt, "--receipt", receipt, "--model", "fixture", "--provider", "openai", "--base-url", gateway.URL + "/v1"}
				if tc.timeout {
					args = append(args, "--timeout", "20ms")
				}
				code = runOpsRun(io.Discard, io.Discard, args)

				data, err := os.ReadFile(receipt)
				if err != nil {
					t.Fatal(err)
				}
				var got struct {
					InferencePreflight struct {
						ReceiptRef string `json:"receipt_ref"`
					} `json:"inference_preflight"`
				}
				if err := json.Unmarshal(data, &got); err != nil {
					t.Fatal(err)
				}
				if got.InferencePreflight.ReceiptRef == "" {
					t.Fatalf("receipt missing inference_preflight.receipt_ref: %s", data)
				}
			}
			if probes.Load() != 1 {
				t.Fatalf("inference preflight probes = %d, want exactly 1 across %d unchanged invocation(s)", probes.Load(), tc.invokes)
			}
			wantLaunches := int32(0)
			if tc.wantLaunch {
				wantLaunches = int32(tc.invokes)
			}
			if launches.Load() != wantLaunches {
				t.Fatalf("child launches = %d, want %d (exit=%d)", launches.Load(), wantLaunches, code)
			}
			if tc.wantLaunch != (code == 0) {
				t.Fatalf("exit = %d, want success=%v", code, tc.wantLaunch)
			}
		})
	}

	for _, provider := range []string{"openai", "gemini"} {
		t.Run(provider+"_without_explicit_route_fails_closed", func(t *testing.T) {
			dir := t.TempDir()
			prompt := filepath.Join(dir, "prompt.txt")
			receipt := filepath.Join(dir, "receipt.json")
			if err := os.WriteFile(prompt, []byte("private prompt\n"), 0600); err != nil {
				t.Fatal(err)
			}
			if provider == "gemini" {
				t.Setenv("FAK_OPS_GEMINI_TEST_KEY", "fixture-only")
			}

			old := opsRunExecute
			t.Cleanup(func() { opsRunExecute = old })
			var launches atomic.Int32
			opsRunExecute = func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
				launches.Add(1)
				return 0, true, false, nil
			}

			args := []string{"--prompt-file", prompt, "--receipt", receipt, "--model", "fixture", "--provider", provider}
			if provider == "gemini" {
				args = append(args, "--api-key-env", "FAK_OPS_GEMINI_TEST_KEY")
			}
			if code := runOpsRun(io.Discard, io.Discard, args); code == 0 {
				t.Fatalf("%s without an explicit qualified route returned success", provider)
			}
			if launches.Load() != 0 {
				t.Fatalf("%s without an explicit qualified route bypassed preflight and launched child %d time(s)", provider, launches.Load())
			}
		})
	}
}

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
