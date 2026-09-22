package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/processalive"
)

// ops_run_pi_test.go — the witness for the `fak ops run --harness pi` arm (#13380).
//
// What is proven at which seam:
//
//   - PURE units: opsPiClassify, opsPiComplete, opsPiProviderWire and opsPiChildArgv
//     (including the @file prompt delivery that keeps the prompt off argv).
//   - DISPATCHER: runOpsRun selects Pi once and hands the args to runOpsPi; a wrong or
//     unsupported selection is refused before any child is launched (no harness retry).
//   - BOUNDED CHILD: opsPiExecute is exercised against REAL child processes (this test
//     binary re-exec'd in helper mode) for the success, error-event, malformed-event,
//     timeout, cancellation and process-tree cases.
//   - RUN SPINE: runOpsPi is driven end to end (real gateway preflight via httptest, real
//     receipt write, real identity) with only the single `opsPiRunExecute` seam swapped so
//     no `fak guard`/pi binary is needed; the swap STILL runs a real fake-Pi child and the
//     real opsPiEvents classifier, and asserts the child consumed the injected gateway
//     configuration. The outer `fak guard ... -- pi ...` argv is asserted, not executed:
//     the guard injection itself is not proven here.
//
// Nothing reaches the network except the in-process httptest gateway, and every child wait
// is bounded.

// fakePiChildEnv turns this test binary into a stand-in for the `pi --mode json` child.
// The value selects the scripted turn; opsPiExecute is the only launcher, so the helper
// never sees a non-Pi argv it must parse.
const fakePiChildEnv = "FAK_OPS_PI_FAKE_CHILD"

// fakePiGatewayEnv carries the injected gateway configuration the fake Pi is expected to
// consume — the same origin the guard wrapper receives via --base-url.
const fakePiGatewayEnv = "FAK_OPS_PI_FAKE_GATEWAY"

// fakePiSpawnFileEnv names a file the helper writes the spawned grandchild PID into, so
// the process-tree test can assert the grandchild is dead after cancellation.
const fakePiSpawnFileEnv = "FAK_OPS_PI_FAKE_SPAWN_FILE"

func TestOpsRunPi(t *testing.T) {
	if mode := os.Getenv(fakePiChildEnv); mode != "" {
		runFakePiChild(mode)
		return // runFakePiChild exits; unreachable in the parent.
	}

	t.Run("classify", func(t *testing.T) {
		for _, line := range []string{`{"type":"error"}`, `{"type":"run_error","message":"boom"}`} {
			if got := opsPiClassify([]byte(line)); got != "error" {
				t.Errorf("opsPiClassify(%s)=%q want error", line, got)
			}
		}
		for _, line := range []string{`{"type":"assistant"}`, `{"type":"message"}`, `{"type":"result"}`, `{"type":"run_result"}`, `{"type":"finish"}`} {
			if got := opsPiClassify([]byte(line)); got != "assistant" {
				t.Errorf("opsPiClassify(%s)=%q want assistant", line, got)
			}
		}
		for _, line := range []string{"", "   ", "garbage", "not json", `{"type":"tool_call"}`, `[1,2,3]`} {
			if got := opsPiClassify([]byte(line)); got != "" {
				t.Errorf("opsPiClassify(%q)=%q want empty", line, got)
			}
		}
	})

	t.Run("complete", func(t *testing.T) {
		for _, tc := range []struct {
			name         string
			assistant    bool
			failed       bool
			wantComplete bool
		}{
			{"assistant_clean", true, false, true},
			{"assistant_then_error", true, true, false},
			{"error_only", false, true, false},
			{"silent", false, false, false},
		} {
			got := opsPiComplete(&opsPiEvents{assistant: tc.assistant, failed: tc.failed})
			if got != tc.wantComplete {
				t.Errorf("%s: opsPiComplete(assistant=%v failed=%v)=%v want %v", tc.name, tc.assistant, tc.failed, got, tc.wantComplete)
			}
		}
	})

	t.Run("provider_wire", func(t *testing.T) {
		if wire, ok := opsPiProviderWire("fak"); !ok || wire != "openai" {
			t.Errorf("fak => (%q,%v) want (openai,true)", wire, ok)
		}
		if wire, ok := opsPiProviderWire("anthropic"); !ok || wire != "anthropic" {
			t.Errorf("anthropic => (%q,%v) want (anthropic,true)", wire, ok)
		}
		if wire, ok := opsPiProviderWire("openai"); ok {
			t.Errorf("openai => (%q,%v) want refused", wire, ok)
		}
	})

	t.Run("child_argv", func(t *testing.T) {
		promptPath := filepath.Join(t.TempDir(), "task.txt")
		if err := os.WriteFile(promptPath, bytes.Repeat([]byte("task body\n"), 4096), 0o600); err != nil {
			t.Fatal(err)
		}
		body, err := os.ReadFile(promptPath)
		if err != nil {
			t.Fatal(err)
		}
		argv := opsPiChildArgv("pi", promptPath, "fak", "qwen3.8-27b", "high")
		joined := strings.Join(argv, " ")
		if !strings.Contains(joined, "@"+promptPath) {
			t.Fatalf("prompt not delivered as @file: %q", argv)
		}
		if strings.Contains(joined, string(body)) || strings.Contains(joined, "task body") {
			t.Fatalf("prompt leaked inline into argv: %q", argv)
		}
		if argv[0] != "pi" || argv[1] != "-p" || argv[2] != opsPiDirective {
			t.Fatalf("unexpected head argv: %q", argv)
		}
		for _, want := range []string{"--mode", "json", "--no-session", "--provider", "fak", "--model", "qwen3.8-27b", "--thinking", "high"} {
			if !strings.Contains(joined, want) {
				t.Errorf("argv missing %q: %q", want, argv)
			}
		}
		if len(opsPiDirective) > 200 {
			t.Errorf("inline directive too long for the Win32 argv ceiling: %d bytes", len(opsPiDirective))
		}
	})

	t.Run("dispatcher_selects_pi_once", func(t *testing.T) {
		dir := t.TempDir()
		prompt := writeOpsPiPrompt(t, dir)
		var stdout, stderr bytes.Buffer
		if got := runOpsRun(&stdout, &stderr, []string{"--harness", "pi", "--dry-run", "--model", "fixture-model", "--prompt-file", prompt, "--base-url", "http://127.0.0.1:1"}); got != 0 {
			t.Fatalf("runOpsRun --harness pi --dry-run exit=%d stderr=%s", got, stderr.String())
		}
		var plan map[string]any
		if err := json.Unmarshal(stdout.Bytes(), &plan); err != nil {
			t.Fatalf("decode plan: %v; out=%s", err, stdout.String())
		}
		if plan["schema"] != "fak-ops-run-plan/1" || plan["harness"] != "pi" || plan["guarded"] != true || plan["prompt_delivery"] != "file" {
			t.Fatalf("plan not the Pi plan: %v", plan)
		}
		if plan["wire"] != "openai" {
			t.Fatalf("plan wire=%v want openai", plan["wire"])
		}
		argv, _ := plan["argv"].([]any)
		parts := make([]string, 0, len(argv))
		for _, a := range argv {
			parts = append(parts, fmt.Sprint(a))
		}
		joined := strings.Join(parts, " ")
		if !strings.Contains(joined, "guard --provider openai") || !strings.Contains(joined, " -- pi -p ") || !strings.Contains(joined, "@") {
			t.Fatalf("plan argv is not the guarded Pi launch: %v", argv)
		}
	})

	t.Run("unsupported_selection_refused_without_retry", func(t *testing.T) {
		dir := t.TempDir()
		prompt := writeOpsPiPrompt(t, dir)
		oldPi, oldRun := opsPiRunExecute, opsRunExecute
		t.Cleanup(func() { opsPiRunExecute, opsRunExecute = oldPi, oldRun })
		opsPiRunExecute = func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
			t.Fatal("a refused Pi selection launched a child")
			return 0, true, false, nil
		}
		// The Pi arm must never fall through to the OpenCode executor (no harness retry).
		opsRunExecute = func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
			t.Fatal("Pi dispatch fell through to the OpenCode executor (harness retry)")
			return 0, true, false, nil
		}
		receipt := filepath.Join(dir, "refused.json")
		for _, args := range [][]string{
			{"--harness", "pi", "--provider", "openai", "--workspace", dir, "--prompt-file", prompt, "--receipt", receipt, "--model", "m", "--base-url", "http://127.0.0.1:1"},
			{"--harness", "pi", "--workspace", dir, "--prompt-file", prompt, "--receipt", prompt, "--model", "m", "--base-url", "http://127.0.0.1:1"},
			{"--harness", "pi", "--positional"},
		} {
			var stderr bytes.Buffer
			code := runOpsRun(io.Discard, &stderr, args)
			if code != 2 {
				t.Fatalf("args %q: exit=%d want 2 (refused); stderr=%s", args, code, stderr.String())
			}
		}
		// A missing base URL is refused before launch too (no silent fallback route).
		var stderr bytes.Buffer
		if code := runOpsPi(io.Discard, &stderr, []string{"--harness", "pi", "--workspace", dir, "--prompt-file", prompt, "--receipt", receipt, "--model", "m"}); code != 1 {
			t.Fatalf("missing --base-url exit=%d want 1; stderr=%s", code, stderr.String())
		}
	})

	t.Run("happy_path_writes_pi_receipt", func(t *testing.T) {
		dir := t.TempDir()
		gateway := newOpsRunQualifiedGateway(t)
		prompt := writeOpsPiPrompt(t, dir)
		receipt := filepath.Join(dir, "receipt.json")
		model := "pi-witness-model"

		old := opsPiRunExecute
		t.Cleanup(func() { opsPiRunExecute = old })
		var childSawGateway string
		opsPiRunExecute = func(ctx context.Context, stdout, stderr io.Writer, argv, env []string, promptBytes []byte) (int, bool, bool, []opsRunLifecycleRecord) {
			// The prompt must never appear on argv; it travels as @file.
			joined := strings.Join(argv, " ")
			if strings.Contains(joined, "opswitness") {
				t.Fatalf("prompt leaked into argv: %q", argv)
			}
			if !strings.Contains(joined, "@"+prompt) || !strings.Contains(joined, "--provider fak") {
				t.Fatalf("pi child argv not pinned: %q", argv)
			}
			// Inject the gateway configuration the guard wrapper received and run a REAL
			// fake-Pi child; classification uses the production opsPiEvents reader.
			childEnv := append(append([]string{}, env...),
				fakePiChildEnv+"=complete",
				fakePiGatewayEnv+"="+gateway.URL,
			)
			events := &opsPiEvents{output: stdout}
			// Tee the child stream so the test can prove the fake Pi actually consumed the
			// injected gateway configuration (the url it echoes back in its turn events).
			tap := &bytes.Buffer{}
			events.output = io.MultiWriter(stdout, tap)
			cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestOpsRunPi$")
			cmd.Env = childEnv
			cmd.Stdout, cmd.Stderr = events, stderr
			if err := cmd.Run(); err != nil {
				fmt.Fprintf(stderr, "fake pi: %v\n", err)
				return 1, false, false, nil
			}
			var turn struct {
				Gateway string `json:"gateway"`
			}
			for _, line := range strings.Split(tap.String(), "\n") {
				if json.Unmarshal([]byte(strings.TrimSpace(line)), &turn) == nil && turn.Gateway != "" {
					childSawGateway = turn.Gateway
				}
			}
			return 0, opsPiComplete(events), events.failed, nil
		}

		var stdout, stderr bytes.Buffer
		code := runOpsPi(&stdout, &stderr, []string{
			"--harness", "pi",
			"--provider", "fak",
			"--model", model,
			"--prompt-file", prompt,
			"--workspace", dir,
			"--receipt", receipt,
			"--base-url", gateway.URL + "/v1",
			"--pi-bin", os.Args[0],
			"--timeout", "30s",
		})
		if code != 0 {
			t.Fatalf("runOpsPi exit=%d want 0; stderr=%s", code, stderr.String())
		}
		data, err := os.ReadFile(receipt)
		if err != nil {
			t.Fatal(err)
		}
		var r opsRunReceipt
		if err := json.Unmarshal(data, &r); err != nil {
			t.Fatalf("decode receipt: %v; raw=%s", err, data)
		}
		if r.Schema != "fak-ops-run/1" || r.Harness != "pi" || r.Status != "succeeded" || r.ExitCode != 0 || r.Finished.IsZero() {
			t.Fatalf("receipt not a successful Pi run: %s", data)
		}
		if r.InferencePreflight == nil || r.InferencePreflight.Status != "qualified" {
			t.Fatalf("inference preflight not qualified: %s", data)
		}
		if r.LaunchIdentity == nil || r.LaunchIdentity.Harness != "pi" || !strings.HasPrefix(r.LaunchIdentity.RunID, "ops-") {
			t.Fatalf("launch identity missing: %s", data)
		}
		// Source-session identity: the run is identified by its own run id, and the route
		// digest binds provider + gateway + model for that run.
		if r.LaunchIdentity.ModelDigest == "" || r.LaunchIdentity.RouteDigest == "" || r.LaunchIdentity.ModelDigest != opsRunDigest(model) {
			t.Fatalf("route/model identity not bound: %s", data)
		}
		if r.LaunchIdentity.InferenceProbeRef != r.InferencePreflight.ReceiptRef {
			t.Fatalf("receipt ref not persisted into identity: %s", data)
		}
		if strings.Contains(string(data), "opswitness") {
			t.Fatalf("prompt text leaked into receipt: %s", data)
		}
		if childSawGateway != gateway.URL {
			t.Fatalf("fake Pi did not consume the injected gateway configuration: got %q want %q", childSawGateway, gateway.URL)
		}
	})

	t.Run("malformed_or_error_event_is_non_success", func(t *testing.T) {
		dir := t.TempDir()
		gateway := newOpsRunQualifiedGateway(t)
		prompt := writeOpsPiPrompt(t, dir)
		if err := os.WriteFile(prompt, []byte("opswitness task\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		for _, tc := range []struct {
			name      string
			mode      string
			wantExit  int
			wantState string
		}{
			{"assistant_clean", "complete", 0, "succeeded"},
			{"error_event", "error", 1, "failed"},
			{"malformed_event", "garbage", 1, "failed"},
			{"side_effect_then_failure", "spawn_then_fail", 3, "failed"},
		} {
			t.Run(tc.name, func(t *testing.T) {
				old := opsPiRunExecute
				spawnFile := filepath.Join(t.TempDir(), "side-effect.pid")
				t.Cleanup(func() {
					opsPiRunExecute = old
					reapFakePiPID(spawnFile)
				})
				opsPiRunExecute = fakePiExecute(t, tc.mode, spawnFile)
				receipt := filepath.Join(dir, tc.name+".json")
				var stdout, stderr bytes.Buffer
				code := runOpsPi(&stdout, &stderr, []string{
					"--harness", "pi", "--model", "pi-witness-model-" + tc.name,
					"--prompt-file", prompt, "--workspace", dir, "--receipt", receipt,
					"--base-url", gateway.URL + "/v1", "--pi-bin", os.Args[0], "--timeout", "30s",
				})
				if code != tc.wantExit {
					t.Fatalf("%s exit=%d want %d; stderr=%s", tc.name, code, tc.wantExit, stderr.String())
				}
				var r opsRunReceipt
				raw, err := os.ReadFile(receipt)
				if err != nil {
					t.Fatal(err)
				}
				if err := json.Unmarshal(raw, &r); err != nil {
					t.Fatal(err)
				}
				if r.Status != tc.wantState {
					t.Fatalf("%s receipt status=%q want %q: %s", tc.name, r.Status, tc.wantState, raw)
				}
			})
		}
	})

	t.Run("bound_child_timeout_and_cancellation", func(t *testing.T) {
		exe, err := os.Executable()
		if err != nil {
			t.Fatal(err)
		}
		workspace := t.TempDir()

		t.Run("timeout_is_non_success", func(t *testing.T) {
			// Direct seam: the lifecycle records a timeout reason and the child does not
			// report a clean exit. (An assistant event may already have arrived; a
			// completed-looking turn that blows the deadline is still non-success.)
			ctx, cancel := context.WithTimeout(withOpsRunWorkspace(context.Background(), workspace), 2*time.Second)
			defer cancel()
			start := time.Now()
			code, _, _, lifecycle := opsPiExecute(ctx, io.Discard, io.Discard, []string{exe, "-test.run=^TestOpsRunPi$"}, fakePiChildEnvN(t, "sleep", ""), nil)
			if time.Since(start) > 30*time.Second {
				t.Fatal("timeout wait was not bounded")
			}
			if code == 0 {
				t.Fatalf("timed-out child reported exit 0: lifecycle=%v", lifecycle)
			}
			if len(lifecycle) == 0 || lifecycle[0].Reason != "timeout" {
				t.Fatalf("timeout lifecycle=%v want reason=timeout", lifecycle)
			}

			// Run level: runOpsPi must normalize the deadline to 124 / timed_out, not
			// succeed because the assistant event arrived before the deadline fired.
			dir := t.TempDir()
			gateway := newOpsRunQualifiedGateway(t)
			prompt := writeOpsPiPrompt(t, dir)
			receipt := filepath.Join(dir, "timeout.json")
			old := opsPiRunExecute
			t.Cleanup(func() { opsPiRunExecute = old })
			opsPiRunExecute = mockPiExecuteFromEnv(t, "sleep")
			var stderr bytes.Buffer
			if got := runOpsPi(io.Discard, &stderr, []string{
				"--harness", "pi", "--model", "pi-witness-timeout",
				"--prompt-file", prompt, "--workspace", dir, "--receipt", receipt,
				"--base-url", gateway.URL + "/v1", "--pi-bin", os.Args[0], "--timeout", "2s",
			}); got != 124 {
				t.Fatalf("timed-out run exit=%d want 124; stderr=%s", got, stderr.String())
			}
			var r opsRunReceipt
			raw, err := os.ReadFile(receipt)
			if err != nil {
				t.Fatal(err)
			}
			if err := json.Unmarshal(raw, &r); err != nil {
				t.Fatal(err)
			}
			if r.Status != "timed_out" || r.ExitCode != 124 {
				t.Fatalf("timeout receipt status=%q exit=%d want timed_out/124: %s", r.Status, r.ExitCode, raw)
			}
		})

		t.Run("process_tree_dies_on_cancel", func(t *testing.T) {
			spawnFile := filepath.Join(t.TempDir(), "grandchild.pid")
			ctx, cancel := context.WithCancel(withOpsRunWorkspace(context.Background(), workspace))
			env := fakePiChildEnvN(t, "spawn_grandchild", spawnFile)
			done := make(chan int, 1)
			go func() {
				code, _, _, _ := opsPiExecute(ctx, io.Discard, io.Discard, []string{exe, "-test.run=^TestOpsRunPi$"}, env, nil)
				done <- code
			}()
			pid := waitFakePiPID(t, spawnFile)
			if pid <= 0 || !processalive.Check(pid) {
				t.Fatalf("grandchild pid %d never became alive", pid)
			}
			cancel()
			select {
			case code := <-done:
				if code == 0 {
					t.Fatal("cancelled child returned success")
				}
			case <-time.After(20 * time.Second):
				t.Fatal("child survived cancellation")
			}
			deadline := time.Now().Add(10 * time.Second)
			for processalive.Check(pid) && time.Now().Before(deadline) {
				time.Sleep(100 * time.Millisecond)
			}
			if processalive.Check(pid) {
				t.Fatalf("grandchild %d survived cancellation (process tree not reaped)", pid)
			}
		})
	})
}

// fakePiChildEnvN builds the helper environment for a scripted fake-Pi child. It is used
// by the direct opsPiExecute cases, which do not carry a gateway env.
func fakePiChildEnvN(t *testing.T, mode, spawnFile string) []string {
	t.Helper()
	env := append(os.Environ(), fakePiChildEnv+"="+mode)
	if spawnFile != "" {
		env = append(env, fakePiSpawnFileEnv+"="+spawnFile)
	}
	return env
}

// fakePiExecute returns an opsPiRunExecute stand-in that launches the fake Pi helper as a
// REAL child through the production opsPiEvents reader. The mode selects the scripted turn;
// spawnFile, when nonempty, is where a spawning mode records its grandchild PID for reaping.
func fakePiExecute(t *testing.T, mode, spawnFile string) func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
	t.Helper()
	return func(ctx context.Context, stdout, stderr io.Writer, argv, env []string, promptBytes []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		events := &opsPiEvents{output: stdout}
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestOpsRunPi$")
		cmd.Env = append(append([]string{}, env...), fakePiChildEnv+"="+mode)
		if spawnFile != "" {
			cmd.Env = append(cmd.Env, fakePiSpawnFileEnv+"="+spawnFile)
		}
		cmd.Stdout, cmd.Stderr = events, stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(stderr, "fake pi: %v\n", err)
			if exitErr, ok := err.(*exec.ExitError); ok {
				return exitErr.ExitCode(), opsPiComplete(events), events.failed, nil
			}
			return 1, opsPiComplete(events), events.failed, nil
		}
		return 0, opsPiComplete(events), events.failed, nil
	}
}

// runFakePiChild is the fake-Pi helper entry point. It emits the newline-delimited JSON a
// `pi --mode json` run would emit for the scripted turn, then exits.
func runFakePiChild(mode string) {
	fail := func(format string, args ...any) {
		fmt.Fprintf(os.Stderr, "fake pi: "+format+"\n", args...)
		os.Exit(9)
	}
	gateway := os.Getenv(fakePiGatewayEnv)
	emit := func(v map[string]any) {
		data, err := json.Marshal(v)
		if err != nil {
			fail("marshal event: %v", err)
		}
		fmt.Println(string(data))
	}
	switch mode {
	case "complete":
		// Consume the injected gateway configuration: the emitted turn records the origin
		// the fake Pi was pointed at, so the test can prove it observed the route.
		emit(map[string]any{"type": "assistant", "text": "working", "gateway": gateway})
		emit(map[string]any{"type": "result", "text": "done", "gateway": gateway})
		os.Exit(0)
	case "error":
		emit(map[string]any{"type": "assistant", "text": "starting"})
		emit(map[string]any{"type": "error", "message": "provider refused"})
		os.Exit(0)
	case "garbage":
		fmt.Println("pi: not-json banner line")
		fmt.Println("{\"type\":")
		os.Exit(0)
	case "sleep":
		fmt.Println(`{"type":"assistant","text":"waiting"}`)
		time.Sleep(10 * time.Minute)
		os.Exit(0)
	case "spawn_grandchild":
		spawnGrandchild(mode)
		time.Sleep(10 * time.Minute)
		os.Exit(0)
	case "spawn_then_fail":
		// The side effect lands (a real grandchild is spawned and its pid recorded), then
		// the child exits non-zero. The grandchild is deliberately short-lived so a
		// non-success turn cannot leak a long-lived process into CI.
		spawnGrandchild(mode)
		os.Exit(3)
	case "grandchild":
		time.Sleep(20 * time.Second)
		os.Exit(0)
	default:
		fail("unknown mode %q", mode)
	}
}

// spawnGrandchild forks a real grandchild (this binary in `grandchild` mode) and records
// its PID. The grandchild's stdio is deliberately detached (devnull): an inherited stdout
// pipe would keep the parent's cmd.Run() blocked until the grandchild exited, which is the
// hang this witness must never introduce.
func spawnGrandchild(mode string) {
	exe, err := os.Executable()
	if err != nil {
		fmt.Fprintf(os.Stderr, "fake pi: executable: %v\n", err)
		os.Exit(9)
	}
	cmd := exec.Command(exe, "-test.run=^TestOpsRunPi$")
	cmd.Env = append(os.Environ(), fakePiChildEnv+"=grandchild")
	cmd.Stdin, cmd.Stdout, cmd.Stderr = nil, nil, nil
	if err := cmd.Start(); err != nil {
		fmt.Fprintf(os.Stderr, "fake pi: start grandchild: %v\n", err)
		os.Exit(9)
	}
	if spawnFile := os.Getenv(fakePiSpawnFileEnv); spawnFile != "" {
		_ = os.WriteFile(spawnFile, []byte(strconv.Itoa(cmd.Process.Pid)), 0o600)
	}
	fmt.Println(`{"type":"assistant","text":"spawned a side effect"}`)
	_ = mode
}

// mockPiExecuteFromEnv is a minimal opsPiRunExecute stand-in for the run-level deadline
// test: it launches the fake-Pi helper under the caller's context and lets the context kill
// it, so runOpsPi's own timeout normalization is what is being observed.
func mockPiExecuteFromEnv(t *testing.T, mode string) func(context.Context, io.Writer, io.Writer, []string, []string, []byte) (int, bool, bool, []opsRunLifecycleRecord) {
	t.Helper()
	return func(ctx context.Context, stdout, stderr io.Writer, argv, env []string, promptBytes []byte) (int, bool, bool, []opsRunLifecycleRecord) {
		cmd := exec.CommandContext(ctx, os.Args[0], "-test.run=^TestOpsRunPi$")
		cmd.Env = append(append([]string{}, env...), fakePiChildEnv+"="+mode)
		cmd.Stdout, cmd.Stderr = stdout, stderr
		if err := cmd.Run(); err != nil {
			if exitErr, ok := err.(*exec.ExitError); ok {
				return exitErr.ExitCode(), false, false, nil
			}
			return 1, false, false, nil
		}
		return 0, true, false, nil
	}
}

// reapFakePiPID kills a grandchild a spawning fake-Pi mode recorded, so a non-success turn
// cannot leak a live process out of the test.
func reapFakePiPID(spawnFile string) {
	data, err := os.ReadFile(spawnFile)
	if err != nil {
		return
	}
	pid, err := strconv.Atoi(strings.TrimSpace(string(data)))
	if err != nil || pid <= 0 {
		return
	}
	if proc, err := os.FindProcess(pid); err == nil {
		_ = proc.Kill()
	}
	deadline := time.Now().Add(5 * time.Second)
	for processalive.Check(pid) && time.Now().Before(deadline) {
		time.Sleep(50 * time.Millisecond)
	}
}

// waitFakePiPID polls the helper-written PID file under a bounded deadline.
func waitFakePiPID(t *testing.T, path string) int {
	t.Helper()
	deadline := time.Now().Add(20 * time.Second)
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil {
			if pid, err := strconv.Atoi(strings.TrimSpace(string(data))); err == nil && pid > 0 {
				return pid
			}
		}
		time.Sleep(100 * time.Millisecond)
	}
	t.Fatal("fake pi grandchild never reported its pid")
	return 0
}

func writeOpsPiPrompt(t *testing.T, dir string) string {
	t.Helper()
	path := filepath.Join(dir, "task.txt")
	if err := os.WriteFile(path, []byte("opswitness: complete one deterministic turn\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return path
}
