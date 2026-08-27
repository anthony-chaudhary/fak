package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/journal"
)

// TestGuardMaybeRestartOnCrash pins the in-place crash-restart admission discipline (#4686)
// against REAL child exits (via runToExit, shared with guard_crash_classify_test.go) so the
// decision is exercised on genuine *exec.ExitError / *os.ProcessState values, not hand-forged ones.
func TestGuardMaybeRestartOnCrash(t *testing.T) {
	crashErr, crashState := runToExit(t, 2) // a plain non-zero exit (a panic-turned-code) — a real crash
	cleanErr, _ := runToExit(t, 0)          // the agent finished

	t.Run("crash within budget restarts", func(t *testing.T) {
		class, code, ok := guardMaybeRestartOnCrash(crashErr, crashState.ProcessState, 0, 3)
		if !ok || class != journal.CrashNonzeroExit || code != 2 {
			t.Fatalf("crash within budget -> class=%q code=%d ok=%v, want NONZERO_EXIT/2/true", class, code, ok)
		}
	})

	t.Run("clean exit never restarts", func(t *testing.T) {
		if _, _, ok := guardMaybeRestartOnCrash(cleanErr, nil, 0, 3); ok {
			t.Fatal("clean exit admitted for crash restart")
		}
	})

	t.Run("nil runErr never restarts", func(t *testing.T) {
		if _, _, ok := guardMaybeRestartOnCrash(nil, nil, 0, 3); ok {
			t.Fatal("nil runErr admitted for crash restart")
		}
	})

	t.Run("budget spent surfaces the crash", func(t *testing.T) {
		// restartsSoFar == limit: the bound is reached, so the crash is surfaced (master exits),
		// never masked by an unbounded relaunch.
		if _, _, ok := guardMaybeRestartOnCrash(crashErr, crashState.ProcessState, 3, 3); ok {
			t.Fatal("crash admitted after the budget was spent")
		}
		if _, _, ok := guardMaybeRestartOnCrash(crashErr, crashState.ProcessState, 4, 3); ok {
			t.Fatal("crash admitted past the budget")
		}
	})

	t.Run("explicit zero disables restart", func(t *testing.T) {
		for _, limit := range []int{0, -1} {
			if _, _, ok := guardMaybeRestartOnCrash(crashErr, crashState.ProcessState, 0, limit); ok {
				t.Fatalf("crash admitted with limit=%d (crash-restart must be off)", limit)
			}
		}
	})
}

func TestGuardCodexCLIUsageFailureClassification(t *testing.T) {
	exit2, state2 := runToExit(t, 2)
	exit17, state17 := runToExit(t, 17)
	observed := "error: unexpected argument '--full-auto' found\n\nUsage: codex exec [OPTIONS] [PROMPT]\n"

	if !guardIsCodexCLIUsageFailure(exit2, state2.ProcessState, "codex", observed) {
		t.Fatal("directly observed Codex exit-2 usage envelope was not classified")
	}
	for name, stderr := range map[string]string{
		"unrelated exit 2":     "panic: index out of range\n",
		"diagnostic only":      "error: unexpected argument '--full-auto' found\n",
		"usage only":           "Usage: codex exec [OPTIONS] [PROMPT]\n",
		"different usage root": "error: unexpected argument '--full-auto' found\nUsage: codex [OPTIONS]\n",
	} {
		t.Run(name, func(t *testing.T) {
			if guardIsCodexCLIUsageFailure(exit2, state2.ProcessState, "codex", stderr) {
				t.Fatalf("overgeneralized stderr %q as Codex CLI usage", stderr)
			}
		})
	}
	if guardIsCodexCLIUsageFailure(exit2, state2.ProcessState, "claude", observed) {
		t.Fatal("Codex-shaped stderr from a non-Codex harness was classified")
	}
	if guardIsCodexCLIUsageFailure(exit17, state17.ProcessState, "codex", observed) {
		t.Fatal("non-exit-2 Codex failure was classified as CLI usage")
	}
	if _, code, ok := guardMaybeRestartOnCrash(exit2, state2.ProcessState, 0, 3); !ok || code != 2 {
		t.Fatalf("unrelated exit 2 lost generic restart admission: code=%d ok=%v", code, ok)
	}
	if _, code, ok := guardMaybeRestartOnCrash(exit17, state17.ProcessState, 0, 3); !ok || code != 17 {
		t.Fatalf("transient crash lost generic restart admission: code=%d ok=%v", code, ok)
	}
}

func TestGuardChildStderrCapturePreservesAndBoundsStream(t *testing.T) {
	payload := bytes.Repeat([]byte("x"), guardChildStderrCaptureLimit+257)
	var original bytes.Buffer
	capture := newGuardChildStderrCapture(&original)
	if n, err := capture.Write(payload); err != nil || n != len(payload) {
		t.Fatalf("capture.Write = %d, %v; want %d, nil", n, err, len(payload))
	}
	if !bytes.Equal(original.Bytes(), payload) {
		t.Fatal("capture changed the original stderr stream")
	}
	if got := capture.String(); len(got) != guardChildStderrCaptureLimit || got != string(payload[:guardChildStderrCaptureLimit]) {
		t.Fatalf("captured prefix length=%d, want bounded unchanged prefix length=%d", len(got), guardChildStderrCaptureLimit)
	}
}

func TestGuardRefuseCodexCLIUsageWritesOneTypedWitness(t *testing.T) {
	exit2, state2 := runToExit(t, 2)
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = j.Close() }()
	var stderr bytes.Buffer
	if !guardRefuseCodexCLIUsage(exit2, state2.ProcessState, "codex", "trace-usage", "error: unexpected argument '--full-auto' found\nUsage: codex exec [OPTIONS] [PROMPT]\n", time.Now(), j, &stderr) {
		t.Fatal("recognized usage failure was not refused")
	}
	rows := j.Recent(4)
	if len(rows) != 1 || rows[0].Kind != "CHILD_CRASH" || rows[0].Reason != guardCodexCLIUsageReason || rows[0].ExitCode != 2 {
		t.Fatalf("typed usage witness = %+v", rows)
	}
	if got := stderr.String(); !strings.Contains(got, guardCodexCLIUsageReason) || strings.Contains(got, guardCrashRestartExhaustedReason) {
		t.Fatalf("typed final status = %q", got)
	}
}

// TestGuardCrashRestartLimit pins default-on child isolation and the explicit env override.
func TestGuardCrashRestartLimit(t *testing.T) {
	old, had := os.LookupEnv(guardCrashRestartLimitEnv)
	t.Cleanup(func() {
		if had {
			_ = os.Setenv(guardCrashRestartLimitEnv, old)
		} else {
			_ = os.Unsetenv(guardCrashRestartLimitEnv)
		}
	})
	_ = os.Unsetenv(guardCrashRestartLimitEnv)
	if got := guardCrashRestartLimit(); got != guardCrashRestartDefaultLimit {
		t.Fatalf("unset limit=%d, want default %d", got, guardCrashRestartDefaultLimit)
	}
	for _, c := range []struct {
		val  string
		want int
	}{
		{"", guardCrashRestartDefaultLimit}, {"0", 0}, {"-2", guardCrashRestartDefaultLimit},
		{"abc", guardCrashRestartDefaultLimit}, {"3", 3}, {" 5 ", 5},
	} {
		if err := os.Setenv(guardCrashRestartLimitEnv, c.val); err != nil {
			t.Fatal(err)
		}
		if got := guardCrashRestartLimit(); got != c.want {
			t.Fatalf("limit with %q=%d, want %d", c.val, got, c.want)
		}
	}
}

// TestGuardCrashRestartHopReattaches pins the lineage record: for a recognized agent the hop is a
// --continue reattach under the SAME trace (from==to==child==guardTraceID, handback=continue,
// status=ok), so a crash restart folds into the same restart chain as a budget restart / wire retry
// rather than being an invisible relaunch.
func TestGuardCrashRestartHopReattaches(t *testing.T) {
	hop := guardCrashRestartHop("guard-abc", "claude", 2)
	if hop.FromTrace != "guard-abc" || hop.ToTrace != "guard-abc" || hop.Child != "guard-abc" {
		t.Fatalf("crash hop trace lineage = from=%q to=%q child=%q, want all guard-abc", hop.FromTrace, hop.ToTrace, hop.Child)
	}
	if hop.Handback != guardRestartHandbackContinue || hop.Status != journal.RestartHopOK {
		t.Fatalf("crash hop for a recognized agent = handback=%q status=%q, want continue/ok", hop.Handback, hop.Status)
	}
	if hop.Hop != 2 || hop.Schema != journal.RestartChainSchema {
		t.Fatalf("crash hop = hop=%d schema=%q, want 2/%s", hop.Hop, hop.Schema, journal.RestartChainSchema)
	}
}

func TestGuardCrashRestartNoProgressReap(t *testing.T) {
	head, stalled, reap := guardCrashNoProgressStep("sha-a", "sha-a", 0, 2)
	if head != "sha-a" || stalled != 1 || reap {
		t.Fatalf("first stall = head %q count %d reap %v", head, stalled, reap)
	}
	head, stalled, reap = guardCrashNoProgressStep(head, "sha-a", stalled, 2)
	if stalled != 2 || !reap {
		t.Fatalf("second stall = count %d reap %v, want early reap", stalled, reap)
	}
	head, stalled, reap = guardCrashNoProgressStep(head, "sha-b", stalled, 2)
	if head != "sha-b" || stalled != 0 || reap {
		t.Fatalf("HEAD progress = head %q count %d reap %v, want reset", head, stalled, reap)
	}
}

func TestGuardCrashRestartGiveUpReason(t *testing.T) {
	line := guardCrashRestartGiveUpStatus(2, "trace-crash")
	for _, want := range []string{guardCrashRestartExhaustedReason, "2 consecutive", "trace-crash"} {
		if !strings.Contains(line, want) {
			t.Fatalf("give-up status %q missing %q", line, want)
		}
	}
	path := filepath.Join(t.TempDir(), "audit.jsonl")
	j, err := journal.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	// Close before the test returns: the journal holds an open handle on audit.jsonl, and on
	// Windows an open handle makes t.TempDir()'s RemoveAll cleanup fail the test.
	defer func() { _ = j.Close() }()
	guardRecordCrashRestartGiveUp(j, "claude", "trace-crash")
	rows := j.Recent(1)
	if len(rows) != 1 || rows[0].Kind != "CHILD_CRASH" || rows[0].Reason != guardCrashRestartExhaustedReason {
		t.Fatalf("give-up audit row = %+v", rows)
	}
}

func TestGuardCrashNoProgressLimit(t *testing.T) {
	t.Setenv(guardCrashNoProgressLimitEnv, "")
	if got := guardCrashNoProgressLimit(3); got != 2 {
		t.Fatalf("default limit = %d, want 2", got)
	}
	t.Setenv(guardCrashNoProgressLimitEnv, "1")
	if got := guardCrashNoProgressLimit(3); got != 1 {
		t.Fatalf("override limit = %d, want 1", got)
	}
	t.Setenv(guardCrashNoProgressLimitEnv, "0")
	if got := guardCrashNoProgressLimit(3); got != 0 {
		t.Fatalf("disabled limit = %d, want 0", got)
	}
}
func TestGuardCrashRestartBackoff(t *testing.T) {
	want := []time.Duration{0, 250 * time.Millisecond, 500 * time.Millisecond, time.Second, 2 * time.Second, 2 * time.Second}
	for attempt, expected := range want {
		if got := guardCrashRestartDelay(attempt); got != expected {
			t.Fatalf("attempt %d delay=%s, want %s", attempt, got, expected)
		}
	}
}

func TestGuardReportCrashRestartSaysParentStaysUp(t *testing.T) {
	var stderr bytes.Buffer
	guardReportCrashRestart(&stderr, "claude", journal.CrashNonzeroExit, 17, 1, 3, []string{"claude"})
	got := stderr.String()
	for _, want := range []string{
		"claude harness crashed",
		"NONZERO_EXIT",
		"exit 17",
		"guard remains up",
		"restarting the child in place",
		"crash restart 1/3",
		"claude --continue",
	} {
		if !strings.Contains(got, want) {
			t.Fatalf("restart signal %q missing %q", got, want)
		}
	}
}

// TestGuardCodexCLIUsageFailureStopsBeforeRestart is the supervision witness for #9346. The
// copied test binary is deliberately named codex so the real launch-plan classifier recognizes
// it, then its child mode emits the captured field envelope and exits 2. The launch counter and
// audit journal prove that classification happened before generic crash-restart admission.
func TestGuardCodexCLIUsageFailureStopsBeforeRestart(t *testing.T) {
	dir := t.TempDir()
	codexPath := guardNamedCodexTestBinary(t, dir)
	statePath := filepath.Join(dir, "child-state")
	auditPath := filepath.Join(dir, "audit.jsonl")

	cmd := exec.Command(os.Args[0], "guard")
	guardArgs := strings.Join([]string{
		"--quiet", "--provider", "openai",
		"--api-key-env", "FAK_GUARD_CRASH_WITNESS_KEY",
		"--audit", auditPath,
		"--", codexPath, "--fak-guard-crash-witness-child",
	}, " ")
	cmd.Env = append(os.Environ(),
		guardE2EHelperEnv+"="+guardArgs,
		"FAK_GUARD_CRASH_WITNESS_KEY=test-only",
		"FAK_GUARD_CRASH_WITNESS_STATE="+statePath,
		"FAK_GUARD_CRASH_WITNESS_MODE=codex-usage",
		"FAK_FLEET_BUS="+filepath.Join(dir, "fleet-bus"),
		guardCrashRestartLimitEnv+"=3",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	var exitErr *exec.ExitError
	if !errors.As(err, &exitErr) || exitErr.ExitCode() != 2 {
		t.Fatalf("guard exit=%v, want preserved child exit 2\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}
	gotStderr := stderr.String()
	for _, want := range []string{
		"error: unexpected argument '--full-auto' found",
		"Usage: codex exec [OPTIONS] [PROMPT]",
		guardCodexCLIUsageReason,
	} {
		if !strings.Contains(gotStderr, want) {
			t.Fatalf("stderr missing %q:\n%s", want, gotStderr)
		}
	}
	if strings.Contains(gotStderr, "restarting the child") || strings.Contains(gotStderr, guardCrashRestartExhaustedReason) {
		t.Fatalf("usage failure entered crash restart:\n%s", gotStderr)
	}
	state, err := os.ReadFile(statePath)
	if err != nil {
		t.Fatal(err)
	}
	if got := strings.TrimSpace(string(state)); got != "1" {
		t.Fatalf("child launch count=%s, want exactly one launch", got)
	}

	data, err := os.ReadFile(auditPath)
	if err != nil {
		t.Fatal(err)
	}
	var usageWitnesses, restartHops int
	scan := bufio.NewScanner(bytes.NewReader(data))
	for scan.Scan() {
		var row journal.Row
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil {
			t.Fatalf("decode audit row %q: %v", scan.Text(), err)
		}
		if row.Kind == "CHILD_CRASH" && row.Reason == guardCodexCLIUsageReason && row.ExitCode == 2 {
			usageWitnesses++
		}
		if row.Kind == "RESTART_HOP" {
			restartHops++
		}
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
	if usageWitnesses != 1 || restartHops != 0 {
		t.Fatalf("audit usage witnesses=%d restart hops=%d, want 1/0\n%s", usageWitnesses, restartHops, data)
	}
}

func guardNamedCodexTestBinary(t *testing.T, dir string) string {
	t.Helper()
	source, err := os.Executable()
	if err != nil {
		t.Fatal(err)
	}
	name := "codex"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	destination := filepath.Join(dir, name)
	if err := os.Link(source, destination); err == nil {
		return destination
	}
	in, err := os.Open(source)
	if err != nil {
		t.Fatal(err)
	}
	defer in.Close()
	out, err := os.OpenFile(destination, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o700)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := io.Copy(out, in); err != nil {
		_ = out.Close()
		t.Fatal(err)
	}
	if err := out.Close(); err != nil {
		t.Fatal(err)
	}
	return destination
}

// TestGuardParentSurvivesHarnessCrash is the behavior witness for #4686. It starts the REAL
// fak guard command as a parent process and gives it this test binary as a child harness. The
// first child invocation exits non-zero; the restarted invocation probes the guard-owned
// gateway through the injected ANTHROPIC_BASE_URL and exits cleanly. The observation file
// records the stable parent PID and gateway URL from both child generations, proving the
// harness is a separate child and its crash neither replaced nor tore down the guard.
func TestGuardParentSurvivesHarnessCrash(t *testing.T) {
	dir := t.TempDir()
	statePath := filepath.Join(dir, "child-state")
	observedPath := filepath.Join(dir, "observed.jsonl")
	auditPath := filepath.Join(dir, "audit.jsonl")

	// Keep a real verb-shaped argv for recordGuardUsage, while TestMain takes the guard
	// command from guardE2EHelperEnv. The production binary always has os.Args[1]="guard".
	cmd := exec.Command(os.Args[0], "guard")
	guardArgs := strings.Join([]string{
		"--quiet", "--provider", "anthropic",
		"--api-key-env", "FAK_GUARD_CRASH_WITNESS_KEY",
		"--audit", auditPath,
		"--", os.Args[0], "--fak-guard-crash-witness-child",
	}, " ")
	cmd.Env = append(os.Environ(),
		guardE2EHelperEnv+"="+guardArgs,
		"FAK_GUARD_CRASH_WITNESS_KEY=test-only",
		"FAK_GUARD_CRASH_WITNESS_STATE="+statePath,
		"FAK_GUARD_CRASH_WITNESS_OBSERVED="+observedPath,
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		t.Fatalf("guard process did not converge after child crash: %v\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}
	if got := stderr.String(); !strings.Contains(got, "guard remains up and is restarting the child in place") {
		t.Fatalf("guard did not report parent/child crash isolation:\n%s", got)
	}

	data, err := os.ReadFile(observedPath)
	if err != nil {
		t.Fatalf("read child observations: %v", err)
	}
	var observations []guardCrashWitnessObservation
	scan := bufio.NewScanner(bytes.NewReader(data))
	for scan.Scan() {
		var row guardCrashWitnessObservation
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil {
			t.Fatalf("decode observation %q: %v", scan.Text(), err)
		}
		observations = append(observations, row)
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
	if len(observations) != 2 {
		t.Fatalf("child generations=%d, want 2; rows=%s", len(observations), data)
	}
	first, second := observations[0], observations[1]
	if first.Generation != 1 || second.Generation != 2 {
		t.Fatalf("generations=(%d,%d), want (1,2)", first.Generation, second.Generation)
	}
	if first.ParentPID == 0 || first.ParentPID != second.ParentPID {
		t.Fatalf("guard parent PIDs=(%d,%d), want one stable non-zero PID", first.ParentPID, second.ParentPID)
	}
	if first.ChildPID == second.ChildPID || first.ChildPID == first.ParentPID || second.ChildPID == second.ParentPID {
		t.Fatalf("process isolation missing: first=%+v second=%+v", first, second)
	}
	if first.GatewayURL == "" || first.GatewayURL != second.GatewayURL {
		t.Fatalf("gateway URLs=(%q,%q), want one stable guard-owned URL", first.GatewayURL, second.GatewayURL)
	}
	if second.HealthStatus != http.StatusOK {
		t.Fatalf("restarted child reached gateway status=%d, want 200; second=%+v", second.HealthStatus, second)
	}
}

type guardCrashWitnessObservation struct {
	Generation   int    `json:"generation"`
	ParentPID    int    `json:"parent_pid"`
	ChildPID     int    `json:"child_pid"`
	GatewayURL   string `json:"gateway_url"`
	HealthStatus int    `json:"health_status,omitempty"`
}

func runGuardCrashWitnessChild() {
	statePath := os.Getenv("FAK_GUARD_CRASH_WITNESS_STATE")
	observedPath := os.Getenv("FAK_GUARD_CRASH_WITNESS_OBSERVED")
	mode := os.Getenv("FAK_GUARD_CRASH_WITNESS_MODE")
	generation := 1
	if data, err := os.ReadFile(statePath); err == nil {
		if prior, convErr := strconv.Atoi(strings.TrimSpace(string(data))); convErr == nil {
			generation = prior + 1
		}
	}
	if err := os.WriteFile(statePath, []byte(strconv.Itoa(generation)), 0o600); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(91)
	}
	if mode == "codex-usage" {
		fmt.Fprintln(os.Stderr, "error: unexpected argument '--full-auto' found")
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Usage: codex exec [OPTIONS] [PROMPT]")
		os.Exit(2)
	}

	base := strings.TrimRight(os.Getenv("ANTHROPIC_BASE_URL"), "/")
	row := guardCrashWitnessObservation{
		Generation: generation,
		ParentPID:  os.Getppid(),
		ChildPID:   os.Getpid(),
		GatewayURL: base,
	}
	if generation > 1 {
		client := http.Client{Timeout: 2 * time.Second}
		resp, err := client.Get(base + "/healthz")
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(92)
		}
		row.HealthStatus = resp.StatusCode
		_ = resp.Body.Close()
	}
	encoded, _ := json.Marshal(row)
	f, err := os.OpenFile(observedPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(93)
	}
	_, err = fmt.Fprintln(f, string(encoded))
	_ = f.Close()
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(94)
	}
	if generation == 1 {
		os.Exit(17)
	}
}
