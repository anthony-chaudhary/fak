package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
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

	t.Run("disabled by default never restarts", func(t *testing.T) {
		for _, limit := range []int{0, -1} {
			if _, _, ok := guardMaybeRestartOnCrash(crashErr, crashState.ProcessState, 0, limit); ok {
				t.Fatalf("crash admitted with limit=%d (crash-restart must be off)", limit)
			}
		}
	})
}

// TestGuardCrashRestartLimit pins the env knob: unset/0/garbage/negative all read as OFF, a positive
// integer is the budget. The default MUST be 0 so a crash tears the master down exactly as today
// until an operator opts in.
func TestGuardCrashRestartLimit(t *testing.T) {
	cases := []struct {
		set  bool
		val  string
		want int
	}{
		{false, "", 0},
		{true, "", 0},
		{true, "0", 0},
		{true, "-2", 0},
		{true, "abc", 0},
		{true, "3", 3},
		{true, " 5 ", 5},
	}
	for _, c := range cases {
		if c.set {
			t.Setenv("FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT", c.val)
		} else {
			// t.Setenv is the only way to guarantee cleanup; an unset case just sets empty above.
			t.Setenv("FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT", "")
		}
		if got := guardCrashRestartLimit(); got != c.want {
			t.Fatalf("guardCrashRestartLimit() with %q = %d, want %d", c.val, got, c.want)
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
		"FLEET_CLAUDE_GUARD_CRASH_RESTART_LIMIT=1",
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
