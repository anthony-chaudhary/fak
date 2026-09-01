//go:build darwin

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
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
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

func TestGuardParentSurvivesDarwinRSSContainment(t *testing.T) {
	if os.Getenv("FAK_DARWIN_RESOURCE_RELAUNCH_HELPER") != "" {
		runDarwinGuardResourceRelaunchHelper(t)
		return
	}
	for _, tc := range []struct {
		name        string
		maxDuration string
	}{
		{name: "unsupervised"},
		{name: "supervised", maxDuration: "1m"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			runGuardParentSurvivesDarwinRSSContainment(t, tc.maxDuration)
		})
	}
}

func TestGuardParentSurvivesDarwinCodexRSSContainment(t *testing.T) {
	if os.Getenv("FAK_DARWIN_CODEX_RESOURCE_RELAUNCH_HELPER") != "" {
		runDarwinGuardCodexResourceRelaunchHelper(t)
		return
	}

	dir := t.TempDir()
	statePath := filepath.Join(dir, "child-state")
	observedPath := filepath.Join(dir, "observed.jsonl")
	resourceJournal := filepath.Join(dir, "child-resource.jsonl")
	auditPath := filepath.Join(dir, "audit.jsonl")
	codexPath := filepath.Join(dir, "codex")
	wrapper := `#!/bin/sh
unset FAK_GUARD_E2E_HELPER
exec "$FAK_DARWIN_CODEX_RESOURCE_TEST_BINARY" -test.run=^TestGuardParentSurvivesDarwinCodexRSSContainment$ -- "$@"
`
	if err := os.WriteFile(codexPath, []byte(wrapper), 0o700); err != nil {
		t.Fatalf("write Darwin Codex resource wrapper: %v", err)
	}

	const guardTrace = "guard-trace-9734"
	guardArgs := strings.Join([]string{
		"--quiet", "--provider", "openai", "--session-id", guardTrace,
		"--api-key-env", "FAK_GUARD_RESOURCE_WITNESS_KEY", "--audit", auditPath,
		"--child-max-memory-mb", "48", "--child-resource-poll", "100ms", "--child-resource-journal", resourceJournal,
		"--", codexPath, "original-prompt-must-not-replay",
	}, " ")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "guard")
	cmd.Env = append(os.Environ(),
		guardE2EHelperEnv+"="+guardArgs,
		"FAK_GUARD_RESOURCE_WITNESS_KEY=test-only",
		"FAK_DARWIN_CODEX_RESOURCE_RELAUNCH_HELPER=1",
		"FAK_DARWIN_CODEX_RESOURCE_STATE="+statePath,
		"FAK_DARWIN_CODEX_RESOURCE_OBSERVED="+observedPath,
		"FAK_DARWIN_CODEX_RESOURCE_TEST_BINARY="+os.Args[0],
		guardResourceRestartLimitEnv+"=1",
		guardResourceNoProgressLimitEnv+"=0",
		guardCrashRestartLimitEnv+"=0",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("Codex guard did not converge after RSS containment: %v\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}
	if ctx.Err() != nil {
		t.Fatalf("Codex guard RSS containment witness timed out: %v\nstderr:\n%s", ctx.Err(), stderr.String())
	}

	observations := readDarwinResourceRelaunchObservations(t, observedPath)
	if len(observations) != 2 {
		t.Fatalf("Codex child generations=%d, want 2: %+v", len(observations), observations)
	}
	first, second := observations[0], observations[1]
	if first.ParentPID <= 0 || first.ParentPID != second.ParentPID || first.GatewayURL == "" || first.GatewayURL != second.GatewayURL {
		t.Fatalf("guard parent/gateway did not survive: first=%+v second=%+v", first, second)
	}
	if first.ChildPID == second.ChildPID || second.HealthStatus != http.StatusOK {
		t.Fatalf("replacement child/gateway witness failed: first=%+v second=%+v", first, second)
	}
	const threadID = "0198f76a-67c2-7d11-a8f5-8f3d82149734"
	if len(second.Argv) < 2 {
		t.Fatalf("Codex replacement argv too short: %v", second.Argv)
	}
	if got := second.Argv[len(second.Argv)-2:]; got[0] != "resume" || got[1] != threadID {
		t.Fatalf("Codex replacement suffix=%v, want [resume %s]; argv=%v", got, threadID, second.Argv)
	}
	joined := strings.Join(second.Argv, " ")
	if strings.Count(joined, "resume") != 1 || strings.Contains(joined, "original-prompt-must-not-replay") || strings.Contains(joined, guardTrace+" resume") {
		t.Fatalf("Codex replacement was not one exact provider-thread resume: %v", second.Argv)
	}
	if got, want := guardCodexConfigOverrides(second.Argv), guardCodexConfigOverrides(first.Argv); strings.Join(got, "\x00") != strings.Join(want, "\x00") {
		t.Fatalf("Codex root config/wire overrides changed: first=%v second=%v", want, got)
	}

	receipts := readDarwinResourceReceipts(t, resourceJournal)
	if len(receipts) != 1 || receipts[0].RootPID != first.ChildPID || receipts[0].Reason != "CHILD_TREE_RSS_LIMIT" {
		t.Fatalf("Codex resource receipt=%+v, want one generation-1 RSS receipt", receipts)
	}
	rows, err := journal.ReadRows(auditPath)
	if err != nil {
		t.Fatalf("read durable Codex guard audit: %v", err)
	}
	hops := 0
	for _, row := range rows {
		if row.Kind != journal.KindRestartHop {
			continue
		}
		hops++
		if row.Restart == nil || row.Restart.Status != journal.RestartHopOK || row.Restart.Handback != guardRestartHandbackContinue {
			t.Fatalf("Codex resource restart hop was not engaged: %+v", row)
		}
	}
	if hops != 1 {
		t.Fatalf("Codex durable restart hops=%d, want exactly 1: %+v", hops, rows)
	}
	assertDarwinWitnessPIDsGone(t, 3*time.Second, first.ChildPID)
}

func TestGuardDarwinCodexRSSContainmentRefusesUnsafeBinding(t *testing.T) {
	if os.Getenv("FAK_DARWIN_CODEX_RESOURCE_RELAUNCH_HELPER") != "" {
		runDarwinGuardCodexResourceRelaunchHelper(t)
		return
	}
	for _, mode := range []string{"missing", "malformed"} {
		t.Run(mode, func(t *testing.T) {
			dir := t.TempDir()
			observedPath := filepath.Join(dir, "observed.jsonl")
			auditPath := filepath.Join(dir, "audit.jsonl")
			codexPath := filepath.Join(dir, "codex")
			wrapper := `#!/bin/sh
unset FAK_GUARD_E2E_HELPER
exec "$FAK_DARWIN_CODEX_RESOURCE_TEST_BINARY" -test.run=^TestGuardDarwinCodexRSSContainmentRefusesUnsafeBinding$ -- "$@"
`
			if err := os.WriteFile(codexPath, []byte(wrapper), 0o700); err != nil {
				t.Fatal(err)
			}
			guardArgs := strings.Join([]string{
				"--quiet", "--provider", "openai", "--session-id", "guard-trace-9734",
				"--api-key-env", "FAK_GUARD_RESOURCE_WITNESS_KEY", "--audit", auditPath,
				"--child-max-memory-mb", "48", "--child-resource-poll", "100ms", "--child-resource-journal", filepath.Join(dir, "child-resource.jsonl"),
				"--", codexPath, "original-prompt-must-not-replay",
			}, " ")
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			defer cancel()
			cmd := exec.CommandContext(ctx, os.Args[0], "guard")
			cmd.Env = append(os.Environ(),
				guardE2EHelperEnv+"="+guardArgs,
				"FAK_GUARD_RESOURCE_WITNESS_KEY=test-only",
				"FAK_DARWIN_CODEX_RESOURCE_RELAUNCH_HELPER=1",
				"FAK_DARWIN_CODEX_RESOURCE_BINDING_MODE="+mode,
				"FAK_DARWIN_CODEX_RESOURCE_STATE="+filepath.Join(dir, "child-state"),
				"FAK_DARWIN_CODEX_RESOURCE_OBSERVED="+observedPath,
				"FAK_DARWIN_CODEX_RESOURCE_TEST_BINARY="+os.Args[0],
				guardResourceRestartLimitEnv+"=1", guardResourceNoProgressLimitEnv+"=0", guardCrashRestartLimitEnv+"=0",
			)
			var stdout, stderr bytes.Buffer
			cmd.Stdout, cmd.Stderr = &stdout, &stderr
			err := cmd.Run()
			var exitErr *exec.ExitError
			if !errors.As(err, &exitErr) || exitErr.ExitCode() != 1 {
				t.Fatalf("unsafe %s binding guard exit=%v, want exit 1\nstdout:\n%s\nstderr:\n%s", mode, err, stdout.String(), stderr.String())
			}
			if !strings.Contains(stderr.String(), guardResourceReattachUnavailable) || !strings.Contains(stderr.String(), "fak guard -- codex resume") || !strings.Contains(stderr.String(), "refusing a cold relaunch") {
				t.Fatalf("unsafe %s binding lacks typed recovery guidance:\n%s", mode, stderr.String())
			}
			observations := readDarwinResourceRelaunchObservations(t, observedPath)
			if len(observations) != 1 {
				t.Fatalf("unsafe %s binding launched %d generations, want 1: %+v", mode, len(observations), observations)
			}
			rows, err := journal.ReadRows(auditPath)
			if err != nil {
				t.Fatal(err)
			}
			causes, hops := 0, 0
			for _, row := range rows {
				if row.Kind == journal.KindChildCrash && row.Reason == guardResourceReattachUnavailable {
					causes++
				}
				if row.Kind == journal.KindRestartHop {
					hops++
				}
			}
			if causes != 1 || hops != 0 {
				t.Fatalf("unsafe %s binding durable causes=%d hops=%d, want 1/0: %+v", mode, causes, hops, rows)
			}
		})
	}
}

func runGuardParentSurvivesDarwinRSSContainment(t *testing.T, maxDuration string) {
	t.Helper()
	dir := t.TempDir()
	statePath := filepath.Join(dir, "child-state")
	observedPath := filepath.Join(dir, "observed.jsonl")
	resourceJournal := filepath.Join(dir, "child-resource.jsonl")
	auditPath := filepath.Join(dir, "audit.jsonl")
	wrapperPath := filepath.Join(dir, "claude")
	wrapper := `#!/bin/sh
unset FAK_GUARD_E2E_HELPER
continue_seen=0
for arg in "$@"; do
  if [ "$arg" = "--continue" ]; then
    continue_seen=1
  fi
done
export FAK_DARWIN_RESOURCE_RELAUNCH_CONTINUE="$continue_seen"
exec "$FAK_DARWIN_RESOURCE_RELAUNCH_TEST_BINARY" -test.run=^TestGuardParentSurvivesDarwinRSSContainment$
`
	if err := os.WriteFile(wrapperPath, []byte(wrapper), 0o700); err != nil {
		t.Fatalf("write Darwin resource relaunch wrapper: %v", err)
	}

	guardArgv := []string{
		"--quiet", "--provider", "anthropic",
		"--api-key-env", "FAK_GUARD_RESOURCE_WITNESS_KEY",
		"--audit", auditPath,
		"--child-max-memory-mb", "48", "--child-resource-poll", "100ms", "--child-resource-journal", resourceJournal,
	}
	if maxDuration != "" {
		guardArgv = append(guardArgv, "--max-duration", maxDuration)
	}
	guardArgv = append(guardArgv, "--", wrapperPath)
	guardArgs := strings.Join(guardArgv, " ")
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, os.Args[0], "guard")
	cmd.Env = append(os.Environ(),
		guardE2EHelperEnv+"="+guardArgs,
		"FAK_GUARD_RESOURCE_WITNESS_KEY=test-only",
		"FAK_DARWIN_RESOURCE_RELAUNCH_HELPER=1",
		"FAK_DARWIN_RESOURCE_RELAUNCH_STATE="+statePath,
		"FAK_DARWIN_RESOURCE_RELAUNCH_OBSERVED="+observedPath,
		"FAK_DARWIN_RESOURCE_RELAUNCH_TEST_BINARY="+os.Args[0],
		guardResourceRestartLimitEnv+"=1",
		guardResourceNoProgressLimitEnv+"=0",
		guardCrashRestartLimitEnv+"=0",
	)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		t.Fatalf("guard did not converge after RSS containment: %v\nstdout:\n%s\nstderr:\n%s", err, stdout.String(), stderr.String())
	}
	if ctx.Err() != nil {
		t.Fatalf("guard RSS containment witness timed out: %v\nstderr:\n%s", ctx.Err(), stderr.String())
	}
	for _, want := range []string{"CHILD_TREE_RSS_LIMIT", "verified tree reap and receipt complete", "guard remains up", "resource restart 1/1"} {
		if !strings.Contains(stderr.String(), want) {
			t.Fatalf("guard containment report missing %q:\n%s", want, stderr.String())
		}
	}

	observations := readDarwinResourceRelaunchObservations(t, observedPath)
	if len(observations) != 2 {
		t.Fatalf("child generations=%d, want 2: %+v", len(observations), observations)
	}
	first, second := observations[0], observations[1]
	if first.Generation != 1 || second.Generation != 2 {
		t.Fatalf("generations=(%d,%d), want (1,2)", first.Generation, second.Generation)
	}
	if first.ParentPID <= 0 || first.ParentPID != second.ParentPID {
		t.Fatalf("guard parent PIDs=(%d,%d), want one stable non-zero PID", first.ParentPID, second.ParentPID)
	}
	if first.ChildPID == second.ChildPID || first.ChildPID == first.ParentPID || second.ChildPID == second.ParentPID {
		t.Fatalf("process generations were not isolated: first=%+v second=%+v", first, second)
	}
	if first.GatewayURL == "" || first.GatewayURL != second.GatewayURL {
		t.Fatalf("gateway URLs=(%q,%q), want one stable guard-owned URL", first.GatewayURL, second.GatewayURL)
	}
	if first.Continued || !second.Continued {
		t.Fatalf("reattach flags=(%v,%v), want false then true", first.Continued, second.Continued)
	}
	if second.HealthStatus != http.StatusOK {
		t.Fatalf("generation 2 gateway health=%d, want 200: %+v", second.HealthStatus, second)
	}

	receipts := readDarwinResourceReceipts(t, resourceJournal)
	if len(receipts) != 1 {
		t.Fatalf("resource receipts=%d, want exactly generation 1: %+v", len(receipts), receipts)
	}
	receipt := receipts[0]
	if receipt.RootPID != first.ChildPID || receipt.Reason != "CHILD_TREE_RSS_LIMIT" || receipt.MemoryMetric != string(procguard.MemoryMetricRSS) {
		t.Fatalf("generation 1 receipt does not identify RSS containment: %+v first=%+v", receipt, first)
	}
	if receipt.Action != "reap_tree" || receipt.DescendantsSurvive || receipt.TreeRSSBytes == nil || *receipt.TreeRSSBytes < receipt.ThresholdBytes {
		t.Fatalf("generation 1 receipt does not prove verified reap at threshold: %+v", receipt)
	}
	rows, err := journal.ReadRows(auditPath)
	if err != nil {
		t.Fatalf("read durable guard audit: %v", err)
	}
	var restart *journal.Row
	for i := range rows {
		if rows[i].Kind == journal.KindRestartHop {
			restart = &rows[i]
			break
		}
	}
	if restart == nil || restart.Restart == nil {
		t.Fatalf("guard audit has no resource restart hop: %+v", rows)
	}
	if restart.TraceID == "" || restart.Restart.FromTrace != restart.TraceID || restart.Restart.ToTrace != restart.TraceID || restart.Restart.Child != restart.TraceID {
		t.Fatalf("resource restart did not preserve one guard trace: %+v", restart)
	}
	if restart.Restart.Handback != guardRestartHandbackContinue || restart.Restart.Status != journal.RestartHopOK {
		t.Fatalf("resource restart did not record an engaged reattach: %+v", restart.Restart)
	}
	assertDarwinWitnessPIDsGone(t, 3*time.Second, first.ChildPID)
}

type darwinResourceRelaunchObservation struct {
	Generation   int      `json:"generation"`
	ParentPID    int      `json:"parent_pid"`
	ChildPID     int      `json:"child_pid"`
	GatewayURL   string   `json:"gateway_url"`
	Continued    bool     `json:"continued"`
	HealthStatus int      `json:"health_status,omitempty"`
	Argv         []string `json:"argv,omitempty"`
}

func runDarwinGuardCodexResourceRelaunchHelper(t *testing.T) {
	statePath := os.Getenv("FAK_DARWIN_CODEX_RESOURCE_STATE")
	generation := 1
	if data, err := os.ReadFile(statePath); err == nil {
		if prior, convErr := strconv.Atoi(strings.TrimSpace(string(data))); convErr == nil {
			generation = prior + 1
		}
	}
	if err := os.WriteFile(statePath, []byte(strconv.Itoa(generation)), 0o600); err != nil {
		t.Fatalf("write Codex resource generation: %v", err)
	}
	argv := guardTestArgsAfterDelimiter(os.Args)
	if generation == 1 {
		hookState := guardCodexHookStatePath(argv)
		if hookState == "" {
			t.Fatalf("fake Codex found no launch-scoped SessionStart state path in %v", argv)
		}
		switch os.Getenv("FAK_DARWIN_CODEX_RESOURCE_BINDING_MODE") {
		case "missing":
		case "malformed":
			if err := os.WriteFile(hookState, []byte(`{not-json`), 0o600); err != nil {
				t.Fatalf("fake malformed Codex SessionStart binding: %v", err)
			}
		default:
			if err := writeGuardCodexSessionBinding(hookState, "0198f76a-67c2-7d11-a8f5-8f3d82149734", "guard-trace-9734"); err != nil {
				t.Fatalf("fake Codex SessionStart binding: %v", err)
			}
		}
	}

	base := strings.TrimRight(os.Getenv("OPENAI_BASE_URL"), "/")
	if base == "" {
		base = guardCodexConfigValue(argv, "model_providers.fak.base_url")
	}
	base = strings.TrimSuffix(base, "/v1")
	row := darwinResourceRelaunchObservation{Generation: generation, ParentPID: os.Getppid(), ChildPID: os.Getpid(), GatewayURL: base, Argv: argv}
	if generation > 1 {
		client := http.Client{Timeout: 2 * time.Second}
		resp, err := client.Get(base + "/healthz")
		if err != nil {
			t.Fatalf("probe stable Codex guard gateway: %v", err)
		}
		row.HealthStatus = resp.StatusCode
		_ = resp.Body.Close()
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(os.Getenv("FAK_DARWIN_CODEX_RESOURCE_OBSERVED"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := fmt.Fprintln(f, string(encoded)); err != nil {
		_ = f.Close()
		t.Fatal(err)
	}
	if err := f.Close(); err != nil {
		t.Fatal(err)
	}
	if generation > 1 {
		return
	}
	memory := make([]byte, 96<<20)
	for i := 0; i < len(memory); i += 4096 {
		memory[i] = byte(i)
	}
	for {
		time.Sleep(100 * time.Millisecond)
		memory[0]++
	}
}

func guardTestArgsAfterDelimiter(argv []string) []string {
	for i, arg := range argv {
		if arg == "--" {
			return append([]string(nil), argv[i+1:]...)
		}
	}
	return nil
}

func guardCodexHookStatePath(argv []string) string {
	const marker = "'--state' '"
	for _, arg := range argv {
		start := strings.Index(arg, marker)
		if start < 0 {
			continue
		}
		rest := arg[start+len(marker):]
		if end := strings.IndexByte(rest, '\''); end >= 0 {
			return rest[:end]
		}
	}
	return ""
}

func guardCodexConfigOverrides(argv []string) []string {
	var out []string
	for i := 0; i < len(argv); i++ {
		if (argv[i] == "-c" || argv[i] == "--config") && i+1 < len(argv) {
			out = append(out, argv[i], argv[i+1])
			i++
		} else if strings.HasPrefix(argv[i], "--config=") {
			out = append(out, argv[i])
		}
	}
	return out
}

func guardCodexConfigValue(argv []string, key string) string {
	for i := 0; i+1 < len(argv); i++ {
		if argv[i] != "-c" && argv[i] != "--config" {
			continue
		}
		configKey, value, ok := strings.Cut(argv[i+1], "=")
		if ok && configKey == key {
			return strings.Trim(value, `"`)
		}
		i++
	}
	return ""
}

func runDarwinGuardResourceRelaunchHelper(t *testing.T) {
	statePath := os.Getenv("FAK_DARWIN_RESOURCE_RELAUNCH_STATE")
	generation := 1
	if data, err := os.ReadFile(statePath); err == nil {
		if prior, convErr := strconv.Atoi(strings.TrimSpace(string(data))); convErr == nil {
			generation = prior + 1
		}
	}
	if err := os.WriteFile(statePath, []byte(strconv.Itoa(generation)), 0o600); err != nil {
		t.Fatalf("write resource relaunch generation: %v", err)
	}

	base := strings.TrimRight(os.Getenv("ANTHROPIC_BASE_URL"), "/")
	row := darwinResourceRelaunchObservation{
		Generation: generation,
		ParentPID:  os.Getppid(),
		ChildPID:   os.Getpid(),
		GatewayURL: base,
		Continued:  os.Getenv("FAK_DARWIN_RESOURCE_RELAUNCH_CONTINUE") == "1",
	}
	if generation > 1 {
		client := http.Client{Timeout: 2 * time.Second}
		resp, err := client.Get(base + "/healthz")
		if err != nil {
			t.Fatalf("probe stable guard gateway: %v", err)
		}
		row.HealthStatus = resp.StatusCode
		_ = resp.Body.Close()
	}
	encoded, err := json.Marshal(row)
	if err != nil {
		t.Fatal(err)
	}
	f, err := os.OpenFile(os.Getenv("FAK_DARWIN_RESOURCE_RELAUNCH_OBSERVED"), os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatalf("open resource relaunch observations: %v", err)
	}
	if _, err = fmt.Fprintln(f, string(encoded)); err != nil {
		_ = f.Close()
		t.Fatalf("write resource relaunch observation: %v", err)
	}
	if err := f.Close(); err != nil {
		t.Fatalf("close resource relaunch observations: %v", err)
	}
	if generation > 1 {
		return
	}

	memory := make([]byte, 96<<20)
	for i := 0; i < len(memory); i += 4096 {
		memory[i] = byte(i)
	}
	for {
		time.Sleep(100 * time.Millisecond)
		memory[0]++
	}
}

func readDarwinResourceRelaunchObservations(t *testing.T, path string) []darwinResourceRelaunchObservation {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open resource relaunch observations: %v", err)
	}
	defer f.Close()
	var rows []darwinResourceRelaunchObservation
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		var row darwinResourceRelaunchObservation
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil {
			t.Fatalf("decode resource relaunch observation %q: %v", scan.Text(), err)
		}
		rows = append(rows, row)
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
	return rows
}

func readDarwinResourceReceipts(t *testing.T, path string) []guardResourceReceipt {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatalf("open resource receipts: %v", err)
	}
	defer f.Close()
	var rows []guardResourceReceipt
	scan := bufio.NewScanner(f)
	for scan.Scan() {
		var row guardResourceReceipt
		if err := json.Unmarshal(scan.Bytes(), &row); err != nil {
			t.Fatalf("decode resource receipt %q: %v", scan.Text(), err)
		}
		rows = append(rows, row)
	}
	if err := scan.Err(); err != nil {
		t.Fatal(err)
	}
	return rows
}

func TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts(t *testing.T) {
	if mode := os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_HELPER"); mode != "" {
		runDarwinGuardResourceWitnessHelper(t, mode)
		return
	}

	dir := t.TempDir()
	childPIDPath := filepath.Join(dir, "child.pid")
	grandchildPIDPath := filepath.Join(dir, "grandchild.pid")
	journal := filepath.Join(dir, "child-resource.jsonl")
	cmd := exec.CommandContext(context.Background(), os.Args[0], "-test.run=^TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts$")
	cmd.Env = append(os.Environ(),
		"FAK_DARWIN_RESOURCE_WITNESS_HELPER=root",
		"FAK_DARWIN_RESOURCE_WITNESS_CHILD_PID_PATH="+childPIDPath,
		"FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH="+grandchildPIDPath,
	)
	procguard.ConfigureProcessTreeCancel(cmd)
	if err := cmd.Start(); err != nil {
		t.Fatalf("start contained Darwin witness tree: %v", err)
	}
	wait := make(chan error, 1)
	go func() { wait <- cmd.Wait() }()
	stopped := false
	t.Cleanup(func() {
		if !stopped {
			_, _ = procguard.KillPID(cmd.Process.Pid)
			select {
			case <-wait:
			case <-time.After(5 * time.Second):
			}
		}
	})

	childPID := waitForDarwinWitnessPID(t, childPIDPath, 5*time.Second)
	grandchildPID := waitForDarwinWitnessPID(t, grandchildPIDPath, 5*time.Second)
	const threshold = uint64(48) << 20
	before, supported, detail := procguard.CollectMemorySnapshot(cmd.Process.Pid)
	if !supported || detail != "" {
		t.Fatalf("read fak-owned witness tree: supported=%v detail=%q snapshot=%+v", supported, detail, before)
	}
	if before.Metric != procguard.MemoryMetricRSS || before.TreeBytes <= threshold {
		t.Fatalf("bounded witness did not cross RSS threshold: %+v", before)
	}
	wantOwned := map[int]bool{cmd.Process.Pid: false, childPID: false, grandchildPID: false}
	for _, process := range before.Processes {
		if _, ok := wantOwned[process.PID]; ok {
			wantOwned[process.PID] = true
		}
	}
	for pid, found := range wantOwned {
		if !found {
			t.Fatalf("fak-owned PID %d missing from pre-intervention snapshot: %+v", pid, before.Processes)
		}
	}

	stop := make(chan struct{})
	started := time.Now()
	resource := startGuardChildResourceMonitor(cmd.Process.Pid, "trace-darwin-witness", "codex", guardResourcePolicy{
		PollInterval: 100 * time.Millisecond,
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: threshold,
		Stop:         stop,
	})
	ev := <-resource
	close(stop)
	if ev.Kind != guardChildResourceLimit || ev.Resource == nil || !ev.Resource.Stop || ev.Resource.Reason != "CHILD_TREE_RSS_LIMIT" {
		t.Fatalf("resource event=%+v", ev)
	}
	if elapsed := time.Since(started); elapsed > 3*time.Second {
		t.Fatalf("intervention took %v, want <=3s", elapsed)
	}

	_ = stopGuardChild(cmd, wait, 0)
	stopped = true
	oldConfig := guardResourceConfigured
	setGuardResourceConfig(guardResourceConfig{ReceiptPath: journal})
	t.Cleanup(func() { setGuardResourceConfig(oldConfig) })
	if err := guardWriteResourceReceipt(ev, "trace-darwin-witness", "codex", cmd.Process.Pid); err != nil {
		t.Fatalf("write Darwin resource receipt: %v", err)
	}
	assertDarwinWitnessPIDsGone(t, 3*time.Second, cmd.Process.Pid, childPID, grandchildPID)

	data, err := os.ReadFile(journal)
	if err != nil {
		t.Fatalf("read Darwin resource receipt: %v", err)
	}
	var receipt guardResourceReceipt
	if err := json.Unmarshal(data, &receipt); err != nil {
		t.Fatalf("decode Darwin resource receipt: %v\n%s", err, data)
	}
	if receipt.RootPID != cmd.Process.Pid || receipt.OffenderPID == 0 || receipt.ThresholdBytes != threshold || receipt.TreeRSSBytes == nil || *receipt.TreeRSSBytes <= receipt.ThresholdBytes {
		t.Fatalf("receipt lacks identity/threshold/RSS evidence: %+v", receipt)
	}
	if receipt.MemoryMetric != string(procguard.MemoryMetricRSS) || receipt.TreeCommitBytes != nil {
		t.Fatalf("Darwin RSS was mislabeled as commit: %+v", receipt)
	}
	if receipt.Action != "reap_tree" || receipt.DescendantsSurvive {
		t.Fatalf("receipt lacks successful reap readback: %+v", receipt)
	}
}

func TestGuardChildResourceMonitorDarwinVanishedRootDefersToChildWait(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	called := make(chan struct{}, 1)
	resource := startGuardChildResourceMonitorWithCollector(1234, "trace-darwin-exit", "codex", guardResourcePolicy{
		PollInterval: 10 * time.Millisecond,
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: 1,
		Stop:         stop,
	}, func(pid int) (procguard.MemorySnapshot, bool, string) {
		if pid != 1234 {
			t.Fatalf("collector pid=%d", pid)
		}
		called <- struct{}{}
		return procguard.MemorySnapshot{Metric: procguard.MemoryMetricRSS}, true, ""
	})

	select {
	case <-called:
	case <-time.After(time.Second):
		t.Fatal("resource collector was not polled")
	}
	select {
	case event := <-resource:
		t.Fatalf("vanished root emitted false resource event: %+v", event)
	case <-time.After(50 * time.Millisecond):
	}
}

func TestGuardChildResourceMonitorDarwinSurfacesCollectorError(t *testing.T) {
	stop := make(chan struct{})
	resource := startGuardChildResourceMonitor(-1, "trace-darwin-error", "codex", guardResourcePolicy{
		PollInterval: 100 * time.Millisecond,
		Metric:       procguard.MemoryMetricRSS,
		MaxTreeBytes: 1 << 30,
		Stop:         stop,
	})
	defer close(stop)
	select {
	case event := <-resource:
		if event.Resource == nil || event.Resource.Stop || event.Resource.Reason != "CHILD_RESOURCE_COLLECTOR_FAILURE" || event.Resource.Metric != procguard.MemoryMetricRSS {
			t.Fatalf("collector error was not a typed nonterminal diagnostic: %+v", event)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("Darwin collector error was silent")
	}
}

func runDarwinGuardResourceWitnessHelper(t *testing.T, mode string) {
	const allocation = 16 << 20
	memory := make([]byte, allocation)
	for i := 0; i < len(memory); i += 4096 {
		memory[i] = byte(i)
	}
	if mode == "root" {
		child := exec.Command(os.Args[0], "-test.run=^TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts$")
		child.Env = append(os.Environ(),
			"FAK_DARWIN_RESOURCE_WITNESS_HELPER=child",
			"FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH="+os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH"),
		)
		if err := child.Start(); err != nil {
			t.Fatalf("start Darwin child: %v", err)
		}
		if err := os.WriteFile(os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_CHILD_PID_PATH"), []byte(strconv.Itoa(child.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write Darwin child pid: %v", err)
		}
	} else if mode == "child" {
		grandchild := exec.Command(os.Args[0], "-test.run=^TestGuardChildResourceMonitorDarwinReapsOwnedTreeAndReceipts$")
		grandchild.Env = append(os.Environ(), "FAK_DARWIN_RESOURCE_WITNESS_HELPER=grandchild")
		if err := grandchild.Start(); err != nil {
			t.Fatalf("start Darwin grandchild: %v", err)
		}
		if err := os.WriteFile(os.Getenv("FAK_DARWIN_RESOURCE_WITNESS_GRANDCHILD_PID_PATH"), []byte(strconv.Itoa(grandchild.Process.Pid)), 0o600); err != nil {
			t.Fatalf("write Darwin grandchild pid: %v", err)
		}
	}
	for {
		time.Sleep(time.Second)
		memory[0]++
	}
}

func waitForDarwinWitnessPID(t *testing.T, path string, timeout time.Duration) int {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if data, err := os.ReadFile(path); err == nil {
			pid, err := strconv.Atoi(string(data))
			if err == nil && pid > 0 {
				return pid
			}
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("Darwin witness pid was not published within %v", timeout)
	return 0
}

func assertDarwinWitnessPIDsGone(t *testing.T, timeout time.Duration, pids ...int) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for {
		rows, detail := procguard.CollectRelations()
		if detail != "" {
			t.Fatalf("Darwin process readback: %s", detail)
		}
		alive := map[int]bool{}
		for _, row := range rows {
			alive[row.PID] = true
		}
		any := false
		for _, pid := range pids {
			any = any || alive[pid]
		}
		if !any {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("owned Darwin process IDs still alive after %v: %v", timeout, pids)
		}
		time.Sleep(25 * time.Millisecond)
	}
}
