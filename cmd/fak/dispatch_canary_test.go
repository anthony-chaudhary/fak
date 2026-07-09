package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// healthyCanaryRecord / unhealthyCanaryRecord synthesize the route-health record the probe
// seam returns, so the canary's command/environment shape and its refusal gate are exercised
// without a live provider key.
func healthyCanaryRecord(spec routeProbeSpec, now int64) routeHealthRecord {
	return routeHealthRecord{
		Schema:       routeHealthSchema,
		Route:        routeHealthKey(spec.Provider, spec.Account, spec.Model),
		Provider:     spec.Provider,
		Account:      spec.Account,
		Model:        spec.Model,
		BaseURL:      spec.BaseURL,
		Class:        string(routeClassHealthy),
		Status:       200,
		ProbedAtUnix: now,
		Recheck:      routeHealthRecheckCommand(spec),
	}
}

func unhealthyCanaryRecord(spec routeProbeSpec, now int64) routeHealthRecord {
	return routeHealthRecord{
		Schema:            routeHealthSchema,
		Route:             routeHealthKey(spec.Provider, spec.Account, spec.Model),
		Provider:          spec.Provider,
		Account:           spec.Account,
		Model:             spec.Model,
		BaseURL:           spec.BaseURL,
		Class:             string(routeClassRateLimited),
		Status:            429,
		Detail:            "HTTP 429: quota or rate limit",
		ProbedAtUnix:      now,
		CooldownUntilUnix: now + 900,
		Recheck:           routeHealthRecheckCommand(spec),
	}
}

// installCanarySeams swaps the probe and launch seams for the duration of a test and returns a
// pointer that captures the plan the launch seam last received (nil if it was never called).
func installCanarySeams(t *testing.T, rec func(routeProbeSpec, int64) routeHealthRecord, now int64, launchCode int) (captured **canaryPlan) {
	t.Helper()
	origProbe, origLaunch := canaryRouteProbe, canaryLaunch
	var got *canaryPlan
	canaryRouteProbe = func(spec routeProbeSpec) (routeHealthRecord, error) {
		return rec(spec, now), nil
	}
	canaryLaunch = func(_, _ io.Writer, plan canaryPlan) int {
		c := plan
		got = &c
		return launchCode
	}
	t.Cleanup(func() { canaryRouteProbe = origProbe; canaryLaunch = origLaunch })
	return &got
}

func TestBuildCanaryPlanProofOnlyShape(t *testing.T) {
	plan := buildCanaryPlan(canaryParams{
		Workspace: "/repo",
		Provider:  "deepseek",
		Model:     "deepseek-chat",
		BaseURL:   "https://api.example.com/v1",
		APIKeyEnv: "MY_KEY",
		Now:       1000,
		FakBin:    "/usr/bin/fak",
	})
	if plan.Mode != canaryModeProofOnly {
		t.Fatalf("mode = %q, want %q", plan.Mode, canaryModeProofOnly)
	}
	if !strings.HasPrefix(plan.RunID, "canary-") || strings.Contains(plan.RunID, "issue") {
		t.Errorf("proof-only run-id = %q, want canary-<route>-<now> without issue", plan.RunID)
	}
	// The command is the guard-wrapped Claude Code harness in headless print mode, and the route
	// key reaches guard via --api-key-env.
	argv := strings.Join(plan.Argv, " ")
	for _, want := range []string{"guard", "--api-key-env MY_KEY", "-- claude", "--dangerously-skip-permissions", "--model deepseek-chat", "-p"} {
		if !strings.Contains(argv, want) {
			t.Errorf("argv missing %q; got: %s", want, argv)
		}
	}
	// The route/metadata env is canary-namespaced and carries no issue in proof-only mode.
	env := strings.Join(plan.Env, "\n")
	for _, want := range []string{"FAK_CANARY_MODE=proof-only", "FAK_CANARY_MODEL=deepseek-chat", "FAK_CANARY_BASE_URL=https://api.example.com/v1", "FAK_CANARY_API_KEY_ENV=MY_KEY"} {
		if !strings.Contains(env, want) {
			t.Errorf("env missing %q; got:\n%s", want, env)
		}
	}
	if strings.Contains(env, "FAK_CANARY_ISSUE") {
		t.Errorf("proof-only env must not carry FAK_CANARY_ISSUE; got:\n%s", env)
	}
	if !strings.Contains(plan.Prompt, "CANARY-OK") {
		t.Errorf("proof-only prompt should ask for the CANARY-OK proof; got: %q", plan.Prompt)
	}
}

func TestBuildCanaryPlanSingleIssueShape(t *testing.T) {
	plan := buildCanaryPlan(canaryParams{
		Workspace: "/repo",
		Provider:  "deepseek",
		Model:     "deepseek-chat",
		BaseURL:   "https://api.example.com/v1",
		Issue:     3036,
		Now:       2000,
		FakBin:    "/usr/bin/fak",
	})
	if plan.Mode != canaryModeSingleIssue {
		t.Fatalf("mode = %q, want %q", plan.Mode, canaryModeSingleIssue)
	}
	if !strings.HasPrefix(plan.RunID, "canary-issue3036-") {
		t.Errorf("single-issue run-id = %q, want canary-issue3036-...", plan.RunID)
	}
	env := strings.Join(plan.Env, "\n")
	if !strings.Contains(env, "FAK_CANARY_ISSUE=3036") {
		t.Errorf("single-issue env must carry FAK_CANARY_ISSUE=3036; got:\n%s", env)
	}
	if !strings.Contains(plan.Prompt, "#3036") {
		t.Errorf("single-issue prompt should point at issue #3036; got: %q", plan.Prompt)
	}
	// --proof-only forces proof-only even with an issue set.
	forced := buildCanaryPlan(canaryParams{Provider: "p", Model: "m", BaseURL: "u", Issue: 5, ProofOnly: true, Now: 1, FakBin: "fak"})
	if forced.Mode != canaryModeProofOnly {
		t.Errorf("--proof-only with --issue should stay proof-only, got %q", forced.Mode)
	}
}

func TestRunDispatchCanaryRequiresBaseURLAndModel(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runDispatchCanary(&stdout, &stderr, []string{"--model", "m"}); code != 2 {
		t.Errorf("missing --base-url should exit 2, got %d", code)
	}
	if !strings.Contains(stderr.String(), "--base-url and --model are required") {
		t.Errorf("stderr should name the missing flags; got %q", stderr.String())
	}
}

func TestRunDispatchCanaryRefusesUnhealthyRoute(t *testing.T) {
	ws := t.TempDir()
	captured := installCanarySeams(t, unhealthyCanaryRecord, 5000, 0)

	var stdout, stderr bytes.Buffer
	code := runDispatchCanary(&stdout, &stderr, []string{
		"--base-url", "https://api.example.com/v1", "--model", "deepseek-chat",
		"--provider", "deepseek", "--workspace", ws, "--now", "5000", "--json",
	})
	if code != 3 {
		t.Fatalf("unhealthy route should refuse launch with exit 3, got %d (stderr=%q)", code, stderr.String())
	}
	if *captured != nil {
		t.Error("launch seam must NOT be called on a refused route")
	}
	if !strings.Contains(stdout.String(), canaryVerdictRefused) {
		t.Errorf("output should carry %s; got:\n%s", canaryVerdictRefused, stdout.String())
	}
	// The refusal is auditable: metadata + route-health sidecars written, but no prompt (no turn).
	runDir := filepath.Join(ws, timeoutLedgerRunsDir, "canary")
	entries, _ := os.ReadDir(runDir)
	if len(entries) != 1 {
		t.Fatalf("want exactly one canary run dir, got %d", len(entries))
	}
	dir := filepath.Join(runDir, entries[0].Name())
	assertFileExists(t, filepath.Join(dir, "metadata.json"))
	assertFileExists(t, filepath.Join(dir, "route-health.json"))
	if _, err := os.Stat(filepath.Join(dir, "prompt.txt")); !os.IsNotExist(err) {
		t.Error("a refused canary must not write a worker prompt sidecar")
	}
	// The probe row is persisted to the shared route-health ledger.
	assertFileExists(t, routeHealthLedgerPath(ws))
}

func TestRunDispatchCanaryLaunchesHealthyRouteAndWritesSidecars(t *testing.T) {
	ws := t.TempDir()
	captured := installCanarySeams(t, healthyCanaryRecord, 7000, 0)

	var stdout, stderr bytes.Buffer
	code := runDispatchCanary(&stdout, &stderr, []string{
		"--base-url", "https://api.example.com/v1", "--model", "deepseek-chat",
		"--provider", "deepseek", "--api-key-env", "MY_KEY", "--workspace", ws, "--now", "7000",
	})
	if code != 0 {
		t.Fatalf("healthy route should launch and exit 0, got %d (stderr=%q)", code, stderr.String())
	}
	if *captured == nil {
		t.Fatal("launch seam must be called on a healthy route")
	}
	if (*captured).Verdict != canaryVerdictLaunched {
		t.Errorf("launched plan verdict = %q, want %s", (*captured).Verdict, canaryVerdictLaunched)
	}
	// The full worker sidecar set is on disk when the canary launches.
	dir := (*captured).Sidecars.Dir
	for _, name := range []string{"prompt.txt", "transcript.jsonl", "guard-audit.jsonl", "metadata.json", "lease.json", "route-health.json"} {
		assertFileExists(t, filepath.Join(dir, name))
	}
	// The lease sidecar records the lane the canary held.
	var lease canaryLeaseMetadata
	readJSONFile(t, filepath.Join(dir, "lease.json"), &lease)
	if lease.Lane != "canary/deepseek" {
		t.Errorf("lease lane = %q, want canary/deepseek", lease.Lane)
	}
	// The route-health sidecar carries the healthy probe result.
	var rh routeHealthRecord
	readJSONFile(t, filepath.Join(dir, "route-health.json"), &rh)
	if rh.Class != string(routeClassHealthy) {
		t.Errorf("route-health sidecar class = %q, want healthy", rh.Class)
	}
}

func TestRunDispatchCanaryDryRunDoesNotLaunch(t *testing.T) {
	ws := t.TempDir()
	captured := installCanarySeams(t, healthyCanaryRecord, 9000, 0)

	var stdout, stderr bytes.Buffer
	code := runDispatchCanary(&stdout, &stderr, []string{
		"--base-url", "https://api.example.com/v1", "--model", "deepseek-chat",
		"--provider", "deepseek", "--workspace", ws, "--now", "9000", "--dry-run", "--json",
	})
	if code != 0 {
		t.Fatalf("dry-run of a healthy route should exit 0, got %d (stderr=%q)", code, stderr.String())
	}
	if *captured != nil {
		t.Error("dry-run must NOT call the launch seam")
	}
	if !strings.Contains(stdout.String(), canaryVerdictPlanned) {
		t.Errorf("dry-run output should carry %s; got:\n%s", canaryVerdictPlanned, stdout.String())
	}
	// Even in dry-run the full sidecar set is materialized so the plan is fully auditable.
	runDir := filepath.Join(ws, timeoutLedgerRunsDir, "canary")
	entries, _ := os.ReadDir(runDir)
	if len(entries) != 1 {
		t.Fatalf("want exactly one canary run dir, got %d", len(entries))
	}
	dir := filepath.Join(runDir, entries[0].Name())
	assertFileExists(t, filepath.Join(dir, "prompt.txt"))
	assertFileExists(t, filepath.Join(dir, "lease.json"))
}

func assertFileExists(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); err != nil {
		t.Errorf("expected sidecar %s to exist: %v", path, err)
	}
}

func readJSONFile(t *testing.T, path string, v any) {
	t.Helper()
	b, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	if err := json.Unmarshal(b, v); err != nil {
		t.Fatalf("unmarshal %s: %v", path, err)
	}
}
