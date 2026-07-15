package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestBenchFleetDispatchClaimsOnceAndWritesWitness(t *testing.T) {
	dir := t.TempDir()
	q := filepath.Join(dir, "requests")
	os.MkdirAll(q, 0755)
	req := benchFleetRequest{Schema: "fak.bench-fleet.request.v1", ID: "abc", Machine: "gcp-g2-l4", Command: "echo ok", State: "queued"}
	if err := writeBenchFleetRequest(filepath.Join(q, "gcp-abc.json"), req); err != nil {
		t.Fatal(err)
	}
	fake := func(name string, args ...string) ([]byte, int, error) {
		return []byte("FAK_BENCH_NODE=fak-cuda-build-l4\ngpu=NVIDIA L4\n"), 0, nil
	}
	var out, errOut bytes.Buffer
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var got benchFleetDispatchReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Claimed != 1 || got.Succeeded != 1 {
		t.Fatalf("report=%+v", got)
	}
	out.Reset()
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 0 {
		t.Fatal(code)
	}
	json.Unmarshal(out.Bytes(), &got)
	if got.Claimed != 0 {
		t.Fatalf("duplicate claim: %+v", got)
	}
	if _, err := os.Stat(filepath.Join(dir, "witnesses", "abc.json")); err != nil {
		t.Fatal(err)
	}
}

func TestBenchFleetRouteLeavesUnconfiguredMacWaiting(t *testing.T) {
	t.Setenv("FAK_BENCH_MAC_HOST", "")
	_, _, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "node-macos-a"})
	if err == nil || route != "mac:tailscale" || state != "waiting_session" {
		t.Fatalf("route=%q state=%q err=%v", route, state, err)
	}
}

func TestBenchFleetRouteUsesConfiguredMacSession(t *testing.T) {
	t.Setenv("FAK_BENCH_MAC_HOST", "mac-node")
	name, args, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "node-macos-a", Command: "go run ./cmd/livecodebench --check --json"})
	if err != nil || name != "tailscale" || route != "mac:tailscale/mac-node" || state != "running" {
		t.Fatalf("name=%q args=%q route=%q state=%q err=%v", name, args, route, state, err)
	}
	if got := strings.Join(args, " "); !strings.Contains(got, "ssh mac-node") || !strings.Contains(got, "FAK_BENCH_NODE") {
		t.Fatalf("args=%q", args)
	}
}

func TestBenchFleetRouteRunsWorkstationLocally(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("Windows control route")
	}
	name, args, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "workstation-a", Command: "go run ./cmd/livecodebench --check --json"})
	if err != nil || name != "powershell.exe" || route != "local-control" || state != "running" {
		t.Fatalf("name=%q args=%q route=%q state=%q err=%v", name, args, route, state, err)
	}
	if got := strings.Join(args, " "); !strings.Contains(got, "FAK_BENCH_NODE") || !strings.Contains(got, "livecodebench") {
		t.Fatalf("args=%q", args)
	}
}
func TestBenchFleetFailedExecutionRemainsWitnessedAndNotReclaimed(t *testing.T) {
	dir := t.TempDir()
	q := filepath.Join(dir, "requests")
	if err := os.MkdirAll(q, 0o755); err != nil {
		t.Fatal(err)
	}
	req := benchFleetRequest{Schema: "fak.bench-fleet.request.v1", ID: "bad", Machine: "gcp-g2-l4", Command: "false", State: "queued"}
	if err := writeBenchFleetRequest(filepath.Join(q, "bad.json"), req); err != nil {
		t.Fatal(err)
	}
	fake := func(string, ...string) ([]byte, int, error) { return []byte("remote failure"), 7, errors.New("exit 7") }
	var out, errOut bytes.Buffer
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 1 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	got, err := readBenchFleetRequest(filepath.Join(q, "bad.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "failed" {
		t.Fatalf("state=%s", got.State)
	}
	out.Reset()
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 0 {
		t.Fatal(code)
	}
	var report benchFleetDispatchReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Claimed != 0 {
		t.Fatalf("failed request reclaimed: %+v", report)
	}
}

func TestBenchFleetSuccessfulWitnessIsIngestedAsRunArtifact(t *testing.T) {
	root := t.TempDir()
	q := filepath.Join(root, ".fak", "bench-fleet", "requests")
	if err := os.MkdirAll(q, 0o755); err != nil {
		t.Fatal(err)
	}
	req := benchFleetRequest{Schema: "fak.bench-fleet.request.v1", ID: "abc", Machine: "gcp-g2-l4", NodeClass: "gpu", Benchmark: "gpu-benchmark", Model: "qwen", Precision: "q8", Command: "echo ok", State: "queued"}
	if err := writeBenchFleetRequest(filepath.Join(q, "gcp-abc.json"), req); err != nil {
		t.Fatal(err)
	}
	fake := func(string, ...string) ([]byte, int, error) { return []byte("FAK_BENCH_NODE=node\nOK\n"), 0, nil }
	var out, errOut bytes.Buffer
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--workspace", root, "--json"}, fake); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	matches, err := filepath.Glob(filepath.Join(root, "experiments", "benchmark", "runs", "by-machine", "gcp-g2-l4", "*-bench-fleet-abc", "manifest.json"))
	if err != nil || len(matches) != 1 {
		t.Fatalf("matches=%q err=%v", matches, err)
	}
	b, err := os.ReadFile(matches[0])
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Contains(b, []byte(`"request_id": "abc"`)) || !bytes.Contains(b, []byte(`"witness": "witness.json"`)) {
		t.Fatalf("manifest=%s", b)
	}
}

func TestBenchFleetDGXCredentialFailureIsTypedWaiting(t *testing.T) {
	root := t.TempDir()
	bridge := filepath.Join(root, "..", "fak-private", ".dgxbridge-verify")
	if err := os.MkdirAll(bridge, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(bridge, "dgxbridge-fresh.exe"), []byte("stub"), 0o755); err != nil {
		t.Fatal(err)
	}
	w := executeBenchFleetRequest(root, benchFleetRequest{ID: "dgx", Machine: "a100", Command: "echo ok"}, func(string, ...string) ([]byte, int, error) {
		return []byte("no Slack channel set"), 1, errors.New("exit 1")
	})
	if w.State != "waiting_credentials" {
		t.Fatalf("witness=%+v", w)
	}
}

func TestBenchFleetRejectsEmptyRemoteSuccess(t *testing.T) {
	w := executeBenchFleetRequest(t.TempDir(), benchFleetRequest{ID: "empty", Machine: "gcp-g2-l4", Command: "true"}, func(string, ...string) ([]byte, int, error) { return nil, 0, nil })
	if w.State != "failed" || w.Error != "remote witness missing FAK_BENCH_NODE marker" {
		t.Fatalf("witness=%+v", w)
	}
}

func TestBenchNodeWitnessRequiresResolvedIdentity(t *testing.T) {
	for _, bad := range []string{"", "FAK_BENCH_NODE=", "FAK_BENCH_NODE=$(hostname)"} {
		if hasBenchNodeWitness(bad) {
			t.Fatalf("accepted %q", bad)
		}
	}
	if !hasBenchNodeWitness("noise\nFAK_BENCH_NODE=fak-cuda-build-l4\n") {
		t.Fatal("rejected resolved identity")
	}
}

func TestBenchFleetGCPProvisionFailureIsRetryable(t *testing.T) {
	w := executeBenchFleetRequest(t.TempDir(), benchFleetRequest{ID: "setup", Machine: "gcp-g2-l4", Command: "gpucheck"}, func(string, ...string) ([]byte, int, error) {
		return []byte("FAK_BENCH_NODE=l4\nbash: go: command not found"), 127, errors.New("exit 127")
	})
	if w.State != "waiting_provision" {
		t.Fatalf("witness=%+v", w)
	}
}

func TestBenchFleetQwen36HintIsNeverExecutedLiterally(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "qwen36", Command: "fak serve + fak agent (qwen3.6-27b via gateway)"})
		if err == nil || state != "waiting_provision" {
			t.Fatalf("%s: state=%q err=%v, want waiting_provision error", machine, state, err)
		}
		if name != "" || len(args) != 0 {
			t.Fatalf("%s: executable hint leaked into route: %q %q", machine, name, args)
		}
	}
}

func TestBenchFleetWorkstationSessionUsesBoundedRecipe(t *testing.T) {
	if runtime.GOOS != "windows" {
		t.Skip("workstation route is Windows-only")
	}
	_, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "workstation-a", Benchmark: "session-benchmark", Command: "go run ./cmd/sessionbench"})
	if err != nil || state != "running" {
		t.Fatalf("state=%q err=%v", state, err)
	}
	command := strings.Join(args, " ")
	for _, want := range []string{"sessionbench", "-synthetic tiny", "-agents 2", "-turns 2", "FAK_BENCH_NODE="} {
		if !strings.Contains(command, want) {
			t.Fatalf("command %q missing %q", command, want)
		}
	}
}

func TestBenchFleetL4UsesProvisionedGPURecipe(t *testing.T) {
	cmd := benchFleetRemoteCommand(benchFleetRequest{Machine: "gcp-g2-l4", Benchmark: "gpu-benchmark", Command: "generic"})
	for _, want := range []string{"FAK_BENCH_NODE=", "$HOME/.local/go/bin", "build_cuda.sh binary", "~/models/qwen05"} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command %q missing %q", cmd, want)
		}
	}
}

func TestBenchFleetA100UsesProvisionedGPURecipe(t *testing.T) {
	cmd := benchFleetRemoteCommand(benchFleetRequest{Machine: "gcp-a3-high-h100-1g", Benchmark: "gpu-benchmark", Command: "generic"})
	for _, want := range []string{"FAK_CUDA_ARCH=sm_80", "build_cuda.sh binary", "~/models/qwen05"} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command %q missing %q", cmd, want)
		}
	}
}

func TestBenchFleetCOSModelBenchmarkUsesContainerRecipe(t *testing.T) {
	cmd := benchFleetRemoteCommand(benchFleetRequest{Machine: "gcp-g2-l4-32", Benchmark: "model-benchmark", Command: "generic"})
	for _, want := range []string{"FAK_BENCH_NODE=", "docker run --rm", "golang:1.26", "/usr/local/go/bin/go", "/models/smollm2-135m"} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command %q missing %q", cmd, want)
		}
	}
}

func TestBenchFleetL4ServeUsesLiveDecodeRecipe(t *testing.T) {
	cmd := benchFleetRemoteCommand(benchFleetRequest{Machine: "gcp-g2-l4-32", Benchmark: "gpu-benchmark", Command: "generic"})
	for _, want := range []string{"FAK_BENCH_NODE=", "FAK_BENCH_HTTP=", "/v1/chat/completions", "max_tokens"} {
		if !strings.Contains(cmd, want) {
			t.Fatalf("command %q missing %q", cmd, want)
		}
	}
}

func TestBenchFleetCatalogDoesNotRegressLastRun(t *testing.T) {
	root := t.TempDir()
	runDir := filepath.Join(root, "experiments", "benchmark", "runs", "by-machine", "node-a", "20260101T000000Z-bench-fleet-old")
	if err := os.MkdirAll(runDir, 0755); err != nil {
		t.Fatal(err)
	}
	catalog := `{"last_updated":"2026-01-02T00:00:00Z","machines":{"node-a":{"runs":1,"last_run":"20260102T000000Z"}},"runs":[{"run_id":"newer","machine_id":"node-a","timestamp":"20260102T000000Z"}]}`
	catalogPath := filepath.Join(root, "experiments", "benchmark", "catalog.json")
	if err := os.MkdirAll(filepath.Dir(catalogPath), 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(catalogPath, []byte(catalog), 0644); err != nil {
		t.Fatal(err)
	}
	manifest := benchFleetRunManifest{Schema: "fak.benchmark.run.v1", RunID: "older", MachineID: "node-a", Timestamp: "20260101T000000Z"}
	b, err := json.Marshal(manifest)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(runDir, "manifest.json"), b, 0644); err != nil {
		t.Fatal(err)
	}
	if err := updateBenchFleetCatalog(root); err != nil {
		t.Fatal(err)
	}
	var got struct {
		Machines map[string]struct {
			LastRun string `json:"last_run"`
			Runs    int    `json:"runs"`
		} `json:"machines"`
	}
	out, err := os.ReadFile(catalogPath)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatal(err)
	}
	if got.Machines["node-a"].LastRun != "20260102T000000Z" {
		t.Fatalf("last_run regressed to %q", got.Machines["node-a"].LastRun)
	}
	if got.Machines["node-a"].Runs != 2 {
		t.Fatalf("runs = %d, want 2", got.Machines["node-a"].Runs)
	}
}
