package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/benchloop"
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

func TestBenchFleetStaleRunningClaimIsReconciledAndMeasured(t *testing.T) {
	// A dispatcher killed mid-run left its row marked "running" forever: no later
	// tick would claim it, and the queue reported a cell that nothing was executing
	// (#6503). The lock read-back returns it to the queue instead.
	root := t.TempDir()
	q := filepath.Join(root, ".fak", "bench-fleet", "requests")
	if err := os.MkdirAll(q, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(q, "stuck.json")
	req := benchFleetRequest{
		Schema: "fak.bench-fleet.request.v1", ID: "stuck", Machine: "gcp-g2-l4",
		Command: "echo ok", State: "running", Attempts: 1,
		LastAttemptAt: time.Now().UTC().Add(-time.Hour).Format(time.RFC3339),
	}
	if err := writeBenchFleetRequest(path, req); err != nil {
		t.Fatal(err)
	}
	fake := func(string, ...string) ([]byte, int, error) {
		return []byte("FAK_BENCH_NODE=fak-cuda-build-l4\nFAK_BENCH_SECONDS=1.5\n"), 0, nil
	}
	var out, errOut bytes.Buffer
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--workspace", root, "--json"}, fake); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report benchFleetDispatchReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Reconciled != 1 || report.Claimed != 1 || report.Succeeded != 1 {
		t.Fatalf("stale claim not reconciled and rerun: %+v", report)
	}
	if report.Utility.Measured != 1 || !report.Utility.Healthy {
		t.Fatalf("utility=%+v, want one witnessed numeric measurement", report.Utility)
	}
	got, err := readBenchFleetRequest(path)
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "succeeded" || !got.Measured || got.Attempts != 2 {
		t.Fatalf("request=%+v", got)
	}
	if _, err := os.Stat(path + ".claim"); !os.IsNotExist(err) {
		t.Fatalf("claim lock left behind: %v", err)
	}
}

func TestBenchFleetClaimReadsBackItsOwnDispatcher(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "req.json")
	host, _ := os.Hostname()
	stamp := time.Now().UTC().Format(time.RFC3339)
	if err := os.WriteFile(path+".claim", fmt.Appendf(nil, "%d %s %s\n", os.Getpid(), host, stamp), 0o600); err != nil {
		t.Fatal(err)
	}
	claim := readBenchFleetClaim(path, benchFleetRequest{})
	if !claim.Present || !claim.Local || !claim.Alive || claim.PID != os.Getpid() {
		t.Fatalf("claim=%+v, want this live dispatcher", claim)
	}
	if claim.Started.IsZero() {
		t.Fatalf("claim start not read back: %+v", claim)
	}
	if state, _ := benchloop.ReconcileFleetRunning(claim, time.Now().UTC()); state != benchloop.FleetRunning {
		t.Fatalf("live claim reconciled to %q", state)
	}
	if err := os.Remove(path + ".claim"); err != nil {
		t.Fatal(err)
	}
	if claim := readBenchFleetClaim(path, benchFleetRequest{}); claim.Present || claim.Alive {
		t.Fatalf("missing lock read back as held: %+v", claim)
	}
}

func TestBenchFleetRemoteCommandProvisionsLabNodes(t *testing.T) {
	cases := []struct {
		machine   string
		benchmark string
		want      string
	}{
		{machine: "a100", benchmark: "gpu-benchmark", want: "FAK_CUDA_ARCH=sm_80 bash tools/run_485_acceptance_on_gpu.sh"},
		{machine: "a100", benchmark: "radix-benchmark", want: "go run ./cmd/radixbench -live=false"},
		{machine: "cpu-server-a", benchmark: "model-benchmark", want: "go run ./cmd/modelbench -synthetic tiny"},
		{machine: "cpu-server-a", benchmark: "qwen36", want: "go run ./cmd/sessionbench -synthetic tiny"},
	}
	for _, tc := range cases {
		t.Run(tc.machine+"/"+tc.benchmark, func(t *testing.T) {
			got := benchFleetRemoteCommand(benchFleetRequest{Machine: tc.machine, Benchmark: tc.benchmark})
			for _, required := range []string{"FAK_BENCH_NODE=", "mktemp -d /tmp/fak-bench.XXXXXX", "git clone --depth 1", tc.want} {
				if !strings.Contains(got, required) {
					t.Fatalf("command %q does not contain %q", got, required)
				}
			}
			if strings.Contains(got, "cd ~/fak") {
				t.Fatalf("lab command assumes an unprovisioned checkout: %q", got)
			}
		})
	}
}

func TestBenchFleetRouteUsesMachineSpecificBridgeChannels(t *testing.T) {
	root := t.TempDir()
	bridgeDir := filepath.Clean(filepath.Join(root, "..", "fak-private", ".dgxbridge-verify"))
	if err := os.MkdirAll(bridgeDir, 0755); err != nil {
		t.Fatal(err)
	}
	bridgeName := "dgxbridge-fresh"
	if runtime.GOOS == "windows" {
		bridgeName += ".exe"
	}
	bridge := filepath.Join(bridgeDir, bridgeName)
	if err := os.WriteFile(bridge, []byte("bridge"), 0755); err != nil {
		t.Fatal(err)
	}
	configDir := filepath.Join(root, ".fak", "bench-fleet")
	if err := os.MkdirAll(configDir, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(configDir, "routes.json"), []byte(`{"a100_channel":"gpu-channel","cpu_server_channel":"cpu-channel"}`), 0600); err != nil {
		t.Fatal(err)
	}

	for _, tc := range []struct {
		machine string
		channel string
	}{
		{machine: "a100", channel: "gpu-channel"},
		{machine: "cpu-server-a", channel: "cpu-channel"},
	} {
		t.Run(tc.machine, func(t *testing.T) {
			name, args, route, state, err := benchFleetRoute(root, benchFleetRequest{ID: "req-123", Machine: tc.machine, Command: "hostname"})
			if err != nil || name != bridge || route != "dgxbridge" || state != "running" {
				t.Fatalf("route=(%q %v %q %q %v), want configured dgxbridge", name, args, route, state, err)
			}
			if len(args) < 8 || args[0] != "-channel" || args[1] != tc.channel || args[2] != "-timeout" || args[3] != "4m" || args[4] != "-remote-out" || args[5] != "/tmp/fak-bench-results/req-123.bridge.out" || args[6] != "run" {
				t.Fatalf("args=%v, want machine-specific channel and remote-file readback before run", args)
			}
		})
	}
}

func TestBenchFleetRouteLeavesUnconfiguredMacWaiting(t *testing.T) {
	t.Setenv("FAK_BENCH_MAC_HOST", "")
	_, _, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "node-macos-a"})
	if err == nil || route != "mac:ssh" || state != "waiting_session" {
		t.Fatalf("route=%q state=%q err=%v", route, state, err)
	}
}

func TestBenchFleetRouteUsesConfiguredMacSession(t *testing.T) {
	t.Setenv("FAK_BENCH_MAC_HOST", "mac-node")
	name, args, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "node-macos-a", Command: "go run ./cmd/livecodebench --check --json"})
	if err != nil || name != "ssh" || route != "mac:ssh/mac-node" || state != "running" {
		t.Fatalf("name=%q args=%q route=%q state=%q err=%v", name, args, route, state, err)
	}
	if got := strings.Join(args, " "); !strings.Contains(got, "IdentitiesOnly=yes mac-node") || !strings.Contains(got, "FAK_BENCH_NODE") || !strings.Contains(got, ".fak-mac-bench/fak") || !strings.Contains(got, "GOTOOLCHAIN=auto") {
		t.Fatalf("args=%q", args)
	}
}

func TestBenchFleetRouteLeavesUnconfiguredWorkstationWaiting(t *testing.T) {
	t.Setenv("FAK_BENCH_WORKSTATION_HOST", "")
	t.Setenv("FAK_BENCH_WORKSTATION_USER", "")
	t.Setenv("FAK_BENCH_WORKSTATION_IDENTITY_FILE", "")
	_, _, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: "workstation-a"})
	if err == nil || route != "workstation:ssh" || state != "waiting_session" {
		t.Fatalf("route=%q state=%q err=%v", route, state, err)
	}
}

func TestBenchFleetRouteUsesConfiguredWorkstationWSL(t *testing.T) {
	t.Setenv("FAK_BENCH_WORKSTATION_HOST", "workstation-node")
	t.Setenv("FAK_BENCH_WORKSTATION_USER", "runner")
	t.Setenv("FAK_BENCH_WORKSTATION_IDENTITY_FILE", `C:\keys\workstation`)
	t.Setenv("FAK_BENCH_WORKSTATION_DISTRO", "Ubuntu")
	name, args, route, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{ID: "abc", Machine: "workstation-a", Benchmark: "gpu-benchmark", Command: "generic"})
	if err != nil || name != "ssh" || route != "workstation:ssh/workstation-node" || state != "running" {
		t.Fatalf("name=%q args=%q route=%q state=%q err=%v", name, args, route, state, err)
	}
	got := strings.Join(args, " ")
	for _, want := range []string{"BatchMode=yes", `C:\keys\workstation`, "runner@workstation-node", "wsl.exe", "FromBase64String"} {
		if !strings.Contains(got, want) {
			t.Fatalf("args=%q missing %q", args, want)
		}
	}
}

func TestBenchFleetWorkstationGPURecipeUsesRealRemoteCUDA(t *testing.T) {
	got := benchFleetWorkstationScript(benchFleetRequest{Machine: "workstation-a", Benchmark: "gpu-benchmark", Command: "generic"})
	for _, want := range []string{"FAK_BENCH_NODE=", "FAK_BENCH_GPU=", "nvidia-smi", "reset --hard origin/main", "FAK_CUDA_ARCH=sm_89", "run_485_acceptance_on_gpu.sh"} {
		if !strings.Contains(got, want) {
			t.Fatalf("script missing %q:\n%s", want, got)
		}
	}
	if strings.Contains(got, "generic") {
		t.Fatalf("planner hint leaked into GPU recipe: %s", got)
	}
}

func TestBenchFleetWorkstationConnectionFailureWaitsForSession(t *testing.T) {
	dir := t.TempDir()
	q := filepath.Join(dir, "requests")
	if err := os.MkdirAll(q, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_BENCH_WORKSTATION_HOST", "workstation-node")
	t.Setenv("FAK_BENCH_WORKSTATION_USER", "runner")
	t.Setenv("FAK_BENCH_WORKSTATION_IDENTITY_FILE", `C:\keys\workstation`)
	req := benchFleetRequest{Schema: "fak.bench-fleet.request.v1", ID: "offline", Machine: "workstation-a", Benchmark: "gpu-benchmark", Command: "generic", State: "queued"}
	if err := writeBenchFleetRequest(filepath.Join(q, "offline.json"), req); err != nil {
		t.Fatal(err)
	}
	fake := func(string, ...string) ([]byte, int, error) {
		return []byte("ssh: connect to host workstation-node port 22: Connection timed out"), 255, errors.New("exit 255")
	}
	var out, errOut bytes.Buffer
	// A tick whose only cell is waiting on an unconfigured session measured nothing,
	// so it cannot report the scheduler a healthy 0 (#6503).
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 3 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	got, err := readBenchFleetRequest(filepath.Join(q, "offline.json"))
	if err != nil {
		t.Fatal(err)
	}
	if got.State != "waiting_session" {
		t.Fatalf("state=%s", got.State)
	}
	if got.HeldSince == "" || got.HeldReason != "session" {
		t.Fatalf("hold not typed: %+v", got)
	}
	// The held node is not re-dispatched on the next tick.
	out.Reset()
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 3 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	var report benchFleetDispatchReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Claimed != 0 || len(report.Skipped) != 1 || report.Utility.Held != 1 {
		t.Fatalf("held node re-dispatched: %+v", report)
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
	// The next tick claims nothing -- and a queue whose only cell is failed is still
	// not a healthy fleet, so the result stays nonzero instead of reporting 0 because
	// this tick happened to do no work (#6503).
	if code := runBenchFleetDispatchWithExec(&out, &errOut, []string{"--queue", q, "--json"}, fake); code != 1 {
		t.Fatal(code)
	}
	var report benchFleetDispatchReport
	if err := json.Unmarshal(out.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.Claimed != 0 {
		t.Fatalf("failed request reclaimed: %+v", report)
	}
	if report.Utility.Failed != 1 || report.Utility.Successful != 0 || report.Utility.Healthy {
		t.Fatalf("utility hides the failed cell: %+v", report.Utility)
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

func TestBenchFleetQwen36UsesProvisionedGatewayRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "qwen36", Command: "fak serve + fak agent (qwen3.6-27b via gateway)"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v, want running gcloud route", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "qwen/qwen3.6-27b", ".config/fak/groq.key", "api.groq.com/openai/v1/chat/completions", "FAK_BENCH_MODEL="} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if strings.Contains(command, "fak serve + fak agent") {
			t.Fatalf("%s: planner prose leaked into executable command: %q", machine, command)
		}
	}
}

func TestBenchFleetGCPAgentLiveUsesBoundedGatewayRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "agent-live", Command: "go run ./cmd/fak agent --task <task>"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "fak agent", "qwen/qwen3.6-27b", "api.groq.com/openai/v1", "-max-turns 10", ".config/fak/groq.key"} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if strings.Contains(command, "<task>") {
			t.Fatalf("%s: planner placeholder leaked into command: %q", machine, command)
		}
	}
}

func TestBenchFleetExecCommandBoundsRuntimeAndPipeWait(t *testing.T) {
	cmd, cancel := newBenchFleetExecCommand("definitely-not-a-real-fak-command", 25*time.Millisecond)
	defer cancel()
	if cmd.WaitDelay != benchFleetWaitDelay {
		t.Fatalf("WaitDelay=%s want %s", cmd.WaitDelay, benchFleetWaitDelay)
	}
	if cmd.Cancel == nil {
		t.Fatal("CommandContext cancellation is not configured")
	}
}

func TestBenchFleetGCPTurnTaxUsesBoundedIsolatedRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "turn-tax", Command: "go run ./cmd/fak turntax --suite turntax-airline"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "turntax", "-suite turntax-airline", "-out /tmp/fak-turntax-report.json"} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if strings.Contains(command, "--suite") {
			t.Fatalf("%s: planner's invalid double-dash flag leaked into command: %q", machine, command)
		}
		if machine == "gcp-g2-l4-32" && !strings.Contains(command, "docker run") {
			t.Fatalf("COS route must use container: %q", command)
		}
	}
}

func TestBenchFleetGCPSessionUsesBoundedSyntheticRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "session-benchmark", Command: "go run ./cmd/sessionbench"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "sessionbench", "-synthetic tiny", "-agents 2", "-turns 2", "-reps 1", "/tmp/fak-sessionbench.json"} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if machine == "gcp-g2-l4-32" && !strings.Contains(command, "docker run") {
			t.Fatalf("COS route must use container: %q", command)
		}
	}
}

func TestBenchFleetGCPParityUsesIsolatedArtifactRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "parity", Command: "go run ./cmd/paritybench"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "paritybench", "-out-json /tmp/fak-parity.json", "-out-md /tmp/fak-parity.md"} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if machine == "gcp-g2-l4-32" && !strings.Contains(command, "docker run") {
			t.Fatalf("COS route must use container: %q", command)
		}
	}
}

func TestBenchFleetGCPFanUsesBoundedTopologyRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "fan-benchmark", Command: "go run ./cmd/fanbench"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "fanbench", "-agents 1,4", "-sub-turns 1", "-trials 1", "-prefixes smoke", "/tmp/fak-fanbench.json"} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if machine == "gcp-g2-l4-32" && !strings.Contains(command, "docker run") {
			t.Fatalf("COS route must use container: %q", command)
		}
	}
}

func TestBenchFleetGCPConceptUsesBoundedReplayRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "concept-benchmark", Command: "go run ./cmd/conceptbench --replay cmd/conceptbench/testdata/replay"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "conceptbench", "-replay cmd/conceptbench/testdata/replay"} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if strings.Contains(command, "--replay") {
			t.Fatalf("%s: planner's invalid double-dash flag leaked into command: %q", machine, command)
		}
		if machine == "gcp-g2-l4-32" && !strings.Contains(command, "docker run") {
			t.Fatalf("COS route must use container: %q", command)
		}
	}
}

func TestBenchFleetGCPRadixUsesProvisionedRealModelRecipe(t *testing.T) {
	for _, machine := range []string{"gcp-g2-l4", "gcp-g2-l4-32", "gcp-a3-high-h100-1g"} {
		name, args, _, state, err := benchFleetRoute(t.TempDir(), benchFleetRequest{Machine: machine, Benchmark: "radix-benchmark", Command: "go run ./cmd/radixbench"})
		if err != nil || state != "running" || name != "gcloud" {
			t.Fatalf("%s: name=%q state=%q err=%v", machine, name, state, err)
		}
		command := strings.Join(args, " ")
		for _, want := range []string{"FAK_BENCH_NODE=", "radixbench", "smollm2-135m", "-lean", "-quant", "-reps 1", "-only few-shot"} {
			if !strings.Contains(command, want) {
				t.Fatalf("%s: command missing %q: %q", machine, want, command)
			}
		}
		if machine == "gcp-g2-l4-32" && !strings.Contains(command, "docker run") {
			t.Fatalf("COS route must use container: %q", command)
		}
	}
}

func TestBenchFleetWorkstationSessionUsesBoundedRecipe(t *testing.T) {
	command := benchFleetWorkstationScript(benchFleetRequest{Machine: "workstation-a", Benchmark: "session-benchmark", Command: "go run ./cmd/sessionbench"})
	for _, want := range []string{"sessionbench", "-synthetic tiny", "-agents 2", "-turns 2", "FAK_BENCH_NODE="} {
		if !strings.Contains(command, want) {
			t.Fatalf("command %q missing %q", command, want)
		}
	}
}

func TestBenchFleetWorkstationModelUsesProvisionedSnapshot(t *testing.T) {
	command := benchFleetWorkstationScript(benchFleetRequest{Machine: "workstation-a", Benchmark: "model-benchmark", Command: "go run ./cmd/modelbench -quant"})
	for _, want := range []string{"modelbench", "-hf internal/model/.cache/smollm2-135m", "-decode-steps 4", "FAK_BENCH_NODE="} {
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
