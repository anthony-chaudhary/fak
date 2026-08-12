package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

func TestBenchFleetQueuesEveryPlannedNodeAndDeduplicates(t *testing.T) {
	// Not parallel: the route preflight (#6503) reads the fleet's session
	// configuration, which this test pins to "nothing is configured".
	t.Setenv("FAK_BENCH_MAC_HOST", "")
	t.Setenv("FAK_BENCH_A100_CHANNEL", "")
	t.Setenv("FAK_BENCH_CPU_SERVER_CHANNEL", "")
	dir := t.TempDir()
	planPath := filepath.Join(dir, "plan.json")
	plan := `{"per_machine_next":{
		"a100":{"machine_id":"a100","workload_kind":"gpu-benchmark","model":"qwen","precision":"Q8_0","reason":"coverage","suggested_command":"on a100: go run -tags cuda ./cmd/gpucheck  # HINT"},
		"cpu-server-a":{"machine_id":"cpu-server-a","workload_kind":"model-benchmark","model":"smollm","precision":"q8_0","reason":"coverage","suggested_command":"on cpu: go run ./cmd/modelbench -quant  # HINT"},
		"node-macos-a":{"machine_id":"node-macos-a","workload_kind":"livecodebench","model":"qwen","precision":"official","reason":"coverage","suggested_command":"on mac: go run ./cmd/livecodebench --check --json  # HINT"}
	}}`
	if err := os.WriteFile(planPath, []byte(plan), 0o644); err != nil {
		t.Fatal(err)
	}
	queue := filepath.Join(dir, "queue")

	var out, errOut bytes.Buffer
	code := runBenchFleet(&out, &errOut, []string{"--apply", "--json", "--plan-json", planPath, "--queue", queue, "--workspace", dir})
	if code != 0 {
		t.Fatalf("first run code=%d stderr=%s", code, errOut.String())
	}
	var got benchFleetReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Machines != 3 || got.Enqueued != 3 || got.Existing != 0 {
		t.Fatalf("first report=%+v", got)
	}
	// None of the three nodes has a configured session or credential, so the plan
	// enqueues them already held instead of spending a dispatch on each one every
	// fifteen minutes (#6503).
	if got.Held != 3 {
		t.Fatalf("held=%d, want every unconfigured node preflighted as held: %+v", got.Held, got)
	}
	entries, err := os.ReadDir(queue)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 {
		t.Fatalf("queue entries=%d want 3", len(entries))
	}
	classes := map[string]bool{}
	wantState := map[string]string{"a100": "waiting_credentials", "cpu-server-a": "waiting_credentials", "node-macos-a": "waiting_session"}
	for _, req := range got.Requests {
		classes[req.NodeClass] = true
		if req.State != wantState[req.Machine] {
			t.Fatalf("%s state=%s, want %s", req.Machine, req.State, wantState[req.Machine])
		}
		if req.HeldReason == "" || req.HeldSince == "" {
			t.Fatalf("%s held without naming the gap: %+v", req.Machine, req)
		}
	}
	for _, class := range []string{"gpu", "cpu", "mac"} {
		if !classes[class] {
			t.Fatalf("missing class %s: %#v", class, classes)
		}
	}

	out.Reset()
	errOut.Reset()
	code = runBenchFleet(&out, &errOut, []string{"--apply", "--json", "--plan-json", planPath, "--queue", queue, "--workspace", dir})
	if code != 0 {
		t.Fatalf("second run code=%d stderr=%s", code, errOut.String())
	}
	got = benchFleetReport{}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Enqueued != 0 || got.Existing != 3 {
		t.Fatalf("second report=%+v", got)
	}
}

func TestBenchFleetDryRunWritesNothing(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	planPath := filepath.Join(dir, "plan.json")
	queue := filepath.Join(dir, "queue")
	if err := os.WriteFile(planPath, []byte(`{"per_machine_next":{"gcp-g2-l4":{"machine_id":"gcp-g2-l4","workload_kind":"gpu-benchmark","suggested_command":"on l4: gpucheck  # HINT"}}}`), 0o644); err != nil {
		t.Fatal(err)
	}
	var out, errOut bytes.Buffer
	if code := runBenchFleet(&out, &errOut, []string{"--json", "--plan-json", planPath, "--queue", queue}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errOut.String())
	}
	if _, err := os.Stat(queue); !os.IsNotExist(err) {
		t.Fatalf("dry run created queue: %v", err)
	}
	var got benchFleetReport
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.Enqueued != 0 || got.Requests[0].State != "planned" || got.Requests[0].NodeClass != "gpu" {
		t.Fatalf("report=%+v", got)
	}
}
func TestBenchFleetTaskSpawnsAreWindowless(t *testing.T) {
	src, err := os.ReadFile("benchloop_fleet.go")
	if err != nil {
		t.Fatal(err)
	}
	if findings := windowgate.GoExecViolations("cmd/fak/benchloop_fleet.go", string(src)); len(findings) != 0 {
		t.Fatalf("Scheduled Task helper can flash a console window: %v", findings)
	}
}
