package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// h2hModelsHandler answers GET /v1/models with the given model ids and 404s elsewhere.
func h2hModelsHandler(ids ...string) http.Handler {
	var data []string
	for _, id := range ids {
		b, _ := json.Marshal(id)
		data = append(data, `{"id":`+string(b)+`}`)
	}
	body := `{"object":"list","data":[` + strings.Join(data, ",") + `]}`
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/models" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, body)
	})
}

func TestBenchH2HServeWaitAllArmsReady(t *testing.T) {
	a := httptest.NewServer(h2hModelsHandler("m1", "m2"))
	defer a.Close()
	b := httptest.NewServer(h2hModelsHandler("m1"))
	defer b.Close()
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"serve-wait",
		"--arm", "fak=" + a.URL + "/v1", "--arm", "llama=" + b.URL + "/v1",
		"--model", "m1", "--interval", "10ms", "--timeout", "5s"})
	if code != 0 {
		t.Fatalf("serve-wait exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "ready fak "+a.URL+"/v1") || !strings.Contains(out.String(), "ready llama "+b.URL+"/v1") {
		t.Fatalf("serve-wait output: %s", out.String())
	}
}

func TestBenchH2HServeWaitJSONEnvelope(t *testing.T) {
	a := httptest.NewServer(h2hModelsHandler("m1"))
	defer a.Close()
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"serve-wait",
		"--arm", "fak=" + a.URL + "/v1", "--model", "m1", "--interval", "10ms", "--timeout", "5s", "--json"})
	if code != 0 {
		t.Fatalf("serve-wait --json exit %d: %s", code, errb.String())
	}
	var env h2hServeWaitEnvelope
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("serve-wait json: %v\n%s", err, out.String())
	}
	if env.Schema != h2hServeWaitSchema || len(env.Ready) != 1 || len(env.NotReady) != 0 || env.Waited == "" {
		t.Fatalf("envelope = %+v", env)
	}
	if !strings.Contains(out.String(), `"not_ready":[]`) {
		t.Fatalf("success envelope must carry an empty not_ready array, got: %s", out.String())
	}
	if env.Ready[0].Name != "fak" || env.Ready[0].URL != a.URL+"/v1" {
		t.Fatalf("ready entry = %+v", env.Ready[0])
	}
}

func TestBenchH2HServeWaitModelMissingTimesOut(t *testing.T) {
	srv := httptest.NewServer(h2hModelsHandler("other-model"))
	defer srv.Close()
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"serve-wait",
		"--arm", "fak=" + srv.URL + "/v1", "--model", "missing", "--interval", "20ms", "--timeout", "200ms"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1: %s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "serve-wait: timeout after 200ms waiting for") {
		t.Fatalf("stderr: %s", errb.String())
	}
	if !strings.Contains(errb.String(), `model "missing" not listed by`) {
		t.Fatalf("stderr: %s", errb.String())
	}
}

func TestBenchH2HServeWaitConnectionRefusedRetriesUntilDeadline(t *testing.T) {
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	addr := ln.Addr().String()
	_ = ln.Close() // nothing listens here: every probe is connection-refused
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"serve-wait",
		"--arm", "fak=http://" + addr + "/v1", "--interval", "20ms", "--timeout", "200ms"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1: %s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "serve-wait: timeout after 200ms waiting for") {
		t.Fatalf("stderr: %s", errb.String())
	}
}

func TestBenchH2HPlanEmitsPerArmArgvInOrder(t *testing.T) {
	ledgerDir := strings.ReplaceAll(t.TempDir(), "\\", "/")
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"plan",
		"--model", "m", "--run-id", "r1",
		"--arm", "fak=http://127.0.0.1:8080/v1", "--arm", "llama=http://127.0.0.1:8091/v1",
		"--reps", "5", "--ledger-dir", ledgerDir})
	if code != 0 {
		t.Fatalf("plan exit %d: %s", code, errb.String())
	}
	lines := strings.Split(strings.TrimSpace(out.String()), "\n")
	if len(lines) != 2 {
		t.Fatalf("want 2 argv lines, got %d:\n%s", len(lines), out.String())
	}
	wantFak := "fak bench h2h run --model m --arm fak=http://127.0.0.1:8080/v1 --ledger " + ledgerDir + "/h2h-fak.jsonl --run-id r1-fak --reps 5"
	if lines[0] != wantFak {
		t.Fatalf("line 1 = %q, want %q", lines[0], wantFak)
	}
	wantLlama := "fak bench h2h run --model m --arm llama=http://127.0.0.1:8091/v1 --ledger " + ledgerDir + "/h2h-llama.jsonl --run-id r1-llama --reps 5"
	if lines[1] != wantLlama {
		t.Fatalf("line 2 = %q, want %q", lines[1], wantLlama)
	}
	for _, omitted := range []string{"--host", "--speculative", "--cold", "--decode", "--turns", "--system", "--turn-tokens", "--answer", "--timeout", "--keep-thinking", "--slots-url", "--gpu-busy", "--api-key-env"} {
		if strings.Contains(out.String(), omitted) {
			t.Fatalf("default-valued flag %s must be omitted from emitted argvs:\n%s", omitted, out.String())
		}
	}
}

func TestBenchH2HPlanJSONDefaultsAndLedgerDirFromLedger(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"plan", "--model", "m", "--arm", "fak=http://127.0.0.1:8080/v1", "--json"})
	if code != 0 {
		t.Fatalf("plan --json exit %d: %s", code, errb.String())
	}
	var env h2hPlanEnvelope
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("plan json: %v\n%s", err, out.String())
	}
	if env.Schema != h2hPlanSchema || env.Model != "m" || env.LedgerDir != ".fak/nightrun" {
		t.Fatalf("envelope = %+v", env)
	}
	if len(env.Arms) != 1 || env.Arms[0].Ledger != ".fak/nightrun/h2h-fak.jsonl" {
		t.Fatalf("arms = %+v", env.Arms)
	}
	argv := env.Arms[0].Argv
	if argv[0] != "fak" || argv[1] != "bench" || argv[2] != "h2h" || argv[3] != "run" {
		t.Fatalf("argv prefix = %v", argv[:4])
	}
	for _, absent := range []string{"--run-id", "--reps"} {
		if containsArgv(argv, absent) {
			t.Fatalf("default-valued flag %s must be omitted: %v", absent, argv)
		}
	}

	out.Reset()
	errb.Reset()
	code = runBenchH2H(&out, &errb, []string{"plan", "--model", "m", "--arm", "fak=http://127.0.0.1:8080/v1", "--ledger", "/tmp/x/combined.jsonl", "--json"})
	if code != 0 {
		t.Fatalf("plan --ledger exit %d: %s", code, errb.String())
	}
	if err := json.Unmarshal(out.Bytes(), &env); err != nil {
		t.Fatalf("plan json: %v\n%s", err, out.String())
	}
	if env.LedgerDir != "/tmp/x" || env.Arms[0].Ledger != "/tmp/x/h2h-fak.jsonl" {
		t.Fatalf("ledger dir from --ledger = %+v", env)
	}
}

func containsArgv(argv []string, flag string) bool {
	for _, a := range argv {
		if a == flag {
			return true
		}
	}
	return false
}

func TestBenchH2HPlanQuotesArgsWithSpaces(t *testing.T) {
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"plan", "--model", "my model", "--arm", "fak=http://127.0.0.1:8080/v1"})
	if code != 0 {
		t.Fatalf("plan exit %d: %s", code, errb.String())
	}
	line := strings.TrimSpace(out.String())
	if !strings.Contains(line, `--model "my model"`) {
		t.Fatalf("model arg with spaces must be quoted: %s", line)
	}
	if !strings.HasPrefix(line, "fak bench h2h run ") {
		t.Fatalf("line must start bare: %s", line)
	}
}

func TestBenchH2HGPUIdePassesWhenEnoughIdle(t *testing.T) {
	old := gpuIdleSample
	defer func() { gpuIdleSample = old }()
	gpuIdleSample = func() ([]compute.GPUStat, bool) {
		return []compute.GPUStat{
			{Index: 0, VRAMUsedBytes: 512 << 20, VRAMTotalBytes: 24576 << 20, UtilizationPct: 0},
			{Index: 1, VRAMUsedBytes: 2048 << 20, VRAMTotalBytes: 24576 << 20, UtilizationPct: 3},
		}, true
	}
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"gpu-idle",
		"--require", "2", "--max-util-pct", "5", "--max-mem-used-mib", "4096",
		"--interval", "10ms", "--timeout", "5s"})
	if code != 0 {
		t.Fatalf("gpu-idle exit %d: %s", code, errb.String())
	}
	if !strings.Contains(out.String(), "idle 0 util=0% mem=512/24576MiB") {
		t.Fatalf("output: %s", out.String())
	}
	if !strings.Contains(out.String(), "idle 1 util=3% mem=2048/24576MiB") {
		t.Fatalf("output: %s", out.String())
	}
}

func TestBenchH2HGPUIdePollsUntilIdle(t *testing.T) {
	old := gpuIdleSample
	defer func() { gpuIdleSample = old }()
	calls := 0
	busy := compute.GPUStat{Index: 0, VRAMUsedBytes: 20000 << 20, VRAMTotalBytes: 24576 << 20, UtilizationPct: 90}
	idle := compute.GPUStat{Index: 0, VRAMUsedBytes: 100 << 20, VRAMTotalBytes: 24576 << 20, UtilizationPct: 0}
	gpuIdleSample = func() ([]compute.GPUStat, bool) {
		calls++
		if calls < 3 {
			return []compute.GPUStat{busy}, true
		}
		return []compute.GPUStat{idle}, true
	}
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"gpu-idle", "--require", "1", "--interval", "10ms", "--timeout", "5s"})
	if code != 0 {
		t.Fatalf("gpu-idle exit %d: %s", code, errb.String())
	}
	if calls < 3 {
		t.Fatalf("expected at least 3 samples (busy, busy, idle), got %d", calls)
	}
	if !strings.Contains(out.String(), "idle 0 util=0% mem=100/24576MiB") {
		t.Fatalf("output: %s", out.String())
	}
}

func TestBenchH2HGPUIdeAbsentIsTypedFailure(t *testing.T) {
	old := gpuIdleSample
	defer func() { gpuIdleSample = old }()
	gpuIdleSample = func() ([]compute.GPUStat, bool) { return nil, false }
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"gpu-idle", "--interval", "10ms", "--timeout", "5s"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1: %s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "nvidia-smi absent or unreadable") {
		t.Fatalf("stderr: %s", errb.String())
	}
	if strings.Contains(out.String(), "idle ") {
		t.Fatalf("absent nvidia-smi must never print an idle pass: %s", out.String())
	}
}

func TestBenchH2HGPUIdeTimeoutReportsLastSample(t *testing.T) {
	old := gpuIdleSample
	defer func() { gpuIdleSample = old }()
	gpuIdleSample = func() ([]compute.GPUStat, bool) {
		return []compute.GPUStat{
			{Index: 0, VRAMUsedBytes: 100 << 20, VRAMTotalBytes: 24576 << 20, UtilizationPct: 0},
			{Index: 1, VRAMUsedBytes: 20000 << 20, VRAMTotalBytes: 24576 << 20, UtilizationPct: 90},
		}, true
	}
	var out, errb bytes.Buffer
	code := runBenchH2H(&out, &errb, []string{"gpu-idle",
		"--require", "2", "--max-util-pct", "5", "--max-mem-used-mib", "1024",
		"--interval", "20ms", "--timeout", "200ms"})
	if code != 1 {
		t.Fatalf("exit = %d, want 1: %s", code, errb.String())
	}
	if !strings.Contains(errb.String(), "gpu-idle: timeout after 200ms: 1/2 GPUs idle (need 2)") {
		t.Fatalf("stderr: %s", errb.String())
	}
	if !strings.Contains(errb.String(), "last sample: gpu 1 util=90% mem=20000/24576MiB") {
		t.Fatalf("stderr: %s", errb.String())
	}
}
