package buildwitness

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// fak-test:justify why=regression when=changed:cmd/fak/webbench.go,internal/webbench/serving_capacity_search.go
// fak-test:runtime integration est=5m lane=default
func TestWebbenchCapacitySearchCLIIntegration(t *testing.T) {
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	home := t.TempDir()
	binary := filepath.Join(home, "fak-capacity")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	ctx, cancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer cancel()
	build := exec.CommandContext(ctx, "go", "build", "-o", binary, "./cmd/fak")
	build.Dir = root
	procguard.ConfigureProcessTreeCancel(build)
	build.WaitDelay = 10 * time.Second
	var diagnostics serveProofCLIBuffer
	build.Stdout = &diagnostics
	build.Stderr = &diagnostics
	if err := build.Run(); err != nil || diagnostics.truncated {
		t.Fatalf("CLI compiler/setup failure, not behavioral RED: %v %s", err, diagnostics.String())
	}
	var requests, bad atomic.Int32
	var fail atomic.Bool
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		requests.Add(1)
		var body struct {
			Model string `json:"model"`
		}
		if r.Method != "POST" || r.URL.Path != "/v1/chat/completions" || r.Header.Get("Authorization") != "" || json.NewDecoder(r.Body).Decode(&body) != nil || body.Model != "synthetic-capacity" {
			bad.Add(1)
			http.Error(w, "fixture contract", 400)
			return
		}
		if fail.Load() {
			http.Error(w, "synthetic failure", 500)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream")
		fmt.Fprint(w, "data: {\"choices\":[{\"delta\":{\"content\":\"a\"}}]}\n\ndata: {\"choices\":[{\"delta\":{\"content\":\"b\"}}]}\n\ndata: [DONE]\n\n")
	}))
	defer server.Close()
	run := func(invalid bool, extra ...string) map[string]any {
		t.Helper()
		out := filepath.Join(home, "capacity.json")
		ctx, cancel := context.WithTimeout(t.Context(), 10*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, binary, "webbench", "serving", "--dataset", filepath.Join(root, "testdata/webbench/sample-tasks.jsonl"), "--tracks", "ours", "--endpoints", "ours="+server.URL+"/v1", "--model", "synthetic-capacity", "--agents", "2", "--concurrency", "2", "--engines", "ours=fixture", "--engine-receipts", "ours=sha256:"+strings.Repeat("a", 64), "--batch-capacities", "ours=2", "--capacity-sources", "ours=synthetic", "--ttft-p99-budget-ms", "1000", "--capacity-min-qps", "10", "--capacity-max-qps", "20", "--capacity-tolerance-qps", "1", "--capacity-max-probes", "2", "--capacity-max-requests", "4", "--capacity-max-duration", "3s", "--out", out)
		cmd.Args = append(cmd.Args, extra...)
		cmd.Dir = home
		cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + home, "USERPROFILE=" + home, "TMPDIR=" + home, "TMP=" + home, "TEMP=" + home, "XDG_CONFIG_HOME=" + home, "XDG_CACHE_HOME=" + home}
		procguard.ConfigureProcessTreeCancel(cmd)
		cmd.WaitDelay = 5 * time.Second
		var output serveProofCLIBuffer
		cmd.Stdout = &output
		cmd.Stderr = &output
		runErr := cmd.Run()
		if invalid {
			if runErr == nil || output.truncated || !strings.Contains(output.String(), "invalid capacity search") {
				t.Fatalf("invalid bounds not refused before dataset I/O: %v %s", runErr, output.String())
			}
			return nil
		}
		if err := runErr; err != nil || output.truncated {
			t.Fatalf("actual CLI failed: %v %s", err, output.String())
		}
		data, err := os.ReadFile(out)
		if err != nil {
			t.Fatal(err)
		}
		var report map[string]any
		if err = json.Unmarshal(data, &report); err != nil {
			t.Fatal(err)
		}
		result, ok := report["capacity_search"].(map[string]any)
		if !ok {
			t.Fatal("actual CLI did not emit capacity result")
		}
		contract := result["contract"].(map[string]any)
		if contract["model"] != "synthetic-capacity" || contract["engine"] != "fixture" || result["workload_digest"] != report["workload"].(map[string]any)["digest"] || result["qualification"] != "bounded_client_probe_not_capacity_certification" {
			t.Fatalf("identity/qualification lost: %s", data)
		}
		return result
	}
	run(true, "--capacity-max-duration", "0s", "--dataset", filepath.Join(home, "missing-dataset"))
	if requests.Load() != 0 {
		t.Fatal("invalid bounds reached HTTP consumer")
	}
	result := run(false)
	if result["status"] != "bounded" || result["best_passing_qps"] != float64(20) || requests.Load() != 4 || bad.Load() != 0 {
		t.Fatalf("bounded CLI: %+v requests=%d bad=%d", result, requests.Load(), bad.Load())
	}
	fail.Store(true)
	result = run(false)
	if result["status"] != "unknown" || result["best_passing_qps"] != nil || requests.Load() != 6 || bad.Load() != 0 {
		t.Fatalf("erroring HTTP probe claimed success: %+v requests=%d", result, requests.Load())
	}
}
