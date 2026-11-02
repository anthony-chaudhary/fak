package buildwitness

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
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

	"github.com/anthony-chaudhary/fak/internal/gateway"
	"github.com/anthony-chaudhary/fak/internal/inferencetopology"
	"github.com/anthony-chaudhary/fak/internal/procguard"
)

// fak-test:justify why=regression when=changed:cmd/fak/serve.go,cmd/fak/serve_topology_reconcile.go,internal/gateway/topology_reconcile.go
// fak-test:runtime integration est=5m lane=default
// TestServeTopologyShadowReconcile runs ordinary serve against owned loopback
// readiness fixtures. It is not external fleet load or hardware qualification.
func TestServeTopologyShadowReconcile(t *testing.T) {
	root, e := filepath.Abs(filepath.Join("..", ".."))
	if e != nil {
		t.Fatal(e)
	}
	dir := t.TempDir()
	binary := filepath.Join(dir, "fak-shadow")
	if runtime.GOOS == "windows" {
		binary += ".exe"
	}
	buildCtx, buildCancel := context.WithTimeout(t.Context(), 5*time.Minute)
	defer buildCancel()
	build := exec.CommandContext(buildCtx, "go", "build", "-o", binary, "./cmd/fak")
	build.Dir = root
	procguard.ConfigureProcessTreeCancel(build)
	build.WaitDelay = 10 * time.Second
	var diagnostic serveProofCLIBuffer
	build.Stdout = &diagnostic
	build.Stderr = &diagnostic
	if e = build.Run(); e != nil {
		t.Fatalf("CLI compiler error (not behavioral RED): %v\n%s", e, diagnostic.String())
	}
	if diagnostic.truncated {
		t.Fatal("build diagnostics truncated")
	}
	f, e := os.Open(filepath.Join(root, "internal/inferencetopology/testdata/runtime_search.json"))
	if e != nil {
		t.Fatal(e)
	}
	search, e := inferencetopology.DecodeRuntimeSearch(f)
	f.Close()
	if e != nil {
		t.Fatal(e)
	}
	now := time.Now().UTC()
	search.Workload.Concurrency = 4
	for i := range search.Options {
		search.Options[i].Topology.Observed.At = now.Format(time.RFC3339Nano)
		search.Options[i].Topology.Observed.TTLSeconds = 60
	}
	for i := range search.Observations {
		search.Observations[i].Decode.MeasuredAt = now.Format(time.RFC3339Nano)
		search.Observations[i].ExpiresAt = now.Add(time.Minute).Format(time.RFC3339Nano)
	}
	selected, e := inferencetopology.SearchRuntimeTopology(t.Context(), search, now)
	if e != nil || selected.Chosen == nil {
		t.Fatalf("fixture search setup: error=%v result=%+v", e, selected)
	}
	input := gateway.TopologyShadowInput{Schema: gateway.TopologyShadowSchema, Search: search, Incumbent: 3, HysteresisTicks: 2, Objective: gateway.PoolObjective{MinReplicas: 1, MaxReplicas: 3, ScaleStep: 1, QueueHigh: 4, QueueLow: 1, KVUtilHigh: .9, KVUtilLow: .5}}
	waiting, running, kv := 0.0, 0.0, .2
	for i, id := range []string{"r0", "r1", "r2"} {
		input.Workers = append(input.Workers, gateway.TopologyPressureObservation{TopologyID: selected.Chosen.Config.TopologyID, RankID: id, WorkerID: id, Incarnation: uint64(i + 1), Epoch: selected.Chosen.ObservationEpoch, Source: "owned-loopback-fixture-not-hardware", ModelRevision: search.ModelRevision, Backend: search.Backend, Kernel: search.Kernel, LayoutDigest: selected.Chosen.LayoutDigest, LinkDigest: selected.Chosen.LinkDigest, ObservedAt: now, TTLSeconds: 8, Waiting: &waiting, Running: &running, KVUtilization: &kv})
	}
	raw, e := json.Marshal(input)
	if e != nil {
		t.Fatal(e)
	}
	path := filepath.Join(dir, "shadow.json")
	if e = os.WriteFile(path, raw, 0600); e != nil {
		t.Fatal(e)
	}
	var unhealthy atomic.Bool
	args := []string{"serve", "--addr", "127.0.0.1:0", "--topology-shadow-input", path}
	for _, id := range []string{"r0", "r1", "r2"} {
		id := id
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if id == "r2" && unhealthy.Load() {
				w.WriteHeader(http.StatusServiceUnavailable)
				return
			}
			w.WriteHeader(http.StatusOK)
		}))
		defer srv.Close()
		args = append(args, "--replica-base-url", id+"="+srv.URL)
	}
	// A nonregular source is refused before the server can enter its lifecycle.
	refusalCtx, refusalCancel := context.WithTimeout(t.Context(), 10*time.Second)
	refusalArgs := append([]string(nil), args...)
	refusalArgs[4] = dir
	refusal := exec.CommandContext(refusalCtx, binary, refusalArgs...)
	refusal.Dir = dir
	refusal.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + dir, "USERPROFILE=" + dir, "XDG_CONFIG_HOME=" + dir, "XDG_CACHE_HOME=" + dir}
	procguard.ConfigureProcessTreeCancel(refusal)
	refusal.WaitDelay = 5 * time.Second
	var refusalOutput serveProofCLIBuffer
	refusal.Stdout = &refusalOutput
	refusal.Stderr = &refusalOutput
	refusalErr := refusal.Run()
	refusalCancel()
	if refusalErr == nil || !strings.Contains(refusalOutput.String(), "regular file") || refusalOutput.truncated {
		t.Fatalf("nonregular input refusal failed: %v %s", refusalErr, refusalOutput.String())
	}
	ctx, cancel := context.WithTimeout(t.Context(), 30*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, binary, args...)
	cmd.Dir = dir
	cmd.Env = []string{"PATH=" + os.Getenv("PATH"), "HOME=" + dir, "USERPROFILE=" + dir, "XDG_CONFIG_HOME=" + dir, "XDG_CACHE_HOME=" + dir, "TMPDIR=" + dir, "TMP=" + dir, "TEMP=" + dir}
	procguard.ConfigureProcessTreeCancel(cmd)
	cmd.WaitDelay = 10 * time.Second
	cmd.Stdout = io.Discard
	stderr, e := cmd.StderrPipe()
	if e != nil {
		t.Fatal(e)
	}
	if e = cmd.Start(); e != nil {
		t.Fatalf("CLI setup failed: %v", e)
	}
	reports := make(chan gateway.TopologyShadowReport, 8)
	scanDone := make(chan string, 1)
	go func() {
		scanner := bufio.NewScanner(stderr)
		scanner.Buffer(make([]byte, 4096), 1<<20)
		var log strings.Builder
		for scanner.Scan() {
			line := scanner.Text()
			if strings.HasPrefix(line, "fak topology-shadow: ") {
				var r gateway.TopologyShadowReport
				if json.Unmarshal([]byte(strings.TrimPrefix(line, "fak topology-shadow: ")), &r) == nil {
					select {
					case reports <- r:
					default:
					}
				}
			}
			if log.Len() < 1<<16 {
				log.WriteString(line)
				log.WriteByte('\n')
			}
		}
		if scanner.Err() != nil {
			log.WriteString(scanner.Err().Error())
		}
		scanDone <- log.String()
	}()
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()
	defer func() {
		cancel()
		select {
		case <-done:
		case <-time.After(12 * time.Second):
			t.Error("CLI child did not terminate")
		}
	}()
	first := false
	for {
		select {
		case r := <-reports:
			if r.Evidence != "SUPPLIED_OBSERVATIONS_SHADOW_ONLY_NO_CAPACITY_ACTION" || r.Decision.Action != gateway.ScaleActionHold || len(r.Decision.Workers) != 0 {
				t.Fatalf("false action/evidence %+v", r)
			}
			if !first {
				if r.Decision.CurrentReplicas != 3 || r.Decision.DesiredReplicas != 3 {
					t.Fatalf("ordinary first lifecycle %+v", r)
				}
				first = true
				unhealthy.Store(true)
				continue
			}
			if r.Decision.CurrentReplicas == 2 && r.Backpressure {
				if r.Decision.DesiredReplicas != 2 || r.Decision.AppliedReplicas != 2 {
					t.Fatalf("stale/unhealthy lifecycle %+v", r)
				}
				t.Log("ordinary serve fleet-health lifecycle: healthy then unhealthy/expired supplied observations; no capacity action")
				return
			}
		case err := <-done:
			// Replace the consumed terminal receipt for deferred owned-process join.
			done <- err
			t.Fatalf("ordinary serve exited before witness: %v\n%s", err, <-scanDone)
		case <-ctx.Done():
			t.Fatal("ordinary serve did not produce bounded shadow lifecycle witness")
		}
	}
}
